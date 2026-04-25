"""Ingenieria de variables para el dataset de interrupciones SUNASS.

El .dta crudo trae columnas separadas para fecha y hora, y los numericos vienen
como string con cadenas vacias. Este modulo reconstruye las variables derivadas
que el mentor usa en su baseline (y que NO existen en el raw):

    - duracion_horas       diferencia entre inicio y reanudacion prevista
    - impacto              duracion_horas * N_conexiones_afectadas
    - evento_critico       impacto > 100_000 (umbral del mentor)

Tambien agrega features temporales (hora_inicio, dow, mes_num) y un encoding
ligero de categoricas de alta cardinalidad via frequency encoding.

Convenciones:
    - Entrada: polars.DataFrame tal como lo devuelve src.io.load_interrupciones
    - Salida: polars.DataFrame con columnas adicionales; no muta el input.
    - Nunca imprime; todo por logger.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

import polars as pl

logger = logging.getLogger(__name__)

# Umbral del mentor para definir evento critico (ver docs/EXPLORACION.md).
IMPACTO_CRITICO_DEFAULT = 100_000.0

# Columnas fecha y hora esperadas en el .dta.
COL_FECHA_INICIO = "Fechadeinicio"
COL_HORA_INICIO = "Horadeinicio"
COL_FECHA_FIN = "Fechaprevistaderestablecimien"
COL_HORA_FIN = "Horaprevistaderestablecimient"

COL_CONEXIONES = "Ndeconexionesdomiciliariasa"
COL_UNIDADES = "Unidadesdeusoafectadas"


def _combine_fecha_hora(
    df: pl.DataFrame, fecha_col: str, hora_col: str, alias: str
) -> pl.Expr:
    """Construye una expresion Datetime combinando fecha y hora.

    Soporta las formas en que pyreadstat puede devolver la hora de un .dta:
        - pl.Time             objeto datetime.time
        - pl.Datetime         Stata %tc (ms desde 1960); se extrae time portion
        - pl.Duration         ya es un offset desde medianoche
        - pl.Utf8             string "HH:MM:SS" o "HH:MM"
        - numerico (Int/Float) interpretado como ms desde medianoche (Stata %tcHH:MM:SS
                              sin fecha) o como segundos si el rango lo sugiere.

    Si la fecha es null, el resultado es null. Si la hora es null, usa 00:00:00.
    """
    if fecha_col not in df.columns:
        raise KeyError(f"Columna '{fecha_col}' ausente")
    if hora_col not in df.columns:
        raise KeyError(f"Columna '{hora_col}' ausente")

    fecha = pl.col(fecha_col).cast(pl.Datetime, strict=False)
    fecha_00 = fecha.dt.date().cast(pl.Datetime)
    hora_dtype = df.schema[hora_col]
    logger.debug("_combine_fecha_hora(%s): hora_dtype=%s", hora_col, hora_dtype)

    if hora_dtype == pl.Time:
        offset = pl.col(hora_col).cast(pl.Duration, strict=False)
    elif hora_dtype == pl.Datetime:
        # Stata %tc guarda "date+time" en un solo float; nos quedamos con la
        # hora como offset desde medianoche.
        offset = (
            pl.col(hora_col) - pl.col(hora_col).dt.date().cast(pl.Datetime)
        ).cast(pl.Duration, strict=False)
    elif hora_dtype == pl.Duration:
        offset = pl.col(hora_col)
    elif hora_dtype == pl.Utf8:
        hora_str = pl.col(hora_col).str.strip_chars()
        parsed = hora_str.str.to_time(format="%H:%M:%S", strict=False)
        # Fallback a "HH:MM" si to_time fallo (polars deja null):
        parsed = pl.when(parsed.is_null()).then(
            hora_str.str.to_time(format="%H:%M", strict=False)
        ).otherwise(parsed)
        offset = parsed.cast(pl.Duration, strict=False)
    elif hora_dtype.is_numeric():
        # Heuristica: si el max esperado es ~86_400_000, es ms; si es ~86_400,
        # son segundos. Delegamos a polars: cast a int + usar como ms.
        ms = pl.col(hora_col).cast(pl.Int64, strict=False)
        offset = pl.duration(milliseconds=ms)
    else:
        logger.warning("hora_dtype inesperado: %s — devolviendo solo fecha", hora_dtype)
        offset = pl.lit(None).cast(pl.Duration)

    offset_safe = pl.when(offset.is_null()).then(pl.duration(seconds=0)).otherwise(offset)
    return (fecha_00 + offset_safe).alias(alias)


def add_timestamps(df: pl.DataFrame) -> pl.DataFrame:
    """Agrega columnas `ts_inicio` y `ts_fin` como Datetime combinados.

    Si las columnas fuente no existen, las ausentes se rellenan con null y se
    registra un warning. Nunca falla en runtime por ausencia.
    """
    casts: list[pl.Expr] = []
    if COL_FECHA_INICIO in df.columns and COL_HORA_INICIO in df.columns:
        casts.append(_combine_fecha_hora(df, COL_FECHA_INICIO, COL_HORA_INICIO, "ts_inicio"))
    else:
        logger.warning("ts_inicio: faltan %s / %s", COL_FECHA_INICIO, COL_HORA_INICIO)
        casts.append(pl.lit(None, dtype=pl.Datetime).alias("ts_inicio"))

    if COL_FECHA_FIN in df.columns and COL_HORA_FIN in df.columns:
        casts.append(_combine_fecha_hora(df, COL_FECHA_FIN, COL_HORA_FIN, "ts_fin"))
    else:
        logger.warning("ts_fin: faltan %s / %s", COL_FECHA_FIN, COL_HORA_FIN)
        casts.append(pl.lit(None, dtype=pl.Datetime).alias("ts_fin"))

    return df.with_columns(casts)


def add_duracion_impacto(
    df: pl.DataFrame,
    umbral_critico: float = IMPACTO_CRITICO_DEFAULT,
) -> pl.DataFrame:
    """Agrega `duracion_horas`, `impacto`, `evento_critico`.

    - duracion_horas = (ts_fin - ts_inicio) en horas; null si alguno falta o si
      duracion < 0 (orden invertido, dato corrupto).
    - impacto = duracion_horas * N_conexiones (fallback a Unidadesdeusoafectadas
      si Ndeconexiones es null).
    - evento_critico = (impacto > umbral_critico). Null si impacto es null.

    Requiere que `add_timestamps` se haya corrido antes.
    """
    required = ("ts_inicio", "ts_fin")
    for col in required:
        if col not in df.columns:
            raise KeyError(
                f"Columna '{col}' ausente. Corre add_timestamps() antes de add_duracion_impacto()."
            )

    diff_seconds = (pl.col("ts_fin") - pl.col("ts_inicio")).dt.total_seconds()
    duracion_horas = pl.when(diff_seconds >= 0).then(diff_seconds / 3600.0).otherwise(None)

    conexiones_expr = (
        pl.col(COL_CONEXIONES).cast(pl.Float64, strict=False)
        if COL_CONEXIONES in df.columns
        else pl.lit(None, dtype=pl.Float64)
    )
    unidades_expr = (
        pl.col(COL_UNIDADES).cast(pl.Float64, strict=False)
        if COL_UNIDADES in df.columns
        else pl.lit(None, dtype=pl.Float64)
    )
    conexiones_fill = pl.coalesce([conexiones_expr, unidades_expr])

    df_out = df.with_columns(
        duracion_horas.alias("duracion_horas"),
        conexiones_fill.alias("n_afectadas"),
    ).with_columns(
        (pl.col("duracion_horas") * pl.col("n_afectadas")).alias("impacto"),
    ).with_columns(
        pl.when(pl.col("impacto").is_not_null())
        .then(pl.col("impacto") > umbral_critico)
        .otherwise(None)
        .alias("evento_critico"),
    )

    n_nulos = df_out.get_column("evento_critico").null_count()
    n_crit = int(df_out.filter(pl.col("evento_critico")).height)
    logger.info(
        "Duracion/impacto: %d criticos, %d nulos (de %d)",
        n_crit,
        n_nulos,
        df_out.height,
    )
    return df_out


def add_temporal_features(df: pl.DataFrame, ts_col: str = "ts_inicio") -> pl.DataFrame:
    """Agrega features temporales derivadas de ts_col.

    Columnas: hora, dow (0=lun), mes_num, trimestre_num, es_finde, es_madrugada.
    No sobreescribe las originales del .dta (mes, anio, trimestre).
    """
    if ts_col not in df.columns:
        raise KeyError(f"Columna '{ts_col}' ausente")

    hora = pl.col(ts_col).dt.hour()
    dow = pl.col(ts_col).dt.weekday() - 1  # polars: 1..7 (lun..dom) -> 0..6
    return df.with_columns(
        hora.alias("hora"),
        dow.alias("dow"),
        pl.col(ts_col).dt.month().alias("mes_num"),
        pl.col(ts_col).dt.quarter().alias("trimestre_num"),
        (dow >= 5).alias("es_finde"),
        ((hora >= 0) & (hora < 6)).alias("es_madrugada"),
    )


def frequency_encode(
    df: pl.DataFrame,
    columns: Sequence[str],
    suffix: str = "_freq",
) -> pl.DataFrame:
    """Codifica columnas categoricas por frecuencia relativa.

    Para cada col, agrega col{suffix} con la proporcion que representa cada
    categoria en el dataset. Util para LR/logit y arboles sin one-hot explosivo.
    """
    n_total = df.height or 1
    exprs: list[pl.Expr] = []
    for col in columns:
        if col not in df.columns:
            logger.warning("frequency_encode: columna ausente: %s", col)
            continue
        exprs.append(
            (pl.col(col).count().over(col) / n_total).alias(f"{col}{suffix}")
        )
    if not exprs:
        return df
    return df.with_columns(exprs)


def build_feature_matrix(
    df: pl.DataFrame,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    target_col: str = "evento_critico",
) -> tuple[pl.DataFrame, pl.Series]:
    """Ensambla X (features) e y (target) listos para sklearn.

    - Filtra filas con target null (no se pueden evaluar).
    - Aplica frequency_encode a las categoricas.
    - Retorna solo columnas numericas + las encodings; X como polars DataFrame.

    Args:
        df: DataFrame con features ya agregadas.
        numeric_cols: columnas numericas a incluir en X.
        categorical_cols: columnas categoricas a frequency-encodear.
        target_col: columna target (default 'evento_critico').

    Returns:
        (X, y) donde X es pl.DataFrame solo con numericos, y es pl.Series bool/int.
    """
    if target_col not in df.columns:
        raise KeyError(f"Target '{target_col}' ausente")

    df_filtered = df.filter(pl.col(target_col).is_not_null())
    df_encoded = frequency_encode(df_filtered, categorical_cols)

    encoded_names = [f"{c}_freq" for c in categorical_cols if f"{c}_freq" in df_encoded.columns]
    missing_num = [c for c in numeric_cols if c not in df_encoded.columns]
    if missing_num:
        logger.warning("build_feature_matrix: numericos ausentes %s", missing_num)
    selected_num = [c for c in numeric_cols if c in df_encoded.columns]

    feature_cols = [*selected_num, *encoded_names]
    if not feature_cols:
        raise ValueError("No hay features utilizables para X")

    X = df_encoded.select(feature_cols).fill_null(0.0)
    y = df_encoded.get_column(target_col).cast(pl.Int8)
    logger.info(
        "Feature matrix: %d filas, %d cols (%d num, %d cat-freq)",
        X.height,
        X.width,
        len(selected_num),
        len(encoded_names),
    )
    return X, y


def normalize_to_ubigeo(
    df: pl.DataFrame,
    departamento_col: str = "Departamento",
    provincia_col: str = "Provincia",
    distrito_col: str = "Distrito",
    ubigeo_table: pl.DataFrame | None = None,  # reservado para compatibilidad futura
) -> pl.DataFrame:
    """Normaliza dep/prov/dist y agrega columna `ubigeo_distrito` (6 digitos).

    Usa `ubigeos-peru` con busqueda dirigida por provincia para resolver
    homonimos: en lugar de aceptar el primer match global de un distrito,
    itera los distritos reales de la provincia dada y elige el de mayor
    similitud (rapidfuzz token_sort_ratio >= 80).

    Filas sin match -> ubigeo_distrito = null. Emite warning si nulos > 5%.

    Requiere: pip install ubigeos-peru
    """
    try:
        import ubigeos_peru as ubg
        from rapidfuzz import fuzz
    except ImportError as exc:
        raise ImportError(
            "Dependencia no instalada. Ejecuta: pip install ubigeos-peru"
        ) from exc

    for col in (departamento_col, provincia_col, distrito_col):
        if col not in df.columns:
            raise KeyError(f"Columna '{col}' ausente en el DataFrame")

    # Cache de distritos por provincia: prov_code_4d -> {ubigeo: nombre_upper}
    _prov_district_cache: dict[str, dict[str, str]] = {}

    def _get_prov_districts(prov_code: str) -> dict[str, str]:
        """Devuelve {ubigeo_6d: nombre_upper} para todos los distritos de la provincia."""
        if prov_code in _prov_district_cache:
            return _prov_district_cache[prov_code]
        result: dict[str, str] = {}
        for i in range(1, 100):
            ubigeo = prov_code + str(i).zfill(2)
            try:
                name = ubg.get_distrito(ubigeo)
                if name:
                    result[ubigeo] = name.upper()
            except Exception:
                pass
        _prov_district_cache[prov_code] = result
        return result

    def _resolve(dep: str, prov: str, dist: str) -> str | None:
        if not dist:
            return None

        # 1. Obtener prefijo de departamento (2 digitos)
        dep_code: str | None = None
        if dep:
            try:
                raw = ubg.get_ubigeo(dep, "departamentos")
                if raw:
                    dep_code = str(raw).zfill(2)[:2]
            except Exception:
                pass

        # 2. Obtener prefijo de provincia (4 digitos), validado contra dep
        prov_code: str | None = None
        if prov:
            try:
                raw = ubg.get_ubigeo(prov, "provincia")
                if raw:
                    candidate = str(raw).zfill(4)[:4]
                    # Aceptar solo si pertenece al departamento correcto
                    if dep_code and candidate[:2] == dep_code:
                        prov_code = candidate
                    elif not dep_code:
                        prov_code = candidate
            except Exception:
                pass

        # 3a. Busqueda dirigida: iterar distritos de la provincia y fuzzy-match
        if prov_code:
            candidates = _get_prov_districts(prov_code)
            if candidates:
                dist_upper = dist.upper()
                best_score = 0
                best_ubigeo: str | None = None
                for ubigeo, name in candidates.items():
                    score = fuzz.token_sort_ratio(dist_upper, name)
                    if score > best_score:
                        best_score = score
                        best_ubigeo = ubigeo
                if best_score >= 80:
                    return best_ubigeo

        # 3b. Fallback: busqueda global + validacion por departamento
        try:
            raw = ubg.get_ubigeo(dist, "distritos")
            if raw:
                ub6 = str(raw).zfill(6)
                if dep_code and ub6[:2] == dep_code:
                    return ub6
                elif not dep_code and not prov_code:
                    return ub6  # sin contexto geografico, aceptar
        except Exception:
            pass

        return None

    # Iterar combinaciones unicas
    unique_rows = (
        df.select([departamento_col, provincia_col, distrito_col])
        .unique()
        .iter_rows(named=True)
    )
    mapping: dict[tuple[str, str, str], str | None] = {}
    for row in unique_rows:
        dep  = (row[departamento_col] or "").strip()
        prov = (row[provincia_col]    or "").strip()
        dist = (row[distrito_col]     or "").strip()
        mapping[(dep, prov, dist)] = _resolve(dep, prov, dist)

    # Aplicar mapping vectorizado
    dep_list  = df.get_column(departamento_col).cast(pl.Utf8).str.strip_chars().to_list()
    prov_list = df.get_column(provincia_col).cast(pl.Utf8).str.strip_chars().to_list()
    dist_list = df.get_column(distrito_col).cast(pl.Utf8).str.strip_chars().to_list()

    ubigeos = [
        mapping.get((d or "", p or "", di or ""))
        for d, p, di in zip(dep_list, prov_list, dist_list)
    ]

    n_null = sum(1 for v in ubigeos if v is None)
    pct_null = n_null / max(len(ubigeos), 1) * 100
    if pct_null > 5.0:
        logger.warning(
            "normalize_to_ubigeo: %.1f%% de filas sin UBIGEO (%d/%d). "
            "Revisa columnas %s/%s/%s.",
            pct_null, n_null, len(ubigeos),
            departamento_col, provincia_col, distrito_col,
        )
    else:
        logger.info(
            "normalize_to_ubigeo: %.1f%% con UBIGEO (%d sin match de %d)",
            100.0 - pct_null, n_null, len(ubigeos),
        )

    return df.with_columns(
        pl.Series("ubigeo_distrito", ubigeos, dtype=pl.Utf8)
    )


# Bandas DIGESA para calidad de agua potable (violacion = fuera de rango).
_DIGESA_CLORO = (0.5, 5.0)   # mg/L
_DIGESA_PH = (6.5, 8.5)


def join_eventos_calidad(
    df_eventos: pl.DataFrame,
    df_morea: pl.DataFrame,
    df_estaciones: pl.DataFrame,
    ventana_horas_pre: float = 24.0,
    ventana_horas_post: float = 12.0,
    cloro_col: str = "cloro",
    ph_col: str = "ph",
    ts_col: str = "ts_inicio",
    estacion_id_col: str = "estacion_id",
    morea_ts_col: str = "fecha",
) -> pl.DataFrame:
    """Para cada evento de interrupcion, agrega features de calidad de agua MOREA.

    Columnas agregadas (null si no hay lecturas MOREA en el distrito/ventana):
        - cloro_pre_min / cloro_pre_max / cloro_pre_mean
        - ph_pre_min    / ph_pre_max    / ph_pre_mean
        - n_lecturas_morea_pre   (>= 0 siempre)
        - n_violaciones_pre      (lecturas fuera de bandas DIGESA)

    Estrategia de join:
        1. Estaciones MOREA se enriquecen con ubigeo via REGION/PROVINCIA/DISTRITO.
           El link sensores→estaciones usa row_index (1-based) como id implicito.
        2. Eventos se enriquecen con ubigeo si la columna aun no existe.
        3. Join por ubigeo_distrito + filtro de ventana temporal.

    Args:
        df_eventos:       interrupciones enriquecidas (requiere ts_col, Departamento,
                          Provincia, Distrito o ubigeo_distrito ya calculado).
        df_morea:         lecturas MOREA (estacion_id, cloro, ph, fecha UTC).
        df_estaciones:    catalogo estaciones MOREA (REGION/PROVINCIA/DISTRITO).
        ventana_horas_pre:  horas antes de ts_inicio a incluir en la ventana.
        ventana_horas_post: horas despues de ts_inicio a incluir en la ventana.
    """
    # ------------------------------------------------------------------
    # 1. Ubigeo de estaciones MOREA
    # ------------------------------------------------------------------
    # Las columnas geograficas en el catalogo pueden llamarse REGION o REGION
    # con tilde; normalizamos los nombres para robustez.
    col_map = {
        c.upper().replace("Ó", "O").replace("Á", "A").replace("É", "E")
        .replace("Í", "I").replace("Ú", "U"): c
        for c in df_estaciones.columns
    }
    region_col  = col_map.get("REGION",     col_map.get("REGIÓN",     None))
    prov_col    = col_map.get("PROVINCIA",  None)
    dist_col    = col_map.get("DISTRITO",   None)

    if not all([region_col, prov_col, dist_col]):
        logger.warning(
            "join_eventos_calidad: catalogo de estaciones sin columnas geo "
            "esperadas (REGION/PROVINCIA/DISTRITO). Columnas: %s",
            df_estaciones.columns,
        )
        # Devolver df con columnas nulas para no romper el pipeline
        null_cols = [
            pl.lit(None, dtype=pl.Float64).alias(c)
            for c in (
                "cloro_pre_min", "cloro_pre_max", "cloro_pre_mean",
                "ph_pre_min", "ph_pre_max", "ph_pre_mean",
            )
        ] + [
            pl.lit(0, dtype=pl.Int32).alias("n_lecturas_morea_pre"),
            pl.lit(0, dtype=pl.Int32).alias("n_violaciones_pre"),
        ]
        return df_eventos.with_columns(null_cols)

    # Renombrar columnas geo a nombres estandar para normalize_to_ubigeo
    est_geo = df_estaciones.rename({
        region_col: "Departamento",
        prov_col:   "Provincia",
        dist_col:   "Distrito",
    }).with_row_index("_est_id", offset=1)  # id 1-based

    est_geo = normalize_to_ubigeo(
        est_geo,
        departamento_col="Departamento",
        provincia_col="Provincia",
        distrito_col="Distrito",
    ).select(["_est_id", "ubigeo_distrito"])

    # ------------------------------------------------------------------
    # 2. Enriquecer lecturas MOREA con ubigeo
    # ------------------------------------------------------------------
    if estacion_id_col not in df_morea.columns:
        logger.warning(
            "join_eventos_calidad: columna '%s' ausente en df_morea.", estacion_id_col
        )
        return df_eventos.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(c) for c in (
                "cloro_pre_min", "cloro_pre_max", "cloro_pre_mean",
                "ph_pre_min", "ph_pre_max", "ph_pre_mean",
            )]
            + [pl.lit(0, dtype=pl.Int32).alias("n_lecturas_morea_pre"),
               pl.lit(0, dtype=pl.Int32).alias("n_violaciones_pre")]
        )

    morea_geo = (
        df_morea
        .with_columns(
            pl.col(estacion_id_col).cast(pl.Int64).alias("_est_id")
        )
        .join(est_geo.with_columns(pl.col("_est_id").cast(pl.Int64)), on="_est_id", how="left")
        .filter(pl.col("ubigeo_distrito").is_not_null())
    )

    # Normalizar timezone de morea a naive UTC para comparacion uniforme
    if morea_geo.schema.get(morea_ts_col) == pl.Datetime("us", "UTC"):
        morea_geo = morea_geo.with_columns(
            pl.col(morea_ts_col).dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        )

    morea_geo = morea_geo.select([
        "ubigeo_distrito",
        pl.col(morea_ts_col).alias("_morea_ts"),
        pl.col(cloro_col).alias("_cloro") if cloro_col in morea_geo.columns else pl.lit(None, dtype=pl.Float64).alias("_cloro"),
        pl.col(ph_col).alias("_ph") if ph_col in morea_geo.columns else pl.lit(None, dtype=pl.Float64).alias("_ph"),
    ])

    # ------------------------------------------------------------------
    # 3. Ubigeo de eventos (si no esta calculado)
    # ------------------------------------------------------------------
    if "ubigeo_distrito" not in df_eventos.columns:
        geo_cols = {"Departamento", "Provincia", "Distrito"}
        if geo_cols.issubset(set(df_eventos.columns)):
            df_eventos = normalize_to_ubigeo(df_eventos)
        else:
            logger.warning(
                "join_eventos_calidad: df_eventos sin ubigeo_distrito ni columnas geo. "
                "Todas las features MOREA seran null."
            )
            return df_eventos.with_columns(
                [pl.lit(None, dtype=pl.Float64).alias(c) for c in (
                    "cloro_pre_min", "cloro_pre_max", "cloro_pre_mean",
                    "ph_pre_min", "ph_pre_max", "ph_pre_mean",
                )]
                + [pl.lit(0, dtype=pl.Int32).alias("n_lecturas_morea_pre"),
                   pl.lit(0, dtype=pl.Int32).alias("n_violaciones_pre")]
            )

    # ------------------------------------------------------------------
    # 4. Join por ubigeo + ventana temporal
    # ------------------------------------------------------------------
    pre_us  = int(ventana_horas_pre  * 3_600_000_000)  # microsegundos
    post_us = int(ventana_horas_post * 3_600_000_000)

    ev = (
        df_eventos
        .with_row_index("_ev_idx")
        .filter(pl.col("ubigeo_distrito").is_not_null() & pl.col(ts_col).is_not_null())
        .select([
            "_ev_idx",
            "ubigeo_distrito",
            pl.col(ts_col)
              .cast(pl.Datetime("us"))
              .dt.replace_time_zone(None)
              .alias("_ts"),
        ])
        .with_columns([
            (pl.col("_ts") - pl.duration(microseconds=pre_us)).alias("_win_start"),
            (pl.col("_ts") + pl.duration(microseconds=post_us)).alias("_win_end"),
        ])
    )

    # join morea → events on ubigeo, then filter time window
    combined = (
        morea_geo.lazy()
        .join(ev.lazy(), on="ubigeo_distrito", how="inner")
        .filter(
            (pl.col("_morea_ts") >= pl.col("_win_start")) &
            (pl.col("_morea_ts") <= pl.col("_win_end"))
        )
        .with_columns(
            (
                (pl.col("_cloro") < _DIGESA_CLORO[0]) | (pl.col("_cloro") > _DIGESA_CLORO[1]) |
                (pl.col("_ph") < _DIGESA_PH[0]) | (pl.col("_ph") > _DIGESA_PH[1])
            ).cast(pl.Int8).alias("_viola")
        )
        .group_by("_ev_idx")
        .agg([
            pl.col("_cloro").min().alias("cloro_pre_min"),
            pl.col("_cloro").max().alias("cloro_pre_max"),
            pl.col("_cloro").mean().alias("cloro_pre_mean"),
            pl.col("_ph").min().alias("ph_pre_min"),
            pl.col("_ph").max().alias("ph_pre_max"),
            pl.col("_ph").mean().alias("ph_pre_mean"),
            pl.len().alias("n_lecturas_morea_pre"),
            pl.col("_viola").sum().alias("n_violaciones_pre"),
        ])
        .collect()
    )

    # ------------------------------------------------------------------
    # 5. Left join de vuelta a todos los eventos
    # ------------------------------------------------------------------
    result = (
        df_eventos
        .with_row_index("_ev_idx")
        .join(combined, on="_ev_idx", how="left")
        .drop("_ev_idx")
        .with_columns([
            pl.col("n_lecturas_morea_pre").fill_null(0).cast(pl.Int32),
            pl.col("n_violaciones_pre").fill_null(0).cast(pl.Int32),
        ])
    )

    n_matched = combined.height
    pct = n_matched / max(df_eventos.height, 1) * 100
    logger.info(
        "join_eventos_calidad: %d/%d eventos con lecturas MOREA (%.1f%%), "
        "ventana=[-%gh, +%gh]",
        n_matched, df_eventos.height, pct, ventana_horas_pre, ventana_horas_post,
    )
    return result


_SENAMHI_SENTINEL = -99.9
_SENAMHI_CLIMATE_COLS: tuple[str, ...] = ("precip_acum", "tmax", "tmin")


def clean_senamhi_missing(
    df: pl.DataFrame,
    sentinel: float = _SENAMHI_SENTINEL,
    numeric_cols: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Reemplaza el centinela de dato faltante SENAMHI (-99.9) por null.

    Args:
        df:           DataFrame con datos SENAMHI.
        sentinel:     Valor centinela a reemplazar (default -99.9).
        numeric_cols: Columnas a limpiar; si None, aplica a todas las numéricas.

    Returns:
        DataFrame con los mismos tipos y shape pero centinelas → null.
        df.height no cambia.
    """
    if numeric_cols is None:
        cols = [c for c, t in df.schema.items() if t.is_numeric()]
    else:
        cols = [c for c in numeric_cols if c in df.columns]

    if not cols:
        return df

    exprs = [
        pl.when(pl.col(c) == sentinel)
        .then(None)
        .otherwise(pl.col(c))
        .alias(c)
        for c in cols
    ]
    result = df.with_columns(exprs)

    n_replaced = sum(
        int((df.get_column(c) == sentinel).sum())
        for c in cols
    )
    logger.debug("clean_senamhi_missing: %d centinelas → null", n_replaced)
    return result


def ensure_daily_frequency(
    df: pl.DataFrame,
    date_col: str = "fecha",
    group_cols: tuple[str, ...] = ("estacion",),
) -> pl.DataFrame:
    """Asegura que cada serie (por estación) tenga frecuencia diaria completa.

    Genera todas las fechas entre min y max por grupo y hace left join con los
    datos originales. NO elimina filas existentes. Las fechas faltantes aparecen
    con valores null en variables climáticas.

    Args:
        df:         DataFrame con al menos date_col y group_cols.
        date_col:   Nombre de la columna de fecha (dtype pl.Date o casteable).
        group_cols: Columnas que identifican cada serie independiente.

    Returns:
        DataFrame con mismas columnas pero sin huecos temporales,
        ordenable por (group_cols..., date_col). df.height puede crecer.
    """
    if date_col not in df.columns:
        raise KeyError(f"Columna de fecha '{date_col}' ausente")
    for gc in group_cols:
        if gc not in df.columns:
            raise KeyError(f"Columna de grupo '{gc}' ausente")

    if df.schema[date_col] != pl.Date:
        df = df.with_columns(pl.col(date_col).cast(pl.Date, strict=False))

    bounds = df.group_by(list(group_cols)).agg(
        pl.col(date_col).min().alias("_min_d"),
        pl.col(date_col).max().alias("_max_d"),
    )

    spine_parts: list[pl.DataFrame] = []
    for row in bounds.iter_rows(named=True):
        min_d = row["_min_d"]
        max_d = row["_max_d"]
        if min_d is None or max_d is None:
            continue
        dates = pl.date_range(min_d, max_d, interval="1d", eager=True)
        part = pl.DataFrame({date_col: dates})
        for gc in group_cols:
            part = part.with_columns(
                pl.lit(row[gc]).cast(df.schema[gc]).alias(gc)
            )
        spine_parts.append(part)

    if not spine_parts:
        logger.warning("ensure_daily_frequency: no se pudo construir el spine (sin datos)")
        return df

    spine = pl.concat(spine_parts)

    result = spine.join(df, on=[date_col, *list(group_cols)], how="left")

    n_inserted = result.height - df.height
    logger.info(
        "ensure_daily_frequency: %d filas originales + %d fechas insertadas = %d total",
        df.height,
        n_inserted,
        result.height,
    )
    return result


def add_climate_lags(
    df: pl.DataFrame,
    target_cols: tuple[str, ...] = _SENAMHI_CLIMATE_COLS,
    lags: tuple[int, ...] = (1, 3, 7, 14, 30),
    group_col: str = "estacion",
) -> pl.DataFrame:
    """Agrega columnas de lag temporal para variables climáticas SENAMHI.

    Para cada combinación (col, lag) agrega `{col}_lag{lag}` usando
    `.shift(lag).over(group_col)` para respetar los límites de cada estación.
    El lag de hoy es el valor de hace N días reales (requiere frecuencia diaria
    completa; ver ensure_daily_frequency).

    Args:
        df:          DataFrame climático ordenado por (group_col, fecha).
        target_cols: Columnas sobre las que calcular lags.
        lags:        Tamaños de lag en días.
        group_col:   Columna de agrupación (default "estacion").

    Returns:
        df con columnas adicionales `{col}_lag{n}`. df.height no cambia.
    """
    use_over = group_col in df.columns
    exprs: list[pl.Expr] = []

    for col in target_cols:
        if col not in df.columns:
            logger.warning("add_climate_lags: columna '%s' ausente, se omite", col)
            continue
        for lag in lags:
            shifted = pl.col(col).shift(lag)
            expr = (
                shifted.over(group_col).alias(f"{col}_lag{lag}")
                if use_over
                else shifted.alias(f"{col}_lag{lag}")
            )
            exprs.append(expr)

    if not exprs:
        return df

    result = df.with_columns(exprs)
    logger.debug(
        "add_climate_lags: +%d columnas (%d cols × %d lags)",
        len(exprs),
        sum(1 for c in target_cols if c in df.columns),
        len(lags),
    )
    return result


def add_climate_rollings(
    df: pl.DataFrame,
    target_cols: tuple[str, ...] = _SENAMHI_CLIMATE_COLS,
    windows: tuple[int, ...] = (3, 7, 14, 30),
    group_col: str = "estacion",
) -> pl.DataFrame:
    """Agrega estadísticas rolling (media, std, max, min) sin data leakage.

    Aplica shift(1) antes de cada rolling para que la ventana solo incluya
    información pasada respecto a la fila actual (sin fuga del presente).

    Prerequisito: df debe estar ordenado por (group_col, fecha) con frecuencia
    diaria completa (ver ensure_daily_frequency).

    Args:
        df:          DataFrame climático ordenado.
        target_cols: Columnas sobre las que calcular rollings.
        windows:     Tamaños de ventana en días.
        group_col:   Columna de agrupación (default "estacion").

    Returns:
        df con columnas `{col}_roll{w}_{mean|std|max|min}`. df.height no cambia.
    """
    present_cols = [c for c in target_cols if c in df.columns]
    if not present_cols:
        logger.warning("add_climate_rollings: ninguna columna target encontrada")
        return df

    missing = set(target_cols) - set(present_cols)
    if missing:
        logger.warning("add_climate_rollings: columnas ausentes: %s", missing)

    use_over = group_col in df.columns

    shift_exprs = [
        (
            pl.col(c).shift(1).over(group_col).alias(f"_shft_{c}")
            if use_over
            else pl.col(c).shift(1).alias(f"_shft_{c}")
        )
        for c in present_cols
    ]
    df_shifted = df.with_columns(shift_exprs)

    _STATS: list[tuple[str, str]] = [
        ("mean", "rolling_mean"),
        ("std", "rolling_std"),
        ("max", "rolling_max"),
        ("min", "rolling_min"),
    ]
    roll_exprs: list[pl.Expr] = []
    for col in present_cols:
        shft_col = f"_shft_{col}"
        for w in windows:
            for stat_suffix, method_name in _STATS:
                base_expr = getattr(pl.col(shft_col), method_name)(window_size=w)
                expr = (
                    base_expr.over(group_col).alias(f"{col}_roll{w}_{stat_suffix}")
                    if use_over
                    else base_expr.alias(f"{col}_roll{w}_{stat_suffix}")
                )
                roll_exprs.append(expr)

    result = (
        df_shifted
        .with_columns(roll_exprs)
        .drop([f"_shft_{c}" for c in present_cols])
    )
    logger.debug(
        "add_climate_rollings: +%d columnas (%d cols × %d ventanas × 4 stats)",
        len(roll_exprs),
        len(present_cols),
        len(windows),
    )
    return result


def add_climate_interactions(df: pl.DataFrame) -> pl.DataFrame:
    """Agrega features de interacción climática derivadas.

    Columnas generadas (si los inputs están presentes):
        - rango_termico                = tmax - tmin
        - indice_calor                 = tmax * humedad  (solo si humedad existe)
        - dias_consecutivos_sin_lluvia = días pasados consecutivos sin lluvia,
          calculado sin leakage (usa el precipitado del día anterior).
          Null para fechas faltantes (NaN) en precip_acum → tratado como sin lluvia.

    Prerequisito: df debe estar ordenado por (estacion, fecha) con frecuencia
    diaria completa (ver ensure_daily_frequency). df.height no cambia.

    Returns:
        df con columnas adicionales. df.height no cambia.
    """
    exprs: list[pl.Expr] = []

    if "tmax" in df.columns and "tmin" in df.columns:
        exprs.append((pl.col("tmax") - pl.col("tmin")).alias("rango_termico"))

    if "tmax" in df.columns and "humedad" in df.columns:
        exprs.append((pl.col("tmax") * pl.col("humedad")).alias("indice_calor"))

    df_out = df.with_columns(exprs) if exprs else df

    # Implementación sin leakage usando shift(1):
    #   1. _prev_rain: indicador de si el día ANTERIOR tuvo lluvia
    #   2. _streak_id: cumsum de _prev_rain por estación → identificador único
    #      por cada racha seca (se incrementa cada vez que el día anterior llovió)
    #   3. Posición 0-indexed dentro de (estacion, _streak_id) = días consecutivos
    #      sin lluvia hasta hoy (sin incluir hoy).
    if "precip_acum" in df_out.columns:
        use_over = "estacion" in df_out.columns

        prev_precip = (
            pl.col("precip_acum").fill_null(0.0).shift(1).over("estacion")
            if use_over
            else pl.col("precip_acum").fill_null(0.0).shift(1)
        )
        df_out = df_out.with_columns(
            (prev_precip > 0).cast(pl.Int32).alias("_prev_rain")
        )

        streak_id_expr = (
            pl.col("_prev_rain").cum_sum().over("estacion")
            if use_over
            else pl.col("_prev_rain").cum_sum()
        )
        df_out = df_out.with_columns(streak_id_expr.alias("_streak_id"))

        over_keys = ["estacion", "_streak_id"] if use_over else ["_streak_id"]
        df_out = df_out.with_columns(
            pl.int_range(pl.len()).over(over_keys).alias("dias_consecutivos_sin_lluvia")
        ).drop(["_prev_rain", "_streak_id"])

    added = [
        c for c in ("rango_termico", "indice_calor", "dias_consecutivos_sin_lluvia")
        if c in df_out.columns and c not in df.columns
    ]
    logger.debug("add_climate_interactions: columnas agregadas: %s", added)
    return df_out


def build_senamhi_features(
    df: pl.DataFrame,
    date_col: str = "fecha",
    group_col: str = "estacion",
    target_cols: tuple[str, ...] = _SENAMHI_CLIMATE_COLS,
    lags: tuple[int, ...] = (1, 3, 7, 14, 30),
    windows: tuple[int, ...] = (3, 7, 14, 30),
    sentinel: float = _SENAMHI_SENTINEL,
) -> pl.DataFrame:
    """Pipeline completo de feature engineering para datos SENAMHI diarios.

    Orden garantizado:
        1. clean_senamhi_missing   → -99.9 a null
        2. ensure_daily_frequency  → rellena huecos temporales por estación
        3. sort por (group_col, date_col)
        4. add_climate_lags        → lags temporales sin leakage
        5. add_climate_rollings    → ventanas rolling sin leakage
        6. add_climate_interactions → rango térmico, índice calor, racha seca

    Args:
        df:          DataFrame SENAMHI con columnas fecha, estacion y climáticas.
        date_col:    Columna de fecha.
        group_col:   Columna de agrupación por estación.
        target_cols: Columnas climáticas objetivo para lags y rollings.
        lags:        Tamaños de lag en días.
        windows:     Tamaños de ventana rolling en días.
        sentinel:    Valor centinela a reemplazar por null.

    Returns:
        DataFrame enriquecido con todas las features. df.height puede crecer
        (por fechas insertadas) pero nunca encoge.
    """
    return (
        df
        .pipe(clean_senamhi_missing, sentinel=sentinel)
        .pipe(ensure_daily_frequency, date_col=date_col, group_cols=(group_col,))
        .sort([group_col, date_col])
        .pipe(add_climate_lags, target_cols=target_cols, lags=lags, group_col=group_col)
        .pipe(add_climate_rollings, target_cols=target_cols, windows=windows, group_col=group_col)
        .pipe(add_climate_interactions)
    )
