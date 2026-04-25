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
