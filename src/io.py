"""Carga y normalizacion de los datasets oficiales del Datathon SUNASS 2026.

Fuentes soportadas:
    - Interrupciones (.dta)           - historico_interrupciones_limpio.dta
    - MOREA sensores (.parquet)       - datos_morea.parquet
    - MOREA estaciones (.xlsx)        - ubicacion_estaciones_MOREA.xlsx
    - Dataset del concurso (.csv/.xlsx) — formato sintetico 12 meses del dia D
    - SENAMHI diario (.csv/.xlsx)     - formato (Año, Mes, Dia, Precip, Tmax, Tmin)

Convencion: los dataframes se devuelven como polars.DataFrame con nulos reales
y tipos correctos. Los numericos que vienen como string con "" se convierten a
Float64. Las fechas se parsean a Datetime.
"""
from __future__ import annotations

import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pyreadstat
from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)

# Columnas conocidas del .dta de interrupciones con quirks a manejar.
_INTERRUP_NUMERIC_AS_STRING = (
    "Ndeconexionesdomiciliariasa",
    "Unidadesdeusoafectadas",
)

# Prefijos que sugieren columna temporal en el .dta (auto-deteccion).
_DATE_COL_PREFIXES = ("Fecha", "fecha", "date", "Date", "FECHA")

_EMPTY_TOKENS = ("", " ", "NA", "N/A", "null", "None", ".")

# Formatos de fecha a probar en cascada para columnas que vienen como str.
# El .dta de interrupciones trae fechas como string (Stata export crudo); el
# formato real depende de como se exporto. Probamos los mas comunes en LATAM.
_DATE_FORMATS = (
    # Año 4 digitos (Fechadeinicio, Fechaprevistaderestablecimien).
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y",
    "%Y/%m/%d",
    "%d.%m.%Y",
    # Año 2 digitos (Fechaderegistropreliminar viene '01/01/22 01:07:24').
    "%d/%m/%y %H:%M:%S",
    "%d/%m/%y",
    "%d-%m-%y %H:%M:%S",
    "%d-%m-%y",
)

# Engine explicito para xlsx; polars 1.40 pide fastexcel por default y no lo
# tenemos, asi que forzamos openpyxl que si esta instalado.
_XLSX_ENGINE = "openpyxl"


@dataclass(frozen=True)
class DatasetPaths:
    """Rutas resueltas a los datasets oficiales."""

    interrupciones: Path
    morea_parquet: Path
    morea_estaciones: Path


def paths_from_env(env_file: Path | str | None = None) -> DatasetPaths:
    """Construye DatasetPaths desde variables de entorno del .env.

    Las rutas relativas en el .env se resuelven contra el directorio donde
    esta el .env (no contra cwd). Esto permite invocar el pipeline desde
    cualquier cwd — scripts en repo/, notebooks en analisis/, Quarto en
    reports/ — sin reescribir el .env. Rutas absolutas se respetan tal cual.

    Args:
        env_file: ruta al .env. Si None, find_dotenv camina hacia arriba
            desde cwd hasta encontrarlo.

    Returns:
        DatasetPaths con rutas absolutas.

    Raises:
        KeyError: si alguna variable requerida no esta definida.
    """
    if env_file is not None:
        dotenv_path = Path(env_file).resolve()
    else:
        found = find_dotenv(usecwd=True)
        dotenv_path = Path(found).resolve() if found else Path.cwd() / ".env"

    load_dotenv(dotenv_path=dotenv_path if dotenv_path.exists() else None)

    base = dotenv_path.parent if dotenv_path.exists() else Path.cwd()

    def _resolve(var: str) -> Path:
        raw = Path(os.environ[var])
        return (raw if raw.is_absolute() else (base / raw)).resolve()

    try:
        return DatasetPaths(
            interrupciones=_resolve("INTERRUPCIONES_PATH"),
            morea_parquet=_resolve("MOREA_PARQUET_PATH"),
            morea_estaciones=_resolve("MOREA_ESTACIONES_PATH"),
        )
    except KeyError as exc:
        raise KeyError(
            f"Variable {exc.args[0]} no definida. Copia .env.example a .env y edita las rutas."
        ) from exc


def _snake_case(name: str) -> str:
    """Convierte PascalCase/CamelCase/espacios a snake_case ASCII."""
    s1 = re.sub(r"[\s\-]+", "_", name.strip())
    s2 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", s1)
    s3 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s2)
    return s3.lower().strip("_")


def _to_numeric(series: pl.Series) -> pl.Series:
    """Convierte una serie (probablemente Utf8) a Float64 con nulos reales."""
    if series.dtype.is_numeric():
        return series
    return (
        series.cast(pl.Utf8, strict=False)
        .str.strip_chars()
        .replace({tok: None for tok in _EMPTY_TOKENS})
        .cast(pl.Float64, strict=False)
    )


def _ensure_datetime(series: pl.Series) -> pl.Series:
    """Garantiza que la serie sea Datetime. No-op si ya lo es."""
    if series.dtype in (pl.Datetime, pl.Date):
        return series.cast(pl.Datetime, strict=False)
    return series.cast(pl.Datetime, strict=False)


def _parse_date_cascade(col: str) -> pl.Expr:
    """Parsea una columna Utf8 de fecha probando _DATE_FORMATS en cascada.

    El primer formato que no produzca null se queda con el valor; si ninguno
    parsea, el resultado final es null. Evita el fallo silencioso de
    `cast(pl.Datetime, strict=False)` cuando el string no es ISO.
    """
    base = pl.col(col).cast(pl.Utf8, strict=False).str.strip_chars()
    result: pl.Expr = base.str.to_datetime(format=_DATE_FORMATS[0], strict=False)
    for fmt in _DATE_FORMATS[1:]:
        fallback = base.str.to_datetime(format=fmt, strict=False)
        result = pl.when(result.is_null()).then(fallback).otherwise(result)
    return result.alias(col)


def load_interrupciones(path: Path | str | None = None) -> pl.DataFrame:
    """Carga el historico de interrupciones desde .dta y normaliza tipos.

    Args:
        path: ruta al archivo .dta. Si None, se toma de INTERRUPCIONES_PATH.

    Returns:
        polars.DataFrame con numericos-como-string castead os a Float64,
        fechas parseadas a Datetime, y columnas originales preservadas.

    Raises:
        FileNotFoundError: si la ruta no existe.
    """
    resolved = Path(path) if path else paths_from_env().interrupciones
    if not resolved.exists():
        raise FileNotFoundError(f"No existe {resolved}")

    logger.info("Cargando interrupciones desde %s", resolved)
    df_pd, meta = pyreadstat.read_dta(str(resolved))
    df = pl.from_pandas(df_pd)
    logger.debug("Metadata Stata: %d columnas, encoding=%s", len(meta.column_names), meta.file_encoding)

    numeric_casts: list[pl.Expr] = []
    for col in _INTERRUP_NUMERIC_AS_STRING:
        if col in df.columns:
            numeric_casts.append(_to_numeric(df.get_column(col)).alias(col))
        else:
            logger.debug("Columna numerica-string esperada ausente: %s", col)

    # Auto-deteccion de columnas de fecha por prefijo (Fecha*, date*, etc).
    date_cols_str = [
        c for c in df.columns
        if any(c.startswith(pref) for pref in _DATE_COL_PREFIXES)
        and df.schema[c] == pl.Utf8
    ]
    date_casts: list[pl.Expr] = [_parse_date_cascade(c) for c in date_cols_str]
    if date_cols_str:
        logger.info(
            "Columnas de fecha auto-detectadas (parse en cascada): %s",
            date_cols_str,
        )

    exprs = [e for e in (*numeric_casts, *date_casts)]
    if exprs:
        df = df.with_columns(exprs)

    logger.info("interrupciones cargadas: %d filas x %d cols", df.height, df.width)
    return df


def load_morea(
    parquet_path: Path | str | None = None,
    estaciones_path: Path | str | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Carga sensores MOREA + catalogo de estaciones.

    Args:
        parquet_path: datos_morea.parquet. Si None, de MOREA_PARQUET_PATH.
        estaciones_path: ubicacion_estaciones_MOREA.xlsx. Si None, de MOREA_ESTACIONES_PATH.

    Returns:
        (df_sensores, df_estaciones) polars.DataFrame.
    """
    if parquet_path is None or estaciones_path is None:
        env_paths = paths_from_env()
        resolved_parquet = Path(parquet_path) if parquet_path else env_paths.morea_parquet
        resolved_estaciones = (
            Path(estaciones_path) if estaciones_path else env_paths.morea_estaciones
        )
    else:
        resolved_parquet = Path(parquet_path)
        resolved_estaciones = Path(estaciones_path)

    if not resolved_parquet.exists():
        raise FileNotFoundError(f"No existe {resolved_parquet}")
    if not resolved_estaciones.exists():
        raise FileNotFoundError(f"No existe {resolved_estaciones}")

    logger.info("Cargando MOREA parquet desde %s", resolved_parquet)
    df_sensores = pl.read_parquet(resolved_parquet)

    logger.info("Cargando estaciones MOREA desde %s", resolved_estaciones)
    df_estaciones = pl.read_excel(resolved_estaciones, engine=_XLSX_ENGINE)

    logger.info(
        "MOREA: %d lecturas, %d estaciones, %d columnas sensores",
        df_sensores.height,
        df_estaciones.height,
        df_sensores.width,
    )
    return df_sensores, df_estaciones


def load_datathon_tabular(
    path: Path | str,
    sheet: str | int | None = None,
    normalize_columns: bool = True,
) -> pl.DataFrame:
    """Carga un CSV o XLSX sintetico del concurso.

    Args:
        path: ruta al archivo.
        sheet: solo para xlsx; nombre o indice de hoja. None usa la primera.
        normalize_columns: si True, renombra columnas a snake_case.

    Returns:
        polars.DataFrame con columnas normalizadas y tipos inferidos.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"No existe {resolved}")

    suffix = resolved.suffix.lower()
    logger.info("Cargando dataset concurso (%s) desde %s", suffix, resolved)
    if suffix == ".csv":
        df = pl.read_csv(resolved, try_parse_dates=True, infer_schema_length=10_000)
    elif suffix in (".xlsx", ".xls"):
        if isinstance(sheet, int):
            df = pl.read_excel(resolved, sheet_id=sheet + 1, engine=_XLSX_ENGINE)
        elif isinstance(sheet, str):
            df = pl.read_excel(resolved, sheet_name=sheet, engine=_XLSX_ENGINE)
        else:
            df = pl.read_excel(resolved, engine=_XLSX_ENGINE)
    else:
        raise ValueError(f"Extension no soportada: {suffix}. Usa .csv, .xlsx o .xls.")

    if normalize_columns:
        df = df.rename({c: _snake_case(c) for c in df.columns})

    logger.info("dataset concurso cargado: %d filas x %d cols", df.height, df.width)
    return df


def join_morea_estaciones(
    sensores: pl.DataFrame,
    estaciones: pl.DataFrame,
    sensor_key: str = "estacion_id",
    station_key: str = "ESTACIÓN",
) -> pl.DataFrame:
    """Join defensivo entre sensores y estaciones.

    Maneja diferencias de mayusculas/espacios en la llave y registra cuantas
    estaciones no matchearon.
    """
    if sensor_key not in sensores.columns:
        raise KeyError(f"'{sensor_key}' ausente en sensores. Cols: {sensores.columns}")
    if station_key not in estaciones.columns:
        raise KeyError(f"'{station_key}' ausente en estaciones. Cols: {estaciones.columns}")

    sensores_norm = sensores.with_columns(
        pl.col(sensor_key).cast(pl.Utf8).str.strip_chars().str.to_uppercase().alias("_join_key")
    )
    estaciones_norm = estaciones.with_columns(
        pl.col(station_key).cast(pl.Utf8).str.strip_chars().str.to_uppercase().alias("_join_key")
    )

    joined = sensores_norm.join(estaciones_norm, on="_join_key", how="left", suffix="_est")

    # Polars solo agrega el suffix "_est" cuando hay colision de nombres. Si no
    # hay colision, la columna del lado derecho conserva su nombre original.
    probe_col = f"{station_key}_est" if f"{station_key}_est" in joined.columns else station_key
    if probe_col in joined.columns:
        sin_match = (
            joined.filter(pl.col(probe_col).is_null()).get_column("_join_key").unique()
        )
        if sin_match.len() > 0:
            logger.warning(
                "Estaciones sin catalogo (%d de %d llaves): %s",
                sin_match.len(),
                joined.get_column("_join_key").n_unique(),
                sin_match.head(5).to_list(),
            )
    else:
        logger.warning(
            "No se pudo verificar match: columna '%s' ausente del join. "
            "Revisa que sensor_key y station_key tengan valores comparables.",
            station_key,
        )

    return joined.drop("_join_key")


def join_morea_by_row_index(
    sensores: pl.DataFrame,
    estaciones: pl.DataFrame,
    sensor_key: str = "estacion_id",
) -> pl.DataFrame:
    """Join sensores <-> estaciones asumiendo que `estacion_id` (entero 1..N)
    corresponde al indice de fila (1-based) del catalogo Excel.

    Esta funcion existe porque los ids numericos de sensores no coinciden con
    los nombres ESTACION del catalogo — no hay llave natural comun. La unica
    asuncion razonable es que el orden de las filas del Excel coincide con la
    enumeracion 1..26 de MOREA. El usuario debe **verificar esto** contra una
    estacion conocida antes de confiar en los resultados.

    Args:
        sensores: df con columna `estacion_id` numerica (1..N).
        estaciones: df del Excel MOREA, N filas, preservado en orden original.
        sensor_key: nombre de la columna entera en sensores.

    Returns:
        DataFrame de sensores con columnas de estaciones agregadas.
    """
    if sensor_key not in sensores.columns:
        raise KeyError(f"'{sensor_key}' ausente en sensores")

    estaciones_indexed = estaciones.with_row_index(name="_row", offset=1).with_columns(
        pl.col("_row").cast(sensores.schema[sensor_key], strict=False).alias("_row")
    )

    joined = sensores.join(
        estaciones_indexed, left_on=sensor_key, right_on="_row", how="left"
    )
    n_sens_ids = sensores.get_column(sensor_key).n_unique()
    n_est_rows = estaciones.height
    logger.info(
        "join_morea_by_row_index: %d ids en sensores, %d filas en catalogo. "
        "Asume orden Excel == enumeracion MOREA; VERIFICAR con una estacion conocida.",
        n_sens_ids,
        n_est_rows,
    )
    if n_sens_ids > n_est_rows:
        logger.warning(
            "Sensores tiene mas ids (%d) que filas en catalogo (%d). Ids sin match tendran nulls.",
            n_sens_ids,
            n_est_rows,
        )
    return joined


# --------------------------------------------------------------- SENAMHI diario

# Diccionario de sinonimos para mapear variantes de headers al nombre canonico.
# Se matchea despues de normalizar (sin acentos, minusculas, strip, "_" por espacios).
_SENAMHI_HEADER_ALIAS: dict[str, tuple[str, ...]] = {
    "year": ("year", "anio", "año", "a"),
    "month": ("month", "mes", "b"),
    "day": ("day", "dia", "c"),
    "precip_acum": (
        "precipitacion_acumulada",
        "precipitacion",
        "precip_acumulada",
        "precip",
        "pp",
        "d",
    ),
    "tmax": (
        "temperatura_maxima",
        "temp_maxima",
        "tmax",
        "t_max",
        "tx",
        "e",
    ),
    "tmin": (
        "temperatura_minima",
        "temp_minima",
        "tmin",
        "t_min",
        "tn",
        "f",
    ),
}

# Bandas fisicamente posibles para filtrar errores de estacion SENAMHI.
_SENAMHI_PHYS_RANGES: dict[str, tuple[float, float]] = {
    "precip_acum": (0.0, 500.0),  # mm/dia; records mundiales estan bajo 2000 pero local <500
    "tmax": (-10.0, 50.0),
    "tmin": (-25.0, 40.0),
}


def _ascii_lower(text: str) -> str:
    """Quita acentos y lowercasea para matching de headers."""
    if text is None:
        return ""
    nfkd = unicodedata.normalize("NFKD", str(text))
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    return stripped.strip().lower().replace(" ", "_")


def _canonical_senamhi_headers(columns: list[str]) -> dict[str, str]:
    """Mapea columnas originales a nombres canonicos SENAMHI.

    Devuelve un dict original -> canonico solo para columnas reconocidas.
    """
    mapping: dict[str, str] = {}
    seen_canonical: set[str] = set()
    for original in columns:
        key = _ascii_lower(original)
        for canonical, aliases in _SENAMHI_HEADER_ALIAS.items():
            if canonical in seen_canonical:
                continue
            if key in aliases:
                mapping[original] = canonical
                seen_canonical.add(canonical)
                break
    return mapping


def load_senamhi_daily(
    path: Path | str,
    sheet: str | int | None = None,
    enforce_physical_range: bool = True,
) -> pl.DataFrame:
    """Carga serie diaria SENAMHI con schema Año/Mes/Dia + precip + tmax + tmin.

    Args:
        path: ruta al CSV o XLSX. Headers pueden venir con tildes o en mayusculas.
        sheet: para XLSX, nombre o indice (1-based por compat; None usa el primero).
        enforce_physical_range: si True, setea a null los valores fuera de bandas fisicas
            (tmax [-10,50], tmin [-25,40], precip [0,500]). No elimina filas — se pierden
            solo los valores imposibles para que el imputador corriente downstream los vea.

    Returns:
        pl.DataFrame con columnas: fecha (Datetime), year, month, day (Int32),
        precip_acum, tmax, tmin (Float64). Columnas adicionales del archivo se
        preservan con sus nombres originales.

    Raises:
        FileNotFoundError si la ruta no existe.
        KeyError si no se pueden identificar year, month, day en los headers.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"No existe {resolved}")

    suffix = resolved.suffix.lower()
    logger.info("Cargando SENAMHI diario (%s) desde %s", suffix, resolved)
    if suffix == ".csv":
        df = pl.read_csv(resolved, try_parse_dates=False, infer_schema_length=2000)
    elif suffix in (".xlsx", ".xls"):
        if isinstance(sheet, int):
            df = pl.read_excel(resolved, sheet_id=sheet, engine=_XLSX_ENGINE)
        elif isinstance(sheet, str):
            df = pl.read_excel(resolved, sheet_name=sheet, engine=_XLSX_ENGINE)
        else:
            df = pl.read_excel(resolved, engine=_XLSX_ENGINE)
    else:
        raise ValueError(f"Extension no soportada: {suffix}. Usa .csv, .xlsx o .xls.")

    mapping = _canonical_senamhi_headers(df.columns)
    if not any(c in mapping.values() for c in ("year", "month", "day")):
        raise KeyError(
            f"No se pudieron identificar year/month/day en headers {df.columns}. "
            f"Esperados: Año/Mes/Dia (con o sin acento) o A/B/C."
        )
    df = df.rename(mapping)

    required = ("year", "month", "day")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Columnas fecha ausentes tras mapping: {missing}. Recibidas: {df.columns}")

    cast_exprs: list[pl.Expr] = [
        pl.col("year").cast(pl.Int32, strict=False),
        pl.col("month").cast(pl.Int32, strict=False),
        pl.col("day").cast(pl.Int32, strict=False),
    ]
    for numeric in ("precip_acum", "tmax", "tmin"):
        if numeric in df.columns:
            cast_exprs.append(pl.col(numeric).cast(pl.Float64, strict=False))
    df = df.with_columns(cast_exprs)

    # Componer fecha.
    df = df.with_columns(
        pl.datetime(pl.col("year"), pl.col("month"), pl.col("day")).alias("fecha")
    )

    # Orden: fecha al frente, luego year/month/day, luego numericos conocidos, luego resto.
    numeric_known = [c for c in ("precip_acum", "tmax", "tmin") if c in df.columns]
    others = [c for c in df.columns if c not in ("fecha", "year", "month", "day", *numeric_known)]
    df = df.select(["fecha", "year", "month", "day", *numeric_known, *others]).sort("fecha")

    if enforce_physical_range:
        clamp_exprs: list[pl.Expr] = []
        for col, (lo, hi) in _SENAMHI_PHYS_RANGES.items():
            if col in df.columns:
                clamp_exprs.append(
                    pl.when((pl.col(col) < lo) | (pl.col(col) > hi))
                    .then(None)
                    .otherwise(pl.col(col))
                    .alias(col)
                )
        if clamp_exprs:
            before_nulls = {c: df.get_column(c).null_count() for c in numeric_known}
            df = df.with_columns(clamp_exprs)
            after_nulls = {c: df.get_column(c).null_count() for c in numeric_known}
            dropped = {c: after_nulls[c] - before_nulls[c] for c in numeric_known}
            if any(v > 0 for v in dropped.values()):
                logger.info("SENAMHI: valores fuera de rango fisico convertidos a null: %s", dropped)

    n_rows = df.height
    n_valid_dates = int(df.get_column("fecha").is_not_null().sum())
    logger.info(
        "SENAMHI diario cargado: %d filas, %d fechas validas (%.1f%%), cols=%s",
        n_rows,
        n_valid_dates,
        100 * n_valid_dates / max(n_rows, 1),
        df.columns,
    )
    return df
