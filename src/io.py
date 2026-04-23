"""Carga y normalizacion de los datasets oficiales del Datathon SUNASS 2026.

Fuentes soportadas:
    - Interrupciones (.dta)           - historico_interrupciones_limpio.dta
    - MOREA sensores (.parquet)       - datos_morea.parquet
    - MOREA estaciones (.xlsx)        - ubicacion_estaciones_MOREA.xlsx
    - Dataset del concurso (.csv/.xlsx) — formato sintetico 12 meses del dia D

Convencion: los dataframes se devuelven como polars.DataFrame con nulos reales
y tipos correctos. Los numericos que vienen como string con "" se convierten a
Float64. Las fechas se parsean a Datetime.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pyreadstat
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Columnas conocidas del .dta de interrupciones con quirks a manejar.
_INTERRUP_NUMERIC_AS_STRING = (
    "Ndeconexionesdomiciliariasa",
    "Unidadesdeusoafectadas",
)

# Prefijos que sugieren columna temporal en el .dta (auto-deteccion).
_DATE_COL_PREFIXES = ("Fecha", "fecha", "date", "Date", "FECHA")

_EMPTY_TOKENS = ("", " ", "NA", "N/A", "null", "None", ".")

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

    Args:
        env_file: ruta al .env. Si None, busca automaticamente en el cwd.

    Returns:
        DatasetPaths con rutas absolutas.

    Raises:
        KeyError: si alguna variable requerida no esta definida.
    """
    load_dotenv(dotenv_path=env_file)
    try:
        return DatasetPaths(
            interrupciones=Path(os.environ["INTERRUPCIONES_PATH"]).resolve(),
            morea_parquet=Path(os.environ["MOREA_PARQUET_PATH"]).resolve(),
            morea_estaciones=Path(os.environ["MOREA_ESTACIONES_PATH"]).resolve(),
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
    date_cols = [
        c for c in df.columns
        if any(c.startswith(pref) for pref in _DATE_COL_PREFIXES)
        and df.schema[c] not in (pl.Datetime, pl.Date)
    ]
    date_casts: list[pl.Expr] = [
        pl.col(c).cast(pl.Datetime, strict=False).alias(c) for c in date_cols
    ]
    if date_cols:
        logger.info("Columnas de fecha auto-detectadas: %s", date_cols)

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
    sin_match = joined.filter(pl.col(f"{station_key}_est").is_null()).get_column("_join_key").unique()
    if sin_match.len() > 0:
        logger.warning("Estaciones sin catalogo (%d): %s", sin_match.len(), sin_match.head(5).to_list())

    return joined.drop("_join_key")
