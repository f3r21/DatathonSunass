"""Funciones de perfilado y validacion de datasets SUNASS.

Este modulo contiene utilidades puras (sin side effects de IO) para:
    - Perfilar un DataFrame (shape, tipos, nulos, cardinalidad).
    - Describir variables numericas con percentiles.
    - Describir variables categoricas con top-K.
    - Detectar duplicados por llave compuesta.
    - Detectar violaciones de bandas DIGESA (cloro, pH).
    - Analizar regularidad temporal (gaps entre lecturas).

Convenciones:
    - Entrada: polars.DataFrame.
    - Salida: polars.DataFrame o dataclasses frozen.
    - Ninguna funcion imprime; todo va por el logger del modulo.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import polars as pl

logger = logging.getLogger(__name__)

# Bandas de cumplimiento DIGESA para calidad de agua potable.
DIGESA_CLORO_LIBRE = (0.5, 5.0)  # mg/L
DIGESA_PH = (6.5, 8.5)
DIGESA_TURBIEDAD_MAX = 5.0  # NTU


@dataclass(frozen=True)
class ColumnProfile:
    """Perfil por columna."""

    name: str
    dtype: str
    null_count: int
    null_pct: float
    n_unique: int
    sample: list[object] = field(default_factory=list)


@dataclass(frozen=True)
class DataFrameProfile:
    """Perfil completo de un DataFrame."""

    n_rows: int
    n_cols: int
    memory_mb: float
    columns: list[ColumnProfile]

    def to_polars(self) -> pl.DataFrame:
        """Convierte el perfil de columnas a DataFrame para mostrar en tablas."""
        rows = [
            {
                "columna": c.name,
                "tipo": c.dtype,
                "nulos": c.null_count,
                "nulos_pct": c.null_pct,
                "unicos": c.n_unique,
                "muestra": str(c.sample),
            }
            for c in self.columns
        ]
        return pl.DataFrame(rows)


def profile_dataframe(df: pl.DataFrame, sample_size: int = 3) -> DataFrameProfile:
    """Genera un perfil completo del DataFrame.

    Args:
        df: DataFrame a perfilar.
        sample_size: valores de muestra a retener por columna.

    Returns:
        DataFrameProfile con shape, perfil por columna y memoria estimada.
    """
    n_rows = df.height
    column_profiles: list[ColumnProfile] = []
    for col in df.columns:
        series = df.get_column(col)
        null_count = int(series.null_count())
        null_pct = (null_count / n_rows * 100.0) if n_rows else 0.0
        try:
            n_unique = int(series.n_unique())
        except Exception:
            n_unique = -1
        sample = series.drop_nulls().head(sample_size).to_list()
        column_profiles.append(
            ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                null_count=null_count,
                null_pct=round(null_pct, 2),
                n_unique=n_unique,
                sample=sample,
            )
        )
    memory_mb = df.estimated_size("mb")
    return DataFrameProfile(
        n_rows=n_rows,
        n_cols=df.width,
        memory_mb=round(memory_mb, 2),
        columns=column_profiles,
    )


def describe_numeric(
    df: pl.DataFrame, columns: Sequence[str] | None = None
) -> pl.DataFrame:
    """Descriptivos numericos con percentiles 1/25/50/75/99.

    Si columns es None, se perfilan todas las columnas numericas.
    """
    if columns is None:
        columns = [c for c, dt in zip(df.columns, df.dtypes) if dt.is_numeric()]
    if not columns:
        return pl.DataFrame()

    rows: list[dict] = []
    for col in columns:
        if col not in df.columns:
            logger.warning("Columna numerica no existe: %s", col)
            continue
        s = df.get_column(col).drop_nulls()
        if s.is_empty():
            continue
        rows.append(
            {
                "columna": col,
                "n": s.len(),
                "media": float(s.mean()),
                "std": float(s.std(ddof=1)) if s.len() > 1 else 0.0,
                "min": float(s.min()),
                "p01": float(s.quantile(0.01) or 0.0),
                "p25": float(s.quantile(0.25) or 0.0),
                "p50": float(s.quantile(0.50) or 0.0),
                "p75": float(s.quantile(0.75) or 0.0),
                "p99": float(s.quantile(0.99) or 0.0),
                "max": float(s.max()),
            }
        )
    return pl.DataFrame(rows)


def describe_categorical(
    df: pl.DataFrame,
    columns: Sequence[str] | None = None,
    top_k: int = 10,
) -> dict[str, pl.DataFrame]:
    """Para cada columna categorica, devuelve top-K valores con conteos."""
    if columns is None:
        columns = [
            c for c, dt in zip(df.columns, df.dtypes) if dt in (pl.Utf8, pl.Categorical)
        ]
    out: dict[str, pl.DataFrame] = {}
    for col in columns:
        if col not in df.columns:
            logger.warning("Columna categorica no existe: %s", col)
            continue
        counts = (
            df.select(pl.col(col))
            .group_by(col)
            .agg(pl.len().alias("conteo"))
            .sort("conteo", descending=True)
            .head(top_k)
        )
        total = counts.get_column("conteo").sum() or 1
        out[col] = counts.with_columns(
            (pl.col("conteo") / total * 100).round(2).alias("pct")
        )
    return out


def detect_duplicates(df: pl.DataFrame, key_cols: Sequence[str]) -> pl.DataFrame:
    """Filas cuya combinacion de `key_cols` aparece mas de una vez.

    Raises:
        KeyError: si alguna columna no existe.
    """
    missing = [c for c in key_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columnas inexistentes: {missing}")
    mask = df.select(key_cols).is_duplicated()
    return df.filter(mask)


def detect_digesa_violations(
    df: pl.DataFrame,
    column: str,
    low: float,
    high: float,
) -> pl.DataFrame:
    """Filas con valores fuera de la banda [low, high] inclusiva.

    Args:
        df: DataFrame con la columna de sensor.
        column: nombre de la columna (p.e. 'cloro', 'ph').
        low, high: limites inclusivos aceptables.

    Returns:
        Sub-DataFrame con solo las filas violatorias.
    """
    if column not in df.columns:
        raise KeyError(f"Columna '{column}' ausente. Disponibles: {df.columns[:10]}")
    return df.filter((pl.col(column) < low) | (pl.col(column) > high))


def summarize_digesa_compliance(
    df: pl.DataFrame,
    columns_ranges: dict[str, tuple[float, float]] | None = None,
    group_col: str | None = None,
) -> pl.DataFrame:
    """Resumen de cumplimiento DIGESA por columna y opcionalmente por estacion.

    Args:
        df: DataFrame de sensores MOREA.
        columns_ranges: dict {col: (low, high)}. Default: cloro y pH.
        group_col: si se provee, agrupa por estacion/ubigeo.

    Returns:
        DataFrame con [group_col?, columna, n_total, n_violaciones, pct_violacion].
    """
    if columns_ranges is None:
        columns_ranges = {"cloro": DIGESA_CLORO_LIBRE, "ph": DIGESA_PH}

    rows: list[dict] = []
    for col, (low, high) in columns_ranges.items():
        if col not in df.columns:
            logger.warning("Columna DIGESA ausente: %s", col)
            continue
        violation_expr = ((pl.col(col) < low) | (pl.col(col) > high)).cast(pl.Int64)
        if group_col:
            if group_col not in df.columns:
                raise KeyError(f"group_col '{group_col}' ausente")
            grouped = (
                df.filter(pl.col(col).is_not_null())
                .group_by(group_col)
                .agg(
                    pl.len().alias("n_total"),
                    violation_expr.sum().alias("n_violaciones"),
                )
                .with_columns(
                    (pl.col("n_violaciones") / pl.col("n_total") * 100)
                    .round(2)
                    .alias("pct_violacion")
                )
                .with_columns(pl.lit(col).alias("columna"))
                .sort("pct_violacion", descending=True)
            )
            rows.extend(grouped.to_dicts())
        else:
            n_total = df.get_column(col).drop_nulls().len()
            n_viol = df.select(violation_expr.sum()).item() or 0
            rows.append(
                {
                    "columna": col,
                    "n_total": n_total,
                    "n_violaciones": int(n_viol),
                    "pct_violacion": round(n_viol / n_total * 100, 2) if n_total else 0.0,
                }
            )
    return pl.DataFrame(rows)


def timestamp_gaps(
    df: pl.DataFrame,
    ts_col: str,
    group_col: str | None = None,
) -> pl.DataFrame:
    """Resumen de gaps consecutivos entre timestamps.

    Args:
        df: DataFrame con columna temporal.
        ts_col: nombre de la columna Datetime.
        group_col: si se provee, analiza gaps por grupo (p.e. estacion).

    Returns:
        DataFrame con [group_col?, min_gap, p50_gap, max_gap, n_gaps].
    """
    if ts_col not in df.columns:
        raise KeyError(f"Columna '{ts_col}' ausente")

    sort_keys = [group_col, ts_col] if group_col else [ts_col]
    sorted_df = df.sort(sort_keys)

    if group_col:
        if group_col not in df.columns:
            raise KeyError(f"group_col '{group_col}' ausente")
        gaps = sorted_df.with_columns(
            pl.col(ts_col).diff().over(group_col).alias("_gap")
        ).drop_nulls("_gap")
        return (
            gaps.group_by(group_col)
            .agg(
                pl.col("_gap").min().alias("min_gap"),
                pl.col("_gap").median().alias("p50_gap"),
                pl.col("_gap").max().alias("max_gap"),
                pl.len().alias("n_gaps"),
            )
            .sort(group_col)
        )
    gaps = sorted_df.with_columns(pl.col(ts_col).diff().alias("_gap")).drop_nulls("_gap")
    return gaps.select(
        pl.col("_gap").min().alias("min_gap"),
        pl.col("_gap").median().alias("p50_gap"),
        pl.col("_gap").max().alias("max_gap"),
        pl.len().alias("n_gaps"),
    )


def coverage_by_group(
    df: pl.DataFrame,
    group_col: str,
    date_col: str | None = None,
) -> pl.DataFrame:
    """Conteo de registros por grupo y, si aplica, rango temporal observado.

    Util para detectar estaciones con baja cobertura o huecos prolongados.
    """
    if group_col not in df.columns:
        raise KeyError(f"group_col '{group_col}' ausente")
    aggs: list[pl.Expr] = [pl.len().alias("n_registros")]
    if date_col:
        if date_col not in df.columns:
            raise KeyError(f"date_col '{date_col}' ausente")
        aggs.extend(
            [
                pl.col(date_col).min().alias("fecha_min"),
                pl.col(date_col).max().alias("fecha_max"),
            ]
        )
    return df.group_by(group_col).agg(aggs).sort("n_registros", descending=True)
