"""Inspector generico de DataFrames — detecta tipos heterogeneos.

Cubre el pedido del jurado: la app debe aceptar **cualquier** dataset
operacional, no solo el formato MOREA/Interrupciones. Detecta:

    - Numericos (int/float, con o sin nulls)
    - Categoricos (string baja cardinalidad)
    - Fechas (datetime nativo o string parseable)
    - Horas (string HH:MM o time)
    - Geodata: latitud + longitud en dos columnas, o "lat,lon" en una sola
    - Identificadores (string alta cardinalidad)
    - Texto libre

El reporte resultante alimenta:
    - Home/Ejecutivo (KPIs autodetectados)
    - 3_Modelo (multiselect de features)
    - 8_Mapa (lat/lon detectados)
    - 7_Reportes (cualquier tabla cargada)
"""
from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import polars as pl

logger = logging.getLogger(__name__)

ColumnKind = Literal[
    "numeric",
    "categorical",
    "datetime",
    "time",
    "geo_lat",
    "geo_lon",
    "geo_combined",
    "identifier",
    "text",
    "boolean",
    "unknown",
]


@dataclass(frozen=True)
class ColumnInfo:
    """Metadata detectada para una columna."""

    name: str
    kind: ColumnKind
    dtype: str
    null_count: int
    null_pct: float
    n_unique: int
    sample_values: list[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass(frozen=True)
class InspectionReport:
    """Reporte completo de inspeccion."""

    n_rows: int
    n_cols: int
    columns: list[ColumnInfo]

    def by_kind(self, kind: ColumnKind) -> list[ColumnInfo]:
        return [c for c in self.columns if c.kind == kind]

    def names_by_kind(self, kind: ColumnKind) -> list[str]:
        return [c.name for c in self.columns if c.kind == kind]

    def has_geo(self) -> bool:
        return bool(self.names_by_kind("geo_lat") and self.names_by_kind("geo_lon")) or bool(
            self.names_by_kind("geo_combined")
        )

    def detect_lat_lon(self) -> tuple[str | None, str | None]:
        """Devuelve (lat_col, lon_col) si los hay; (None, None) si no."""
        lats = self.names_by_kind("geo_lat")
        lons = self.names_by_kind("geo_lon")
        return (lats[0] if lats else None, lons[0] if lons else None)


# ----------------------------------------------------------------- regex helpers


_LAT_NAMES = re.compile(
    r"^(lat|latitude|latitud)$", re.IGNORECASE
)
_LON_NAMES = re.compile(
    r"^(lon|lng|long|longitude|longitud)$", re.IGNORECASE
)
_GEO_COMBINED = re.compile(
    r"^(coords?|coordenadas?|latlon|geo|geometry|geom|location|ubicacion)$", re.IGNORECASE
)
_TIME_NAMES = re.compile(r"^(hora|horario|time|hour)$", re.IGNORECASE)
_DATE_NAMES = re.compile(r"^(fecha|date|dia|day|timestamp|ts|datetime)", re.IGNORECASE)
_ID_NAMES = re.compile(r"(^id$|_id$|codigo|ubigeo)", re.IGNORECASE)

_HHMM_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")
_LATLON_RE = re.compile(r"^\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*$")


def _ascii_lower(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).strip().lower()


def _values_in_lat_range(s: pl.Series) -> bool:
    if not s.dtype.is_numeric():
        return False
    clean = s.drop_nulls()
    if clean.len() == 0:
        return False
    mn, mx = float(clean.min()), float(clean.max())
    return -90.0 <= mn and mx <= 90.0


def _values_in_lon_range(s: pl.Series) -> bool:
    if not s.dtype.is_numeric():
        return False
    clean = s.drop_nulls()
    if clean.len() == 0:
        return False
    mn, mx = float(clean.min()), float(clean.max())
    return -180.0 <= mn and mx <= 180.0


def _string_looks_like_time(s: pl.Series, threshold: float = 0.8) -> bool:
    if s.dtype != pl.Utf8:
        return False
    sample = s.drop_nulls().head(50).to_list()
    if not sample:
        return False
    matches = sum(1 for v in sample if _HHMM_RE.match(str(v)))
    return matches / len(sample) >= threshold


def _string_looks_like_latlon(s: pl.Series, threshold: float = 0.8) -> bool:
    if s.dtype != pl.Utf8:
        return False
    sample = s.drop_nulls().head(50).to_list()
    if not sample:
        return False
    matches = sum(1 for v in sample if _LATLON_RE.match(str(v)))
    return matches / len(sample) >= threshold


def _detect_kind(name: str, series: pl.Series, n_rows: int) -> tuple[ColumnKind, float]:
    """Heuristica multi-criterio. Retorna (kind, confidence)."""
    name_norm = _ascii_lower(name)
    dtype = series.dtype
    n_unique = series.n_unique()

    # Booleano explicito.
    if dtype == pl.Boolean:
        return "boolean", 1.0

    # Datetime / Date nativos.
    if dtype in (pl.Datetime, pl.Date):
        return "datetime", 1.0
    if dtype == pl.Time:
        return "time", 1.0

    # Por nombre + valores: lat/lon.
    if _LAT_NAMES.match(name_norm) and _values_in_lat_range(series):
        return "geo_lat", 0.95
    if _LON_NAMES.match(name_norm) and _values_in_lon_range(series):
        return "geo_lon", 0.95

    # Combinado tipo "lat,lon" en una sola columna.
    if _GEO_COMBINED.match(name_norm) or _string_looks_like_latlon(series):
        if dtype == pl.Utf8:
            return "geo_combined", 0.85

    # String que parece hora.
    if _TIME_NAMES.match(name_norm) or _string_looks_like_time(series):
        return "time", 0.8

    # String que parece fecha (heuristica por nombre).
    if dtype == pl.Utf8 and _DATE_NAMES.match(name_norm):
        return "datetime", 0.7

    # Numerico generico.
    if dtype.is_numeric():
        # Solo lat/lon por valores aunque el nombre no calce.
        if _values_in_lat_range(series) and not _values_in_lon_range(series):
            # Range de latitudes pero no longitudes -> probablemente lat sin nombre.
            return ("geo_lat", 0.6) if "lat" in name_norm else ("numeric", 1.0)
        return "numeric", 1.0

    # String: identificador (alta cardinalidad), categorico (baja) o texto libre.
    if dtype == pl.Utf8:
        if _ID_NAMES.search(name_norm):
            return "identifier", 0.9
        if n_rows == 0:
            return "categorical", 0.5
        ratio = n_unique / n_rows
        if ratio > 0.9:
            return "identifier", 0.7
        if ratio > 0.5:
            return "text", 0.6
        return "categorical", 0.85

    return "unknown", 0.3


def inspect_dataframe(
    df: pl.DataFrame,
    sample_size: int = 5,
) -> InspectionReport:
    """Profila un DataFrame y devuelve un InspectionReport.

    Args:
        df: cualquier polars.DataFrame.
        sample_size: cuantos valores de ejemplo guardar por columna.
    """
    columns: list[ColumnInfo] = []
    n_rows = df.height
    for name in df.columns:
        s = df.get_column(name)
        kind, conf = _detect_kind(name, s, n_rows)
        nulls = s.null_count()
        sample_values: list[str] = []
        if not s.is_empty():
            for v in s.drop_nulls().head(sample_size).to_list():
                sample_values.append(str(v)[:80])
        columns.append(
            ColumnInfo(
                name=name,
                kind=kind,
                dtype=str(s.dtype),
                null_count=nulls,
                null_pct=round(100 * nulls / max(n_rows, 1), 2),
                n_unique=s.n_unique(),
                sample_values=sample_values,
                confidence=conf,
            )
        )
    logger.info(
        "inspect_dataframe: %d cols, kinds=%s",
        df.width,
        {c.name: c.kind for c in columns},
    )
    return InspectionReport(n_rows=n_rows, n_cols=df.width, columns=columns)


def split_combined_latlon(
    df: pl.DataFrame, col: str, sep: str = ","
) -> pl.DataFrame:
    """Si una columna trae 'lat,lon' como string, separa en dos columnas."""
    if col not in df.columns:
        raise KeyError(f"Columna '{col}' ausente")
    parts = pl.col(col).cast(pl.Utf8).str.split_exact(sep, 1).struct.rename_fields(
        ["_lat", "_lon"]
    )
    return df.with_columns(
        parts.struct.field("_lat").str.strip_chars().cast(pl.Float64, strict=False).alias("lat"),
        parts.struct.field("_lon").str.strip_chars().cast(pl.Float64, strict=False).alias("lon"),
    )


def report_to_dataframe(report: InspectionReport) -> pl.DataFrame:
    """Convierte el reporte a DataFrame para mostrar en Streamlit."""
    return pl.DataFrame(
        [
            {
                "columna": c.name,
                "tipo": c.kind,
                "dtype": c.dtype,
                "nulls_pct": c.null_pct,
                "n_unique": c.n_unique,
                "confianza": round(c.confidence, 2),
                "ejemplos": ", ".join(c.sample_values[:3]),
            }
            for c in report.columns
        ]
    )
