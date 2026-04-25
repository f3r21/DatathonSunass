"""Construccion y agregacion de alertas a partir de violaciones detectadas.

Contrasta con `detect_violations` (que etiqueta lectura por lectura): aqui
colapsamos rachas consecutivas en un solo AlertEvent con duracion, intensidad
y severidad final.
"""
from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import polars as pl

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"

    @classmethod
    def from_label(cls, label: str) -> Severity:
        try:
            return cls(label)
        except ValueError:
            return cls.INFO


_SEVERITY_ORDER = {Severity.INFO: 0, Severity.WARN: 1, Severity.CRITICAL: 2}


@dataclass(frozen=True)
class AlertEvent:
    """Un evento de alerta = racha consecutiva de violaciones en una estacion."""

    station_id: str
    parameter: str
    severity: Severity
    start_ts: datetime
    end_ts: datetime
    duration_minutes: float
    peak_value: float
    peak_exceed: float
    n_readings: int

    def as_dict(self) -> dict[str, object]:
        return {
            "station_id": self.station_id,
            "parameter": self.parameter,
            "severity": self.severity.value,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "duration_min": round(self.duration_minutes, 2),
            "peak_value": round(self.peak_value, 3),
            "peak_exceed": round(self.peak_exceed, 3),
            "n_readings": self.n_readings,
        }


def build_alerts(
    df_violations: pl.DataFrame,
    parameter: str,
    station_col: str = "estacion_id",
    ts_col: str = "fecha",
) -> list[AlertEvent]:
    """Colapsa violaciones consecutivas en eventos.

    Requiere el DataFrame que sale de `detect_violations(..., keep_ok=True)`
    para preservar orden temporal y poder detectar rachas.
    """
    required = (station_col, ts_col, parameter, "viola", "exceso", "severidad")
    missing = [c for c in required if c not in df_violations.columns]
    if missing:
        raise KeyError(f"Columnas faltantes para build_alerts: {missing}")

    df_sorted = df_violations.sort([station_col, ts_col])
    # run_id cambia cada vez que viola cambia dentro de la estacion
    df_runs = df_sorted.with_columns(
        (pl.col("viola") != pl.col("viola").shift(1).over(station_col))
        .fill_null(True)
        .cum_sum()
        .over(station_col)
        .alias("_run_id")
    )
    runs_viola = df_runs.filter(pl.col("viola"))
    if runs_viola.is_empty():
        return []

    grouped = runs_viola.group_by([station_col, "_run_id"]).agg(
        pl.col(ts_col).min().alias("start_ts"),
        pl.col(ts_col).max().alias("end_ts"),
        pl.col(parameter).max().alias("peak_value"),
        pl.col("exceso").max().alias("peak_exceed"),
        pl.len().alias("n_readings"),
        pl.col("severidad").mode().first().alias("severidad_moda"),
    )
    events: list[AlertEvent] = []
    for row in grouped.iter_rows(named=True):
        duration = (row["end_ts"] - row["start_ts"]).total_seconds() / 60.0 if row["start_ts"] and row["end_ts"] else 0.0
        events.append(
            AlertEvent(
                station_id=str(row[station_col]),
                parameter=parameter,
                severity=Severity.from_label(row["severidad_moda"] or "INFO"),
                start_ts=row["start_ts"],
                end_ts=row["end_ts"],
                duration_minutes=duration,
                peak_value=float(row["peak_value"]) if row["peak_value"] is not None else 0.0,
                peak_exceed=float(row["peak_exceed"]) if row["peak_exceed"] is not None else 0.0,
                n_readings=int(row["n_readings"]),
            )
        )
    events.sort(key=lambda e: (_SEVERITY_ORDER[e.severity], e.duration_minutes), reverse=True)
    logger.info("build_alerts(%s): %d eventos colapsados", parameter, len(events))
    return events


def summarize_alerts(events: Iterable[AlertEvent]) -> pl.DataFrame:
    """Tabla compacta de alertas lista para mostrar en Streamlit."""
    rows = [e.as_dict() for e in events]
    if not rows:
        return pl.DataFrame(
            schema={
                "station_id": pl.Utf8,
                "parameter": pl.Utf8,
                "severity": pl.Utf8,
                "start_ts": pl.Datetime,
                "end_ts": pl.Datetime,
                "duration_min": pl.Float64,
                "peak_value": pl.Float64,
                "peak_exceed": pl.Float64,
                "n_readings": pl.Int64,
            }
        )
    return pl.DataFrame(rows)
