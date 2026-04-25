"""Deteccion de violaciones DIGESA en lecturas MOREA.

Este modulo no clasifica: etiqueta. Para cada lectura decide si viola la banda
del parametro y, si viola, con que severidad (cuanto se aleja del limite).

Las bandas DIGESA estan publicadas en el Reglamento de la Calidad del Agua
para Consumo Humano (D.S. 031-2010-SA, Peru). Los valores que usamos son los
que cita el mentor en clases/dia2_* y aparecen en docs/.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import polars as pl

logger = logging.getLogger(__name__)

DIGESA_CLORO: tuple[float, float] = (0.5, 5.0)  # mg/L libre
DIGESA_PH: tuple[float, float] = (6.5, 8.5)
DIGESA_TURBIEDAD: tuple[float, float] = (0.0, 5.0)  # NTU

ParameterName = Literal["cloro", "ph", "temperatura", "turbiedad"]


@dataclass(frozen=True)
class ThresholdConfig:
    """Banda DIGESA + politica de severidad."""

    parameter: str
    low: float
    high: float
    unit: str
    warn_fraction: float = 0.10  # margen (% de banda) antes de activar WARN
    critical_fraction: float = 0.30  # margen para CRITICAL

    def band_width(self) -> float:
        return self.high - self.low

    def describe_band(self) -> str:
        return f"[{self.low}, {self.high}] {self.unit}"


DEFAULT_CONFIGS: dict[str, ThresholdConfig] = {
    "cloro": ThresholdConfig("cloro", *DIGESA_CLORO, unit="mg/L"),
    "ph": ThresholdConfig("ph", *DIGESA_PH, unit="-"),
    "turbiedad": ThresholdConfig("turbiedad", *DIGESA_TURBIEDAD, unit="NTU"),
}


def _severity_expr(col: str, cfg: ThresholdConfig) -> pl.Expr:
    """Calcula severidad por lectura basado en cuanto se excede la banda.

    Retorna string: 'OK', 'INFO', 'WARN', 'CRITICAL'.
    """
    band = cfg.band_width()
    warn_margin = band * cfg.warn_fraction
    crit_margin = band * cfg.critical_fraction

    below = pl.col(col) < cfg.low
    above = pl.col(col) > cfg.high
    violation = below | above
    exceed = (
        pl.when(below)
        .then(cfg.low - pl.col(col))
        .when(above)
        .then(pl.col(col) - cfg.high)
        .otherwise(0.0)
    )
    severity = (
        pl.when(~violation)
        .then(pl.lit("OK"))
        .when(exceed >= crit_margin)
        .then(pl.lit("CRITICAL"))
        .when(exceed >= warn_margin)
        .then(pl.lit("WARN"))
        .otherwise(pl.lit("INFO"))
    )
    return severity


def detect_violations(
    df: pl.DataFrame,
    parameter: str,
    config: ThresholdConfig | None = None,
    keep_ok: bool = False,
) -> pl.DataFrame:
    """Agrega columnas `viola`, `exceso`, `severidad` para un parametro.

    Args:
        df: DataFrame con columna del parametro.
        parameter: nombre de la columna a monitorear.
        config: ThresholdConfig custom; si None usa DEFAULT_CONFIGS[parameter].
        keep_ok: si False filtra filas severidad=='OK' (default: solo violaciones).

    Returns:
        DataFrame con columnas originales + viola, exceso, severidad.
    """
    if parameter not in df.columns:
        raise KeyError(f"Parametro '{parameter}' ausente en DataFrame")
    cfg = config or DEFAULT_CONFIGS.get(parameter)
    if cfg is None:
        raise ValueError(f"No hay ThresholdConfig default para '{parameter}'")

    below = pl.col(parameter) < cfg.low
    above = pl.col(parameter) > cfg.high
    violation = below | above
    exceed = (
        pl.when(below)
        .then(cfg.low - pl.col(parameter))
        .when(above)
        .then(pl.col(parameter) - cfg.high)
        .otherwise(0.0)
    )
    df_out = df.with_columns(
        violation.fill_null(False).alias("viola"),
        exceed.alias("exceso"),
        _severity_expr(parameter, cfg).alias("severidad"),
    )
    n_viol = int(df_out.get_column("viola").sum())
    logger.info(
        "thresholds(%s): %d violaciones (%.2f%%) sobre %d lecturas",
        parameter,
        n_viol,
        100 * n_viol / max(df_out.height, 1),
        df_out.height,
    )
    if not keep_ok:
        return df_out.filter(pl.col("viola"))
    return df_out


def stream_scan(
    df: pl.DataFrame,
    parameter: str,
    chunk_size: int = 5000,
    config: ThresholdConfig | None = None,
) -> Iterator[pl.DataFrame]:
    """Simula un stream: itera chunks de tamaño fijo y emite violaciones.

    Util para la pagina Alertas que auto-refresca y "consume" lecturas como si
    llegaran en tiempo real.
    """
    cfg = config or DEFAULT_CONFIGS.get(parameter)
    if cfg is None:
        raise ValueError(f"ThresholdConfig requerido para '{parameter}'")
    total = df.height
    for start in range(0, total, chunk_size):
        chunk = df.slice(start, chunk_size)
        yield detect_violations(chunk, parameter, config=cfg, keep_ok=False)
