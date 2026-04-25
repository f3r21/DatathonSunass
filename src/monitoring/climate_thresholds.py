"""Umbrales climaticos para deteccion de eventos extremos en series SENAMHI.

A diferencia de `thresholds.py` (que usa bandas DIGESA para calidad de agua),
aqui detectamos:

    - Precipitacion extrema (riesgo de huaico / desborde).
    - Ola de calor (consumo anomalo + falla en redes de distribucion).
    - Helada (congelamiento en tuberias altoandinas).

Las bandas pueden ser absolutas (umbral fijo en mm o C) o relativas a la
distribucion historica (percentil empirico por estacion).

Todas las funciones son puras, no mutan el input y devuelven pl.DataFrame.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import polars as pl

from src.monitoring.thresholds import ThresholdConfig

logger = logging.getLogger(__name__)


# Umbrales absolutos por defecto (fuentes: manual SENAMHI de clasificacion de
# eventos extremos para sierra sur del Peru; fallback conservador).
CLIMATE_PRECIP_EXTREMO_MM: float = 30.0
CLIMATE_OLA_DE_CALOR_C: float = 30.0
CLIMATE_HELADA_C: float = 0.0


@dataclass(frozen=True)
class ClimateThreshold:
    """Config de umbral climatico: acepta absoluto o percentil."""

    parameter: str
    direction: str  # 'gt' o 'lt'
    absolute: float | None
    percentile: float | None
    unit: str
    label: str

    def resolve(self, values: pl.Series) -> float:
        """Devuelve el umbral efectivo en unidades del parametro."""
        if self.absolute is not None:
            return float(self.absolute)
        if self.percentile is not None:
            clean = values.drop_nulls()
            if clean.len() == 0:
                raise ValueError(f"Sin datos para calcular percentil {self.percentile}")
            return float(clean.quantile(self.percentile))
        raise ValueError("ClimateThreshold requiere 'absolute' o 'percentile'")


DEFAULT_CLIMATE_THRESHOLDS: dict[str, ClimateThreshold] = {
    "precip_extremo": ClimateThreshold(
        parameter="precip_acum",
        direction="gt",
        absolute=CLIMATE_PRECIP_EXTREMO_MM,
        percentile=None,
        unit="mm/dia",
        label="Precipitacion extrema",
    ),
    "ola_de_calor": ClimateThreshold(
        parameter="tmax",
        direction="gt",
        absolute=CLIMATE_OLA_DE_CALOR_C,
        percentile=None,
        unit="C",
        label="Ola de calor",
    ),
    "helada": ClimateThreshold(
        parameter="tmin",
        direction="lt",
        absolute=CLIMATE_HELADA_C,
        percentile=None,
        unit="C",
        label="Helada",
    ),
    "precip_p95": ClimateThreshold(
        parameter="precip_acum",
        direction="gt",
        absolute=None,
        percentile=0.95,
        unit="mm/dia",
        label="Precipitacion p95 historico",
    ),
    "tmax_p98": ClimateThreshold(
        parameter="tmax",
        direction="gt",
        absolute=None,
        percentile=0.98,
        unit="C",
        label="Temperatura max p98 historico",
    ),
}


def detect_climate_events(
    df: pl.DataFrame,
    threshold: ClimateThreshold,
    ts_col: str = "fecha",
) -> pl.DataFrame:
    """Marca eventos climaticos extremos con severidad.

    Args:
        df: DataFrame con columna del parametro climatico y columna temporal.
        threshold: ClimateThreshold que indica direccion (gt/lt) y valor base.
        ts_col: columna de fecha para ordenar.

    Returns:
        DataFrame con columnas originales + `umbral`, `es_evento`, `exceso`,
        `severidad`. Mantiene TODAS las filas (no filtra).
    """
    param = threshold.parameter
    if param not in df.columns:
        raise KeyError(f"Parametro '{param}' ausente")
    if ts_col not in df.columns:
        raise KeyError(f"Columna temporal '{ts_col}' ausente")

    umbral_value = threshold.resolve(df.get_column(param))
    if threshold.direction == "gt":
        viola = pl.col(param) > umbral_value
        exceso = pl.col(param) - umbral_value
    elif threshold.direction == "lt":
        viola = pl.col(param) < umbral_value
        exceso = umbral_value - pl.col(param)
    else:
        raise ValueError(f"direction debe ser 'gt' o 'lt', recibido {threshold.direction!r}")

    # Severidad basada en cuanto excede (>2 sigma del parametro = CRITICAL).
    series = df.get_column(param).drop_nulls()
    sigma = float(series.std() or 1.0)

    severidad = (
        pl.when(~viola)
        .then(pl.lit("OK"))
        .when(exceso >= 2 * sigma)
        .then(pl.lit("CRITICAL"))
        .when(exceso >= sigma)
        .then(pl.lit("WARN"))
        .otherwise(pl.lit("INFO"))
    )

    df_out = df.sort(ts_col).with_columns(
        pl.lit(umbral_value).alias("umbral"),
        viola.fill_null(False).alias("es_evento"),
        exceso.alias("exceso"),
        severidad.alias("severidad"),
    )
    n_eventos = int(df_out.get_column("es_evento").sum())
    logger.info(
        "climate(%s): %d eventos sobre %d dias (%.2f%%), umbral=%.2f %s",
        threshold.label,
        n_eventos,
        df_out.height,
        100 * n_eventos / max(df_out.height, 1),
        umbral_value,
        threshold.unit,
    )
    return df_out


def climate_to_threshold_config(clim: ClimateThreshold, df: pl.DataFrame) -> ThresholdConfig:
    """Adapta un ClimateThreshold al ThresholdConfig generico (low/high).

    Permite reusar `detect_violations` del modulo thresholds.py cuando se
    quiera bajar a la pagina de alertas con el mismo formato de salida.
    """
    value = clim.resolve(df.get_column(clim.parameter))
    if clim.direction == "gt":
        return ThresholdConfig(
            parameter=clim.parameter,
            low=float("-inf"),
            high=value,
            unit=clim.unit,
        )
    return ThresholdConfig(
        parameter=clim.parameter,
        low=value,
        high=float("inf"),
        unit=clim.unit,
    )
