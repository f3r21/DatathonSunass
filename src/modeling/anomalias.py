"""Plantilla B — Deteccion de anomalias y cumplimiento DIGESA depurado.

Objetivo: del 60% bruto de violaciones de cloro y 34% de pH que aparecen en los
datos MOREA, separar los **spikes de sensor** (transitorios, sugieren falla
del instrumento) de las **violaciones sostenidas** (varios minutos seguidos
fuera de banda, que sugieren un problema operativo real). La cifra depurada
es la que se lleva al jurado.

Pipeline:

    1. `filter_imposibles`                  descarta valores fisicamente imposibles
    2. `sustained_violations`               marca violaciones que duran >= N lecturas
    3. `chronic_stations`                   top-K de estaciones cronicamente fuera de banda
    4. `isolation_forest_scan`              anomalias multivariadas (ph + cloro + temp)
    5. `ruptures_changepoints`              regimenes distintos en una estacion

Todas las funciones son puras, no mutan el input, y registran via logger.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# Limites fisicamente imposibles (mas laxos que DIGESA, para descartar glitches).
# Los valores fuera de estas bandas son casi seguro error de sensor, no realidad.
IMPOSIBLE_CLORO = (0.0, 10.0)  # mg/L; DIGESA exige <=5, pero hasta 10 es plausible
IMPOSIBLE_PH = (3.0, 11.0)      # pH extremo pero tecnicamente medible
IMPOSIBLE_TEMP = (0.0, 50.0)    # agua potable en Peru: 5-30C normal; >50 = glitch


@dataclass(frozen=True)
class FilterResult:
    """Resultado de filter_imposibles — DataFrames separados."""

    depurado: pl.DataFrame
    glitches: pl.DataFrame
    resumen: dict[str, int] = field(default_factory=dict)


def filter_imposibles(
    df: pl.DataFrame,
    cloro_col: str = "cloro",
    ph_col: str = "ph",
    temp_col: str = "temperatura",
    cloro_range: tuple[float, float] = IMPOSIBLE_CLORO,
    ph_range: tuple[float, float] = IMPOSIBLE_PH,
    temp_range: tuple[float, float] = IMPOSIBLE_TEMP,
) -> FilterResult:
    """Parte el DataFrame en (depurado, glitches) segun limites fisicos.

    Una fila va a `glitches` si cualquier sensor esta fuera de su banda fisica.
    Los nulls en un sensor no marcan la fila como glitch por ese sensor.

    Returns:
        FilterResult con los dos sub-DataFrames y un dict con conteos.
    """
    conditions: list[pl.Expr] = []
    if cloro_col in df.columns:
        conditions.append(
            (pl.col(cloro_col) < cloro_range[0]) | (pl.col(cloro_col) > cloro_range[1])
        )
    if ph_col in df.columns:
        conditions.append(
            (pl.col(ph_col) < ph_range[0]) | (pl.col(ph_col) > ph_range[1])
        )
    if temp_col in df.columns:
        conditions.append(
            (pl.col(temp_col) < temp_range[0]) | (pl.col(temp_col) > temp_range[1])
        )

    if not conditions:
        logger.warning("filter_imposibles: ninguna columna de sensor presente")
        return FilterResult(depurado=df, glitches=df.head(0), resumen={"total": df.height})

    is_glitch = conditions[0]
    for cond in conditions[1:]:
        is_glitch = is_glitch | cond

    glitches = df.filter(is_glitch.fill_null(False))
    depurado = df.filter(~is_glitch.fill_null(False))

    resumen = {
        "total": df.height,
        "depurado": depurado.height,
        "glitches": glitches.height,
    }
    logger.info(
        "filter_imposibles: total=%d, depurado=%d (%.1f%%), glitches=%d (%.1f%%)",
        resumen["total"],
        resumen["depurado"],
        100 * resumen["depurado"] / max(resumen["total"], 1),
        resumen["glitches"],
        100 * resumen["glitches"] / max(resumen["total"], 1),
    )
    return FilterResult(depurado=depurado, glitches=glitches, resumen=resumen)


def sustained_violations(
    df: pl.DataFrame,
    col: str,
    low: float,
    high: float,
    station_col: str = "estacion_id",
    ts_col: str = "fecha",
    min_consecutive: int = 3,
) -> pl.DataFrame:
    """Marca lecturas cuya violacion forma parte de una racha >= min_consecutive.

    Idea: agrupar por estacion, ordenar por timestamp, contar rachas
    consecutivas del flag violacion, y marcar como `violacion_sostenida` solo
    si el run length de la racha a la que pertenecen alcanza min_consecutive.

    Returns:
        DataFrame con columnas extra: `viola`, `run_id`, `run_len`, `sostenida`.
        Mantiene TODAS las filas del input (no filtra).
    """
    required = (col, station_col, ts_col)
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Columna '{c}' ausente")

    viola_expr = (pl.col(col) < low) | (pl.col(col) > high)
    df_sorted = df.sort([station_col, ts_col]).with_columns(
        viola_expr.fill_null(False).alias("viola")
    )
    # run_id: cambia cada vez que `viola` cambia de valor dentro de la estacion.
    df_runs = df_sorted.with_columns(
        (pl.col("viola") != pl.col("viola").shift(1).over(station_col))
        .fill_null(True)
        .cum_sum()
        .over(station_col)
        .alias("run_id")
    )
    # run_len: cuantas filas tiene cada run dentro de la estacion.
    df_runs = df_runs.with_columns(
        pl.len().over([station_col, "run_id"]).alias("run_len")
    )
    df_final = df_runs.with_columns(
        (pl.col("viola") & (pl.col("run_len") >= min_consecutive)).alias("sostenida")
    )
    n_viola = int(df_final.get_column("viola").sum())
    n_sost = int(df_final.get_column("sostenida").sum())
    logger.info(
        "sustained_violations(%s): violaciones=%d, sostenidas>=%d lecturas=%d (%.1f%% del bruto)",
        col,
        n_viola,
        min_consecutive,
        n_sost,
        100 * n_sost / max(n_viola, 1),
    )
    return df_final


def chronic_stations(
    df_marked: pl.DataFrame,
    station_col: str = "estacion_id",
    flag_col: str = "sostenida",
    min_hits: int = 10,
    top_k: int = 20,
) -> pl.DataFrame:
    """Top estaciones con mas lecturas en violacion sostenida.

    Espera el output de `sustained_violations` (con columnas viola/sostenida).
    """
    if flag_col not in df_marked.columns:
        raise KeyError(
            f"'{flag_col}' ausente — corre sustained_violations() antes de chronic_stations()"
        )
    n_lecturas_expr = pl.len().alias("n_lecturas")
    n_violaciones_expr = pl.col(flag_col).cast(pl.Int64).sum().alias("n_sostenida")
    resumen = (
        df_marked.group_by(station_col)
        .agg(n_lecturas_expr, n_violaciones_expr)
        .with_columns(
            (pl.col("n_sostenida") / pl.col("n_lecturas") * 100).round(2).alias("pct_sostenida")
        )
        .filter(pl.col("n_sostenida") >= min_hits)
        .sort("pct_sostenida", descending=True)
        .head(top_k)
    )
    logger.info("chronic_stations: %d estaciones con >=%d violaciones sostenidas", resumen.height, min_hits)
    return resumen


def isolation_forest_scan(
    df: pl.DataFrame,
    feature_cols: Sequence[str],
    contamination: float = 0.01,
    random_state: int = 42,
) -> pl.DataFrame:
    """Anomalias multivariadas con IsolationForest.

    Devuelve el DataFrame con columnas extras: `anomaly_score` (cuanto mas
    negativo, mas anomalo) y `is_anomaly` (bool).
    """
    from sklearn.ensemble import IsolationForest

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Features ausentes: {missing}")

    subset = df.select(list(feature_cols)).drop_nulls()
    if subset.is_empty():
        logger.warning("isolation_forest_scan: no hay filas no-null en %s", feature_cols)
        return df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("anomaly_score"),
            pl.lit(False).alias("is_anomaly"),
        )

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    X = subset.to_numpy()
    model.fit(X)

    full_X = df.select(list(feature_cols)).fill_null(strategy="mean").to_numpy()
    scores = model.decision_function(full_X)
    preds = model.predict(full_X)  # -1 anomalo, 1 normal
    df_out = df.with_columns(
        pl.Series("anomaly_score", scores),
        pl.Series("is_anomaly", preds == -1),
    )
    n_anom = int(df_out.get_column("is_anomaly").sum())
    logger.info("isolation_forest: %d anomalias (%.2f%%)", n_anom, 100 * n_anom / max(df.height, 1))
    return df_out


def ruptures_changepoints(
    values: np.ndarray,
    penalty: float = 10.0,
    model: str = "rbf",
    min_size: int = 50,
) -> list[int]:
    """Detecta puntos de cambio en una serie 1D con PELT.

    Args:
        values: array 1D de la metrica (cloro, pH, ...) en orden temporal.
        penalty: mayor => menos cambios detectados.
        model: 'rbf' para cambios en distribucion, 'l2' para cambios en media.
        min_size: minimo de observaciones entre cambios.

    Returns:
        Lista de indices (0-based) donde hay cambio; el ultimo elemento es len(values).
    """
    import ruptures as rpt

    arr = np.asarray(values, dtype=float)
    if arr.size < min_size * 2:
        logger.debug("ruptures: serie muy corta (%d), sin cambios", arr.size)
        return [arr.size]

    algo = rpt.Pelt(model=model, min_size=min_size).fit(arr.reshape(-1, 1))
    breakpoints: list[int] = algo.predict(pen=penalty)
    logger.info("ruptures: %d puntos de cambio (penalty=%s, model=%s)", len(breakpoints) - 1, penalty, model)
    return breakpoints


def compare_bruto_vs_depurado(
    df_raw: pl.DataFrame,
    df_clean: pl.DataFrame,
    col: str,
    low: float,
    high: float,
) -> dict[str, float]:
    """Resumen side-by-side del pct de violacion antes y despues de depurar.

    Este es el numero del deck: cuanto cae la tasa de violacion DIGESA cuando
    descartas los glitches fisicamente imposibles.
    """
    def pct_viol(df: pl.DataFrame) -> float:
        if col not in df.columns:
            return float("nan")
        series = df.get_column(col).drop_nulls()
        if series.is_empty():
            return 0.0
        viol = int(((series < low) | (series > high)).sum())
        return 100.0 * viol / series.len()

    bruto = pct_viol(df_raw)
    depurado = pct_viol(df_clean)
    resumen = {
        "pct_violacion_bruto": round(bruto, 2),
        "pct_violacion_depurado": round(depurado, 2),
        "delta_pp": round(bruto - depurado, 2),
        "n_bruto": df_raw.height,
        "n_depurado": df_clean.height,
    }
    logger.info("compare_bruto_vs_depurado(%s): %s", col, resumen)
    return resumen
