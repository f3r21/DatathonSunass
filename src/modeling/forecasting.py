"""Plantilla C — Forecasting de series temporales SUNASS.

Predice metricas mensuales agregadas (p.e. horas_interrumpidas por region,
n_eventos_criticos por EP) para el horizonte h con tres capas justificadas
por la literatura:

    1. Baselines estadisticos (statsmodels): ETS, SARIMA, Holt-Winters.
       Referencias:
         - Hyndman & Athanasopoulos (2021), Forecasting: Principles and
           Practice (3ra ed.), OTexts — manual canonico de ETS/ARIMA.
         - Hyndman, Koehler, Snyder & Grose (2002), "A state space framework
           for automatic forecasting using exponential smoothing methods",
           IJF — base teorica del ETS.
         - Makridakis et al. (2018), "Statistical and Machine Learning
           forecasting methods: Concerns and ways forward", PLoS ONE —
           evidencia de que los estadisticos siguen siendo competitivos.

    2. ML con feature engineering (LightGBM con lags/rolling): patron
       ganador del M5 Accuracy Competition.
       Referencias:
         - Makridakis, Spiliotis & Assimakopoulos (2022), "The M5 accuracy
           competition: Results, findings, and conclusions", IJF — GBMs con
           lags dominaron.
         - Januschowski et al. (2020), "Criteria for classifying forecasting
           methods", IJF — global vs local, ML vs statistical.

    3. Ensemble por promedio / media ponderada.
       Referencias:
         - Bates & Granger (1969), "The combination of forecasts", OR
           Quarterly — teoria de combinacion de pronosticos.
         - Petropoulos et al. (2022), "Forecasting: theory and practice",
           IJF — estado del arte revisado.

Nota sobre transformers (Informer, Autoformer, PatchTST): no se entrenan aqui
por los 7h de competencia, pero se citan en el deck como "siguientes pasos"
con la advertencia de Zeng et al. (2022), "Are Transformers Effective for
Time Series Forecasting?", AAAI 2023 — una regresion lineal simple (DLinear)
les gana en casi todos los benchmarks de LTSF cuando el dataset es chico.

Convenciones:
    - Entrada: polars.DataFrame con columna temporal (mensual) y target.
    - Salida: ForecastResult con fitted/predicted + metricas.
    - Log via logger, sin prints.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForecastResult:
    """Resultado de un forecast: historial, prediccion, metricas sobre holdout."""

    model_name: str
    y_train: list[float]
    y_test: list[float]
    y_hat: list[float]
    mae: float
    rmse: float
    mape: float
    smape: float
    extra: dict[str, Any] = field(default_factory=dict)

    def summary_row(self) -> dict[str, float | str]:
        """Fila para la tabla comparativa del deck."""
        return {
            "modelo": self.model_name,
            "mae": round(self.mae, 3),
            "rmse": round(self.rmse, 3),
            "mape_pct": round(self.mape, 2),
            "smape_pct": round(self.smape, 2),
            "n_test": len(self.y_test),
        }


# ---------------------------------------------------------------------- metricas


def _mae(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_hat)))


def _rmse(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_hat) ** 2)))


def _mape(y_true: np.ndarray, y_hat: np.ndarray, eps: float = 1e-6) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_hat) / denom)) * 100.0)


def _smape(y_true: np.ndarray, y_hat: np.ndarray, eps: float = 1e-6) -> float:
    """Symmetric MAPE — robusto a ceros en y_true. Rango [0, 200].

    Referencia: Hyndman & Koehler (2006), "Another look at measures of
    forecast accuracy", IJF.
    """
    denom = np.maximum((np.abs(y_true) + np.abs(y_hat)) / 2.0, eps)
    return float(np.mean(np.abs(y_true - y_hat) / denom) * 100.0)


def _pack_metrics(
    name: str,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_hat: np.ndarray,
    extra: dict[str, Any] | None = None,
) -> ForecastResult:
    return ForecastResult(
        model_name=name,
        y_train=y_train.tolist(),
        y_test=y_test.tolist(),
        y_hat=y_hat.tolist(),
        mae=_mae(y_test, y_hat),
        rmse=_rmse(y_test, y_hat),
        mape=_mape(y_test, y_hat),
        smape=_smape(y_test, y_hat),
        extra=extra or {},
    )


# ----------------------------------------------------------------- preparacion


def aggregate_monthly(
    df: pl.DataFrame,
    ts_col: str,
    target_col: str,
    group_col: str | None = None,
    agg: str = "sum",
) -> pl.DataFrame:
    """Agrega a mensual — patron obligatorio antes del forecast clasico.

    Args:
        df: DataFrame crudo.
        ts_col: columna Datetime (p.e. ts_inicio).
        target_col: columna numerica a pronosticar.
        group_col: opcional, agrega por grupo (region/EP).
        agg: 'sum' | 'mean' | 'count' | 'max'.

    Returns:
        DataFrame con columnas [group_col?, ym, <target_col>].
    """
    if ts_col not in df.columns:
        raise KeyError(f"ts_col '{ts_col}' ausente")
    if target_col not in df.columns and agg != "count":
        raise KeyError(f"target_col '{target_col}' ausente")

    df_month = df.with_columns(pl.col(ts_col).dt.truncate("1mo").alias("ym"))
    keys: list[str] = ["ym"] if group_col is None else [group_col, "ym"]
    if agg == "sum":
        aggregated = df_month.group_by(keys).agg(pl.col(target_col).sum().alias(target_col))
    elif agg == "mean":
        aggregated = df_month.group_by(keys).agg(pl.col(target_col).mean().alias(target_col))
    elif agg == "count":
        aggregated = df_month.group_by(keys).agg(pl.len().alias(target_col))
    elif agg == "max":
        aggregated = df_month.group_by(keys).agg(pl.col(target_col).max().alias(target_col))
    else:
        raise ValueError(f"agg desconocido: {agg!r}")
    return aggregated.sort(keys)


def train_test_horizon_split(
    series: pl.Series, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split clasico: ultimas `horizon` observaciones al holdout."""
    arr = series.to_numpy().astype(float)
    if len(arr) <= horizon + 6:
        raise ValueError(
            f"Serie muy corta ({len(arr)}) para horizon={horizon}. Min: horizon+7."
        )
    return arr[:-horizon], arr[-horizon:]


# ------------------------------------------------------------- modelos clasicos


def forecast_naive_seasonal(
    y_train: np.ndarray, y_test: np.ndarray, season_length: int = 12
) -> ForecastResult:
    """Naive estacional: y_hat[t] = y_train[t - season_length].

    Benchmark obligatorio (Hyndman & Athanasopoulos cap. 5). Si un modelo
    sofisticado no le gana a esto, no lo reportes.
    """
    h = len(y_test)
    if season_length <= len(y_train):
        last_season = y_train[-season_length:]
        y_hat = np.tile(last_season, int(np.ceil(h / season_length)))[:h]
    else:
        y_hat = np.full(h, y_train[-1])
    return _pack_metrics("naive_seasonal", y_train, y_test, y_hat, {"season_length": season_length})


def forecast_ets(
    y_train: np.ndarray,
    y_test: np.ndarray,
    seasonal_periods: int = 12,
    trend: str | None = "add",
    seasonal: str | None = "add",
) -> ForecastResult:
    """Exponential smoothing (ETS Holt-Winters aditivo por default).

    Referencia: Hyndman, Koehler, Snyder & Grose (2002).
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    h = len(y_test)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            y_train,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        ).fit(optimized=True, use_brute=False)
    y_hat = np.asarray(model.forecast(h))
    return _pack_metrics(
        f"ets_{trend}_{seasonal}",
        y_train,
        y_test,
        y_hat,
        {"aic": float(getattr(model, "aic", np.nan))},
    )


def forecast_sarima(
    y_train: np.ndarray,
    y_test: np.ndarray,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 12),
) -> ForecastResult:
    """SARIMA — Box-Jenkins con componente estacional.

    Referencia: Hyndman & Athanasopoulos cap. 9. En produccion conviene
    grid-search o auto.arima (pmdarima); aqui fijamos un orden conservador.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    h = len(y_test)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            y_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
    y_hat = np.asarray(model.forecast(h))
    return _pack_metrics(
        f"sarima{order}x{seasonal_order}",
        y_train,
        y_test,
        y_hat,
        {"aic": float(model.aic)},
    )


def forecast_holt_winters(
    y_train: np.ndarray,
    y_test: np.ndarray,
    seasonal_periods: int = 12,
) -> ForecastResult:
    """Holt-Winters multiplicativo — util cuando la varianza crece con la media.

    Referencia: Winters (1960), "Forecasting sales by exponentially weighted
    moving averages", Management Science.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    # Multiplicativo requiere valores estrictamente positivos.
    shift = 0.0
    y_safe = y_train
    if (y_train <= 0).any():
        shift = float(-y_train.min()) + 1.0
        y_safe = y_train + shift

    h = len(y_test)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            y_safe,
            trend="add",
            seasonal="mul",
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        ).fit(optimized=True)
    y_hat = np.asarray(model.forecast(h)) - shift
    return _pack_metrics(
        "holt_winters_mul",
        y_train,
        y_test,
        y_hat,
        {"shift_applied": shift},
    )


# --------------------------------------------------------------- modelos ML


def _lag_feature_names(lags: tuple[int, ...], rollings: tuple[int, ...]) -> list[str]:
    """Genera los nombres de columna en el mismo orden que `_build_lag_features`."""
    names = [f"lag_{lag}" for lag in lags]
    for w in rollings:
        names.extend([f"roll_{w}_mean", f"roll_{w}_std"])
    return names


def _build_lag_features(
    series: np.ndarray, lags: tuple[int, ...], rollings: tuple[int, ...]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Construye matriz de features (con nombres) a partir de lags + rolling means.

    Patron M5: cada fila i usa series[i-lag1], series[i-lag2], ..., y medias
    moviles de distintas ventanas. Descarta las primeras filas sin suficiente
    historia. Retorna pd.DataFrame con columnas nombradas para que LGBM/XGB
    no lancen el warning "X does not have valid feature names" al predecir.
    """
    max_lag = max(max(lags), max(rollings))
    n = len(series)
    if n <= max_lag + 1:
        raise ValueError(f"Serie muy corta ({n}) para lags/rollings requeridos.")
    rows_X: list[list[float]] = []
    rows_y: list[float] = []
    for i in range(max_lag, n):
        feats = [float(series[i - lag]) for lag in lags]
        for w in rollings:
            window = series[i - w : i]
            feats.append(float(np.mean(window)))
            feats.append(float(np.std(window, ddof=0)) if w > 1 else 0.0)
        rows_X.append(feats)
        rows_y.append(float(series[i]))
    columns = _lag_feature_names(lags, rollings)
    X_df = pd.DataFrame(rows_X, columns=columns)
    return X_df, np.asarray(rows_y)


def _single_row_features(
    history: list[float], lags: tuple[int, ...], rollings: tuple[int, ...]
) -> pd.DataFrame:
    """Construye una fila DataFrame para predict iterativo, mismas columnas que fit."""
    arr = np.asarray(history, dtype=float)
    feats = [arr[-lag] for lag in lags]
    for w in rollings:
        window = arr[-w:]
        feats.append(float(np.mean(window)))
        feats.append(float(np.std(window, ddof=0)) if w > 1 else 0.0)
    columns = _lag_feature_names(lags, rollings)
    return pd.DataFrame([feats], columns=columns)


def forecast_lgbm_lags(
    y_train: np.ndarray,
    y_test: np.ndarray,
    lags: tuple[int, ...] = (1, 2, 3, 6, 12),
    rollings: tuple[int, ...] = (3, 6, 12),
    n_estimators: int = 300,
    learning_rate: float = 0.05,
) -> ForecastResult:
    """Forecast iterativo con LightGBM y features de lag / rolling.

    Patron ganador M5 (Makridakis et al. 2022). En cada paso del horizonte, se
    predice un solo valor y se realimenta a la serie para el siguiente lag.
    """
    from lightgbm import LGBMRegressor

    X, y = _build_lag_features(y_train, lags, rollings)
    model = LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=31,
        objective="regression",
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y)

    history = list(y_train)
    y_hat: list[float] = []
    for _ in range(len(y_test)):
        x_pred = _single_row_features(history, lags, rollings)
        next_pred = float(model.predict(x_pred)[0])
        y_hat.append(next_pred)
        history.append(next_pred)

    return _pack_metrics(
        "lgbm_lags",
        y_train,
        y_test,
        np.asarray(y_hat),
        {"lags": list(lags), "rollings": list(rollings)},
    )


def forecast_xgb_lags(
    y_train: np.ndarray,
    y_test: np.ndarray,
    lags: tuple[int, ...] = (1, 2, 3, 6, 12),
    rollings: tuple[int, ...] = (3, 6, 12),
    grid_search: bool = False,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 5,
) -> ForecastResult:
    """XGBoost con features de lag/rolling — gemelo del LGBM.

    Si grid_search=True hace un pequeno GridSearchCV sobre n_estimators,
    max_depth, learning_rate usando KFold=3 (time-aware no estricto, basta
    para muestras mensuales limitadas). Si False usa los hiperparametros dados.
    """
    from xgboost import XGBRegressor

    X, y = _build_lag_features(y_train, lags, rollings)

    if grid_search:
        from sklearn.model_selection import GridSearchCV, KFold

        base = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=1,
            random_state=42,
        )
        grid = GridSearchCV(
            base,
            param_grid={
                "n_estimators": [200, 400],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.03, 0.05, 0.1],
            },
            scoring="neg_mean_absolute_error",
            cv=KFold(n_splits=3, shuffle=False),
            n_jobs=-1,
        )
        grid.fit(X, y)
        model = grid.best_estimator_
        extra_grid: dict[str, Any] = {
            "best_params": dict(grid.best_params_),
            "cv_best_mae": -float(grid.best_score_),
        }
    else:
        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X, y)
        extra_grid = {}

    history = list(y_train)
    y_hat: list[float] = []
    for _ in range(len(y_test)):
        x_pred = _single_row_features(history, lags, rollings)
        next_pred = float(model.predict(x_pred)[0])
        y_hat.append(next_pred)
        history.append(next_pred)

    return _pack_metrics(
        "xgb_lags" if not grid_search else "xgb_lags_grid",
        y_train,
        y_test,
        np.asarray(y_hat),
        {"lags": list(lags), "rollings": list(rollings), **extra_grid},
    )


# ------------------------------------------------------------------ ensemble


def forecast_ensemble(
    results: list[ForecastResult],
    weights: list[float] | None = None,
    name: str = "ensemble_mean",
) -> ForecastResult:
    """Combina pronosticos por promedio (o ponderado).

    Referencia teorica: Bates & Granger (1969). En la practica, el promedio
    simple suele ganarle a la combinacion optima cuando la muestra es chica
    (Clemen 1989, "Combining forecasts: A review and annotated bibliography").

    Todas las ForecastResult deben compartir el mismo y_test.
    """
    if not results:
        raise ValueError("Se requiere al menos un resultado para ensemblar.")
    y_test = np.asarray(results[0].y_test)
    y_train = np.asarray(results[0].y_train)
    preds = np.stack([np.asarray(r.y_hat) for r in results], axis=0)

    if weights is None:
        w = np.ones(len(results)) / len(results)
    else:
        if len(weights) != len(results):
            raise ValueError("weights debe tener la misma longitud que results")
        w = np.asarray(weights, dtype=float)
        w = w / w.sum() if w.sum() > 0 else w
    y_hat = (preds * w.reshape(-1, 1)).sum(axis=0)

    return _pack_metrics(
        name,
        y_train,
        y_test,
        y_hat,
        {
            "components": [r.model_name for r in results],
            "weights": w.tolist(),
        },
    )


def compare_forecasts(results: list[ForecastResult]) -> pl.DataFrame:
    """Tabla comparativa de todos los modelos ordenada por SMAPE ascendente."""
    rows = [r.summary_row() for r in results]
    df = pl.DataFrame(rows)
    if "smape_pct" in df.columns:
        df = df.sort("smape_pct")
    return df


# ------------------------------------------------------------ backtest rolante


def walk_forward_backtest(
    series: np.ndarray,
    forecaster,
    horizon: int = 3,
    initial_train: int = 24,
    step: int = 1,
) -> list[ForecastResult]:
    """Backtest rolling-origin (Tashman 2000, "Out-of-sample tests of
    forecasting accuracy").

    Args:
        series: array 1D completo.
        forecaster: callable (y_train, y_test) -> ForecastResult.
        horizon: cuantos pasos hacia adelante predecir en cada origen.
        initial_train: minimo de observaciones para el primer entrenamiento.
        step: cuantas observaciones avanzar el origen entre iteraciones.

    Returns:
        Lista de ForecastResult (uno por origen), en orden temporal.
    """
    if len(series) < initial_train + horizon:
        raise ValueError(f"Serie de largo {len(series)} < initial_train+horizon={initial_train+horizon}")

    results: list[ForecastResult] = []
    origin = initial_train
    while origin + horizon <= len(series):
        y_tr = series[:origin]
        y_te = series[origin : origin + horizon]
        try:
            result = forecaster(y_tr, y_te)
            results.append(result)
        except Exception as exc:
            logger.warning("backtest origin=%d fallo: %s", origin, exc)
        origin += step
    logger.info("walk_forward: %d origenes evaluados, horizon=%d", len(results), horizon)
    return results
