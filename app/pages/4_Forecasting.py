"""Pagina 4 · Forecasting — comparativa de modelos mensuales."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.components.auth import require_auth  # noqa: E402

from app.components.data_loader import data_available, get_interrupciones  # noqa: E402
from app.components.theme import PALETTE  # noqa: E402
from src.modeling.forecasting import (  # noqa: E402
    ForecastResult,
    aggregate_monthly,
    compare_forecasts,
    forecast_ensemble,
    forecast_ets,
    forecast_lgbm_lags,
    forecast_naive_seasonal,
    forecast_sarima,
    forecast_xgb_lags,
    train_test_horizon_split,
)

st.set_page_config(page_title="Forecasting · SUNASS", layout="wide")
require_auth("Forecasting")
st.title("4 · Pronostico de horas de interrupcion")
st.caption(
    "Seis modelos comparados sobre holdout de 3 meses: naive estacional, ETS, "
    "SARIMA, LightGBM, XGBoost y ensemble promedio."
)

ok, msg = data_available()
if not ok:
    st.error(msg)
    st.stop()


@st.cache_data(show_spinner="Agregando serie mensual...")
def _monthly_series() -> pl.DataFrame:
    df = get_interrupciones(enriched=True)
    if "ts_inicio" not in df.columns or "duracion_horas" not in df.columns:
        raise RuntimeError("Faltan columnas enriquecidas.")
    # Excluir el mes en curso: puede estar incompleto y distorsiona el holdout.
    import datetime as _dt
    today = _dt.date.today()
    month_start = _dt.date(today.year, today.month, 1)
    df_valid = df.filter(
        pl.col("ts_inicio").is_not_null()
        & pl.col("duracion_horas").is_not_null()
        & (pl.col("ts_inicio").dt.date() < pl.lit(month_start))
    )
    return aggregate_monthly(df_valid, "ts_inicio", "duracion_horas", agg="sum")


@st.cache_resource(show_spinner="Entrenando 6 modelos de forecasting...")
def _run_forecasts(horizon: int) -> list[ForecastResult]:
    monthly = _monthly_series()
    if monthly.height < horizon + 12:
        raise RuntimeError(
            f"Serie demasiado corta ({monthly.height} meses) para horizon={horizon}."
        )
    series = monthly.get_column("duracion_horas")
    y_train, y_test = train_test_horizon_split(series, horizon)

    results: list[ForecastResult] = []
    results.append(forecast_naive_seasonal(y_train, y_test, season_length=12))
    try:
        results.append(forecast_ets(y_train, y_test, seasonal_periods=12))
    except Exception as exc:
        st.warning(f"ETS fallo: {exc}")
    try:
        results.append(forecast_sarima(y_train, y_test))
    except Exception as exc:
        st.warning(f"SARIMA fallo: {exc}")
    try:
        results.append(forecast_lgbm_lags(y_train, y_test))
    except Exception as exc:
        st.warning(f"LGBM fallo: {exc}")
    try:
        results.append(forecast_xgb_lags(y_train, y_test, grid_search=True))
    except Exception as exc:
        st.warning(f"XGBoost fallo: {exc}")

    # Ensemble con los 2 mejores modelos por sMAPE (evita arrastrar los peores).
    if len(results) >= 2:
        top2 = sorted(results, key=lambda r: r.smape)[:2]
        results.append(forecast_ensemble(top2, name="ensemble_top2"))
    return results


def _chart_forecast(results: list[ForecastResult], monthly: pl.DataFrame) -> go.Figure:
    fig = go.Figure()
    periods = monthly.get_column("ym").to_list()
    values = monthly.get_column("duracion_horas").to_list()
    n_train = len(results[0].y_train)
    test_x = periods[n_train:]

    fig.add_trace(
        go.Scatter(x=periods, y=values, mode="lines", name="historico", line=dict(color=PALETTE.muted))
    )
    palette = [PALETTE.primary, PALETTE.success, PALETTE.warning, PALETTE.secondary, PALETTE.danger, PALETTE.accent]
    for r, color in zip(results, palette, strict=False):
        fig.add_trace(
            go.Scatter(x=test_x, y=r.y_hat, mode="lines+markers", name=r.model_name, line=dict(color=color, dash="dot"))
        )
    fig.update_layout(
        title="Pronostico de horas de interrupcion (mensual)",
        xaxis_title="periodo",
        yaxis_title="horas totales",
        hovermode="x unified",
        height=480,
    )
    return fig


def _chart_metrics(results: list[ForecastResult]) -> go.Figure:
    table = compare_forecasts(results)
    fig = go.Figure(
        data=[
            go.Bar(
                x=table.get_column("modelo").to_list(),
                y=table.get_column("smape_pct").to_list(),
                marker_color=PALETTE.primary,
                name="sMAPE %",
            )
        ]
    )
    fig.update_layout(
        title="Error sMAPE por modelo (menor es mejor)",
        xaxis_title="modelo",
        yaxis_title="sMAPE %",
        height=360,
    )
    return fig


horizon = st.slider("Horizonte (meses de holdout)", min_value=3, max_value=12, value=3, step=1)
try:
    monthly = _monthly_series()
    results = _run_forecasts(horizon)
except Exception as exc:
    st.error(f"Error en forecasting: {exc}")
    st.stop()

col1, col2 = st.columns([2, 1])
with col1:
    st.plotly_chart(_chart_forecast(results, monthly), width="stretch")
with col2:
    st.plotly_chart(_chart_metrics(results), width="stretch")

st.subheader("Tabla comparativa")
st.dataframe(compare_forecasts(results), width="stretch")
st.caption(
    "El ensemble por promedio suele ganarle a la combinacion optima cuando la "
    "muestra es chica (Clemen 1989)."
)
