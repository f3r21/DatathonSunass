"""Pagina 6 · Modo Dia D — upload de CSV/XLSX climatico SENAMHI y pipeline completo.

Cubre resultado oficial 2 (automatizacion de reportes) + 6 (IA aplicada).
Acepta el schema oficial (Año, Mes, Dia, Precipitacion acumulada, Tmax, Tmin),
corre clean, EDA, deteccion de eventos extremos y forecasting sobre precipitacion.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.components.auth import require_auth  # noqa: E402

from app.components.theme import PALETTE  # noqa: E402
from src.io import load_senamhi_daily  # noqa: E402
from src.modeling.forecasting import (  # noqa: E402
    compare_forecasts,
    forecast_ensemble,
    forecast_ets,
    forecast_lgbm_lags,
    forecast_naive_seasonal,
    forecast_xgb_lags,
    train_test_horizon_split,
)
from src.monitoring.climate_thresholds import (  # noqa: E402
    DEFAULT_CLIMATE_THRESHOLDS,
    detect_climate_events,
)
from src.viz.eda import correlation_heatmap, distribution_histogram  # noqa: E402

st.set_page_config(page_title="Modo Dia D · SUNASS", layout="wide")
require_auth("Modo Dia D")
st.title("6 · Modo Dia D — dataset climatico")
st.caption(
    "Sube el CSV/XLSX oficial del dia D con columnas Año, Mes, Dia, "
    "Precipitacion acumulada, Temperatura maxima, Temperatura minima. "
    "El pipeline corre end-to-end sobre el archivo sin reentrenar los modelos "
    "base con MOREA/Interrupciones."
)

uploaded = st.file_uploader(
    "Arrastra aqui el dataset del dia D",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=False,
)

if uploaded is None:
    st.info(
        "Formatos aceptados: .csv, .xlsx, .xls. Headers pueden venir con tildes "
        "(Año, Día, Precipitación, Temperatura máxima, mínima). El orden no "
        "importa — se detectan los headers automaticamente."
    )
    st.stop()

tmp_path = Path(st.session_state.get("_tmp_dir", "/tmp")) / uploaded.name
tmp_path.parent.mkdir(parents=True, exist_ok=True)
tmp_path.write_bytes(uploaded.getvalue())

try:
    df = load_senamhi_daily(tmp_path)
except Exception as exc:
    st.error(f"No se pudo parsear el archivo: {exc}")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Filas", f"{df.height:,}")
col2.metric("Periodo", f"{df['fecha'].min().date()} a {df['fecha'].max().date()}")
col3.metric("Columnas detectadas", df.width)
col4.metric("Dias validos", int(df['fecha'].is_not_null().sum()))

st.dataframe(df.head(20), width="stretch")

st.markdown("---")

tab_eda, tab_eventos, tab_forecast = st.tabs(
    ["EDA", "Eventos extremos", "Forecasting"]
)

with tab_eda:
    st.subheader("Correlaciones climaticas")
    numeric = [c for c in ("precip_acum", "tmax", "tmin") if c in df.columns]
    if len(numeric) < 2:
        st.info("Se requieren al menos 2 columnas numericas para correlacion.")
    else:
        fig = correlation_heatmap(df, numeric_cols=numeric, method="spearman")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Distribuciones")
    cols = st.columns(len(numeric))
    for col_ui, numeric_col in zip(cols, numeric, strict=False):
        with col_ui:
            fig = distribution_histogram(df, numeric_col, bins=60, log_y=False)
            st.plotly_chart(fig, width="stretch")

with tab_eventos:
    st.subheader("Detector de eventos climaticos extremos")
    if "precip_acum" not in df.columns and "tmax" not in df.columns:
        st.info("No hay columnas climaticas detectadas.")
    else:
        choice = st.selectbox(
            "Umbral",
            options=list(DEFAULT_CLIMATE_THRESHOLDS.keys()),
            index=0,
        )
        cfg = DEFAULT_CLIMATE_THRESHOLDS[choice]
        if cfg.parameter not in df.columns:
            st.warning(f"Parametro '{cfg.parameter}' no esta en el archivo subido.")
        else:
            df_events = detect_climate_events(df, cfg)
            n_events = int(df_events.get_column("es_evento").sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Dias con evento", f"{n_events:,}")
            c2.metric(
                "Umbral efectivo",
                f"{df_events['umbral'][0]:.2f} {cfg.unit}",
            )
            c3.metric(
                "% de la serie",
                f"{100*n_events/max(df_events.height,1):.2f}%",
            )

            # Timeline con marcadores de eventos.
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df_events["fecha"].to_list(),
                    y=df_events[cfg.parameter].to_list(),
                    mode="lines",
                    name=cfg.parameter,
                    line=dict(color=PALETTE.secondary, width=1),
                )
            )
            eventos = df_events.filter(pl.col("es_evento"))
            if eventos.height > 0:
                color_map = {
                    "INFO": PALETTE.muted,
                    "WARN": PALETTE.warning,
                    "CRITICAL": PALETTE.danger,
                }
                for sev, color in color_map.items():
                    sub = eventos.filter(pl.col("severidad") == sev)
                    if sub.height > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=sub["fecha"].to_list(),
                                y=sub[cfg.parameter].to_list(),
                                mode="markers",
                                name=sev,
                                marker=dict(size=7, color=color, line=dict(width=0.5, color="white")),
                            )
                        )
            fig.add_hline(
                y=df_events["umbral"][0],
                line_dash="dash",
                line_color=PALETTE.danger,
                annotation_text=f"umbral {cfg.label}",
                annotation_position="top right",
            )
            fig.update_layout(
                title=f"{cfg.label} sobre {cfg.parameter}",
                xaxis_title="fecha",
                yaxis_title=f"{cfg.parameter} ({cfg.unit})",
                height=440,
                hovermode="x unified",
            )
            st.plotly_chart(fig, width="stretch")

            st.subheader("Tabla de eventos")
            if eventos.height > 0:
                cols_show = ["fecha", cfg.parameter, "exceso", "severidad"]
                st.dataframe(
                    eventos.select([c for c in cols_show if c in eventos.columns]).head(200),
                    width="stretch",
                )
            else:
                st.success("Sin eventos extremos detectados con este umbral.")

with tab_forecast:
    st.subheader("Forecasting de la serie climatica")
    target = st.selectbox(
        "Variable a pronosticar",
        options=[c for c in ("precip_acum", "tmax", "tmin") if c in df.columns],
    )
    agg_choice = st.radio(
        "Granularidad",
        options=["semanal", "mensual"],
        horizontal=True,
    )
    horizon = st.slider("Horizonte", 3, 12, 3)

    trunc = "1w" if agg_choice == "semanal" else "1mo"
    agg = "sum" if target == "precip_acum" else "mean"

    df_valid = df.filter(pl.col(target).is_not_null())
    if df_valid.is_empty():
        st.warning(f"Sin datos validos para {target}.")
        st.stop()

    agg_expr = (
        pl.col(target).sum().alias(target) if agg == "sum" else pl.col(target).mean().alias(target)
    )
    serie = (
        df_valid.with_columns(pl.col("fecha").dt.truncate(trunc).alias("_periodo"))
        .group_by("_periodo")
        .agg(agg_expr)
        .sort("_periodo")
    )
    if serie.height < horizon + 12:
        st.warning(
            f"Serie {agg_choice} tiene {serie.height} puntos; se requieren al "
            f"menos {horizon + 12} para un holdout robusto."
        )
        st.stop()

    values = serie.get_column(target)
    y_train, y_test = train_test_horizon_split(values, horizon)

    season = 52 if agg_choice == "semanal" else 12
    results = []
    results.append(forecast_naive_seasonal(y_train, y_test, season_length=season))
    try:
        results.append(forecast_ets(y_train, y_test, seasonal_periods=season))
    except Exception as exc:
        st.info(f"ETS no aplica: {exc}")
    try:
        results.append(forecast_lgbm_lags(y_train, y_test))
    except Exception as exc:
        st.info(f"LGBM no aplica: {exc}")
    try:
        results.append(forecast_xgb_lags(y_train, y_test, grid_search=False))
    except Exception as exc:
        st.info(f"XGBoost no aplica: {exc}")
    if len(results) >= 2:
        results.append(forecast_ensemble(results, name="ensemble_mean"))

    periodos = serie.get_column("_periodo").to_list()
    train_x = periodos[: len(y_train)]
    test_x = periodos[len(y_train):]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=train_x, y=y_train.tolist(), name="train", mode="lines", line=dict(color=PALETTE.muted))
    )
    fig.add_trace(
        go.Scatter(x=test_x, y=y_test.tolist(), name="real", mode="lines+markers", line=dict(color=PALETTE.text, width=2))
    )
    palette_cycle = [PALETTE.primary, PALETTE.success, PALETTE.warning, PALETTE.secondary, PALETTE.danger]
    for r, color in zip(results, palette_cycle, strict=False):
        fig.add_trace(
            go.Scatter(x=test_x, y=r.y_hat, name=r.model_name, mode="lines+markers", line=dict(color=color, dash="dot"))
        )
    fig.update_layout(
        title=f"Forecast {target} {agg_choice} · horizonte {horizon}",
        xaxis_title="periodo",
        yaxis_title=target,
        height=440,
        hovermode="x unified",
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Ranking sMAPE")
    st.dataframe(compare_forecasts(results), width="stretch")
