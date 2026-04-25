"""Pagina 2 · EDA — heatmap de correlacion + distribuciones + timelines."""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.components.auth import require_auth  # noqa: E402

from app.components.data_loader import (  # noqa: E402
    data_available,
    get_interrupciones,
    get_morea_sensores,
)
from src.modeling.anomalias import filter_imposibles  # noqa: E402
from src.viz.eda import (  # noqa: E402
    boxplot_by_group,
    correlation_heatmap,
    distribution_histogram,
    interrupciones_timeline,
    morea_sensor_timeline,
)

st.set_page_config(page_title="EDA · SUNASS", layout="wide")
require_auth("EDA")
st.title("2 · Analisis exploratorio")
st.caption("Correlaciones, distribuciones, series temporales.")

ok, msg = data_available()
if not ok:
    st.error(msg)
    st.stop()


def _heatmap_block() -> None:
    st.subheader("Matriz de correlacion (Spearman)")
    st.write(
        "Spearman captura relaciones monotonas no-lineales entre duracion, "
        "unidades afectadas y metricas derivadas."
    )
    df = get_interrupciones(enriched=True)
    candidates = [
        "duracion_horas",
        "n_afectadas",
        "impacto",
        "Ndeconexionesdomiciliariasa",
        "Unidadesdeusoafectadas",
    ]
    numeric_cols = [
        c for c in candidates if c in df.columns and df.schema[c].is_numeric()
    ]
    method = st.radio("Metodo", options=["spearman", "pearson"], horizontal=True)
    fig = correlation_heatmap(df, numeric_cols=numeric_cols, method=method)
    st.plotly_chart(fig, width="stretch")


def _distribution_block() -> None:
    st.subheader("Distribucion de variables clave")
    df = get_interrupciones(enriched=True)
    col1, col2 = st.columns(2)
    with col1:
        if "duracion_horas" in df.columns:
            fig = distribution_histogram(
                df, "duracion_horas", bins=80, log_y=True,
                reference_lines=[(24.0, "1 dia"), (72.0, "3 dias")],
            )
            st.plotly_chart(fig, width="stretch")
    with col2:
        if "n_afectadas" in df.columns:
            fig = distribution_histogram(
                df, "n_afectadas", bins=80, log_y=True,
                reference_lines=[(10_000.0, "10k")],
            )
            st.plotly_chart(fig, width="stretch")


def _timeline_block() -> None:
    st.subheader("Evolucion temporal de eventos")
    df = get_interrupciones(enriched=True)
    freq = st.selectbox("Agregacion temporal", options=["1mo", "1w", "1q"], index=0)
    fig = interrupciones_timeline(df, ts_col="ts_inicio", freq=freq)
    st.plotly_chart(fig, width="stretch")


def _morea_block() -> None:
    st.subheader("Series MOREA con bandas DIGESA")
    df = get_morea_sensores()
    depurado = filter_imposibles(df).depurado
    station_ids = depurado.get_column("estacion_id").drop_nulls().unique().sort().to_list()
    if not station_ids:
        st.warning("Sin estaciones con datos tras depuracion")
        return
    station_id = st.selectbox("Estacion", options=station_ids, index=0)
    col1, col2 = st.columns(2)
    with col1:
        fig_cl = morea_sensor_timeline(
            depurado,
            value_col="cloro",
            station_id=station_id,
            band=(0.5, 5.0),
        )
        st.plotly_chart(fig_cl, width="stretch")
    with col2:
        fig_ph = morea_sensor_timeline(
            depurado,
            value_col="ph",
            station_id=station_id,
            band=(6.5, 8.5),
        )
        st.plotly_chart(fig_ph, width="stretch")


def _boxplot_block() -> None:
    st.subheader("Duracion de eventos por motivo")
    df = get_interrupciones(enriched=True)
    motivo_cols = [c for c in df.columns if c.startswith("Motivo")]
    if not motivo_cols or "duracion_horas" not in df.columns:
        st.info("No hay columna de motivo disponible para agrupar.")
        return
    group_col = st.selectbox("Agrupar por", options=motivo_cols)
    fig = boxplot_by_group(
        df.filter(pl.col("duracion_horas") > 0),
        value_col="duracion_horas",
        group_col=group_col,
        top_k=12,
        log_y=True,
    )
    st.plotly_chart(fig, width="stretch")


_heatmap_block()
st.markdown("---")
_distribution_block()
st.markdown("---")
_timeline_block()
st.markdown("---")
_morea_block()
st.markdown("---")
_boxplot_block()
