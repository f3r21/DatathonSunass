"""Home de la app Datathon SUNASS 2026 — Categoria I Operacional.

Renderiza overview institucional + KPIs principales + mapa rapido de secciones.
Las 5 paginas de la izquierda son: Datos, EDA, Modelo, Forecasting, Alertas.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Permite ejecutar `streamlit run app/Home.py` desde cualquier cwd.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.components.auth import require_auth  # noqa: E402

from app.components.data_loader import (  # noqa: E402
    data_available,
    get_interrupciones,
    get_morea_estaciones,
    get_morea_sensores,
)
from app.components.kpi import KPI, render_kpis  # noqa: E402
from app.components.theme import PALETTE  # noqa: E402

st.set_page_config(
    page_title="SUNASS Datathon 2026 · UCSP",
    page_icon="app/assets/logo.svg" if Path("app/assets/logo.svg").exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)
require_auth("Inicio")


def _header() -> None:
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.markdown(
            f"""
            <div style="padding: 1rem 0;">
                <h1 style="color: {PALETTE.primary}; margin-bottom: 0.25rem;">
                    Calidad operacional del servicio de agua
                </h1>
                <p style="color: {PALETTE.muted}; font-size: 1.05rem; margin: 0;">
                    Interrupciones, eventos criticos y calidad MOREA ·
                    Categoria I Operacional · Datathon SUNASS 2026
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div style="text-align: right; padding: 1rem 0;">
                <div style="color: {PALETTE.muted}; font-size: 0.85rem;">Equipo UCSP</div>
                <div style="color: {PALETTE.text}; font-weight: 600;">Escuela de Ciencia de la Computacion</div>
                <div style="color: {PALETTE.muted}; font-size: 0.8rem;">25 de abril, 2026</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("---")


def _data_status() -> bool:
    ok, msg = data_available()
    if not ok:
        st.warning(f"Datos no accesibles. {msg}. Revisa el archivo `.env` del repo.")
        return False
    return True


def _compute_kpis() -> list[KPI]:
    df_int = get_interrupciones(enriched=True)
    df_morea = get_morea_sensores()
    df_est = get_morea_estaciones()

    n_eventos = df_int.height
    n_estaciones = df_est.height
    n_lecturas_morea = df_morea.height

    # Eventos criticos (duracion_horas * unidades > 100_000).
    criticos = 0
    pct_criticos = 0.0
    if "evento_critico" in df_int.columns:
        criticos = int(df_int.get_column("evento_critico").cast(int).sum())
        pct_criticos = 100 * criticos / max(n_eventos, 1)

    # Rango temporal.
    rango = "s/d"
    if "ts_inicio" in df_int.columns:
        nn = df_int.get_column("ts_inicio").drop_nulls()
        if nn.len() > 0:
            rango = f"{nn.min().date()} a {nn.max().date()}"

    return [
        KPI(
            label="Eventos de interrupcion",
            value=f"{n_eventos:,}",
            help_text=f"Periodo: {rango}",
        ),
        KPI(
            label="Eventos criticos",
            value=f"{criticos:,}",
            delta=f"{pct_criticos:.1f}% del total",
            help_text="duracion_horas x unidades_afectadas > 100.000",
        ),
        KPI(
            label="Lecturas MOREA",
            value=f"{n_lecturas_morea/1_000_000:.2f} M",
            help_text="pH, cloro y temperatura cada ~5 min",
        ),
        KPI(
            label="Estaciones MOREA",
            value=f"{n_estaciones}",
            help_text="Puntos de monitoreo de agua potable",
        ),
    ]


def _sections() -> None:
    st.markdown("### Recorrido de la solucion")
    st.write(
        "Cinco vistas, una narrativa: desde el dato crudo hasta la alerta accionable."
    )
    cards = [
        (
            "1 · Datos",
            "Raw -> limpieza -> glitches descartados.",
            "Explora el volumen, los tipos y como se depuran los sensores MOREA.",
            PALETTE.secondary,
        ),
        (
            "2 · EDA",
            "Distribuciones, correlaciones, heatmap.",
            "Donde estan las relaciones entre duracion, impacto y motivo.",
            PALETTE.success,
        ),
        (
            "3 · Modelo",
            "XGBoost con GridSearchCV.",
            "Clasificador de eventos criticos optimizado por PR-AUC.",
            PALETTE.primary,
        ),
        (
            "4 · Forecasting",
            "ETS, SARIMA, LightGBM, XGBoost, ensemble.",
            "Cinco pronosticos de horas interrumpidas mensuales.",
            PALETTE.accent,
        ),
        (
            "5 · Alertas",
            "Umbrales DIGESA en vivo.",
            "Violaciones sostenidas de cloro y pH con severidad.",
            PALETTE.primary_dark,
        ),
    ]
    cols = st.columns(5)
    for col, (title, subtitle, body, color) in zip(cols, cards, strict=True):
        with col:
            st.markdown(
                f"""
                <div style="border-top: 3px solid {color}; padding: 0.8rem;
                            background: {PALETTE.bg_soft}; border-radius: 4px;
                            min-height: 180px;">
                    <div style="font-weight: 700; color: {PALETTE.text};">{title}</div>
                    <div style="color: {color}; font-size: 0.85rem; margin: 0.2rem 0 0.5rem;">
                        {subtitle}
                    </div>
                    <div style="color: {PALETTE.muted}; font-size: 0.85rem;">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _pitch() -> None:
    st.markdown("### Propuesta en una frase")
    st.info(
        "Separar sensor glitches de violaciones reales reduce la tasa DIGESA "
        "reportada en ~30 puntos porcentuales y permite pronosticar horas de "
        "interrupcion a 3 meses con error menor que la variabilidad natural."
    )


def main() -> None:
    _header()
    if not _data_status():
        st.info(
            "Configura el archivo `.env` con `INTERRUPCIONES_PATH`, "
            "`MOREA_PARQUET_PATH` y `MOREA_ESTACIONES_PATH`, luego recarga."
        )
        return
    with st.spinner("Calculando KPIs..."):
        kpis = _compute_kpis()
    render_kpis(kpis, n_cols=4)
    st.markdown("")
    _pitch()
    st.markdown("")
    _sections()
    st.markdown("---")
    st.caption(
        "UCSP - Escuela de Ciencia de la Computacion · "
        "Codigo y notebooks en el repositorio del proyecto."
    )


main()
