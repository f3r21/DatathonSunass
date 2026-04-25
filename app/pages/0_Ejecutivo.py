"""Pagina 0 · Ejecutivo — tablero de control de un vistazo (Resultado oficial 1).

Traffic-light por estacion, alertas activas por severidad, evento critico mas
reciente y tendencia de horas interrumpidas. La pagina por la que el regulador
entraria al dashboard.
"""
from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go
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
from app.components.theme import PALETTE  # noqa: E402
from src.modeling.anomalias import (  # noqa: E402
    chronic_stations,
    filter_imposibles,
    sustained_violations,
)
from src.monitoring.alerts import build_alerts  # noqa: E402
from src.monitoring.thresholds import DEFAULT_CONFIGS, detect_violations  # noqa: E402

st.set_page_config(page_title="Ejecutivo · SUNASS", layout="wide")
require_auth("Ejecutivo")
st.title("0 · Tablero ejecutivo")
st.caption(
    "Estado operacional del servicio en un vistazo. Continuidad, calidad y "
    "atencion de incidencias en una sola pantalla."
)

ok, msg = data_available()
if not ok:
    st.error(msg)
    st.stop()


def _traffic_light(pct_bad: float) -> tuple[str, str]:
    """4 niveles verde-azul-amber-rojo, escalonados para reducir mar de rojo."""
    if pct_bad < 5:
        return "OK", PALETTE.success
    if pct_bad < 20:
        return "VIGILAR", PALETTE.secondary
    if pct_bad < 50:
        return "ALERTA", PALETTE.warning
    return "CRITICO", PALETTE.danger


@st.cache_data(show_spinner="Calculando tablero...")
def _snapshot() -> dict[str, object]:
    df_int = get_interrupciones(enriched=True)
    df_morea_raw = get_morea_sensores()
    df_morea_clean = filter_imposibles(df_morea_raw).depurado

    # Alertas activas por parametro.
    alerts_by_param: dict[str, int] = {}
    severity_counts: dict[str, int] = {"INFO": 0, "WARN": 0, "CRITICAL": 0}
    for param, cfg in DEFAULT_CONFIGS.items():
        if param not in df_morea_clean.columns:
            continue
        df_v = detect_violations(df_morea_clean, param, keep_ok=True)
        events = build_alerts(df_v.filter(pl.col("viola")), parameter=param)
        alerts_by_param[param] = len(events)
        for e in events:
            severity_counts[e.severity.value] = severity_counts.get(e.severity.value, 0) + 1

    # Ranking estaciones cronicas (cloro, sostenida >=3 lecturas).
    df_sost = sustained_violations(
        df_morea_clean, col="cloro", low=0.5, high=5.0, min_consecutive=3
    )
    cronicas = chronic_stations(df_sost, min_hits=5, top_k=10)

    # Evento critico mas reciente.
    ultimo_critico = {}
    if "evento_critico" in df_int.columns and "ts_inicio" in df_int.columns:
        df_crit = df_int.filter(
            pl.col("evento_critico").fill_null(False) & pl.col("ts_inicio").is_not_null()
        ).sort("ts_inicio", descending=True)
        if df_crit.height > 0:
            r = df_crit.head(1).to_dicts()[0]
            ultimo_critico = {
                "ts": r.get("ts_inicio"),
                "duracion_horas": r.get("duracion_horas"),
                "n_afectadas": r.get("n_afectadas"),
                "motivo": next(
                    (r.get(k) for k in r if isinstance(k, str) and k.startswith("Motivo")),
                    None,
                ),
            }

    # Tendencia mensual total + criticos.
    tendencia = pl.DataFrame()
    if "ts_inicio" in df_int.columns:
        tendencia = (
            df_int.filter(pl.col("ts_inicio").is_not_null())
            .with_columns(pl.col("ts_inicio").dt.truncate("1mo").alias("_periodo"))
            .group_by("_periodo")
            .agg(
                pl.len().alias("total"),
                pl.col("evento_critico").fill_null(False).cast(pl.Int32).sum().alias("criticos"),
            )
            .sort("_periodo")
            .tail(24)
        )

    # KPI global de cumplimiento MOREA.
    pct_cumplimiento = 100.0
    if "cloro" in df_morea_clean.columns:
        viol = detect_violations(df_morea_clean, "cloro", keep_ok=False)
        pct_cumplimiento = 100 - 100 * viol.height / max(df_morea_clean.height, 1)

    return {
        "n_eventos": df_int.height,
        "n_criticos": int(df_int.get_column("evento_critico").fill_null(False).cast(int).sum())
        if "evento_critico" in df_int.columns
        else 0,
        "pct_cumplimiento": pct_cumplimiento,
        "alerts_by_param": alerts_by_param,
        "severity_counts": severity_counts,
        "cronicas": cronicas,
        "ultimo_critico": ultimo_critico,
        "tendencia": tendencia,
    }


snap = _snapshot()

# Fila 1: 4 KPIs grandes.
c1, c2, c3, c4 = st.columns(4)
c1.metric("Eventos totales", f"{snap['n_eventos']:,}")
pct_crit = 100 * snap["n_criticos"] / max(snap["n_eventos"], 1)
c2.metric(
    "Eventos criticos",
    f"{snap['n_criticos']:,}",
    delta=f"{pct_crit:.1f}% del total",
    delta_color="inverse",
)
c3.metric("Cumplimiento cloro MOREA", f"{snap['pct_cumplimiento']:.2f}%")
c4.metric(
    "Alertas CRITICAL activas",
    snap["severity_counts"].get("CRITICAL", 0),
    delta=f"WARN: {snap['severity_counts'].get('WARN', 0)}",
    delta_color="inverse",
)

st.markdown("---")

# Fila 2: semaforos por estacion cronica + tendencia mensual.
col_izq, col_der = st.columns([1.2, 1])

with col_izq:
    st.subheader("Semaforo por estacion MOREA")
    cronicas = snap["cronicas"]
    if cronicas.height == 0:
        st.success("Sin estaciones cronicamente fuera de banda.")
    else:
        rows_html = []
        for row in cronicas.to_dicts():
            label, color = _traffic_light(row["pct_sostenida"])
            rows_html.append(
                f"""
                <div style="display:flex; align-items:center; padding:6px 10px;
                            background:{PALETTE.bg_soft}; margin-bottom:4px;
                            border-left: 4px solid {color}; border-radius:3px;">
                    <div style="flex:1; font-weight:600;">Estacion {row['estacion_id']}</div>
                    <div style="flex:1; color:{PALETTE.muted}; font-size:0.85rem;">
                        {row['n_sostenida']} sostenidas / {row['n_lecturas']} lecturas
                    </div>
                    <div style="width:90px; text-align:right;">
                        <span style="background:{color}; color:white;
                                     padding:2px 10px; border-radius:10px;
                                     font-size:0.75rem; font-weight:700;">
                            {label} {row['pct_sostenida']:.1f}%
                        </span>
                    </div>
                </div>
                """
            )
        st.markdown("\n".join(rows_html), unsafe_allow_html=True)

with col_der:
    st.subheader("Tendencia 24 meses")
    tendencia = snap["tendencia"]
    if tendencia.height == 0:
        st.info("Sin serie temporal disponible.")
    else:
        periodos = tendencia.get_column("_periodo").to_list()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=periodos,
                y=tendencia.get_column("total").to_list(),
                mode="lines+markers",
                name="Eventos",
                line=dict(color=PALETTE.primary, width=2),
                fill="tozeroy",
                fillcolor="rgba(13,59,102,0.08)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=periodos,
                y=tendencia.get_column("criticos").to_list(),
                mode="lines+markers",
                name="Criticos",
                line=dict(color=PALETTE.danger, width=2),
            )
        )
        fig.update_layout(
            margin=dict(l=30, r=10, t=10, b=30),
            height=300,
            hovermode="x unified",
            xaxis_title="",
            yaxis_title="",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, width="stretch")

st.markdown("---")

# Fila 3: ultimo critico + breakdown severidad.
col_a, col_b = st.columns([1.3, 1])

with col_a:
    st.subheader("Ultimo evento critico registrado")
    uc = snap["ultimo_critico"]
    if not uc:
        st.info("Sin eventos criticos en el historico.")
    else:
        duracion = uc.get("duracion_horas") or 0
        afectadas = uc.get("n_afectadas") or 0
        motivo = uc.get("motivo") or "sin especificar"
        ts = uc.get("ts")
        st.markdown(
            f"""
            <div style="padding:1rem; background:{PALETTE.bg_soft};
                        border-left: 4px solid {PALETTE.danger}; border-radius:4px;">
                <div style="font-size:0.85rem; color:{PALETTE.muted};">
                    {ts}
                </div>
                <div style="font-size:1.1rem; font-weight:600; margin:0.3rem 0;">
                    {duracion:.1f} h de interrupcion · {int(afectadas):,} unidades afectadas
                </div>
                <div style="color:{PALETTE.text}; font-size:0.9rem;">
                    Motivo: <strong>{motivo}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with col_b:
    st.subheader("Alertas por severidad")
    sev = snap["severity_counts"]
    total_alerts = sum(sev.values())
    if total_alerts == 0:
        st.success("Sin alertas activas.")
    else:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=["INFO", "WARN", "CRITICAL"],
                    y=[sev.get("INFO", 0), sev.get("WARN", 0), sev.get("CRITICAL", 0)],
                    marker_color=[PALETTE.muted, PALETTE.warning, PALETTE.danger],
                )
            ]
        )
        fig.update_layout(margin=dict(l=30, r=10, t=10, b=30), height=240, showlegend=False)
        st.plotly_chart(fig, width="stretch")

st.caption(
    "Este tablero mapea al resultado oficial 1 (construccion de tableros de "
    "control) y al resultado 4 (deteccion temprana de desviaciones) de la "
    "Categoria I Operacional SUNASS 2026."
)
