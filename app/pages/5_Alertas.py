"""Pagina 5 · Alertas — umbrales DIGESA con auto-refresh simulando streaming."""
from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import polars as pl
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.components.auth import require_auth  # noqa: E402

from app.components.data_loader import data_available, get_morea_sensores  # noqa: E402
from app.components.theme import PALETTE  # noqa: E402
from src.modeling.anomalias import filter_imposibles, sustained_violations  # noqa: E402
from src.monitoring.alerts import build_alerts, summarize_alerts  # noqa: E402
from src.monitoring.incidents import (  # noqa: E402
    IncidentRecord,
    IncidentStatus,
    IncidentStore,
    alert_to_incident_key,
)
from src.monitoring.thresholds import (  # noqa: E402
    DEFAULT_CONFIGS,
    detect_violations,
)

_INCIDENT_STORE = IncidentStore(path=_ROOT / "artifacts" / "incidents.json")

st.set_page_config(page_title="Alertas · SUNASS", layout="wide")
require_auth("Alertas")
st.title("5 · Alertas DIGESA en vivo")
st.caption(
    "Lecturas MOREA contra bandas DIGESA. Severidad por cuanto se excede y "
    "cuanto dura la violacion."
)

ok, msg = data_available()
if not ok:
    st.error(msg)
    st.stop()


@st.cache_data(show_spinner="Preparando dataset depurado...")
def _prepared_morea() -> pl.DataFrame:
    df = get_morea_sensores()
    return filter_imposibles(df).depurado


def _severity_badge(severity: str) -> str:
    colors = {
        "OK": PALETTE.success,
        "INFO": PALETTE.muted,
        "WARN": PALETTE.warning,
        "CRITICAL": PALETTE.danger,
    }
    color = colors.get(severity, PALETTE.muted)
    return (
        f"<span style='background:{color}; color:white; padding:2px 8px; "
        f"border-radius:10px; font-size:0.78rem; font-weight:600;'>{severity}</span>"
    )


with st.sidebar:
    st.markdown("### Configuracion de monitoreo")
    parameter = st.selectbox("Parametro", options=list(DEFAULT_CONFIGS.keys()), index=0)
    cfg = DEFAULT_CONFIGS[parameter]
    st.markdown(f"Banda DIGESA: **{cfg.describe_band()}**")
    limit_events = st.slider("Eventos a mostrar", min_value=10, max_value=200, value=50, step=10)
    min_consecutive = st.slider(
        "Min lecturas consecutivas para 'sostenida'",
        min_value=2,
        max_value=20,
        value=3,
    )

df_clean = _prepared_morea()
if parameter not in df_clean.columns:
    st.error(f"Parametro '{parameter}' no presente en el dataset.")
    st.stop()


@st.cache_data(show_spinner="Escaneando violaciones...")
def _compute(param: str, min_con: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    df_v = detect_violations(df_clean, param, keep_ok=True)
    sustained = sustained_violations(
        df_clean,
        col=param,
        low=cfg.low,
        high=cfg.high,
        min_consecutive=min_con,
    )
    return df_v, sustained


df_v, df_sostenida = _compute(parameter, min_consecutive)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Lecturas (depuradas)", f"{df_clean.height:,}")
n_viol = int(df_v.get_column("viola").sum())
col2.metric("Violaciones puntuales", f"{n_viol:,}", f"{100*n_viol/max(df_clean.height,1):.1f}%")
n_sost = int(df_sostenida.get_column("sostenida").sum())
col3.metric(
    f"Sostenidas ≥{min_consecutive}",
    f"{n_sost:,}",
    f"{100*n_sost/max(n_viol,1):.1f}% del bruto",
    delta_color="inverse",
)
pct_ok = 100 - 100 * n_viol / max(df_clean.height, 1)
col4.metric("Cumplimiento estimado", f"{pct_ok:.2f}%")

st.markdown("---")

st.subheader("Eventos activos")
events = build_alerts(df_v.filter(pl.col("viola")), parameter=parameter)
if not events:
    st.success("Sin violaciones detectadas con la configuracion actual.")
else:
    tabla = summarize_alerts(events[:limit_events])
    pdf = tabla.to_pandas()
    pdf["severity"] = pdf["severity"].apply(_severity_badge)
    st.write(
        pdf.to_html(escape=False, index=False, classes="dataframe", border=0),
        unsafe_allow_html=True,
    )
    st.caption(
        f"Mostrando {min(limit_events, len(events))} de {len(events)} eventos "
        "(ordenados por severidad y duracion)."
    )

st.markdown("---")

st.subheader("Distribucion de severidad")
sev_counts = (
    pl.DataFrame({"severity": [e.severity.value for e in events]})
    .group_by("severity")
    .agg(pl.len().alias("n"))
    .sort("n", descending=True)
) if events else pl.DataFrame({"severity": [], "n": []})

if sev_counts.height > 0:
    fig = px.bar(
        sev_counts.to_pandas(),
        x="severity",
        y="n",
        color="severity",
        color_discrete_map={
            "INFO": PALETTE.muted,
            "WARN": PALETTE.warning,
            "CRITICAL": PALETTE.danger,
        },
    )
    fig.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig, width="stretch")

st.markdown("---")

st.subheader("Top estaciones con violaciones sostenidas")
top = (
    df_sostenida.filter(pl.col("sostenida"))
    .group_by("estacion_id")
    .agg(pl.len().alias("n_sostenida"))
    .sort("n_sostenida", descending=True)
    .head(15)
)
if top.height > 0:
    fig = px.bar(
        top.to_pandas(),
        x="estacion_id",
        y="n_sostenida",
        color="n_sostenida",
        color_continuous_scale="Reds",
    )
    fig.update_layout(height=320, xaxis_title="estacion", yaxis_title="lecturas sostenidas")
    st.plotly_chart(fig, width="stretch")
else:
    st.info("No hay estaciones con violaciones sostenidas segun el umbral actual.")

st.markdown("---")
st.subheader("Atencion de incidencias (workflow)")
st.caption(
    "Persiste el estado de cada alerta en artifacts/incidents.json. Cubre el "
    "resultado oficial 7 (Atencion oportuna de incidencias)."
)

if not events:
    st.info("Sin incidencias pendientes.")
else:
    records = _INCIDENT_STORE.load()
    # Reconciliacion: crea IncidentRecord NUEVO para alertas no vistas.
    for e in events[:limit_events]:
        key = alert_to_incident_key(e.station_id, e.parameter, e.start_ts)
        if key not in records:
            rec = IncidentRecord(
                alert_key=key,
                station_id=e.station_id,
                parameter=e.parameter,
                severity=e.severity.value,
                start_ts=e.start_ts.isoformat() if hasattr(e.start_ts, "isoformat") else str(e.start_ts),
                end_ts=e.end_ts.isoformat() if hasattr(e.end_ts, "isoformat") else str(e.end_ts),
                peak_value=float(e.peak_value),
                duration_min=float(e.duration_minutes),
            )
            _INCIDENT_STORE.upsert(rec)
    records = _INCIDENT_STORE.load()

    # Contadores por estado.
    status_counts = {s.value: 0 for s in IncidentStatus}
    for rec in records.values():
        status_counts[rec.status.value] += 1
    cs1, cs2, cs3 = st.columns(3)
    cs1.metric("Nuevas", status_counts["NUEVO"])
    cs2.metric("En revision", status_counts["EN_REVISION"])
    cs3.metric("Resueltas", status_counts["RESUELTO"])

    status_filter = st.multiselect(
        "Filtrar por estado",
        options=[s.value for s in IncidentStatus],
        default=["NUEVO", "EN_REVISION"],
    )
    filtered = [
        r for r in records.values()
        if r.status.value in status_filter and r.parameter == parameter
    ]
    filtered = sorted(filtered, key=lambda r: (r.status.value != "NUEVO", r.start_ts), reverse=False)[:limit_events]

    if not filtered:
        st.caption("Sin incidencias que coincidan con el filtro actual.")
    for rec in filtered:
        with st.expander(
            f"{_severity_badge(rec.severity)} · Estacion {rec.station_id} · "
            f"{rec.parameter} · {rec.status.value} · inicio {rec.start_ts[:19]}",
            expanded=False,
        ):
            cols = st.columns([2, 1, 1])
            with cols[0]:
                st.markdown(
                    f"""
                    **ID:** `{rec.id}` · **Duracion:** {rec.duration_min:.1f} min ·
                    **Peak:** {rec.peak_value:.2f}
                    """
                )
                new_note = st.text_input(
                    "Nota nueva",
                    key=f"note_{rec.id}",
                    placeholder="Ej. Tecnico notificado a las 10:45",
                )
                new_assignee = st.text_input(
                    "Asignado a",
                    value=rec.assignee or "",
                    key=f"assignee_{rec.id}",
                )
            with cols[1]:
                allowed_next = rec.status.next_allowed()
                if allowed_next:
                    target = st.selectbox(
                        "Transicionar a",
                        options=["(mantener)", *[s.value for s in allowed_next]],
                        key=f"target_{rec.id}",
                    )
                else:
                    target = "(mantener)"
                    st.caption("Estado terminal")
            with cols[2]:
                if st.button("Guardar cambios", key=f"save_{rec.id}"):
                    try:
                        if target != "(mantener)":
                            rec.transition_to(IncidentStatus(target), assignee=new_assignee or None, note=new_note)
                        else:
                            if new_assignee and new_assignee != (rec.assignee or ""):
                                rec.assignee = new_assignee
                            if new_note:
                                rec.notes = (
                                    f"{rec.notes}\n" if rec.notes else ""
                                ) + f"[{rec.updated_at}] {new_note}"
                        _INCIDENT_STORE.save(records)
                        st.success("Guardado.")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
            if rec.notes:
                st.code(rec.notes, language="text")

st.caption(
    "Streaming real se habilita montando un productor externo sobre el mismo "
    "detector; aqui se simula consumiendo el historico en chunks."
)
