"""Pagina 9 · Streaming — auto-refresh con notificaciones live de violaciones DIGESA.

Cubre el resultado oficial 4 (deteccion temprana de desviaciones) en modo
"productor en vivo": cada N segundos avanzamos un offset sobre el dataset MOREA
depurado, escaneamos violaciones y mostramos las nuevas como toasts.
"""
from __future__ import annotations

import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.components.auth import require_auth  # noqa: E402
from app.components.data_loader import data_available, get_morea_sensores  # noqa: E402
from app.components.theme import PALETTE  # noqa: E402
from src.modeling.anomalias import filter_imposibles  # noqa: E402
from src.monitoring.thresholds import DEFAULT_CONFIGS, detect_violations  # noqa: E402

st.set_page_config(page_title="Streaming · SUNASS", layout="wide")
require_auth("Streaming")
st.title("9 · Streaming en vivo")
st.caption(
    "El historico se itera en chunks y se reportan violaciones nuevas como si "
    "estuvieran llegando en tiempo real. Toast por cada CRITICAL detectado."
)

ok, msg = data_available()
if not ok:
    st.error(msg)
    st.stop()


@st.cache_data(show_spinner="Preparando flujo MOREA...")
def _stream_source() -> pl.DataFrame:
    return filter_imposibles(get_morea_sensores()).depurado.sort("fecha")


with st.sidebar:
    st.markdown("### Configuracion del stream")
    parametro = st.selectbox(
        "Parametro a monitorear",
        options=list(DEFAULT_CONFIGS.keys()),
        index=0,
    )
    chunk_size = st.slider("Lecturas por tick", 100, 5000, 1000, step=100)
    refresh_seconds = st.slider("Intervalo de refresh (s)", 2, 30, 5)
    auto = st.toggle("Auto-refresh", value=True)
    btn_advance = st.button("Avanzar 1 tick manualmente", width="stretch")
    btn_reset = st.button("Reiniciar stream", type="secondary", width="stretch")


df_stream = _stream_source()

# Estado por parametro: offset y feed acumulado.
state_key = f"stream_{parametro}"
default_state = {"offset": 0, "feed": [], "started_at": datetime.now(UTC).isoformat()}
if btn_reset or state_key not in st.session_state:
    st.session_state[state_key] = default_state.copy()

state = st.session_state[state_key]


def _advance_one_tick() -> int:
    """Avanza el offset, detecta violaciones nuevas y las mete al feed."""
    cfg = DEFAULT_CONFIGS[parametro]
    if parametro not in df_stream.columns:
        return 0
    start = state["offset"]
    end = min(start + chunk_size, df_stream.height)
    if end <= start:
        return 0
    chunk = df_stream.slice(start, end - start)
    detected = detect_violations(chunk, parametro, keep_ok=False)
    n_new = detected.height
    for row in detected.iter_rows(named=True):
        state["feed"].append(
            {
                "ts": row.get("fecha"),
                "estacion_id": row.get("estacion_id"),
                "valor": row.get(parametro),
                "exceso": row.get("exceso"),
                "severidad": row.get("severidad"),
                "detectado_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        if row.get("severidad") == "CRITICAL":
            st.toast(
                f"CRITICAL · estacion {row.get('estacion_id')} · "
                f"{parametro}={row.get(parametro):.2f} (banda {cfg.describe_band()})",
                icon="!",
            )
    # Mantener feed acotado.
    if len(state["feed"]) > 500:
        state["feed"] = state["feed"][-500:]
    state["offset"] = end
    return n_new


if btn_advance or (auto and state["offset"] < df_stream.height):
    n_new = _advance_one_tick()
    if n_new == 0 and auto and state["offset"] >= df_stream.height:
        st.success("Stream agotado. Reinicia o cambia el parametro.")

# KPIs de stream.
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Lecturas procesadas",
    f"{state['offset']:,}",
    delta=f"{state['offset']/max(df_stream.height,1)*100:.1f}% del historico",
)
c2.metric("Pendientes", f"{max(df_stream.height - state['offset'], 0):,}")
c3.metric("Violaciones acumuladas", f"{len(state['feed']):,}")
crits = sum(1 for r in state["feed"] if r["severidad"] == "CRITICAL")
c4.metric("CRITICAL detectadas", f"{crits:,}")

st.markdown("---")

st.subheader("Feed en tiempo real")
if not state["feed"]:
    st.info("Esperando primera violacion. Sube el chunk size o pulsa 'Avanzar 1 tick'.")
else:
    feed_df = pl.DataFrame(state["feed"][-50:][::-1])
    st.dataframe(feed_df, width="stretch", height=380)
    st.caption("Mostrando las 50 violaciones mas recientes. Newest on top.")

# Auto-refresh nativo: re-render con un sleep + rerun.
# Evitamos depender de streamlit-autorefresh (no esta en deps).
if auto and state["offset"] < df_stream.height:
    st.caption(
        f"Auto-refresh activo: proximo tick en {refresh_seconds}s. "
        f"Apaga el toggle del sidebar para detener."
    )
    time.sleep(refresh_seconds)
    st.rerun()
