"""KPI cards compactas para la pagina Home."""
from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from app.components.theme import PALETTE


@dataclass(frozen=True)
class KPI:
    label: str
    value: str
    delta: str | None = None
    help_text: str | None = None


def render_kpis(kpis: list[KPI], n_cols: int = 4) -> None:
    cols = st.columns(n_cols)
    for i, kpi in enumerate(kpis):
        with cols[i % n_cols]:
            st.metric(label=kpi.label, value=kpi.value, delta=kpi.delta, help=kpi.help_text)


def styled_metric(label: str, value: str, color: str = PALETTE.primary) -> None:
    st.markdown(
        f"""
        <div style="border-left: 4px solid {color}; padding: 0.6rem 0.9rem;
                    background: {PALETTE.bg_soft}; border-radius: 4px;">
            <div style="font-size: 0.85rem; color: {PALETTE.muted};">{label}</div>
            <div style="font-size: 1.6rem; font-weight: 600; color: {PALETTE.text};">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
