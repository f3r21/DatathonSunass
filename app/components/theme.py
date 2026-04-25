"""Paleta institucional SUNASS (azul + verde) y template Plotly compartido.

El logo oficial de SUNASS es una gota dividida en azul (izquierda) y verde
(derecha). Replicamos esa identidad: el azul es el color primario, el verde
el de acento positivo. El rojo se usa solo para CRITICAL real.
"""
from __future__ import annotations

from dataclasses import dataclass

import plotly.graph_objects as go
import plotly.io as pio


@dataclass(frozen=True)
class Palette:
    """Colores institucionales SUNASS."""

    # Azules SUNASS — gota lado izquierdo.
    primary: str = "#0d6bc4"
    primary_dark: str = "#0a4d8c"
    secondary: str = "#4fc3f7"
    # Verdes SUNASS — gota lado derecho.
    success: str = "#2e7d32"
    accent: str = "#66bb6a"
    # Estados (uso minimo).
    warning: str = "#fbc02d"
    danger: str = "#c62828"
    # Neutros.
    muted: str = "#6c757d"
    bg_soft: str = "#eaf4fc"
    text: str = "#1a1a1a"

    @property
    def qualitative(self) -> list[str]:
        """Paleta cualitativa para series Plotly. Empieza azul, alterna con verde."""
        return [
            self.primary,
            self.success,
            self.secondary,
            self.accent,
            self.primary_dark,
            self.warning,
            self.muted,
            self.danger,
        ]

    def severity_color(self, severity: str) -> str:
        """Mapea OK/INFO/WARN/CRITICAL al color correcto (verde/azul/amber/rojo)."""
        m = {
            "OK": self.success,
            "INFO": self.secondary,
            "WARN": self.warning,
            "CRITICAL": self.danger,
        }
        return m.get(severity.upper(), self.muted)


PALETTE = Palette()

_layout = go.Layout(
    font=dict(family="Inter, system-ui, sans-serif", size=13, color=PALETTE.text),
    colorway=PALETTE.qualitative,
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor="#e6ecf2", zerolinecolor="#e6ecf2", linecolor="#c5ced8"),
    yaxis=dict(gridcolor="#e6ecf2", zerolinecolor="#e6ecf2", linecolor="#c5ced8"),
    legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#e6ecf2", borderwidth=1),
    hoverlabel=dict(bgcolor="white", font_size=12),
)

PLOTLY_TEMPLATE = go.layout.Template(layout=_layout)
pio.templates["sunass"] = PLOTLY_TEMPLATE
pio.templates.default = "sunass"
