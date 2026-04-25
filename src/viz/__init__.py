"""Visualizaciones compartidas entre app Streamlit y deck Quarto."""
from src.viz.eda import (
    boxplot_by_group,
    correlation_heatmap,
    distribution_histogram,
    interrupciones_timeline,
    morea_sensor_timeline,
)

__all__ = [
    "boxplot_by_group",
    "correlation_heatmap",
    "distribution_histogram",
    "interrupciones_timeline",
    "morea_sensor_timeline",
]
