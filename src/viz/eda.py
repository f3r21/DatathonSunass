"""Plots EDA reutilizables con Plotly.

Todas las funciones:
    - Reciben polars.DataFrame (convierten internamente si numpy/pandas hace falta).
    - Devuelven plotly.graph_objects.Figure (renderizable en Streamlit o exportable).
    - Usan el template 'sunass' si esta cargado; fallback a default Plotly.
    - No mutan el input.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl

logger = logging.getLogger(__name__)

_DIVERGENT = "RdBu"


def correlation_heatmap(
    df: pl.DataFrame,
    numeric_cols: Sequence[str] | None = None,
    method: str = "spearman",
    title: str = "Matriz de correlacion",
) -> go.Figure:
    """Heatmap de correlacion (Spearman por default — robusto a no-linealidad).

    Args:
        df: DataFrame con columnas numericas.
        numeric_cols: lista de columnas. Si None, usa todas las numericas.
        method: 'spearman' (default) o 'pearson'.
        title: titulo del plot.
    """
    if numeric_cols is None:
        numeric_cols = [c for c, dt in df.schema.items() if dt.is_numeric()]
    if len(numeric_cols) < 2:
        logger.warning("correlation_heatmap: menos de 2 columnas numericas")
        return go.Figure().update_layout(title=f"{title} (insuficientes columnas)")

    sub = df.select(numeric_cols).drop_nulls()
    if sub.is_empty():
        return go.Figure().update_layout(title=f"{title} (sin filas no-null)")

    pdf = sub.to_pandas()
    corr = pdf.corr(method=method).round(2)
    fig = px.imshow(
        corr,
        x=corr.columns,
        y=corr.columns,
        color_continuous_scale=_DIVERGENT,
        zmin=-1,
        zmax=1,
        aspect="auto",
        text_auto=True,
    )
    fig.update_layout(
        title=f"{title} · {method} · n={sub.height:,}",
        coloraxis_colorbar=dict(title="r"),
        height=max(360, 40 * len(numeric_cols) + 200),
    )
    fig.update_xaxes(tickangle=45)
    return fig


def distribution_histogram(
    df: pl.DataFrame,
    col: str,
    bins: int = 50,
    log_y: bool = False,
    reference_lines: Sequence[tuple[float, str]] = (),
) -> go.Figure:
    """Histograma con lineas de referencia opcionales (ej. bandas DIGESA).

    Args:
        reference_lines: [(valor, label), ...]. Se dibujan como vlines rojas.
    """
    if col not in df.columns:
        raise KeyError(f"Columna '{col}' ausente")
    series = df.get_column(col).drop_nulls()
    if series.is_empty():
        return go.Figure().update_layout(title=f"Sin datos para {col}")

    values = series.to_numpy()
    fig = go.Figure(
        data=[go.Histogram(x=values, nbinsx=bins, marker_line_width=0, opacity=0.9)]
    )
    for value, label in reference_lines:
        fig.add_vline(
            x=value,
            line_width=2,
            line_dash="dash",
            line_color="#c1121f",
            annotation_text=label,
            annotation_position="top",
        )
    fig.update_layout(
        title=f"Distribucion de {col} · n={series.len():,}",
        xaxis_title=col,
        yaxis_title="frecuencia",
        bargap=0.02,
    )
    if log_y:
        fig.update_yaxes(type="log")
    return fig


def boxplot_by_group(
    df: pl.DataFrame,
    value_col: str,
    group_col: str,
    top_k: int = 15,
    log_y: bool = False,
) -> go.Figure:
    """Boxplot de `value_col` agrupado por `group_col`; muestra top-K grupos."""
    if value_col not in df.columns or group_col not in df.columns:
        raise KeyError(f"Columnas requeridas ausentes: {value_col}, {group_col}")

    top_groups = (
        df.select(group_col)
        .drop_nulls()
        .group_by(group_col)
        .agg(pl.len().alias("_n"))
        .sort("_n", descending=True)
        .head(top_k)
        .get_column(group_col)
        .to_list()
    )
    sub = df.filter(pl.col(group_col).is_in(top_groups)).select([group_col, value_col]).drop_nulls()
    if sub.is_empty():
        return go.Figure().update_layout(title="Sin datos")

    pdf = sub.to_pandas()
    fig = px.box(
        pdf,
        x=group_col,
        y=value_col,
        points="suspectedoutliers",
    )
    fig.update_layout(
        title=f"{value_col} por {group_col} · top {len(top_groups)} grupos",
        xaxis_title=group_col,
        yaxis_title=value_col,
    )
    fig.update_xaxes(tickangle=30)
    if log_y:
        fig.update_yaxes(type="log")
    return fig


def interrupciones_timeline(
    df: pl.DataFrame,
    ts_col: str = "ts_inicio",
    freq: str = "1mo",
    critical_col: str | None = "evento_critico",
) -> go.Figure:
    """Timeline de eventos por periodo (default mensual).

    Si critical_col esta presente, separa en dos trazas (total y criticos).
    """
    if ts_col not in df.columns:
        raise KeyError(f"Columna '{ts_col}' ausente")

    df_valid = df.filter(pl.col(ts_col).is_not_null())
    if df_valid.is_empty():
        return go.Figure().update_layout(title="Sin fechas validas")

    grouped = (
        df_valid.with_columns(pl.col(ts_col).dt.truncate(freq).alias("_periodo"))
        .group_by("_periodo")
        .agg(pl.len().alias("n_total"))
        .sort("_periodo")
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=grouped.get_column("_periodo").to_list(),
            y=grouped.get_column("n_total").to_list(),
            mode="lines+markers",
            name="Todas",
            line=dict(width=2),
        )
    )
    if critical_col and critical_col in df.columns:
        criticos = (
            df_valid.filter(pl.col(critical_col).fill_null(False))
            .with_columns(pl.col(ts_col).dt.truncate(freq).alias("_periodo"))
            .group_by("_periodo")
            .agg(pl.len().alias("n_crit"))
            .sort("_periodo")
        )
        fig.add_trace(
            go.Scatter(
                x=criticos.get_column("_periodo").to_list(),
                y=criticos.get_column("n_crit").to_list(),
                mode="lines+markers",
                name="Criticos",
                line=dict(width=2, color="#c1121f"),
            )
        )
    fig.update_layout(
        title=f"Eventos de interrupcion por {freq}",
        xaxis_title="periodo",
        yaxis_title="cantidad",
        hovermode="x unified",
    )
    return fig


def morea_sensor_timeline(
    df: pl.DataFrame,
    ts_col: str = "fecha",
    value_col: str = "cloro",
    station_col: str = "estacion_id",
    station_id: int | str | None = None,
    band: tuple[float, float] | None = None,
    sample_frac: float = 0.05,
) -> go.Figure:
    """Serie temporal de un sensor con banda DIGESA opcional.

    Si station_id es None, toma la estacion con mas lecturas.
    Submuestrea a sample_frac para renderizar rapido.
    """
    required = (ts_col, value_col, station_col)
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Columna '{c}' ausente")

    if station_id is None:
        top = (
            df.get_column(station_col)
            .drop_nulls()
            .value_counts()
            .sort("count", descending=True)
        )
        if top.height == 0:
            return go.Figure().update_layout(title="Sin estaciones")
        station_id = top.get_column(station_col)[0]

    sub = (
        df.filter(pl.col(station_col) == station_id)
        .select([ts_col, value_col])
        .drop_nulls()
        .sort(ts_col)
    )
    if sub.is_empty():
        return go.Figure().update_layout(title=f"Estacion {station_id} sin datos")
    if sample_frac < 1.0 and sub.height > 2000:
        idx = np.linspace(0, sub.height - 1, int(sub.height * sample_frac)).astype(int)
        sub = sub[idx]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sub.get_column(ts_col).to_list(),
            y=sub.get_column(value_col).to_list(),
            mode="lines",
            name=value_col,
            line=dict(width=1.2),
        )
    )
    if band is not None:
        lo, hi = band
        fig.add_hline(y=lo, line_dash="dot", line_color="#c1121f", annotation_text=f"min {lo}")
        fig.add_hline(y=hi, line_dash="dot", line_color="#c1121f", annotation_text=f"max {hi}")
        fig.add_hrect(y0=lo, y1=hi, fillcolor="#2a9d8f", opacity=0.08, line_width=0)
    fig.update_layout(
        title=f"{value_col} en estacion {station_id} · muestra {sub.height:,}",
        xaxis_title=ts_col,
        yaxis_title=value_col,
        hovermode="x unified",
    )
    return fig
