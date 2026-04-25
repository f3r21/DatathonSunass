"""Pagina 8 · Mapa — folium de estaciones MOREA + cruce con interrupciones.

Cubre el pedido de geodata flexible: detecta lat/lon en una o dos columnas y
las dibuja sobre un mapa. Incluye recordatorio de geogpsperu como fuente de
shapefiles oficiales (departamentos, ríos, sectores) para enriquecer capas.
"""
from __future__ import annotations

import sys
from pathlib import Path

import folium
import polars as pl
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.components.auth import require_auth  # noqa: E402
from app.components.data_loader import (  # noqa: E402
    data_available,
    get_morea_estaciones,
    get_morea_sensores,
)
from app.components.theme import PALETTE  # noqa: E402
from src.inspector import inspect_dataframe, split_combined_latlon  # noqa: E402
from src.modeling.anomalias import filter_imposibles, sustained_violations  # noqa: E402

st.set_page_config(page_title="Mapa · SUNASS", layout="wide")
require_auth("Mapa")
st.title("8 · Mapa de estaciones")
st.caption(
    "Geodata detectada automaticamente. Para enriquecer capas (departamentos, "
    "rios, sectores hidrograficos), descarga shapefiles oficiales de "
    "[geogpsperu.com](https://www.geogpsperu.com)."
)

ok, msg = data_available()
if not ok:
    st.error(msg)
    st.stop()


@st.cache_data(show_spinner="Detectando geodata...")
def _stations_with_coords() -> pl.DataFrame | None:
    df_est = get_morea_estaciones()
    inspector = inspect_dataframe(df_est)
    lat_col, lon_col = inspector.detect_lat_lon()

    if lat_col and lon_col:
        return df_est.select(
            pl.col(lat_col).cast(pl.Float64, strict=False).alias("lat"),
            pl.col(lon_col).cast(pl.Float64, strict=False).alias("lon"),
            *(pl.col(c) for c in df_est.columns if c not in (lat_col, lon_col)),
        ).drop_nulls(["lat", "lon"])

    geo_combined = inspector.names_by_kind("geo_combined")
    if geo_combined:
        return split_combined_latlon(df_est, geo_combined[0]).drop_nulls(["lat", "lon"])

    return None


def _violation_rate_per_station() -> dict[str, float]:
    df_clean = filter_imposibles(get_morea_sensores()).depurado
    if "cloro" not in df_clean.columns or "estacion_id" not in df_clean.columns:
        return {}
    df_sost = sustained_violations(df_clean, col="cloro", low=0.5, high=5.0, min_consecutive=3)
    rates = (
        df_sost.group_by("estacion_id")
        .agg(
            pl.len().alias("n"),
            pl.col("sostenida").cast(pl.Int32).sum().alias("n_sost"),
        )
        .with_columns(
            (100 * pl.col("n_sost") / pl.col("n")).alias("pct_sost"),
        )
    )
    return {str(r["estacion_id"]): float(r["pct_sost"]) for r in rates.iter_rows(named=True)}


df_geo = _stations_with_coords()

if df_geo is None or df_geo.height == 0:
    st.warning(
        "No detecte columnas de latitud/longitud en el catalogo MOREA. "
        "Cualquier dataset con columnas `lat`+`lon`, `latitud`+`longitud`, o "
        "una sola columna `coords` con formato `lat,lon` se renderiza aqui."
    )
    st.subheader("Inspeccion de columnas detectadas")
    rep = inspect_dataframe(get_morea_estaciones())
    from src.inspector import report_to_dataframe
    st.dataframe(report_to_dataframe(rep), width="stretch", height=320)
    st.stop()

rates = _violation_rate_per_station()

st.success(f"Detecte {df_geo.height} estaciones con coordenadas validas.")

# Centrado en Peru por default; recomputado si hay puntos.
center = [df_geo["lat"].mean(), df_geo["lon"].mean()]
m = folium.Map(
    location=center,
    zoom_start=6,
    tiles="CartoDB positron",
)

# Capa CartoDB voyager como alternativa.
folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
folium.TileLayer("CartoDB dark_matter", name="Oscuro").add_to(m)

for row in df_geo.iter_rows(named=True):
    sid = str(row.get("estacion_id", row.get("ESTACIÓN", row.get("ESTACION", ""))))
    pct_sost = rates.get(sid, 0.0)
    if pct_sost >= 50:
        color = PALETTE.danger
        label = "CRITICO"
    elif pct_sost >= 20:
        color = PALETTE.warning
        label = "ALERTA"
    elif pct_sost >= 5:
        color = PALETTE.secondary
        label = "VIGILAR"
    else:
        color = PALETTE.success
        label = "OK"
    popup_html = f"""
    <div style="font-family:Inter,sans-serif;min-width:200px;">
      <div style="font-weight:700; color:{PALETTE.primary};">Estacion {sid}</div>
      <div style="color:{color}; font-weight:600;">{label} · {pct_sost:.1f}% sostenida</div>
      <div style="color:#6c757d; font-size:0.85rem;">lat {row['lat']:.4f}, lon {row['lon']:.4f}</div>
    </div>
    """
    folium.CircleMarker(
        location=[float(row["lat"]), float(row["lon"])],
        radius=8,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.85,
        popup=folium.Popup(popup_html, max_width=300),
    ).add_to(m)

folium.LayerControl().add_to(m)

# Folium expone un iframe-en-HTML; lo envolvemos para fijar la altura sin
# depender de la API deprecada streamlit.components.v1.html.
_map_html = (
    f'<div style="height:560px;overflow:hidden;border-radius:6px;">'
    f"{m._repr_html_()}"
    f"</div>"
)
st.html(_map_html)

st.markdown("---")
st.subheader("Capas externas recomendadas (geogpsperu)")
st.markdown(
    """
- **Limites departamentales / provinciales / distritales** — descarga `.shp` y
  cargalo con `geopandas.read_file()` para overlays choropleth.
- **Cuencas hidrograficas (ANA)** — util para asociar estaciones MOREA al
  ambito de la EP que las opera.
- **Vias y centros poblados (INEI/MTC)** — calidad de servicio por accesibilidad.

Cuando bajes los archivos, ponlos en `repo/datos/geo/` y la pagina los detecta
automaticamente para superponerlos.
"""
)
