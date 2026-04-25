"""Pagina 1 · Datos — Raw to clean, con inventario visual."""
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
    get_morea_estaciones,
    get_morea_sensores,
)
from app.components.theme import PALETTE  # noqa: E402
from src.modeling.anomalias import compare_bruto_vs_depurado, filter_imposibles  # noqa: E402

st.set_page_config(page_title="Datos · SUNASS", layout="wide")
require_auth("Datos")
st.title("1 · Datos del Datathon SUNASS 2026")
st.caption("De datos crudos a dataset limpio, con trazabilidad de cada paso.")

ok, msg = data_available()
if not ok:
    st.error(msg)
    st.stop()


def _inventory_section() -> None:
    st.subheader("Inventario de datasets oficiales")
    df_int = get_interrupciones(enriched=False)
    df_morea = get_morea_sensores()
    df_est = get_morea_estaciones()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Interrupciones** (`.dta`)")
        st.write(f"{df_int.height:,} filas x {df_int.width} columnas")
        st.caption("Eventos de interrupcion reportados por las Empresas Prestadoras (EPs).")
    with col2:
        st.markdown(f"**MOREA sensores** (`.parquet`)")
        st.write(f"{df_morea.height:,} filas x {df_morea.width} columnas")
        st.caption("Lecturas automaticas cada ~5 min: pH, cloro, temperatura.")
    with col3:
        st.markdown(f"**Estaciones MOREA** (`.xlsx`)")
        st.write(f"{df_est.height:,} filas x {df_est.width} columnas")
        st.caption("Catalogo geografico: UBIGEO, coordenadas, nombre EP.")


def _schema_preview() -> None:
    st.subheader("Vista previa del esquema")
    tabs = st.tabs(["Interrupciones", "MOREA sensores", "Estaciones"])

    with tabs[0]:
        df = get_interrupciones(enriched=True)
        st.write("Tipos tras enriquecer con `add_timestamps` + `add_duracion_impacto`:")
        schema = pl.DataFrame(
            {"columna": list(df.schema.keys()), "dtype": [str(v) for v in df.schema.values()]}
        )
        st.dataframe(schema, height=260, width="stretch")
        st.dataframe(df.head(10), width="stretch")

    with tabs[1]:
        df = get_morea_sensores()
        schema = pl.DataFrame(
            {"columna": list(df.schema.keys()), "dtype": [str(v) for v in df.schema.values()]}
        )
        st.dataframe(schema, width="stretch")
        st.dataframe(df.head(10), width="stretch")

    with tabs[2]:
        df = get_morea_estaciones()
        st.dataframe(df, width="stretch")


def _cleaning_section() -> None:
    st.subheader("Pipeline de limpieza MOREA")
    st.write(
        "Antes de comparar lecturas contra bandas DIGESA, descartamos valores "
        "**fisicamente imposibles** (glitches de sensor). Solo entonces la tasa "
        "de violacion reportada es honesta."
    )
    df = get_morea_sensores()
    with st.spinner("Calculando depuracion fisica..."):
        result = filter_imposibles(df)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lecturas totales", f"{result.resumen['total']:,}")
    with col2:
        pct_ok = 100 * result.resumen["depurado"] / max(result.resumen["total"], 1)
        st.metric("Depurado", f"{result.resumen['depurado']:,}", delta=f"{pct_ok:.1f}% valido")
    with col3:
        pct_bad = 100 * result.resumen["glitches"] / max(result.resumen["total"], 1)
        st.metric(
            "Glitches descartados",
            f"{result.resumen['glitches']:,}",
            delta=f"{pct_bad:.2f}% fisicamente imposibles",
            delta_color="inverse",
        )

    st.markdown("### Efecto sobre el cumplimiento DIGESA")
    def _render_band(name: str, comp: dict, banda: str) -> None:
        st.markdown(f"**{name}** — banda DIGESA {banda}")
        c1, c2, c3 = st.columns(3)
        c1.metric("% violacion bruto", f"{comp['pct_violacion_bruto']:.2f}%")
        # delta_pp = bruto - depurado. Mostramos el cambio firmado en pp.
        diff_pp = comp["pct_violacion_depurado"] - comp["pct_violacion_bruto"]
        c2.metric(
            "% violacion depurado",
            f"{comp['pct_violacion_depurado']:.2f}%",
            delta=f"{diff_pp:+.2f} pp",
            delta_color="inverse",
        )
        c3.metric("N filas depurado", f"{comp['n_depurado']:,}")

    if "cloro" in df.columns:
        _render_band(
            "Cloro libre",
            compare_bruto_vs_depurado(df, result.depurado, "cloro", 0.5, 5.0),
            "[0.5, 5.0] mg/L",
        )
    if "ph" in df.columns:
        _render_band(
            "pH",
            compare_bruto_vs_depurado(df, result.depurado, "ph", 6.5, 8.5),
            "[6.5, 8.5]",
        )

    st.info(
        "La diferencia en puntos porcentuales es nuestra propuesta tecnica: "
        "distinguir falla de sensor de falla de servicio."
    )


def _nullability_section() -> None:
    st.subheader("Nulos y calidad por columna")
    df = get_interrupciones(enriched=False)
    rows = []
    for col in df.columns:
        s = df.get_column(col)
        nulls = s.null_count()
        rows.append(
            {
                "columna": col,
                "dtype": str(df.schema[col]),
                "nulls": nulls,
                "pct_null": round(100 * nulls / max(df.height, 1), 2),
                "n_unique": s.n_unique(),
            }
        )
    report = pl.DataFrame(rows).sort("pct_null", descending=True)
    st.dataframe(report, width="stretch", height=360)


_inventory_section()
st.markdown("---")
_cleaning_section()
st.markdown("---")
_schema_preview()
st.markdown("---")
_nullability_section()
