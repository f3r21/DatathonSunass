"""Cargadores de datos con cache de Streamlit.

Envuelve `src.io` y `src.modeling.features` con `@st.cache_data` para evitar
recargar el parquet de 1.5M filas en cada reejecucion del script.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import streamlit as st

from src.io import load_interrupciones, load_morea, paths_from_env
from src.modeling.features import add_duracion_impacto, add_timestamps


@st.cache_resource(show_spinner=False)
def dataset_paths():
    """Resuelve rutas desde .env una sola vez por sesion."""
    return paths_from_env()


@st.cache_data(show_spinner="Cargando historico de interrupciones...")
def get_interrupciones(enriched: bool = True) -> pl.DataFrame:
    """Interrupciones + timestamps + duracion/impacto si enriched."""
    paths = dataset_paths()
    df = load_interrupciones(paths.interrupciones)
    if enriched:
        df = add_timestamps(df)
        df = add_duracion_impacto(df)
    return df


@st.cache_data(show_spinner="Cargando serie MOREA...")
def get_morea_sensores() -> pl.DataFrame:
    paths = dataset_paths()
    df_sensores, _ = load_morea(paths.morea_parquet, paths.morea_estaciones)
    return df_sensores


@st.cache_data(show_spinner="Cargando estaciones MOREA...")
def get_morea_estaciones() -> pl.DataFrame:
    paths = dataset_paths()
    _, df_estaciones = load_morea(paths.morea_parquet, paths.morea_estaciones)
    return df_estaciones


def data_available() -> tuple[bool, str]:
    """Chequeo rapido para header de paginas."""
    try:
        paths = dataset_paths()
    except Exception as exc:
        return False, f"Variables .env no configuradas: {exc}"
    missing = [
        p.name
        for p in (paths.interrupciones, paths.morea_parquet, paths.morea_estaciones)
        if not Path(p).exists()
    ]
    if missing:
        return False, f"Archivos no encontrados: {', '.join(missing)}"
    return True, "Datos disponibles"
