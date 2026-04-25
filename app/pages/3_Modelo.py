"""Pagina 3 · Modelo — XGBoost con GridSearchCV sobre evento_critico."""
from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.components.auth import require_auth  # noqa: E402
from app.components.data_loader import data_available, get_interrupciones  # noqa: E402
from app.components.theme import PALETTE  # noqa: E402
from src.modeling.clasificacion import (  # noqa: E402
    GridSearchReport,
    temporal_train_test_split,
    train_xgboost_grid,
)
from src.modeling.features import (  # noqa: E402
    add_temporal_features,
    frequency_encode,
)

st.set_page_config(page_title="Modelo · SUNASS", layout="wide")
require_auth("Modelo")

st.title("3 · Clasificador de eventos criticos")
st.caption(
    "XGBoost optimizado con GridSearchCV. PR-AUC como metrica primaria "
    "porque el dataset esta desbalanceado (~2-5% positivos). Las features se "
    "eligen aqui — sin features hardcoded — y se evita target leakage."
)

ok, msg = data_available()
if not ok:
    st.error(msg)
    st.stop()


# ----------------------------------------------------------------- column inspector


@st.cache_data(show_spinner="Cargando dataset enriquecido...")
def _df_feat() -> pl.DataFrame:
    df = get_interrupciones(enriched=True)
    if "ts_inicio" not in df.columns or "evento_critico" not in df.columns:
        raise RuntimeError("Faltan columnas enriquecidas (ts_inicio/evento_critico).")
    return add_temporal_features(df)


df_feat = _df_feat()

# Excluimos columnas que son leakage directo o no son features utiles.
_LEAKAGE_COLS = {"duracion_horas", "impacto", "evento_critico", "ts_inicio", "ts_fin"}
_DATE_HINT = ("Fecha", "Hora")


def _is_numeric(dtype) -> bool:
    try:
        return dtype.is_numeric()
    except Exception:
        return False


numeric_candidates = [
    c
    for c in df_feat.columns
    if c not in _LEAKAGE_COLS
    and not any(c.startswith(p) for p in _DATE_HINT)
    and _is_numeric(df_feat.schema[c])
]
categorical_candidates = [
    c
    for c in df_feat.columns
    if c not in _LEAKAGE_COLS
    and not any(c.startswith(p) for p in _DATE_HINT)
    and not _is_numeric(df_feat.schema[c])
]

# Defaults razonables: features temporales derivadas + n_afectadas; motivos como cat.
default_num = [
    c for c in ("hora", "dow", "mes_num", "trimestre_num", "n_afectadas")
    if c in numeric_candidates
]
default_cat = [
    c for c in categorical_candidates if c.startswith("Motivo") or c == "Empresaprestadora"
][:3]


with st.sidebar:
    st.markdown("### Configuracion del modelo")
    sel_num = st.multiselect(
        "Features numericas",
        options=sorted(numeric_candidates),
        default=default_num,
        help="Las temporales y conteos. Excluye duracion_horas e impacto (leakage).",
    )
    sel_cat = st.multiselect(
        "Features categoricas (frequency-encoded)",
        options=sorted(categorical_candidates),
        default=default_cat,
        help="Polars no acepta categoricas crudas en XGB; encodeamos por frecuencia.",
    )
    cutoff = st.text_input(
        "Cutoff temporal (train <= cutoff < test)",
        value="2025-07-01",
        help="Formato ISO. Por defecto deja ~6 meses para test.",
    )
    grid_size = st.radio(
        "Tamano de grid",
        options=["pequeno (rapido)", "mediano", "grande"],
        horizontal=False,
        index=1,
    )
    btn_train = st.button("Entrenar modelo", type="primary", width="stretch")


# ----------------------------------------------------------------- entrenamiento


def _grid_for(size: str) -> dict[str, list]:
    if size.startswith("pequeno"):
        return {
            "n_estimators": [200],
            "max_depth": [4, 6],
            "learning_rate": [0.05, 0.1],
        }
    if size.startswith("grande"):
        return {
            "n_estimators": [200, 400, 600],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1],
            "min_child_weight": [1, 3, 5],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
        }
    return {
        "n_estimators": [200, 400],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "min_child_weight": [1, 5],
        "subsample": [0.8, 1.0],
    }


def _build_xy_temporal(
    df_feat: pl.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    cutoff: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.Series, pl.Series]:
    """Encode + split sin perder columnas. Retorna (X_tr, X_te, y_tr, y_te)."""
    df_with_target = df_feat.filter(pl.col("evento_critico").is_not_null())
    df_encoded = frequency_encode(df_with_target, categorical_cols)

    encoded_names = [f"{c}_freq" for c in categorical_cols if f"{c}_freq" in df_encoded.columns]
    feature_cols = [c for c in numeric_cols if c in df_encoded.columns] + encoded_names
    if not feature_cols:
        raise ValueError("No hay features seleccionadas que existan en el dataset.")

    keep = ["ts_inicio", "evento_critico", *feature_cols]
    df_keep = df_encoded.select(keep).fill_null(0.0)
    tr, te = temporal_train_test_split(df_keep, "ts_inicio", cutoff=cutoff)
    X_tr = tr.select(feature_cols)
    X_te = te.select(feature_cols)
    y_tr = tr.get_column("evento_critico").cast(pl.Int8)
    y_te = te.get_column("evento_critico").cast(pl.Int8)
    return X_tr, X_te, y_tr, y_te


@st.cache_resource(show_spinner="Entrenando XGBoost con GridSearchCV...")
def _train_cached(
    numeric_cols: tuple[str, ...],
    categorical_cols: tuple[str, ...],
    cutoff: str,
    grid_size: str,
) -> GridSearchReport:
    X_tr, X_te, y_tr, y_te = _build_xy_temporal(
        _df_feat(), list(numeric_cols), list(categorical_cols), cutoff
    )
    if X_tr.height == 0 or X_te.height == 0:
        raise RuntimeError(
            f"Split vacio (train={X_tr.height}, test={X_te.height}). Cambia el cutoff."
        )
    _, report = train_xgboost_grid(
        X_tr, y_tr, X_te, y_te, param_grid=_grid_for(grid_size), cv_folds=3
    )
    return report


if "report" not in st.session_state or btn_train:
    if not sel_num and not sel_cat:
        st.warning("Selecciona al menos una feature en el sidebar.")
        st.stop()
    try:
        st.session_state["report"] = _train_cached(
            tuple(sel_num), tuple(sel_cat), cutoff, grid_size
        )
    except Exception as exc:
        st.error(f"Error entrenando el modelo: {exc}")
        st.stop()

report: GridSearchReport = st.session_state["report"]


# ----------------------------------------------------------------- visualizaciones


def _metrics_section() -> None:
    st.subheader("Metricas del mejor modelo")
    eval_ = report.evaluation
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("PR-AUC (test)", f"{eval_.pr_auc:.3f}")
    col2.metric("ROC-AUC (test)", f"{eval_.roc_auc:.3f}")
    col3.metric("Recall @ precision 90%", f"{eval_.recall_at_p90:.2%}")
    col4.metric("Precision @ recall 50%", f"{eval_.precision_at_r50:.2%}")

    st.markdown("**Best params (CV):**")
    st.json(report.best_params, expanded=False)
    st.caption(
        f"CV={report.cv_folds} folds, scoring={report.cv_scoring}, "
        f"mejor score CV={report.cv_best_score:.3f}. Test prevalencia="
        f"{eval_.prevalencia_test:.2%}. Train n={eval_.n_train}, test n={eval_.n_test}."
    )


def _confusion_section() -> None:
    st.subheader("Matriz de confusion @ threshold 0.5")
    cm = report.evaluation.confusion
    fig = px.imshow(
        [[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]],
        x=["pred: no critico", "pred: critico"],
        y=["real: no critico", "real: critico"],
        color_continuous_scale="Blues",
        text_auto=True,
        aspect="auto",
    )
    fig.update_layout(height=320)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(fig, width="stretch")
    with col2:
        st.markdown("**Barrido de umbrales**")
        sweep = pl.DataFrame(report.evaluation.threshold_sweep)
        st.dataframe(sweep, width="stretch")
        st.caption(
            "Ajusta el threshold segun la aversion a falsos positivos o falsos "
            "negativos que tenga el regulador."
        )


def _importance_section() -> None:
    st.subheader("Importancia de features")
    fi = report.feature_importance
    if not fi:
        st.info("No se calcularon importancias.")
        return
    top = list(fi.items())[:15]
    fig = go.Figure(
        data=[
            go.Bar(
                x=[v for _, v in top][::-1],
                y=[k for k, _ in top][::-1],
                orientation="h",
                marker_color=PALETTE.primary,
            )
        ]
    )
    fig.update_layout(
        title="Top 15 features por gain",
        xaxis_title="importancia",
        yaxis_title="feature",
        height=400,
    )
    st.plotly_chart(fig, width="stretch")


def _grid_section() -> None:
    st.subheader("Top 10 combinaciones del GridSearch")
    if not report.cv_results_top:
        st.info("Sin resultados del grid.")
        return
    st.dataframe(pl.DataFrame(report.cv_results_top), width="stretch")


_metrics_section()
st.markdown("---")
_confusion_section()
st.markdown("---")
_importance_section()
st.markdown("---")
_grid_section()
