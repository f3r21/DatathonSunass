"""Plantilla A — Clasificacion desbalanceada de eventos criticos.

Critica explicita al baseline del mentor (logit sin balancear, 85.6% accuracy
pero 0.77% sensitivity) y propone tres modelos alternativos con foco en la
metrica correcta para un dataset ~2-5% positivo:

    - Logistic Regression con class_weight='balanced'
    - XGBoost con scale_pos_weight calibrado
    - LightGBM con is_unbalance=True

Evaluacion: PR-AUC y recall @ precision >= p, NO accuracy. Devuelve un objeto
ModelEvaluation con todas las metricas para poder comparar en el deck.

Split temporal: train <= cutoff, test > cutoff, porque el dataset tiene drift
estacional y un random split sobreestima el rendimiento.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def _xgb_device() -> str:
    """Detecta el mejor dispositivo para XGBoost: cuda > cpu.

    Compatible con Windows (NVIDIA), Linux (NVIDIA) y macOS (sin GPU).
    En Mac o cualquier maquina sin CUDA cae automaticamente a CPU.
    """
    try:
        import numpy as _np
        from xgboost import XGBClassifier as _XGB
        _m = _XGB(n_estimators=1, device="cuda", tree_method="hist")
        _m.fit(_np.array([[1.0]]), _np.array([0]))
        logger.info("XGBoost: usando GPU (cuda)")
        return "cuda"
    except Exception:
        logger.info("XGBoost: GPU no disponible, usando CPU")
        return "cpu"


def _lgbm_device() -> str:
    """Detecta el mejor dispositivo para LightGBM: cuda > gpu > cpu.

    LightGBM 4.x soporta device='cuda' (NVIDIA moderna).
    device='gpu' es la API antigua (OpenCL). Ambos caen a CPU en Mac/sin GPU.
    """
    import numpy as _np
    from lightgbm import LGBMClassifier as _LGBM

    for _dev in ("cuda", "gpu"):
        try:
            _m = _LGBM(n_estimators=1, device=_dev, verbose=-1)
            _m.fit(_np.array([[1.0], [2.0]]), _np.array([0, 1]))
            logger.info("LightGBM: usando GPU (%s)", _dev)
            return _dev
        except Exception:
            pass

    logger.info("LightGBM: GPU no disponible, usando CPU")
    return "cpu"


# Detectar una sola vez al importar el modulo
_XGB_DEVICE: str = _xgb_device()
_LGBM_DEVICE: str = _lgbm_device()


@dataclass(frozen=True)
class ModelEvaluation:
    """Resultado de evaluar un clasificador sobre el test set."""

    model_name: str
    n_train: int
    n_test: int
    prevalencia_test: float
    pr_auc: float
    roc_auc: float
    recall_at_p90: float
    precision_at_r50: float
    confusion: list[list[int]]
    threshold_sweep: list[dict[str, float]] = field(default_factory=list)

    def summary_row(self) -> dict[str, float | str]:
        """Fila compacta para mostrar en tabla comparativa."""
        return {
            "modelo": self.model_name,
            "pr_auc": round(self.pr_auc, 4),
            "roc_auc": round(self.roc_auc, 4),
            "recall@p90": round(self.recall_at_p90, 4),
            "precision@r50": round(self.precision_at_r50, 4),
            "n_test": self.n_test,
            "prevalencia": round(self.prevalencia_test, 4),
        }


def temporal_train_test_split(
    df: pl.DataFrame,
    time_col: str,
    cutoff: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split temporal: train <= cutoff < test.

    Args:
        df: DataFrame con columna temporal.
        time_col: nombre de la columna Datetime.
        cutoff: ISO date/datetime string, p.e. '2025-07-01'.

    Returns:
        (df_train, df_test).
    """
    if time_col not in df.columns:
        raise KeyError(f"time_col '{time_col}' ausente")
    # pl.lit(str).cast(Datetime, strict=False) devuelve null para strings no-ISO.
    # Parseamos en Python y pasamos el datetime directo para evitar el silencio.
    cutoff_dt = datetime.fromisoformat(cutoff)
    train = df.filter(pl.col(time_col) <= cutoff_dt)
    test = df.filter(pl.col(time_col) > cutoff_dt)
    logger.info(
        "Split temporal cutoff=%s: train=%d, test=%d",
        cutoff,
        train.height,
        test.height,
    )
    return train, test


def _scale_pos_weight(y: np.ndarray) -> float:
    """Ratio negativos/positivos para XGBoost."""
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        return 1.0
    return n_neg / n_pos


def _recall_at_precision(
    y_true: np.ndarray, y_proba: np.ndarray, target_precision: float
) -> float:
    """Recall maximo alcanzable con precision >= target."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    valid = precision >= target_precision
    if not valid.any():
        return 0.0
    return float(recall[valid].max())


def _precision_at_recall(
    y_true: np.ndarray, y_proba: np.ndarray, target_recall: float
) -> float:
    """Precision maxima alcanzable con recall >= target."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    valid = recall >= target_recall
    if not valid.any():
        return 0.0
    return float(precision[valid].max())


def _threshold_sweep(
    y_true: np.ndarray, y_proba: np.ndarray, thresholds: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.7)
) -> list[dict[str, float]]:
    """Tabla de precision/recall a umbrales fijos para discutir trade-offs."""
    rows: list[dict[str, float]] = []
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        rows.append({
            "threshold": thr,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })
    return rows


def _evaluate(
    model_name: str,
    y_train: np.ndarray,
    y_test: np.ndarray,
    y_proba: np.ndarray,
) -> ModelEvaluation:
    """Calcula todas las metricas y empaqueta en ModelEvaluation."""
    threshold = 0.5
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()
    pr_auc = float(average_precision_score(y_test, y_proba))
    roc_auc = float(roc_auc_score(y_test, y_proba)) if len(set(y_test)) > 1 else 0.5
    return ModelEvaluation(
        model_name=model_name,
        n_train=int(len(y_train)),
        n_test=int(len(y_test)),
        prevalencia_test=float(y_test.mean()),
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        recall_at_p90=_recall_at_precision(y_test, y_proba, 0.90),
        precision_at_r50=_precision_at_recall(y_test, y_proba, 0.50),
        confusion=cm,
        threshold_sweep=_threshold_sweep(y_test, y_proba),
    )


def train_logit(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    X_test: pl.DataFrame,
    y_test: pl.Series,
    C: float = 1.0,
    max_iter: int = 2000,
) -> tuple[LogisticRegression, ModelEvaluation]:
    """Logistic Regression con class_weight='balanced' — baseline honesto."""
    xa_train = X_train.to_numpy()
    ya_train = y_train.to_numpy().astype(int)
    xa_test = X_test.to_numpy()
    ya_test = y_test.to_numpy().astype(int)

    model = LogisticRegression(
        class_weight="balanced",
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
    )
    model.fit(xa_train, ya_train)
    y_proba = model.predict_proba(xa_test)[:, 1]
    evaluation = _evaluate("logit_balanced", ya_train, ya_test, y_proba)
    logger.info("logit_balanced: %s", evaluation.summary_row())
    return model, evaluation


def train_xgboost(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    X_test: pl.DataFrame,
    y_test: pl.Series,
    n_estimators: int = 400,
    max_depth: int = 6,
    learning_rate: float = 0.05,
) -> tuple[Any, ModelEvaluation]:
    """XGBoost con scale_pos_weight calibrado al ratio del train."""
    from xgboost import XGBClassifier  # import tardio: pesa

    xa_train = X_train.to_numpy()
    ya_train = y_train.to_numpy().astype(int)
    xa_test = X_test.to_numpy()
    ya_test = y_test.to_numpy().astype(int)

    spw = _scale_pos_weight(ya_train)
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        device=_XGB_DEVICE,
        n_jobs=-1 if _XGB_DEVICE == "cpu" else 1,
        random_state=42,
    )
    model.fit(xa_train, ya_train)
    y_proba = model.predict_proba(xa_test)[:, 1]
    evaluation = _evaluate("xgboost_spw", ya_train, ya_test, y_proba)
    logger.info("xgboost_spw (spw=%.2f): %s", spw, evaluation.summary_row())
    return model, evaluation


def train_lightgbm(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    X_test: pl.DataFrame,
    y_test: pl.Series,
    n_estimators: int = 400,
    num_leaves: int = 63,
    learning_rate: float = 0.05,
) -> tuple[Any, ModelEvaluation]:
    """LightGBM con is_unbalance=True — rapido en CPU, interpretable con SHAP."""
    from lightgbm import LGBMClassifier

    xa_train = X_train.to_numpy()
    ya_train = y_train.to_numpy().astype(int)
    xa_test = X_test.to_numpy()
    ya_test = y_test.to_numpy().astype(int)

    model = LGBMClassifier(
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        is_unbalance=True,
        objective="binary",
        device=_LGBM_DEVICE,
        n_jobs=-1 if _LGBM_DEVICE == "cpu" else 1,
        random_state=42,
        verbose=-1,
    )
    model.fit(xa_train, ya_train)
    y_proba = model.predict_proba(xa_test)[:, 1]
    evaluation = _evaluate("lightgbm_unbalanced", ya_train, ya_test, y_proba)
    logger.info("lightgbm_unbalanced: %s", evaluation.summary_row())
    return model, evaluation


def compare_models(evaluations: list[ModelEvaluation]) -> pl.DataFrame:
    """Tabla comparativa de modelos ordenada por PR-AUC descendente."""
    rows = [e.summary_row() for e in evaluations]
    df = pl.DataFrame(rows)
    if "pr_auc" in df.columns:
        df = df.sort("pr_auc", descending=True)
    return df


# --------------------------------------------------------------- grid search XGB


@dataclass(frozen=True)
class GridSearchReport:
    """Resultado de GridSearchCV sobre un clasificador."""

    model_name: str
    best_params: dict[str, Any]
    cv_best_score: float
    cv_scoring: str
    cv_folds: int
    evaluation: ModelEvaluation
    feature_importance: dict[str, float] = field(default_factory=dict)
    cv_results_top: list[dict[str, Any]] = field(default_factory=list)

    def summary_row(self) -> dict[str, float | str]:
        row = self.evaluation.summary_row()
        row["cv_best_score"] = round(self.cv_best_score, 4)
        row["cv_scoring"] = self.cv_scoring
        return row


def train_xgboost_grid(
    X_train: pl.DataFrame,
    y_train: pl.Series,
    X_test: pl.DataFrame,
    y_test: pl.Series,
    param_grid: dict[str, list[Any]] | None = None,
    cv_folds: int = 3,
    scoring: str = "average_precision",
    n_jobs: int = -1,
) -> tuple[Any, GridSearchReport]:
    """XGBoost + GridSearchCV sobre hiperparametros clasicos.

    Scoring default = PR-AUC (average_precision), apropiado para desbalance ~2-5%.
    cv_folds=3 con StratifiedKFold preserva prevalencia en cada fold.

    Args:
        X_train, y_train: split de entrenamiento (ya sin target leakage).
        X_test, y_test: split de evaluacion final (no visto por el grid).
        param_grid: dict sklearn-style. Default: grid chico para ~7h de competencia.
        cv_folds: k del StratifiedKFold.
        scoring: metrica sklearn para seleccionar el mejor.
        n_jobs: paralelismo.

    Returns:
        (modelo_entrenado, GridSearchReport con best_params y ModelEvaluation).
    """
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from xgboost import XGBClassifier

    xa_train = X_train.to_numpy()
    ya_train = y_train.to_numpy().astype(int)
    xa_test = X_test.to_numpy()
    ya_test = y_test.to_numpy().astype(int)

    spw = _scale_pos_weight(ya_train)
    base = XGBClassifier(
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        device=_XGB_DEVICE,
        n_jobs=1,
        random_state=42,
    )
    if param_grid is None:
        param_grid = {
            "n_estimators": [200, 400],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1],
            "min_child_weight": [1, 5],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
    )
    grid.fit(xa_train, ya_train)
    best_model = grid.best_estimator_

    y_proba = best_model.predict_proba(xa_test)[:, 1]
    evaluation = _evaluate("xgboost_grid", ya_train, ya_test, y_proba)

    fi = dict(zip(X_train.columns, best_model.feature_importances_.tolist(), strict=True))
    fi_sorted = dict(sorted(fi.items(), key=lambda kv: kv[1], reverse=True))

    cv_rows: list[dict[str, Any]] = []
    results = grid.cv_results_
    for i in np.argsort(results["mean_test_score"])[::-1][:10]:
        cv_rows.append(
            {
                "rank": int(results["rank_test_score"][i]),
                "mean_score": float(results["mean_test_score"][i]),
                "std_score": float(results["std_test_score"][i]),
                **{k: results["params"][i].get(k) for k in param_grid.keys()},
            }
        )

    report = GridSearchReport(
        model_name="xgboost_grid",
        best_params=dict(grid.best_params_),
        cv_best_score=float(grid.best_score_),
        cv_scoring=scoring,
        cv_folds=cv_folds,
        evaluation=evaluation,
        feature_importance=fi_sorted,
        cv_results_top=cv_rows,
    )
    logger.info(
        "xgboost_grid: best=%s score=%.4f (%s) test=%s",
        report.best_params,
        report.cv_best_score,
        scoring,
        evaluation.summary_row(),
    )
    return best_model, report
