"""Pipeline end-to-end: raw -> clean -> features -> XGBoost con GridSearch -> artefactos.

Uso:
    uv run python scripts/run_pipeline.py [--horizon N] [--cutoff YYYY-MM-DD]

Genera en repo/artifacts/:
    - interrupciones_enriched.parquet    dataset enriquecido
    - morea_depurado.parquet             MOREA sin glitches fisicos
    - modelo_metricas.json               PR-AUC, ROC-AUC, best_params
    - feature_importance.csv             top features
    - forecast_metrics.csv               tabla comparativa de forecasts
    - run_log.txt                        log consolidado

Diseñado para validar que todo el pipeline funciona fuera de Streamlit antes
de construir la imagen Docker. Tambien sirve como referencia de `cuanto tarda
end-to-end` en el workstation del equipo.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import polars as pl

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.io import load_interrupciones, load_morea, paths_from_env  # noqa: E402
from src.modeling.anomalias import filter_imposibles  # noqa: E402
from src.modeling.clasificacion import (  # noqa: E402
    temporal_train_test_split,
    train_xgboost_grid,
)
from src.modeling.features import (  # noqa: E402
    add_duracion_impacto,
    add_temporal_features,
    add_timestamps,
    build_feature_matrix,
)
from src.modeling.forecasting import (  # noqa: E402
    aggregate_monthly,
    compare_forecasts,
    forecast_ensemble,
    forecast_ets,
    forecast_lgbm_lags,
    forecast_naive_seasonal,
    forecast_sarima,
    forecast_xgb_lags,
    train_test_horizon_split,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline end-to-end SUNASS")
    parser.add_argument("--horizon", type=int, default=3, help="meses de holdout forecasting")
    parser.add_argument("--cutoff", type=str, default="2025-07-01", help="corte train/test clasificacion")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=_ROOT / "artifacts",
        help="directorio de salida",
    )
    return parser.parse_args()


def _step_interrupciones() -> pl.DataFrame:
    logger.info("=== Paso 1/4: interrupciones ===")
    paths = paths_from_env()
    df = load_interrupciones(paths.interrupciones)
    df = add_timestamps(df)
    df = add_duracion_impacto(df)
    df = add_temporal_features(df)
    return df


def _step_morea_clean() -> pl.DataFrame:
    logger.info("=== Paso 2/4: MOREA depurado ===")
    paths = paths_from_env()
    df, _ = load_morea(paths.morea_parquet, paths.morea_estaciones)
    result = filter_imposibles(df)
    return result.depurado


def _step_classification(df: pl.DataFrame, cutoff: str) -> dict[str, object]:
    logger.info("=== Paso 3/4: clasificacion XGBoost + GridSearch ===")
    numeric_cols = [
        c for c in ("hora", "dow", "mes_num", "trimestre_num", "n_afectadas")
        if c in df.columns
    ]
    categorical_cols = [
        c for c in df.columns if c.startswith("Motivo") or c == "Empresaprestadora"
    ][:3]
    X, y = build_feature_matrix(df, numeric_cols, categorical_cols)

    df_aligned = df.filter(pl.col("evento_critico").is_not_null())
    df_with_ts = df_aligned.select(["ts_inicio", *X.columns, "evento_critico"])
    tr, te = temporal_train_test_split(df_with_ts, "ts_inicio", cutoff=cutoff)

    X_tr = tr.select(X.columns)
    X_te = te.select(X.columns)
    y_tr = tr.get_column("evento_critico").cast(pl.Int8)
    y_te = te.get_column("evento_critico").cast(pl.Int8)

    # Grid chico para que el pipeline corra en minutos, no horas.
    param_grid = {
        "n_estimators": [200],
        "max_depth": [4, 6],
        "learning_rate": [0.05, 0.1],
        "min_child_weight": [1, 5],
    }
    _, report = train_xgboost_grid(X_tr, y_tr, X_te, y_te, param_grid=param_grid, cv_folds=3)
    return {
        "best_params": report.best_params,
        "cv_best_score": report.cv_best_score,
        "evaluation": asdict(report.evaluation),
        "feature_importance": report.feature_importance,
    }


def _step_forecast(df: pl.DataFrame, horizon: int) -> pl.DataFrame:
    logger.info("=== Paso 4/4: forecasting mensual ===")
    monthly = aggregate_monthly(
        df.filter(pl.col("ts_inicio").is_not_null() & pl.col("duracion_horas").is_not_null()),
        "ts_inicio",
        "duracion_horas",
        agg="sum",
    )
    series = monthly.get_column("duracion_horas")
    y_train, y_test = train_test_horizon_split(series, horizon)

    results = [
        forecast_naive_seasonal(y_train, y_test, season_length=12),
        forecast_ets(y_train, y_test, seasonal_periods=12),
        forecast_sarima(y_train, y_test),
        forecast_lgbm_lags(y_train, y_test),
        forecast_xgb_lags(y_train, y_test, grid_search=False),
    ]
    results.append(forecast_ensemble(results, name="ensemble_mean"))
    return compare_forecasts(results)


def main() -> int:
    args = _parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    t0 = perf_counter()

    df_int = _step_interrupciones()
    df_int.write_parquet(args.artifacts / "interrupciones_enriched.parquet")

    df_morea_clean = _step_morea_clean()
    df_morea_clean.write_parquet(args.artifacts / "morea_depurado.parquet")

    clf_report = _step_classification(df_int, cutoff=args.cutoff)
    (args.artifacts / "modelo_metricas.json").write_text(
        json.dumps(clf_report, indent=2, default=str), encoding="utf-8"
    )
    pl.DataFrame(
        [{"feature": k, "importance": v} for k, v in clf_report["feature_importance"].items()]
    ).write_csv(args.artifacts / "feature_importance.csv")

    forecast_table = _step_forecast(df_int, horizon=args.horizon)
    forecast_table.write_csv(args.artifacts / "forecast_metrics.csv")

    elapsed = perf_counter() - t0
    summary = {
        "tiempo_total_s": round(elapsed, 1),
        "clasificador": {
            "best_params": clf_report["best_params"],
            "pr_auc_test": clf_report["evaluation"]["pr_auc"],
            "recall_at_p90": clf_report["evaluation"]["recall_at_p90"],
        },
        "forecast_best": forecast_table.head(1).to_dicts()[0] if forecast_table.height > 0 else {},
    }
    (args.artifacts / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Pipeline terminado en %.1fs. Artefactos en %s", elapsed, args.artifacts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
