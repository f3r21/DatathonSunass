"""Pagina 7 · Reportes — export institucional XLSX y PDF.

Cubre el resultado oficial 2 (Automatizacion de reportes). Arma un
`ReportContext` con los mismos caches que las otras paginas y entrega dos
archivos listos para enviar por correo o subir a la plataforma.
"""
from __future__ import annotations

import sys
from datetime import datetime
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
    get_morea_sensores,
)
from src.modeling.anomalias import (  # noqa: E402
    chronic_stations,
    filter_imposibles,
    sustained_violations,
)
from src.modeling.forecasting import (  # noqa: E402
    aggregate_monthly,
    compare_forecasts,
    forecast_ensemble,
    forecast_ets,
    forecast_lgbm_lags,
    forecast_naive_seasonal,
    train_test_horizon_split,
)
from src.monitoring.alerts import build_alerts, summarize_alerts  # noqa: E402
from src.monitoring.thresholds import DEFAULT_CONFIGS, detect_violations  # noqa: E402
from src.reports import ReportContext, build_report_pdf, build_report_xlsx  # noqa: E402

st.set_page_config(page_title="Reportes · SUNASS", layout="wide")
require_auth("Reportes")
st.title("7 · Reportes automatizados")
st.caption(
    "Genera un informe XLSX y un PDF con los hallazgos actuales. "
    "Cubre el resultado oficial 2 (Automatizacion de reportes)."
)

ok, msg = data_available()
if not ok:
    st.error(msg)
    st.stop()


@st.cache_data(show_spinner="Consolidando reporte...")
def _build_context() -> ReportContext:
    df_int = get_interrupciones(enriched=True)
    df_morea_clean = filter_imposibles(get_morea_sensores()).depurado

    # KPIs.
    n_eventos = df_int.height
    n_crit = 0
    if "evento_critico" in df_int.columns:
        n_crit = int(df_int.get_column("evento_critico").fill_null(False).cast(int).sum())

    pct_cumplimiento = 100.0
    if "cloro" in df_morea_clean.columns:
        viol = detect_violations(df_morea_clean, "cloro", keep_ok=False)
        pct_cumplimiento = 100 - 100 * viol.height / max(df_morea_clean.height, 1)

    rango = "s/d"
    if "ts_inicio" in df_int.columns:
        nn = df_int.get_column("ts_inicio").drop_nulls()
        if nn.len() > 0:
            rango = f"{nn.min().date()} a {nn.max().date()}"

    kpis = [
        ("Eventos totales de interrupcion", f"{n_eventos:,}"),
        ("Eventos criticos", f"{n_crit:,}  ({100*n_crit/max(n_eventos,1):.1f}%)"),
        ("Lecturas MOREA depuradas", f"{df_morea_clean.height:,}"),
        ("Cumplimiento cloro DIGESA", f"{pct_cumplimiento:.2f}%"),
        ("Rango temporal cubierto", rango),
    ]

    # Alertas activas (cloro por default; parametrizable si hace falta).
    param = "cloro" if "cloro" in df_morea_clean.columns else next(iter(DEFAULT_CONFIGS))
    df_v = detect_violations(df_morea_clean, param, keep_ok=True)
    events = build_alerts(df_v.filter(pl.col("viola")), parameter=param)
    alerts_table = summarize_alerts(events[:30])

    # Forecasting: re-construccion barata con 3 modelos + ensemble.
    forecast_table = pl.DataFrame()
    if "ts_inicio" in df_int.columns and "duracion_horas" in df_int.columns:
        monthly = aggregate_monthly(
            df_int.filter(
                pl.col("ts_inicio").is_not_null() & pl.col("duracion_horas").is_not_null()
            ),
            "ts_inicio",
            "duracion_horas",
            agg="sum",
        )
        if monthly.height >= 15:
            try:
                serie = monthly.get_column("duracion_horas")
                y_tr, y_te = train_test_horizon_split(serie, horizon=3)
                fc_results = [
                    forecast_naive_seasonal(y_tr, y_te, season_length=12),
                    forecast_ets(y_tr, y_te, seasonal_periods=12),
                    forecast_lgbm_lags(y_tr, y_te),
                ]
                fc_results.append(forecast_ensemble(fc_results, name="ensemble_mean"))
                forecast_table = compare_forecasts(fc_results)
            except Exception:
                forecast_table = pl.DataFrame()

    # Estaciones cronicas.
    df_sost = sustained_violations(
        df_morea_clean, col="cloro", low=0.5, high=5.0, min_consecutive=3
    )
    cronicas = chronic_stations(df_sost, min_hits=5, top_k=15)

    insights = [
        "Separar glitches fisicos de violaciones reales reduce la tasa DIGESA reportada drasticamente.",
        "El target evento_critico se define como duracion_horas * unidades_afectadas > 100.000.",
        "Para dataset desbalanceado (~2-5% positivos) reportamos PR-AUC y recall@precision90, no accuracy.",
        "El ensemble por promedio gana consistentemente a los modelos individuales (Clemen 1989).",
    ]

    return ReportContext(
        team_code="SSA11",
        categoria="I Operacional",
        title="Informe operacional — Datathon SUNASS 2026",
        subtitle="UCSP · Escuela de Ciencia de la Computacion",
        generated_at=datetime.now(),
        kpis=kpis,
        alerts_table=alerts_table,
        forecast_table=forecast_table,
        chronic_stations=cronicas,
        model_metrics={
            "cobertura_cumplimiento_cloro_pct": round(pct_cumplimiento, 2),
            "n_estaciones_cronicas_cloro": cronicas.height,
            "n_alertas_activas_cloro": len(events),
        },
        insights=insights,
    )


ctx = _build_context()

st.subheader("Vista previa")
col1, col2, col3 = st.columns(3)
col1.metric("Alertas en el reporte", ctx.alerts_table.height)
col2.metric("Filas de forecasting", ctx.forecast_table.height)
col3.metric("Estaciones criticas", ctx.chronic_stations.height)

with st.expander("Indicadores a incluir", expanded=True):
    for label, value in ctx.kpis:
        st.markdown(f"- **{label}:** {value}")

with st.expander("Insights del reporte"):
    for bullet in ctx.insights:
        st.markdown(f"- {bullet}")

st.markdown("---")
st.subheader("Descargar informe")

colx, coly = st.columns(2)
timestamp = ctx.generated_at.strftime("%Y%m%d_%H%M")

with colx:
    with st.spinner("Armando XLSX..."):
        xlsx_bytes = build_report_xlsx(ctx)
    st.download_button(
        label="Descargar XLSX",
        data=xlsx_bytes,
        file_name=f"informe_SSA11_{timestamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        width="stretch",
    )

with coly:
    with st.spinner("Armando PDF..."):
        try:
            pdf_bytes = build_report_pdf(ctx)
            st.download_button(
                label="Descargar PDF",
                data=pdf_bytes,
                file_name=f"informe_SSA11_{timestamp}.pdf",
                mime="application/pdf",
                width="stretch",
            )
        except Exception as exc:
            st.warning(f"PDF no disponible: {exc}")

st.markdown("---")
st.subheader("Tablas incluidas")
st.markdown("**Alertas activas (top 30)**")
st.dataframe(ctx.alerts_table, width="stretch", height=240)
st.markdown("**Forecasting — comparativa de modelos**")
st.dataframe(ctx.forecast_table, width="stretch")
st.markdown("**Estaciones cronicas (cloro)**")
st.dataframe(ctx.chronic_stations, width="stretch")
