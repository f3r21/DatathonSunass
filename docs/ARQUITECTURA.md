# Arquitectura — Datathon SUNASS 2026 · SSA11

Describe los componentes en uso y como se conectan. **Nada de aqui es
aspiracional**: si un componente no existe en `src/` o `app/`, no esta en este
documento.

## Componentes

### App Streamlit (`app/`)

Multi-pagina con login wall en cada vista (`require_auth`), tema azul+verde
SUNASS y botón **Cerrar sesion** persistente en el sidebar.

| Archivo                         | Resultado oficial cubierto |
|---------------------------------|----------------------------|
| `Home.py`                       | 3 (indicadores)            |
| `pages/0_Ejecutivo.py`          | 1 (tableros) + 4 (deteccion)|
| `pages/1_Datos.py`              | 3 (indicadores)            |
| `pages/2_EDA.py`                | 3 (indicadores)            |
| `pages/3_Modelo.py`             | 6 (IA aplicada)            |
| `pages/4_Forecasting.py`        | 6 (IA aplicada)            |
| `pages/5_Alertas.py`            | 4 (deteccion) + 7 (atencion oportuna)|
| `pages/6_Modo_Dia_D.py`         | 6 (uploader generico SENAMHI)|
| `pages/7_Reportes.py`           | 2 (reportes XLSX/PDF)      |
| `pages/8_Mapa.py`               | 1 + 5 (folium + geogpsperu)|
| `pages/9_Stream.py`             | 4 (notificaciones live)    |
| `components/auth.py`            | login wall + logout sidebar|
| `components/data_loader.py`     | carga cacheada con st.cache|
| `components/theme.py`           | paleta + Plotly template   |
| `components/kpi.py`             | KPI cards reutilizables    |

### Paquete `src/`

Logica pura, sin dependencias de Streamlit. Reusable desde la app, scripts CLI
y notebooks.

```
src/
├── io.py                   load_interrupciones (.dta), load_morea (.parquet
│                           + .xlsx), load_senamhi_daily (.csv/.xlsx),
│                           paths_from_env() resuelve rutas relativas a .env
├── inspector.py            inspect_dataframe → ColumnInfo (kind, dtype, nulls,
│                           confidence). Detecta numerico, categorico, datetime,
│                           time, geo_lat, geo_lon, geo_combined, identifier.
├── eda.py                  perfilado de DataFrames (legacy, opcional)
├── modeling/
│   ├── features.py         add_timestamps, add_duracion_impacto,
│   │                       add_temporal_features, frequency_encode,
│   │                       build_feature_matrix
│   ├── anomalias.py        filter_imposibles (FilterResult),
│   │                       sustained_violations, chronic_stations,
│   │                       isolation_forest_scan, ruptures_changepoints
│   ├── clasificacion.py    train_logit, train_xgboost, train_xgboost_grid
│   │                       (GridSearchCV → GridSearchReport con
│   │                       feature_importance + cv_results_top)
│   └── forecasting.py      forecast_naive_seasonal, forecast_ets,
│                           forecast_sarima, forecast_lgbm_lags,
│                           forecast_xgb_lags, forecast_ensemble,
│                           train_test_horizon_split, aggregate_monthly,
│                           compare_forecasts (sMAPE)
├── monitoring/
│   ├── thresholds.py       DIGESA_CLORO/PH/TURBIEDAD, ThresholdConfig,
│   │                       detect_violations, stream_scan
│   ├── climate_thresholds.py   ClimateThreshold + DEFAULT_CLIMATE_THRESHOLDS
│   │                       (precip extremo, ola de calor, helada, p95/p98)
│   ├── alerts.py           AlertEvent, build_alerts (colapsa rachas),
│   │                       summarize_alerts
│   └── incidents.py        IncidentRecord + IncidentStore (JSON), workflow
│                           NUEVO/EN_REVISION/RESUELTO con transiciones
├── reports/
│   └── export.py           ReportContext, build_report_xlsx (5 hojas),
│                           build_report_pdf (PyMuPDF, A4)
└── viz/
    └── eda.py              correlation_heatmap, distribution_histogram,
                            boxplot_by_group, interrupciones_timeline,
                            morea_sensor_timeline
```

### Scripts CLI

```
scripts/
├── run_pipeline.py     raw → clean → features → XGBoost+Grid → forecast →
│                       artifacts/ (run_summary.json, modelo_metricas.json,
│                       feature_importance.csv, forecast_metrics.csv,
│                       interrupciones_enriched.parquet, morea_depurado.parquet)
└── smoke_setup.py      Valida entorno en <30s: Python 3.12, .env, datasets
                        accesibles, imports, carga real, auth bcrypt.
```

### Contenedor

```
docker/
├── Dockerfile          python:3.12-slim + uv + libgomp; healthcheck en
│                       /_stcore/health; expone 8501
├── docker-compose.yml  Monta ../../datos como /data:ro; inyecta env
└── .dockerignore       Excluye datos pesados, .venv, _site, _freeze
```

### Reportes

```
reports/
├── deck.qmd            Slides Quarto revealjs (en evolucion; ver tarea 28)
├── _quarto.yml         Profile deck + handout PDF
├── assets/custom.scss  Tema SUNASS revealjs
└── references.bib      Hyndman & Athanasopoulos, Clemen, Makridakis, Zeng
```

### Configuracion

```
config/users.yaml       Credenciales bcrypt (fer, fabio admin; jurado, visitante viewer)
.streamlit/config.toml  Tema + server config
.env.example            Plantilla con rutas a datasets
.gitattributes          eol=lf para evitar guerras CRLF Mac vs Windows
```

## Flujo de datos

```
.dta + .parquet + .xlsx (datos/)
       │
       ├──► src.io.load_interrupciones / load_morea / load_senamhi_daily
       │         │
       │         ▼
       │     pl.DataFrame (con tipos correctos, fechas parseadas)
       │         │
       ├─────────┼────────────────────┬────────────────────┐
       │         │                    │                    │
       ▼         ▼                    ▼                    ▼
  src.eda    src.modeling.features   src.modeling.anomalias  src.monitoring
       │         │                    │                    │
       │         ▼                    ▼                    ▼
       │    add_temporal_features    filter_imposibles    detect_violations
       │    build_feature_matrix     sustained_violations build_alerts
       │         │                    │                    │
       │         ▼                    ▼                    ▼
       │   train_xgboost_grid    chronic_stations       AlertEvent[]
       │         │                                        │
       │         ▼                                        ▼
       │   GridSearchReport                          IncidentStore.upsert
       │
       └────► src.viz.eda → Plotly figures → app/ pages
```

## Convenciones

- **Polars first.** Pandas solo para xlsxwriter y Plotly cuando hace falta.
- **No prints.** Todo via `logger = logging.getLogger(__name__)`.
- **Frozen dataclasses** para resultados (`FilterResult`, `ColumnProfile`,
  `ModelEvaluation`, `GridSearchReport`, `ForecastResult`, `AlertEvent`,
  `ColumnInfo`, `ReportContext`, `IncidentRecord`).
- **Type annotations** en todas las firmas, PEP 8.
- **Spanish** en docstrings + comentarios; identificadores en ingles.
- **No emojis** en codigo.
- **`paths_from_env()`** resuelve rutas relativas contra el `.env` parent, asi
  el pipeline corre desde cualquier cwd (app, scripts, reports).

## Excluido del entregable

Componentes considerados y descartados antes del 25-abr-2026:

- **RAG + chatbot** sobre PDFs SUNASS — fuera del scope para este sabado;
  no estaba pidiendo el reto.
- **Ollama / OpenAI dual-mode** — sin LLM no hay deps pesadas (sentence-
  transformers, llama-index, chromadb borradas del lock).
- **Auth multi-rol granular** — solo admin/viewer; no hace falta mas.
- **Mapas con shapefiles cargados** — la pagina 8 esta lista para superponer
  capas de geogpsperu cuando aterrice un `.shp`, pero no incluimos los
  shapefiles en el repo.
