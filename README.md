# Datathon SUNASS 2026 · SSA11 · Categoria I Operacional

Entrega del equipo **SSA11** (UCSP, Escuela de Ciencia de la Computacion) para el
concurso presencial del **sabado 25 de abril de 2026** en Arequipa.

> Categoria I = Operaciones y calidad de servicio. Mapeamos cada vista de la app
> a uno de los 7 resultados oficiales de la categoria.

| Que ver | Donde |
|---|---|
| **Onboarding paso a paso** (Windows + macOS) | [`docs/SETUP_FABIAN.md`](docs/SETUP_FABIAN.md) |
| **Estado del repo + bibliografia** | [`docs/PAPERS_FORECASTING.md`](docs/PAPERS_FORECASTING.md) |
| **Deck Quarto revealjs** | [`reports/deck.qmd`](reports/deck.qmd) |
| **Pipeline reproducible CLI** | `uv run python scripts/run_pipeline.py` |

## Demo

<!-- Placeholder hasta grabar el video. Reemplazar por iframe YouTube/Loom. -->
*Video demo: pendiente de grabacion.*

## Quickstart (60 segundos)

```bash
git clone https://github.com/f3r21/DatathonSunass.git
cd DatathonSunass/repo
cp .env.example .env             # edita las 3 rutas a los datasets
uv sync                          # instala todo
uv run streamlit run app/Home.py # http://localhost:8501
```

Login demo: `fer / ssa11` (admin) o `jurado / sunass2026` (viewer).

Para Docker: `docker compose -f docker/docker-compose.yml up`.

> Si eres Fabian: empieza por [`docs/SETUP_FABIAN.md`](docs/SETUP_FABIAN.md).
> Tiene comandos PowerShell paralelos a los de macOS.

## Mapeo a los 7 resultados oficiales

| # | Resultado oficial SUNASS                              | Donde se cumple                                                                 |
|---|--------------------------------------------------------|---------------------------------------------------------------------------------|
| 1 | Tableros de control                                   | `app/pages/0_Ejecutivo.py` (semaforos + tendencia 24m + KPIs)                   |
| 2 | Automatizacion de reportes                            | `app/pages/7_Reportes.py` + `src/reports/export.py` (XLSX + PDF)                |
| 3 | Analisis de indicadores operativos                    | `app/Home.py` + `app/pages/2_EDA.py` (heatmap, distribuciones, timelines)       |
| 4 | Deteccion temprana de desviaciones                    | `app/pages/5_Alertas.py` + `app/pages/9_Stream.py` + `src/monitoring/`          |
| 5 | Desarrollo de aplicaciones                            | `app/` Streamlit + `docker/` (Dockerfile + compose)                             |
| 6 | IA aplicada a toma de decisiones                      | `app/pages/3_Modelo.py` (XGBoost + GridSearch) + `app/pages/4_Forecasting.py`   |
| 7 | Atencion oportuna de incidencias                      | Workflow NUEVO/EN_REVISION/RESUELTO en `app/pages/5_Alertas.py`                 |

Plus: `app/pages/8_Mapa.py` (folium con estaciones MOREA + capas geogpsperu) y
`app/pages/6_Modo_Dia_D.py` (uploader generico para CSV/XLSX climatico SENAMHI).

## Arquitectura

```
┌────────────────────────────────────────────────────────────────────┐
│                       App Streamlit (app/)                         │
│ Home · Ejecutivo · Datos · EDA · Modelo · Forecasting · Alertas    │
│         · Modo Dia D · Reportes · Mapa · Stream                    │
│   Login wall (streamlit-authenticator) en cada pagina              │
└──────────────────────────┬─────────────────────────────────────────┘
                           │  cache_data + cache_resource
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│                    Paquete src/ (logica pura)                      │
│  io.py            carga .dta / .parquet / .xlsx / SENAMHI csv      │
│  inspector.py     detector generico de tipos + lat/lon             │
│  modeling/        features, clasificacion (XGB+Grid), forecasting  │
│                   (ETS, SARIMA, LGBM, XGB, ensemble), anomalias    │
│  monitoring/      thresholds DIGESA + alertas + workflow incidencias│
│  reports/         build_report_xlsx + build_report_pdf             │
│  viz/             plots Plotly compartidos app+deck                │
└──────────────────────────┬─────────────────────────────────────────┘
                           │  lee
                           ▼
┌────────────────────────────────────────────────────────────────────┐
│        Datasets oficiales (fuera del repo, via .env)               │
│  interrupciones/historico_interrupciones_limpio.dta  (52k filas)   │
│  morea/datos_morea.parquet                          (1.59M filas)  │
│  morea/ubicacion_estaciones_MOREA.xlsx              (26 estaciones)│
└────────────────────────────────────────────────────────────────────┘
```

Pipeline reproducible sin UI: `scripts/run_pipeline.py` corre raw → clean →
features → XGB+Grid → forecast → exporta artefactos a `artifacts/`.

## Stack tecnico

| Capa            | Herramientas                                                       |
|-----------------|--------------------------------------------------------------------|
| Datos           | polars, pandas, pyreadstat, pyarrow, openpyxl, xlsxwriter          |
| ML clasico      | scikit-learn, xgboost, lightgbm, imbalanced-learn, shap, ruptures  |
| Time series     | statsmodels, XGBoost/LightGBM con features de lag                  |
| Geo             | folium, shapely                                                    |
| Visualizacion   | plotly, matplotlib                                                 |
| UI + Auth       | streamlit, streamlit-authenticator, bcrypt                         |
| Reportes        | xlsxwriter (XLSX), pymupdf (PDF)                                   |
| Deck            | Quarto revealjs                                                    |
| Entorno         | uv (Python 3.12), Docker                                           |

## Estructura del repo

```
repo/
├── app/                  Streamlit multipagina + componentes (auth, theme, kpi)
├── src/                  Paquete datathon-sunass (io, inspector, modeling/,
│                          monitoring/, reports/, viz/)
├── scripts/
│   ├── run_pipeline.py   Pipeline end-to-end CLI
│   └── smoke_setup.py    Validacion de entorno en <30s
├── docker/               Dockerfile + docker-compose + .dockerignore
├── reports/              deck.qmd + assets/ + references.bib
├── docs/                 SETUP_FABIAN.md + PAPERS_FORECASTING.md
├── config/users.yaml     Credenciales bcrypt
├── .streamlit/config.toml
├── .env.example          Plantilla de rutas a datasets
├── Makefile              Atajos macOS/Linux
├── run_app.ps1           Atajo Windows PowerShell
├── run_app.bat           Atajo Windows CMD
└── pyproject.toml        Definicion uv (Python 3.12)
```

## Datos compartidos

Los datasets oficiales **no** estan en el repo (tamaño + licencia SUNASS). Se
intercambian por **USB** o **Google Drive** dentro del equipo. Si tienes acceso al
Drive, las rutas en `.env` apuntan a tu copia local descargada en `../datos/`.

## Insights

1. Separar **glitches de sensor** (fisicamente imposibles) de violaciones reales
   en MOREA cambia drasticamente la tasa DIGESA reportada. Es el numero honesto.
2. El baseline del mentor (logit, 85.6% accuracy) tiene **0.77% de sensitivity**.
   Para un target ~2-5% positivo, accuracy es engañoso; reportamos PR-AUC y
   recall@precision90.
3. **`duracion_horas` es leakage** del target (`evento_critico = duracion_horas
   × unidades_afectadas > 100k`). El selector de features en la pagina Modelo
   excluye automaticamente leakage para evitar PR-AUC ~0.999 falso.
4. En forecasting el **ensemble por promedio simple** le gana consistentemente a
   los individuales en sMAPE (Clemen 1989).

## Equipo

- **Fernando Ramirez Arredondo** (`fer`) · 99bigdatacloud@gmail.com · macOS M2
- **Fabian Manuel Espinoza Koctong** (`fabio`) · Windows · foco preprocesamiento

UCSP — Escuela de Ciencia de la Computacion · Arequipa, Peru.

## Licencia

Codigo bajo **MIT**. Datos del concurso SUNASS — no se incluyen en el repo.
