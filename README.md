# Datathon SUNASS 2026 — Categoría I Operacional

Entrega del equipo UCSP para el concurso presencial del **sábado 25 de abril de 2026**.
Reto abordado: **Solo Operacional** (calidad de servicio, interrupciones, cobertura).

## Componentes de la entrega

1. **Análisis reproducible** (`src/`, `notebooks/`) — EDA multi-fuente, deduplicación, análisis espacial, clasificación desbalanceada y detección de anomalías.
2. **Dashboard interactivo** (`dashboard/`) — Streamlit multi-página con autenticación por rol y páginas de KPIs, mapa, alertas y chat.
3. **Chatbot corporativo** (`src/rag/`, `dashboard/pages/4_Chat.py`) — LLM dual (Ollama local + OpenAI fallback) sobre corpus SUNASS + datos del concurso con RAG y citas.
4. **Deck de exposición** (`reports/deck.qmd`) — 10 slides incluyendo roadmap y riesgos.

Detalle arquitectónico en [`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md).

## Requisitos

- **Python** 3.12 (exacto: `>=3.12,<3.13`).
- **uv** como gestor (https://docs.astral.sh/uv/). `pip` no está soportado.
- Para el chatbot en modo local: **Ollama** corriendo en una máquina accesible (3060 remota recomendada, M2 como fallback).
- Para el modo OpenAI: API key válida con acceso a `gpt-4o-mini`.

## Instalación

```bash
git clone https://github.com/f3r21/DatathonSunass.git
cd DatathonSunass
uv sync                     # crea .venv y resuelve dependencias
cp .env.example .env        # editar con rutas de datos y credenciales
```

Los datasets oficiales (`datos/interrupciones/*.dta`, `datos/morea/*`) se mantienen **fuera del repo** por tamaño y licencia. Apuntar a ellos desde `.env` (`DATA_DIR`, `INTERRUPCIONES_PATH`, `MOREA_PARQUET_PATH`, `MOREA_ESTACIONES_PATH`).

## Uso

```bash
make eda          # ejecuta EDA 01_eda.ipynb
make train        # entrena clasificador desbalanceado + anomalías
make index        # construye indice RAG desde docs_sunass/
make dashboard    # lanza Streamlit (login por rol)
```

## Autenticación del dashboard

Tres roles definidos en `config/users.yaml` (generado por script aparte para no comprometer hashes):

| Rol         | Permisos                                                     |
|-------------|--------------------------------------------------------------|
| `admin`     | Acceso total, gestión de usuarios, export de findings        |
| `analista`  | Lectura de KPIs/mapa + chat con RAG                          |
| `visitante` | Demo: solo panel de KPIs agregados                           |

Hashes bcrypt. Credenciales demo generadas con `uv run python -m src.auth_setup`.

## Stack técnico

- **Datos**: polars, pandas, pyreadstat, pyarrow.
- **ML**: scikit-learn, xgboost, lightgbm, imbalanced-learn, shap, ruptures.
- **Geo**: geopandas, folium, shapely.
- **LLM/RAG**: llama-index, Chroma, sentence-transformers, Ollama + OpenAI fallback.
- **UI**: Streamlit + streamlit-authenticator.
- **Deck**: Quarto (fallback Marp).

## Estructura

```
.
├── src/               # paquete datathon_sunass
│   ├── io.py          # carga .dta/.parquet/.xlsx con validación
│   ├── eda.py         # funciones de perfilado
│   ├── features.py    # ingeniería de features (duración, impacto, evento_critico)
│   ├── models.py      # clasificadores desbalanceados y anomalías
│   ├── llm.py         # LLMClient dual Ollama/OpenAI
│   ├── auth_setup.py  # bootstrap de config/users.yaml
│   └── rag/
│       ├── build_index.py   # construye Chroma desde docs_sunass/
│       └── retrieve.py      # consulta con citas
├── dashboard/
│   ├── Home.py
│   └── pages/
│       ├── 1_KPIs.py
│       ├── 2_Mapa.py
│       ├── 3_Alertas.py
│       └── 4_Chat.py
├── notebooks/         # análisis exploratorio
├── reports/
│   ├── deck.qmd
│   ├── figuras/
│   └── tablas/
├── docs/
│   └── ARQUITECTURA.md
├── config/
│   └── users.yaml     # generado (no comiteado)
├── chroma_db/         # indice persistente (gitignored)
├── .env.example
├── Makefile
└── pyproject.toml
```

## Licencia

MIT (código). Los datos provienen del concurso SUNASS y mantienen sus licencias originales.
