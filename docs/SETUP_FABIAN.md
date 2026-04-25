# Onboarding del repo · Fabian Espinoza (SSA11)

Hola Fabian. Esta guia es para que clones el repo en tu **Windows** y lo
levantes en menos de 15 minutos. Donde haya diferencia con macOS dejo los
dos comandos: usa el de **Windows (PowerShell)** salvo que digas lo contrario.

Si algo no funciona, mandame el error textual o la captura — todo lo que
documento aca esta probado en M2 (mio) y validado en Windows (tuyo) por
medio de uv que es identico en los dos.

## 0 · Lo que necesitas instalado

### Windows (lo tuyo)

```powershell
# Python 3.12 — desde winget (recomendado) o https://www.python.org/downloads/
winget install --id Python.Python.3.12

# Verifica
python --version    # debe decir 3.12.x

# uv — el unico gestor de paquetes que usamos
irm https://astral.sh/uv/install.ps1 | iex

# Cierra y reabre la terminal para que el PATH agarre uv
uv --version

# Git (si aun no esta)
winget install --id Git.Git

# (Opcional) Docker Desktop, si quieres probar el contenedor
winget install --id Docker.DockerDesktop

# (Opcional) Quarto solo si vas a regenerar el deck
winget install --id Posit.Quarto
```

### macOS (referencia para mi)

```bash
brew install python@3.12 git
curl -LsSf https://astral.sh/uv/install.sh | sh
brew install --cask quarto docker
```

> **Importante:** no uses `pip` directo y no crees el venv a mano. `uv sync`
> hace ambas cosas. Cualquier dep nueva entra con `uv add <pkg>`.

## 1 · Clonar y sincronizar dependencias

```powershell
# Windows: PowerShell o Git Bash, cualquiera funciona
git clone https://github.com/f3r21/DatathonSunass.git
cd DatathonSunass\repo
uv sync
```

```bash
# macOS / Linux / Git Bash
git clone https://github.com/f3r21/DatathonSunass.git
cd DatathonSunass/repo
uv sync
```

`uv sync` crea `.venv\` (Win) o `.venv/` (Mac) y resuelve `pyproject.toml +
uv.lock`. Tarda ~3-5 min la primera vez (descarga torch, xgboost, lightgbm,
streamlit, plotly...).

### Activar el venv (opcional, solo si quieres `python` directo)

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Si PowerShell se queja por execution policy:
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

```bash
# macOS / Git Bash en Win
source .venv/bin/activate     # Mac
source .venv/Scripts/activate # Git Bash en Win
```

> Para correr cosas, lo mas comodo es **NO** activar y solo prefijar con
> `uv run`. Ej: `uv run streamlit run app/Home.py`. Asi siempre usa el
> entorno correcto del proyecto.

## 2 · Configurar `.env`

```powershell
# Windows PowerShell
Copy-Item .env.example .env
notepad .env
```

```bash
# macOS / Git Bash
cp .env.example .env
$EDITOR .env
```

Edita las **3 rutas de datos**. **Usa siempre forward slashes (`/`)**
incluso en Windows — Python las normaliza y evitas problemas de escapes
con `\`. El default asume que los datasets viven al lado del repo:

```
INTERRUPCIONES_PATH=../datos/interrupciones/historico_interrupciones_limpio.dta
MOREA_PARQUET_PATH=../datos/morea/datos_morea.parquet
MOREA_ESTACIONES_PATH=../datos/morea/ubicacion_estaciones_MOREA.xlsx
```

Si los pones en otro lado (ej. `C:\Users\Fabian\Documents\Datathon\datos\...`),
escribelo asi:

```
INTERRUPCIONES_PATH=C:/Users/Fabian/Documents/Datathon/datos/interrupciones/historico_interrupciones_limpio.dta
```

Forward slashes incluso con la letra de unidad. Polars / pyreadstat lo
manejan sin problema. `\` en `.env` se interpreta como escape y truena.

> Los datasets oficiales NO estan en el repo (son grandes y tienen
> licencia). Comparto por **Google Drive** desde `99bigdatacloud@gmail.com`.
> Si necesitas credenciales adicionales para acceder, te las paso por canal
> privado (no las pongas en commits).

## 3 · Smoke test (30 segundos)

Antes de levantar Streamlit, valida que el entorno esta sano:

```powershell
uv run python scripts\smoke_setup.py
```

```bash
uv run python scripts/smoke_setup.py
```

Si todo va bien, ves al final `[OK] Setup correcto. Listo para correr la app.`
Si truena, copia y pegame el error completo.

## 4 · Levantar la app

`make` no esta en Windows por default, asi que tienes dos opciones:

### Opcion A · Comando `uv run` directo (lo mas portable)

```powershell
uv run streamlit run app/Home.py
```

### Opcion B · Si quieres `make` en Windows

Instala `make` con scoop o chocolatey:

```powershell
scoop install make            # si tienes scoop
# o
choco install make            # si tienes chocolatey

# Despues
make app
```

### macOS (mio)

```bash
make app
# equivalente a: uv run streamlit run app/Home.py
```

Abre `http://localhost:8501`. Aparece el form de login.

**Tu usuario:** `fabio` · **contrasena:** `ssa11` (rol admin).
El sidebar te saluda como "Fabian Manuel Espinoza Koctong @fabio".

## 5 · Tour rapido (5 min)

Las 11 vistas en orden:

| # | Pagina        | Que hace                                                |
|---|---------------|---------------------------------------------------------|
| H | Home          | KPIs y propuesta en una frase                           |
| 0 | Ejecutivo     | Tablero un-vistazo: semaforos + tendencia 24m           |
| 1 | Datos         | Inventario + pipeline de limpieza + nulos               |
| 2 | EDA           | Heatmap correlacion, distribuciones, timelines          |
| 3 | Modelo        | XGBoost + GridSearchCV (tarda ~2 min la primera vez)    |
| 4 | Forecasting   | 6 modelos + sMAPE                                       |
| 5 | Alertas       | Eventos DIGESA + workflow de incidencias                |
| 6 | Modo Dia D    | Sube CSV climatico SENAMHI (Año, Mes, Dia, ...)         |
| 7 | Reportes      | Genera XLSX y PDF para descargar                        |
| 8 | Mapa          | Folium con estaciones MOREA                             |
| 9 | Stream        | Auto-refresh con notificaciones live de violaciones     |

## 6 · Tu foco: preprocesamiento

Estos son los modulos donde pasa la limpieza y la ingenieria de variables.
Modificalos aca, no dupliques codigo en notebooks:

```
src/
├── io.py                        # carga .dta / .parquet / .xlsx + load_senamhi_daily
├── inspector.py                 # detector generico de tipos + geo (lat/lon)
├── eda.py                       # perfilado: ColumnProfile, gaps temporales
└── modeling/
    ├── features.py              # add_timestamps, add_duracion_impacto,
    │                              add_temporal_features, frequency_encode,
    │                              build_feature_matrix
    ├── anomalias.py             # filter_imposibles, sustained_violations,
    │                              chronic_stations, isolation_forest_scan,
    │                              compare_bruto_vs_depurado
    └── clasificacion.py         # train_xgboost_grid, temporal_train_test_split
```

### Convenciones que ya estan en uso

- **Polars first.** Pandas solo cuando hace falta para xlsxwriter / Plotly.
- **No prints.** Usa `logger = logging.getLogger(__name__)`.
- **Frozen dataclasses** para resultados (`FilterResult`, `ColumnProfile`,
  `ModelEvaluation`, `GridSearchReport`, `ForecastResult`).
- **Type annotations** en todas las firmas, PEP 8.
- **No emojis** en codigo (preferencia explicita de fer).
- **Spanish** en docstrings y comentarios; identificadores en ingles.
- **Target leakage** declarado: `evento_critico = duracion_horas * unidades > 100_000`.
  Si agregas una nueva feature derivada de `duracion_horas`, **pasala por
  el filtro de leakage** (`_LEAKAGE_COLS` en `app/pages/3_Modelo.py`).

### Atajos para iterar rapido (Windows)

```powershell
# Editar una funcion en src\modeling\anomalias.py:
uv run python scripts\smoke_modeling.py     # corre las plantillas A+B en CLI

# Ver el efecto en la app sin reiniciar:
# Streamlit recarga solo. Si tocas un modulo cacheado, click "Rerun" arriba.

# Validar todo el pipeline end-to-end sin UI:
uv run python scripts\run_pipeline.py       # raw -> clean -> XGB -> artifacts\

# Inspeccionar un dataset desde Python:
uv run python -c "from src.io import load_interrupciones, paths_from_env; df = load_interrupciones(paths_from_env().interrupciones); print(df.schema); print(df.head())"
```

## 7 · Pendientes que pueden ser tuyos

Si quieres tomar algo concreto:

- **Plantillas D + E + F** — deduplicacion, normalizacion espacial, export
  estructurado de findings. Ver `docs/EXPLORACION.md` (estrategia completa).
- **Ingenieria de variables sobre SENAMHI**: lags, rollings y interacciones
  precip x temp para forecasting climatico. Empieza desde
  `src/io.py::load_senamhi_daily()`.
- **Cross-dataset feature** — joinear interrupciones con MOREA por UBIGEO +
  ventana temporal. Esa feature deberia explotar el modelo si la haces bien.

## 8 · Troubleshooting frecuente (Windows-first)

| Sintoma | Causa probable / fix |
|---|---|
| `uv: command not found` | Cierra y reabre la terminal. Si persiste, agrega `%USERPROFILE%\.cargo\bin` al PATH o reinstala uv. |
| `Set-ExecutionPolicy ... cannot be run` | Ejecuta PowerShell **como administrador** la primera vez. Despues `RemoteSigned` para tu usuario alcanza. |
| `quarto: command not found` | `winget install --id Posit.Quarto` y reabre terminal. |
| `KeyError: INTERRUPCIONES_PATH` | No copiaste `.env.example` a `.env`. Hazlo y edita rutas. |
| `FileNotFoundError: ../datos/...` | Los datasets no estan en la ruta del `.env`. Pidemelos. |
| `.env` con `\` y truena al cargar | Cambia los `\` por `/`. Funcionan iguales en Windows desde Python. |
| `make: *** missing separator` o `'make' no se reconoce` | `make` no esta. Usa `uv run streamlit run app/Home.py` directo, o instala `make` con scoop/chocolatey. |
| Modelo tarda mucho la primera vez | Normal. GridSearchCV. ~2 min en M2, ~3-5 en CPU Windows. Despues queda cacheado. |
| Pagina Modelo no muestra grafico | Sin features seleccionadas. Activa al menos una en el sidebar y dale "Entrenar modelo". |
| `Excel does not support datetimes with timezones` | Ya esta arreglado en `src/reports/export.py::_strip_timezones`. Si vuelve, dime que columna. |
| `KeyError: 'logout'` al hacer login | Cache vieja. Borra `.venv` y `uv sync` otra vez, o `uv sync --reinstall-package datathon-sunass`. |
| Auto-refresh en Stream consume CPU | Apaga el toggle "Auto-refresh" del sidebar y avanza manual. |
| Mapa pagina 8 muestra warning de geodata | El catalogo MOREA no tenia lat/lon. Es esperado por ahora — el detector las busca automatico. |
| Line endings raros en commits (CRLF vs LF) | Ya tienes `.gitattributes` con `* text=auto eol=lf`. Si te olvido, corre `git config core.autocrlf input` una sola vez. |

## 9 · Como contribuir cambios

```powershell
git checkout -b fabian/preprocessing-mejoras
# tus cambios

uv run ruff check src\ app\ scripts\

# Validar syntax (PowerShell)
Get-ChildItem -Recurse -Include *.py src,app,scripts | ForEach-Object { uv run python -m py_compile $_.FullName }

git add -A
git commit -m "feat(preprocessing): ..."
git push origin fabian/preprocessing-mejoras
```

```bash
# macOS / Git Bash equivalentes
uv run ruff check src/ app/ scripts/
uv run python -m py_compile $(find src app scripts -name '*.py')
```

Si tocas `pyproject.toml` (agregas dep), confirma que `uv sync` queda
limpio en mi M2 antes de pushear — pasa `uv.lock` actualizado en el commit.

## 10 · Compatibilidad cross-platform

Tres reglas para que mi M2 y tu Windows no se peleen:

1. **Rutas** siempre con forward slashes en `.env` y en codigo Python.
   Usa `pathlib.Path` en src/, no string concatenation.
2. **Line endings**: el repo tiene `.gitattributes` con `eol=lf`. No
   cambies eso.
3. **Comandos shell** en docs: ofrece variantes Windows + Mac. Si dudas,
   prefija con `uv run` y vas seguro en cualquier SO.

## 11 · Contacto

- fer · 99bigdatacloud@gmail.com (M2, macOS)
- Fabian — tu (Windows)
- Codigo de equipo SUNASS: **SSA11** (Categoria I Operacional)

Si vas a estar dandole hoy/mañana, mandame mensaje y empujamos juntos.
