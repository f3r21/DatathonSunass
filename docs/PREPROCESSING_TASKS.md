# Tareas de preprocesamiento — Fabian (SSA11)

> Scope: lo que falta del lado de **datos**. Modeling y app ya estan estables;
> aqui esta el espacio donde tu trabajo mueve la aguja.

Cada tarea trae: **archivo objetivo**, **firma sugerida**, **acceptance
criteria**, **estimacion**. Toma 1-2 maximo, manda PR con la branch
`fabian/<slug>`. Si dudas algo, abre issue o pingueame.

Ordenadas por impacto sobre la nota final del jurado.

---

## 1. Cross-dataset join MOREA × Interrupciones (alto impacto)

**Por que importa:** unir calidad de agua (MOREA por estacion) con eventos
operativos (interrupciones por EP+UBIGEO) con ventana temporal abre features
nuevas para el clasificador y para narrativa del deck.

**Archivo:** `src/modeling/features.py` (nueva funcion).

**Firma sugerida:**

```python
def join_eventos_calidad(
    df_eventos: pl.DataFrame,             # interrupciones enriquecidas
    df_morea: pl.DataFrame,               # MOREA depurado (con estacion_id, fecha)
    df_estaciones: pl.DataFrame,          # catalogo MOREA con UBIGEO
    ventana_horas_pre: float = 24.0,      # cuantas horas antes del evento mirar
    ventana_horas_post: float = 12.0,     # cuantas horas despues mirar
) -> pl.DataFrame:
    """Para cada evento de interrupcion, agrega:
        - cloro_pre_min / cloro_pre_max / cloro_pre_mean
        - ph_pre_min / ph_pre_max / ph_pre_mean
        - n_lecturas_morea_pre / n_violaciones_pre
    Calculadas sobre las estaciones MOREA en el mismo distrito (UBIGEO),
    en la ventana [ts_inicio - ventana_horas_pre, ts_inicio + ventana_horas_post].
    """
```

**Acceptance criteria:**
- Devuelve `pl.DataFrame` con todas las filas originales de `df_eventos` + 8 cols nuevas.
- Cuando un evento no tiene estacion MOREA en su distrito → cols nuevas con null (no excepcion).
- Test manual: tomar 3 eventos criticos conocidos y verificar que las cols agregadas tienen valores razonables (no todo null, no todo cero).
- `n_lecturas_morea_pre` debe ser >= 0 siempre.
- Pruebalo agregandolo como feature en `app/pages/3_Modelo.py` y mira si sube el PR-AUC.

**Estimacion:** 3-4 horas. La parte fea es el join por UBIGEO porque los strings de distrito en interrupciones no estan normalizados (ver tarea 3).

---

## 2. Normalizacion espacial UBIGEO (medio-alto impacto)

**Por que importa:** habilita la tarea 1 y mejora cualquier agregacion espacial.
Hoy `Departamento`, `Provincia`, `Distrito` en interrupciones llegan como
strings con tildes/mayusculas inconsistentes y NO traen UBIGEO directo.

**Archivo:** `src/inspector.py` (nueva funcion) o `src/modeling/features.py`.

**Firma sugerida:**

```python
def normalize_to_ubigeo(
    df: pl.DataFrame,
    departamento_col: str = "Departamento",
    provincia_col: str = "Provincia",
    distrito_col: str = "Distrito",
    ubigeo_table: pl.DataFrame | None = None,  # tabla INEI 1872 distritos
) -> pl.DataFrame:
    """Normaliza dep/prov/dist (strip + sin acentos + lowercase) y agrega
    columna `ubigeo_distrito` (str de 6 digitos). Filas sin match -> null y
    log.warning con la cantidad.
    """
```

**Acceptance criteria:**
- Tabla INEI de UBIGEO bajada a `datos/geo/ubigeo_inei_2023.csv` (la consigues en geogpsperu o INEI).
- `df.filter(pl.col("ubigeo_distrito").is_null()).height` < 5% del total.
- Las filas con match: `ubigeo_distrito` es 6 digitos string.
- Documenta en docstring de donde viene la tabla y cuando se descargo.

**Estimacion:** 2-3 horas (el grueso es bajar y limpiar la tabla INEI).

---

## 3. Feature engineering SENAMHI climatico (alto impacto si llega data del dia D)

**Por que importa:** el dataset del dia D probablemente sera SENAMHI diario
(Año/Mes/Dia/Precip/Tmax/Tmin). Tener listas las features antes ahorra
tiempo en la competencia.

**Archivo:** `src/modeling/features.py` (nuevas funciones, no toques las
existentes).

**Firma sugerida:**

```python
def add_climate_lags(
    df: pl.DataFrame,
    target_cols: tuple[str, ...] = ("precip_acum", "tmax", "tmin"),
    lags: tuple[int, ...] = (1, 3, 7, 14, 30),
) -> pl.DataFrame:
    """Agrega df con cols `{col}_lag{n}` para cada lag y target."""

def add_climate_rollings(
    df: pl.DataFrame,
    target_cols: tuple[str, ...] = ("precip_acum", "tmax", "tmin"),
    windows: tuple[int, ...] = (3, 7, 14, 30),
) -> pl.DataFrame:
    """Agrega rollings: mean, std, max, min en ventanas pasadas (sin leakage)."""

def add_climate_interactions(df: pl.DataFrame) -> pl.DataFrame:
    """Agrega rango_termico = tmax - tmin, indice_calor = tmax * humedad
    (si humedad existe), dias_consecutivos_sin_lluvia, etc.
    """
```

**Acceptance criteria:**
- Cada funcion preserva `df.height` (no filtra filas).
- Las nuevas cols al inicio del periodo tienen nulls esperables (no datos suficientes para el lag/rolling).
- `add_climate_lags` no debe leakear: el lag de hoy es el valor de hace N dias, no de hoy.
- Smoke test: corre las 3 funciones encadenadas sobre un CSV SENAMHI dummy y verifica shapes + algunos valores especificos.

**Estimacion:** 2-3 horas.

---

## 4. Deduplicacion de interrupciones (medio impacto)

**Por que importa:** algunos eventos parecen duplicados (mismo EP, mismo
motivo, mismo dia, ts_inicio dentro de 1 hora). Si no se deduplican, los
KPIs del Home estan inflados.

**Archivo:** `src/modeling/features.py` (nueva funcion) o `src/eda.py`.

**Firma sugerida:**

```python
def deduplicate_eventos(
    df: pl.DataFrame,
    keys: tuple[str, ...] = ("EP", "Distrito", "Motivodelainterrupcion"),
    ts_col: str = "ts_inicio",
    tolerance_hours: float = 1.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Detecta duplicados blandos: misma llave + ts_inicio dentro de
    tolerance_hours. Retorna (df_deduplicado, df_duplicados_descartados).
    """
```

**Acceptance criteria:**
- `df_deduplicado.height + df_duplicados_descartados.height == df_in.height`.
- En cada grupo de duplicados, conservar el evento con `duracion_horas` mayor (asumir que ese fue el reporte final).
- Loggear cuantos descarto (`logger.info`).

**Estimacion:** 1.5-2 horas.

---

## 5. Imputacion inteligente de N conexiones / Unidades (medio-bajo impacto)

**Por que importa:** `Ndeconexionesdomiciliariasa` y `Unidadesdeusoafectadas`
vienen con strings vacios casteados a null. Cuando un evento no reporta
conexiones pero el motivo es "Reparacion de matriz" → podemos imputar la
mediana del motivo+EP.

**Archivo:** `src/modeling/features.py` (nueva funcion).

**Firma sugerida:**

```python
def impute_n_afectadas(
    df: pl.DataFrame,
    target_col: str = "n_afectadas",
    group_cols: tuple[str, ...] = ("EP", "Motivodelainterrupcion"),
    fallback: float = 0.0,
) -> pl.DataFrame:
    """Imputa nulls en `target_col` con la mediana del grupo. Si el grupo
    completo es null, cae a `fallback`. Agrega col `n_afectadas_imputed: bool`.
    """
```

**Acceptance criteria:**
- `df.filter(pl.col(target_col).is_null()).height == 0` en el output.
- Bandera `n_afectadas_imputed` indica cuales fueron imputadas.
- Loggear pct de filas imputadas.

**Estimacion:** 1.5 horas.

---

## 6. Validacion de orden temporal (bajo impacto, alta higiene)

**Por que importa:** algunas filas tienen `ts_fin < ts_inicio` (errores de
captura). Hoy `add_duracion_impacto` las nullea silenciosamente. Mejor
contarlas y exponerlas.

**Archivo:** `src/eda.py` (extiende `DataFrameProfile`).

**Firma sugerida:**

```python
def validate_temporal_order(
    df: pl.DataFrame,
    start_col: str = "ts_inicio",
    end_col: str = "ts_fin",
) -> dict[str, int]:
    """Devuelve dict con conteos:
        - n_total
        - n_null_inicio
        - n_null_fin
        - n_invertido (fin < inicio)
        - n_simultaneo (fin == inicio)
        - n_validos
    """
```

**Acceptance criteria:**
- Suma de los conteos cuadra con `n_total`.
- Si hay > 1% de invertidos, log.warning.
- Llamado desde la pagina Datos como bloque nuevo "Calidad temporal".

**Estimacion:** 1 hora.

---

## Workflow recomendado

```bash
git checkout -b fabian/<slug-de-la-tarea>
# implementa la funcion + smoke test corto
uv run python -c "from src.modeling.features import join_eventos_calidad; print(join_eventos_calidad.__doc__)"
uv run ruff check src/
git add -A
git commit -m "feat(preprocessing): <slug> + smoke"
git push origin fabian/<slug>
# abre PR contra main; mencionarme en la descripcion
```

Si la tarea agrega una funcion publica, **agregala al `__all__`** del modulo
correspondiente para que Streamlit la importe sin sorpresas.

## Preguntas frecuentes

**¿Puedo usar pandas en vez de polars?** Solo si polars no soporta la
operacion. Convierte a pandas con `df.to_pandas()` al final, no al principio.

**¿Que hago si `df_estaciones` no tiene UBIGEO?** Ya pasa: el `.xlsx` MOREA
trae nombres de estacion pero no codigos. La tarea 2 resuelve esto.

**¿Donde corro mi smoke local?** Cualquier funcion nueva: una sola linea de
`uv run python -c "from ... import ...; print(...)"`. Si quieres algo mas
estructurado, agrega un `tests/test_<modulo>.py` con pytest.
