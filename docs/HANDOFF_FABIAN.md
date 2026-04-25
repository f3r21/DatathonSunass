# Mensaje de handoff para Fabian

> Texto listo para enviar por WhatsApp / Telegram / email. Pega el bloque
> de abajo tal cual.

---

```
Fabian, listo para que pruebes en tu Windows.

Repo:  https://github.com/f3r21/DatathonSunass
Datos: https://drive.google.com/drive/folders/1Zd_7nIvgZE0P1SIoSkayzF5mU3SmksCn?usp=sharing

Pasos (15 min):
1. Lee primero docs/SETUP_FABIAN.md — tiene todos los comandos PowerShell.
2. git clone del repo y entra a la carpeta.
3. uv sync (instala el entorno, ~3 min).
4. Descarga la carpeta datos/ del Drive (boton "Descargar" arriba a la
   derecha) y descomprimela al lado del repo:
     DatathonSunass/repo   <- lo del git clone
     DatathonSunass/datos  <- lo del Drive
5. Copy-Item .env.example .env (no necesitas editarlo si seguiste el paso 4).
6. uv run python scripts/smoke_setup.py — debe terminar con [OK].
7. uv run streamlit run app/Home.py
8. Login: fabio / ssa11

Si algo truena: copia el traceback completo y mandamelo.

Tu foco: preprocesamiento. Lee docs/PREPROCESSING_TASKS.md, tiene 6 tareas
priorizadas con firma + acceptance + estimacion. Las top 3:
  1. Cross-dataset join MOREA x Interrupciones (3-4h, alto impacto)
  2. Normalizacion espacial UBIGEO (2-3h, habilita la 1)
  3. Feature engineering SENAMHI climatico (2-3h, listo para dia D)

Codigo de equipo SUNASS: SSA11 (Categoria I Operacional).
Competencia: sabado 25-abr UCSP.
```

---

## Lo que ya esta entregado

- **Repo limpio**, sin dependencias muertas (pyproject.toml slim).
- **App Streamlit** funcional en 11 vistas con login wall y logout.
- **Pipeline reproducible** (`scripts/run_pipeline.py`) corre raw → modelo
  → artefactos sin UI.
- **Smoke test** (`scripts/smoke_setup.py`) valida entorno en 30s.
- **Atajos Windows**: `run_app.ps1`, `run_app.bat` para reemplazar `make`.
- **`.gitattributes`** con `eol=lf` para evitar guerras CRLF/LF.
- **Credenciales bcrypt** para `fabio / ssa11` (admin).

## Lo que falta del lado de Fabian (su scope)

Ver `docs/PREPROCESSING_TASKS.md`. Hay 6 tareas, idealmente toma 1-2.

## Lo que falta del lado de fer (mi scope)

- Grabar video demo 2-3 min y embebir en README (#27)
- Ajustar deck Quarto con screenshots de la app (#28)
- Bajar dataset SENAMHI de prueba para dry-run de Modo Dia D (#31)
- Validacion de replicabilidad en entorno limpio (#13)
- Confirmar canal y formato de entrega formal con SUNASS (#12)

## Si Fabian no tiene tiempo / no llega

La app ya cubre los 7 resultados oficiales sin su contribucion. Lo de el
es mejora marginal (cross-features, deduplicacion, imputacion). El demo
funciona end-to-end con lo que hay hoy.
