# Arquitectura — Datathon SUNASS 2026

Este documento describe cómo se acoplan los tres subsistemas de la entrega: análisis, dashboard y chatbot corporativo con RAG.

## Diagrama de alto nivel

```
                              +---------------------------+
                              |   Datasets oficiales      |
                              |   - interrupciones .dta   |
                              |   - morea .parquet        |
                              |   - estaciones .xlsx      |
                              +-------------+-------------+
                                            |
                                            v
+----------------------------+    +---------+----------+    +---------------------+
|   docs SUNASS (PDFs)       |    |   src/io.py        |    |   config/users.yaml |
|   + resultados del análisis|    |   src/eda.py       |    |   bcrypt + roles    |
|   + tablas exportadas      |    |   src/features.py  |    +----------+----------+
+-------------+--------------+    |   src/models.py    |               |
              |                   +---------+----------+               |
              v                             |                          |
+-------------+--------------+               v                          |
|   src/rag/build_index.py   |    +---------+----------+               |
|   - PyMuPDF / python-docx  |    |   notebooks/*.ipynb|               |
|   - sentence-transformers  |    |   reports/tablas/  |               |
|   - Chroma persistente     |    |   reports/figuras/ |               |
+-------------+--------------+    +---------+----------+               |
              |                             |                          |
              v                             v                          v
+-------------+-------------------------------------------------------+---+
|                        dashboard/ (Streamlit multi-page)               |
|  Home.py | 1_KPIs.py | 2_Mapa.py | 3_Alertas.py | 4_Chat.py            |
|          |                                                             |
|          +-- streamlit-authenticator  (role-gated: admin/analista/vis)  |
+---------------------------------+--------------------------------------+
                                  |
                                  v
                    +-------------+-------------+
                    |    src/llm.py LLMClient   |
                    |    dual backend           |
                    +------+-------------+------+
                           |             |
                 primario  v             v  fallback
            +--------------+--+    +-----+------------+
            |  Ollama 3060    |    |   OpenAI API     |
            |  remoto:11434   |    |   gpt-4o-mini    |
            |  llama3.1:8b-Q4 |    |   (requiere red) |
            +-----------------+    +------------------+
```

## Decisiones clave

### LLM dual (opensource + fallback)

El reto exige un chatbot opensource. Cumplimos con **Ollama + Llama 3.1 8B Q4** corriendo en la desktop con RTX 3060 (12 GB VRAM, 32 GB RAM), accesible remotamente. OpenAI queda como fallback de red por si el túnel a la 3060 se cae durante la exposición presencial en UCSP Arequipa.

`LLMClient` (`src/llm.py`) se comporta según `LLM_BACKEND`:

- `ollama`: solo servidor local. Si `OLLAMA_BASE_URL` no responde en <2 s, error duro.
- `openai`: solo API remota. Falla si `OPENAI_API_KEY` ausente.
- `auto` (por defecto en demo): probe a `/api/tags`; si responde, usar Ollama; si no, caer a OpenAI con log visible.

### RAG con citas verificables

- Corpus: PDFs oficiales SUNASS (`docs/`), transcripciones de mentoría (`clases/`), tablas exportadas del análisis (`reports/tablas/*.xlsx`), y hallazgos en Markdown (`reports/findings/*.md`).
- Chunking: 512 tokens con overlap 64. Documentos largos (PDFs de reglamento) se pre-segmentan por capítulo cuando la estructura lo permite.
- Embeddings: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (ES-first, 384-dim, corre en CPU).
- Vector store: Chroma persistente en `chroma_db/`. Se reconstruye con `make index` y tarda <2 min en la 3060.
- Recuperación: top-5 con re-rank por similaridad coseno. Cada respuesta del chat cita fuente + offset.

### Autenticación por rol

`streamlit-authenticator` con hashes bcrypt en `config/users.yaml`. Tres roles:

- `admin`: todas las páginas + export de findings + panel de auditoría.
- `analista`: KPIs, mapa, alertas, chat completo.
- `visitante`: solo KPIs agregados (panel de demo para el jurado sin credenciales).

`config/users.yaml` **no se comitea**. Se genera con `uv run python -m src.auth_setup` y el seed queda en `.env` (`AUTH_SEED`).

### Reproducibilidad

- `uv` con `pyproject.toml` + `uv.lock` en git → `uv sync` recrea el entorno.
- Datos grandes fuera del repo, referenciados por `.env`.
- Notebooks ejecutables vía `make eda`; entrenamiento vía `make train`; índice RAG vía `make index`.
- Ninguna ruta absoluta. Todo relativo a la raíz del proyecto o a paths de `.env`.

## Topología en el día del concurso

Máquinas disponibles el sábado 25/04/2026 en UCSP:

| Máquina              | Rol                                                        |
|----------------------|------------------------------------------------------------|
| MacBook Air M2 24 GB | Dashboard + notebooks (entorno de exposición)              |
| Laptop RTX 5060      | Entrenamiento pesado de XGBoost/LightGBM + build del índice |
| Desktop RTX 3060 (remota, 32 GB RAM) | Servidor Ollama via SSH/Tailscale            |
| GCP (Tier 1+)        | Backup cómputo si la 5060 falla                            |

En modo degradado (sin red estable a la 3060): M2 levanta Ollama con Llama 3.2 3B Q4 y el chatbot sigue funcional, con LLMClient anunciándolo al jurado.

## Flujo de datos en una query del chat

1. Usuario envía pregunta en `dashboard/pages/4_Chat.py`.
2. `retrieve.py` embeba la pregunta, consulta Chroma con top-5.
3. Se construye prompt con contexto + instrucciones de citar.
4. `LLMClient.chat(...)` enruta a Ollama o OpenAI según configuración.
5. Respuesta se muestra con streaming; las citas se renderizan como expanders con fragmento + nombre de archivo + offset.
