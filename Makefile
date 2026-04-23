.PHONY: help sync lint format eda train dashboard index chat clean

PY := uv run python
STREAMLIT := uv run streamlit

help:
	@echo "Targets disponibles:"
	@echo "  make sync       - uv sync (resuelve dependencias)"
	@echo "  make lint       - ruff check src/ dashboard/"
	@echo "  make format     - ruff format src/ dashboard/"
	@echo "  make eda        - corre notebook de EDA"
	@echo "  make train      - entrena modelos baseline y reporta metricas"
	@echo "  make index      - construye indice RAG desde docs_sunass/"
	@echo "  make dashboard  - lanza Streamlit en \$$STREAMLIT_SERVER_PORT"
	@echo "  make clean      - elimina chroma_db/ y reports/figuras/ generados"

sync:
	uv sync

lint:
	uv run ruff check src/ dashboard/

format:
	uv run ruff format src/ dashboard/

eda:
	$(PY) -m jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb

train:
	$(PY) -m src.train

index:
	$(PY) -m src.rag.build_index

dashboard:
	$(STREAMLIT) run dashboard/Home.py

clean:
	rm -rf chroma_db/
	find reports/figuras -type f ! -name '.gitkeep' -delete
