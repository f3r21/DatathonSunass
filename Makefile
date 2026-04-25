.PHONY: help install sync app pipeline lint format docker-build docker-up docker-down docker-logs render-deck clean

PY := uv run python
STREAMLIT := uv run python -m streamlit

help:
	@echo "Targets del proyecto Datathon SUNASS 2026:"
	@echo "  make install      - uv sync (resuelve dependencias)"
	@echo "  make app          - corre la app Streamlit local (app/Home.py)"
	@echo "  make pipeline     - corre el pipeline end-to-end (scripts/run_pipeline.py)"
	@echo "  make docker-build - construye la imagen docker"
	@echo "  make docker-up    - levanta docker compose"
	@echo "  make docker-down  - detiene los contenedores"
	@echo "  make docker-logs  - tail de logs del servicio"
	@echo "  make render-deck  - renderiza deck Quarto"
	@echo "  make lint         - ruff check src/ app/ scripts/"
	@echo "  make format       - ruff format src/ app/ scripts/"
	@echo "  make clean        - limpia caches y artefactos temporales"

install sync:
	uv sync

app:
	$(STREAMLIT) run app/Home.py

pipeline:
	$(PY) scripts/run_pipeline.py

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-up:
	docker compose -f docker/docker-compose.yml up -d
	@echo "App en http://localhost:8501"

docker-down:
	docker compose -f docker/docker-compose.yml down

docker-logs:
	docker compose -f docker/docker-compose.yml logs -f app

render-deck:
	uv run quarto render reports/deck.qmd --to revealjs

lint:
	uv run ruff check src/ app/ scripts/

format:
	uv run ruff format src/ app/ scripts/

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache artifacts/
	rm -rf reports/.quarto reports/_freeze reports/_site _freeze
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
