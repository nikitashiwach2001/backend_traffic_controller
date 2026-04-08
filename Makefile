PY ?= python
VENV ?= venv
BIN := $(VENV)/bin

.PHONY: help venv install test lint format run docker clean

help:
	@echo "Targets:"
	@echo "  venv     - create local virtualenv at ./$(VENV)"
	@echo "  install  - install runtime + dev dependencies"
	@echo "  test     - run pytest suite"
	@echo "  lint     - run ruff (lint only, no fixes)"
	@echo "  format   - run ruff format"
	@echo "  run      - start the FastAPI environment on :7860"
	@echo "  docker   - build the docker image"
	@echo "  inspect  - dump current env state as JSON"
	@echo "  clean    - remove caches"

venv:
	$(PY) -m venv $(VENV)

install:
	$(BIN)/pip install -r requirements.txt
	$(BIN)/pip install pytest ruff

test:
	$(BIN)/python -m pytest tests/ -q

lint:
	$(BIN)/ruff check .

format:
	$(BIN)/ruff format .

run:
	$(BIN)/uvicorn environment:app --host 0.0.0.0 --port 7860 --reload

docker:
	docker build -t adaptive-traffic-controller .

inspect:
	$(BIN)/python inspect_env.py

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__ .ruff_cache
