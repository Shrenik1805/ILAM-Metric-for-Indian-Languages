PYTHON ?= python3

.PHONY: install-dev install-transfer lint format typecheck test audit demo clean

install-dev:
	$(PYTHON) -m pip install -e ".[dev]" --no-build-isolation

install-transfer:
	$(PYTHON) -m pip install -e ".[transfer]" --no-build-isolation

lint:
	ruff check .
	black --check .

format:
	black .
	ruff check . --fix

typecheck:
	mypy ilam transfer experiments run_all.py

test:
	pytest

audit:
	pip-audit

demo:
	$(PYTHON) run_all.py --demo

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -f results/* data/translations/*
