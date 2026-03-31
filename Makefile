.PHONY: setup run test clean help

PYTHON = python3
PIP = $(PYTHON) -m pip
PYTEST = $(PYTHON) -m pytest

help:
	@echo "Available commands:"
	@echo "  make setup      Initialize environment and install dependencies"
	@echo "  make run        Execute sample 3D reconstruction via the canonical CLI"
	@echo "  make test       Run the pytest suite"
	@echo "  make clean      Remove build artifacts and cache files"

setup:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,runtime]"

run:
	MPLCONFIGDIR=/tmp/mplcache healingstone-run --data-dir data/sample/3d --output-dir artifacts --min-required-accuracy 0 --allow-overwrite-run

test:
	MPLCONFIGDIR=/tmp/mplcache $(PYTEST) -q

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf *.egg-info
	rm -rf dist
	rm -rf build
