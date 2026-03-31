# Engineering-Grade Makefile for healingstone

.PHONY: setup run test clean help

# Environment configuration
PYTHON = python3
PIP = $(PYTHON) -m pip
PYTEST = pytest
SRC_DIR = src/healingstone

help:
	@echo "Available commands:"
	@echo "  make setup      Initialize environment and install dependencies"
	@echo "  make run        Execute full reconstruction pipeline with default config"
	@echo "  make test       Run all unit and integration tests"
	@echo "  make clean      Remove build artifacts and cache files"

setup:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev,runtime]"

run:
	$(PYTHON) -m healingstone.cli run --config configs/pipeline.yaml

test:
	$(PYTEST) tests/unit tests/integration

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf *.egg-info
	rm -rf dist
	rm -rf build
