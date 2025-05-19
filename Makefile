.PHONY: install_library clean lint test test_all_modules update_deps help

# Default target
help:
	@echo "Available targets:"
	@echo "  install_library  - Install dependencies and package in development mode"
	@echo "  update_deps      - Update dependencies in requirements.txt"
	@echo "  clean            - Remove build artifacts and cache files"
	@echo "  lint             - Run linting checks"
	@echo "  test             - Run unit tests"
	@echo "  test_all_modules - Test all modules in the project"

install_library:
	@echo "Installing dependencies from requirements.txt using uv..."
	uv pip install -r requirements.txt
	@echo "Installing the package in development mode..."
	uv pip install -e .

update_deps:
	@echo "Updating dependencies in requirements.txt..."
	uv pip compile requirements.txt --upgrade

clean:
	@echo "Cleaning up build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	@echo "Running linting checks..."
	uv pip install ruff
	ruff check .

test:
	@echo "Running tests..."
	uv pip install pytest
	pytest

test_all_modules:
	@echo "Testing all modules..."
	python test_all_modules.py
