# =============================================================================
# Threat Intelligence Agent - Makefile
# =============================================================================

.PHONY: help install install-dev test test-unit test-integration lint format \
        type-check clean build run dev docker-build docker-up docker-down \
        docker-logs train model-info api-docs

# Default target
.DEFAULT_GOAL := help

# Colors for terminal output
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RESET := \033[0m

# =============================================================================
# Help
# =============================================================================
help:
	@echo "$(CYAN)Threat Intelligence Agent - Development Commands$(RESET)"
	@echo ""
	@echo "$(GREEN)Installation:$(RESET)"
	@echo "  make install        Install production dependencies"
	@echo "  make install-dev    Install development dependencies"
	@echo ""
	@echo "$(GREEN)Testing:$(RESET)"
	@echo "  make test           Run all tests"
	@echo "  make test-unit      Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make test-cov       Run tests with coverage report"
	@echo ""
	@echo "$(GREEN)Code Quality:$(RESET)"
	@echo "  make lint           Run linting (ruff)"
	@echo "  make format         Format code (black, ruff)"
	@echo "  make type-check     Run type checking (mypy)"
	@echo "  make check          Run all code quality checks"
	@echo ""
	@echo "$(GREEN)Running:$(RESET)"
	@echo "  make run            Run the API server (production)"
	@echo "  make dev            Run the API server (development with reload)"
	@echo "  make cli            Run the CLI interface"
	@echo ""
	@echo "$(GREEN)Docker:$(RESET)"
	@echo "  make docker-build   Build Docker images"
	@echo "  make docker-up      Start all services"
	@echo "  make docker-down    Stop all services"
	@echo "  make docker-logs    View logs"
	@echo ""
	@echo "$(GREEN)Model:$(RESET)"
	@echo "  make train          Train the SLM model"
	@echo "  make model-info     Display model information"
	@echo ""
	@echo "$(GREEN)Documentation:$(RESET)"
	@echo "  make api-docs       Generate API documentation"
	@echo ""
	@echo "$(GREEN)Cleanup:$(RESET)"
	@echo "  make clean          Clean build artifacts"
	@echo "  make clean-all      Clean everything including cache"

# =============================================================================
# Installation
# =============================================================================
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install || true

# =============================================================================
# Testing
# =============================================================================
test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v --tb=short

test-integration:
	pytest tests/integration/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-fast:
	pytest tests/unit/ -v --tb=short -x -q

# =============================================================================
# Code Quality
# =============================================================================
lint:
	ruff check src/ tests/

lint-fix:
	ruff check src/ tests/ --fix

format:
	black src/ tests/
	ruff check src/ tests/ --fix

format-check:
	black src/ tests/ --check
	ruff check src/ tests/

type-check:
	mypy src/ --ignore-missing-imports

check: format-check lint type-check
	@echo "$(GREEN)All checks passed!$(RESET)"

# =============================================================================
# Running
# =============================================================================
run:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

dev:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

cli:
	python -m src.cli

# =============================================================================
# Docker
# =============================================================================
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-restart: docker-down docker-up

docker-clean:
	docker-compose down -v --rmi local

# =============================================================================
# Model Operations
# =============================================================================
train:
	python -m src.slm.train

model-info:
	python -c "from src.slm.model import ThreatIntelSLM, TISLMConfig; \
		model = ThreatIntelSLM(TISLMConfig.small()); \
		print(f'Parameters: {model.num_parameters():,}')"

# =============================================================================
# Documentation
# =============================================================================
api-docs:
	@echo "API documentation available at http://localhost:8000/docs"
	@echo "ReDoc documentation available at http://localhost:8000/redoc"

# =============================================================================
# Cleanup
# =============================================================================
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage 2>/dev/null || true

clean-all: clean
	rm -rf .venv/ node_modules/ 2>/dev/null || true

# =============================================================================
# Development Utilities
# =============================================================================
shell:
	python -c "from src.agent.core import *; from src.memory.memory_system import *; import asyncio"

# Health check
health:
	curl -s http://localhost:8000/health | python -m json.tool

# Quick test of enrichment endpoint
test-enrich:
	curl -s -X POST http://localhost:8000/api/v1/enrich \
		-H "Content-Type: application/json" \
		-d '{"ioc_type": "ipv4", "ioc_value": "8.8.8.8"}' | python -m json.tool

# =============================================================================
# CI/CD Helpers
# =============================================================================
ci-test:
	pytest tests/ -v --tb=short --junitxml=test-results.xml

ci-lint:
	ruff check src/ tests/ --output-format=github

ci-security:
	pip-audit || true
	bandit -r src/ -ll || true
