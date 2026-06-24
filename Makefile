# ============================================================
# Interpretable ML for Credit Risk Assessment — Makefile
# ============================================================

.PHONY: help install install-dev setup-hooks lint format test test-cov \
        train evaluate explain monitor run-app run-api docker-build \
        docker-up docker-down mlflow-ui dvc-init clean docs

PYTHON := python3
PIP    := pip3
APP    := app/main.py
API    := src/api/main.py

# ── Colors ────────────────────────────────────────────────
CYAN  := \033[36m
GREEN := \033[32m
RESET := \033[0m

help: ## Show this help message
	@echo ""
	@echo "$(CYAN)╔══════════════════════════════════════════════════╗$(RESET)"
	@echo "$(CYAN)║   Credit Risk XAI — Development Commands         ║$(RESET)"
	@echo "$(CYAN)╚══════════════════════════════════════════════════╝$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-22s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ── Environment Setup ─────────────────────────────────────
install: ## Install production dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PYTHON) -m spacy download en_core_web_sm 2>/dev/null || true
	@echo "$(GREEN)✓ Dependencies installed$(RESET)"

install-dev: install ## Install dev dependencies + pre-commit
	$(PIP) install -r requirements-dev.txt
	$(MAKE) setup-hooks

setup-hooks: ## Configure pre-commit hooks
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "$(GREEN)✓ Pre-commit hooks installed$(RESET)"

setup-dirs: ## Create data/model directories
	mkdir -p data/{raw,processed,external,interim}
	mkdir -p models/{trained,calibrated,registry}
	mkdir -p reports/generated
	mkdir -p mlruns
	@echo "$(GREEN)✓ Directories created$(RESET)"

# ── Data Pipeline ─────────────────────────────────────────
download-data: ## Download datasets via DVC
	dvc pull
	@echo "$(GREEN)✓ Data downloaded$(RESET)"

preprocess: ## Run data preprocessing pipeline
	$(PYTHON) -m src.data_pipeline.preprocess --config configs/data_config.yaml
	@echo "$(GREEN)✓ Preprocessing complete$(RESET)"

feature-engineering: ## Run feature engineering
	$(PYTHON) -m src.data_pipeline.feature_engineering --config configs/data_config.yaml
	@echo "$(GREEN)✓ Feature engineering complete$(RESET)"

# ── Model Training ────────────────────────────────────────
train: ## Train all models
	$(PYTHON) -m src.models.train_pipeline --config configs/model_config.yaml
	@echo "$(GREEN)✓ All models trained$(RESET)"

train-fast: ## Train baseline models only
	$(PYTHON) -m src.models.train_pipeline --config configs/model_config.yaml --fast
	@echo "$(GREEN)✓ Baseline models trained$(RESET)"

evaluate: ## Evaluate all trained models
	$(PYTHON) -m src.models.evaluate --config configs/model_config.yaml
	@echo "$(GREEN)✓ Evaluation complete$(RESET)"

calibrate: ## Calibrate model probabilities
	$(PYTHON) -m src.models.calibration --config configs/model_config.yaml
	@echo "$(GREEN)✓ Calibration complete$(RESET)"

# ── Explainability ────────────────────────────────────────
explain: ## Generate SHAP + LIME explanations
	$(PYTHON) -m src.explainability.generate_explanations
	@echo "$(GREEN)✓ Explanations generated$(RESET)"

fairness: ## Run fairness analysis
	$(PYTHON) -m src.fairness.fairness_analysis --config configs/fairness_config.yaml
	@echo "$(GREEN)✓ Fairness analysis complete$(RESET)"

# ── Code Quality ──────────────────────────────────────────
lint: ## Run all linters
	flake8 src/ app/ tests/ --max-line-length 88 --extend-ignore E203
	mypy src/ --ignore-missing-imports
	@echo "$(GREEN)✓ Linting passed$(RESET)"

format: ## Auto-format with black + isort
	black src/ app/ tests/
	isort src/ app/ tests/
	@echo "$(GREEN)✓ Formatting complete$(RESET)"

format-check: ## Check formatting without changes
	black --check src/ app/ tests/
	isort --check-only src/ app/ tests/

# ── Testing ───────────────────────────────────────────────
test: ## Run all tests
	pytest tests/ -v --tb=short
	@echo "$(GREEN)✓ Tests passed$(RESET)"

test-unit: ## Run unit tests only
	pytest tests/unit/ -v --tb=short

test-integration: ## Run integration tests
	pytest tests/integration/ -v --tb=short

test-cov: ## Tests with HTML coverage report
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)✓ Coverage report: htmlcov/index.html$(RESET)"

# ── Applications ──────────────────────────────────────────
run-app: ## Launch Streamlit dashboard
	streamlit run $(APP) --server.port 8501 --server.address 0.0.0.0

run-api: ## Launch FastAPI backend
	uvicorn src.api.main:app --reload --port 8000 --host 0.0.0.0

mlflow-ui: ## Open MLflow UI
	mlflow ui --port 5000

# ── Docker ────────────────────────────────────────────────
docker-build: ## Build Docker images
	docker-compose build

docker-up: ## Start all containers
	docker-compose up -d
	@echo "$(GREEN)✓ Services started:$(RESET)"
	@echo "  Streamlit:  http://localhost:8501"
	@echo "  FastAPI:    http://localhost:8000"
	@echo "  MLflow:     http://localhost:5000"

docker-down: ## Stop all containers
	docker-compose down

docker-logs: ## Follow container logs
	docker-compose logs -f

# ── MLOps ─────────────────────────────────────────────────
dvc-init: ## Initialize DVC
	dvc init
	dvc remote add -d myremote s3://your-bucket/dvc-store
	@echo "$(GREEN)✓ DVC initialized$(RESET)"

dvc-push: ## Push data to DVC remote
	dvc push

mlflow-server: ## Start MLflow tracking server
	mlflow server \
		--backend-store-uri sqlite:///mlruns/mlflow.db \
		--default-artifact-root ./mlruns/artifacts \
		--port 5000

# ── Reports & Monitoring ──────────────────────────────────
generate-report: ## Generate full system report
	$(PYTHON) -m src.reporting.report_generator
	@echo "$(GREEN)✓ Report saved to reports/generated/$(RESET)"

monitor: ## Run monitoring checks
	$(PYTHON) -m src.monitoring.drift_detector
	@echo "$(GREEN)✓ Monitoring complete$(RESET)"

# ── Documentation ─────────────────────────────────────────
docs: ## Build documentation
	mkdocs build
	@echo "$(GREEN)✓ Docs built: site/$(RESET)"

docs-serve: ## Serve documentation locally
	mkdocs serve

# ── Full Pipeline ─────────────────────────────────────────
pipeline: preprocess feature-engineering train evaluate calibrate explain fairness ## Run complete ML pipeline

ci: format-check lint test ## Run full CI checks

# ── Cleanup ───────────────────────────────────────────────
clean: ## Remove generated artifacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage
	@echo "$(GREEN)✓ Clean complete$(RESET)"

clean-models: ## Remove trained model files
	rm -rf models/trained/*.pkl models/trained/*.json
	@echo "$(GREEN)✓ Models removed$(RESET)"

nuke: clean clean-models ## Remove everything generated
	rm -rf mlruns/ .dvc/cache/
	@echo "$(GREEN)✓ Full reset complete$(RESET)"
