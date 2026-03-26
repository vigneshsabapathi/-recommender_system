.PHONY: setup data train evaluate serve frontend seed-db test pipeline clean all help

# --------------------------------------------------------------------------
# Variables
# --------------------------------------------------------------------------
PYTHON   ?= python
PIP      ?= pip
UVICORN  ?= uvicorn
NPM      ?= npm
DVC      ?= dvc

BACKEND_DIR  = backend
FRONTEND_DIR = frontend
DATA_RAW     = data/raw
DATA_PROC    = data/processed
MODELS_DIR   = models

# --------------------------------------------------------------------------
# Targets
# --------------------------------------------------------------------------

setup: ## Install all dependencies (Python + Node)
	$(PIP) install -r requirements.txt
	$(PIP) install -r $(BACKEND_DIR)/requirements.txt
	cd $(FRONTEND_DIR) && $(NPM) install
	@echo "Setup complete."

data: ## Download and process MovieLens 20M
	$(PYTHON) -m src.data.ingest
	$(PYTHON) -m src.data.preprocess
	$(PYTHON) -m src.data.feature_engineering
	@echo "Data pipeline complete."

train: ## Train all models (collaborative, content-based, ALS)
	$(PYTHON) -m src.pipeline.train collaborative
	$(PYTHON) -m src.pipeline.train content_based
	$(PYTHON) -m src.pipeline.train als
	@echo "All models trained."

evaluate: ## Evaluate and compare all models
	$(PYTHON) -m src.pipeline.evaluate
	@echo "Evaluation complete. See $(MODELS_DIR)/evaluation_summary.json"

serve: ## Start FastAPI backend (port 8000)
	$(UVICORN) backend.app.main:app --reload --port 8000

frontend: ## Start Next.js frontend dev server (port 3000)
	cd $(FRONTEND_DIR) && $(NPM) run dev

seed-db: ## Seed SQLite database from processed data
	$(PYTHON) -m backend.app.db.seed
	@echo "Database seeded."

test: ## Run all tests
	$(PYTHON) -m pytest tests/ -v --tb=short
	@echo "Tests complete."

pipeline: ## Run full DVC pipeline (reproducible)
	$(DVC) repro
	@echo "DVC pipeline complete."

clean: ## Clean generated files (data, models, caches)
	rm -rf $(DATA_RAW)/*.csv
	rm -rf $(DATA_PROC)/*.csv $(DATA_PROC)/*.npz $(DATA_PROC)/*.json
	rm -rf $(MODELS_DIR)/collaborative $(MODELS_DIR)/content_based $(MODELS_DIR)/als $(MODELS_DIR)/hybrid
	rm -f  $(MODELS_DIR)/evaluation_summary.json
	rm -f  $(BACKEND_DIR)/movies.db
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete."

all: setup data train evaluate seed-db ## Run complete pipeline end-to-end
	@echo ""
	@echo "============================================"
	@echo "  Full pipeline complete!"
	@echo "  - Models: $(MODELS_DIR)/"
	@echo "  - Metrics: $(MODELS_DIR)/evaluation_summary.json"
	@echo "  - Backend: make serve"
	@echo "  - Frontend: make frontend"
	@echo "============================================"

help: ## Show this help
	@echo "Movie Recommender System - Available Commands"
	@echo "=============================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
