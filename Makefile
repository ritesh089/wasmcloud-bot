.PHONY: help install setup-db start-db stop-db init-db ingest run test clean setup start stop dev-env venv-check

SHELL := /usr/bin/env zsh

# Virtual environment settings
VENV_DIR := venv
VENV_ACTIVATE := $(VENV_DIR)/bin/activate
PYTHON := $(VENV_DIR)/bin/python3
PIP := $(VENV_DIR)/bin/pip3

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

venv-check: ## Check if virtual environment exists
	@if [ ! -f $(VENV_ACTIVATE) ]; then \
		echo "❌ Virtual environment not found at $(VENV_DIR)"; \
		echo "Please run: ./scripts/setup.zsh"; \
		exit 1; \
	fi
	@echo "✅ Virtual environment found at $(VENV_DIR)"

install: venv-check ## Install Python dependencies in virtual environment
	@echo "Installing dependencies in virtual environment..."
	@source $(VENV_ACTIVATE) && $(PIP) install -r requirements.txt

setup-env: ## Copy environment template
	cp config.env.example .env
	@echo "Please edit .env with your configuration"

start-db: ## Start PostgreSQL with Docker Compose
	docker-compose up -d postgres
	@echo "Waiting for database to be ready..."
	@sleep 5

stop-db: ## Stop PostgreSQL
	docker-compose down

init-db: venv-check ## Initialize database schema
	@source $(VENV_ACTIVATE) && $(PYTHON) scripts/init_db.py

ingest: venv-check ## Ingest wasmCloud documentation
	@source $(VENV_ACTIVATE) && $(PYTHON) scripts/ingest_docs.py

run: venv-check ## Start the RAG bot server
	@source $(VENV_ACTIVATE) && $(PYTHON) -m server.main

test: venv-check ## Run the test client
	@source $(VENV_ACTIVATE) && $(PYTHON) test_client.py

test-mcp: venv-check ## Test the MCP client
	@source $(VENV_ACTIVATE) && $(PYTHON) mcp_client.py

run-mcp: venv-check ## Start the MCP server
	@source $(VENV_ACTIVATE) && $(PYTHON) mcp_server.py

setup: ## Complete setup using zsh script
	@./scripts/setup.zsh

start: ## Start all services using zsh script
	@./scripts/start.zsh

start-mcp: ## Start all services including MCP server
	@./scripts/start.zsh --with-mcp

stop: ## Stop all services using zsh script
	@./scripts/stop.zsh

stop-clean: ## Stop all services and clean up logs/data
	@./scripts/stop.zsh --all

dev-env: ## Start interactive development environment
	@./scripts/dev.zsh

dev-full: ## Start development environment with all features
	@./scripts/dev.zsh --with-tests --with-mcp

dev: ## Start development environment (database + server)
	make start-db
	@echo "Waiting for database..."
	@sleep 10
	make init-db
	make ingest
	make run

clean: ## Clean up Docker containers and volumes
	docker-compose down -v
	docker system prune -f

clean-venv: ## Remove virtual environment
	@if [ -d $(VENV_DIR) ]; then \
		echo "Removing virtual environment..."; \
		rm -rf $(VENV_DIR); \
		echo "Virtual environment removed"; \
	else \
		echo "Virtual environment not found"; \
	fi

logs: ## Show database logs
	docker-compose logs -f postgres

adminer: ## Open database admin interface
	@echo "Database admin available at: http://localhost:8080"
	@echo "Server: postgres"
	@echo "Username: wasmcloud_user" 
	@echo "Password: wasmcloud_password"
	@echo "Database: wasmcloud_rag"

# Additional help for zsh scripts
zsh-help: ## Show zsh script usage
	@echo ""
	@echo "🚀 zsh Script Commands (recommended):"
	@echo "  ./scripts/setup.zsh                    - Complete automated setup with venv"
	@echo "  ./scripts/start.zsh                    - Start all services (uses venv)"
	@echo "  ./scripts/start.zsh --with-mcp         - Start with MCP server"
	@echo "  ./scripts/stop.zsh                     - Stop all services"
	@echo "  ./scripts/stop.zsh --all               - Stop and clean everything"
	@echo "  ./scripts/dev.zsh                      - Interactive development environment"
	@echo "  ./scripts/dev.zsh --with-tests --with-mcp - Full development setup"
	@echo ""
	@echo "🐍 Virtual Environment:"
	@echo "  source $(VENV_ACTIVATE)                - Manually activate venv"
	@echo "  source activate_wasmcloud_rag.zsh      - Activate with helper script"
	@echo "  make venv-check                        - Check if venv exists"
	@echo "  make clean-venv                        - Remove virtual environment"
	@echo ""
	@echo "📝 Make targets (require virtual environment):"
	@echo "  make setup        - Complete setup"
	@echo "  make start        - Start services"
	@echo "  make start-mcp    - Start with MCP"
	@echo "  make stop         - Stop services"
	@echo "  make stop-clean   - Stop and clean"
	@echo "  make dev-env      - Development environment"
	@echo "  make dev-full     - Full development setup" 