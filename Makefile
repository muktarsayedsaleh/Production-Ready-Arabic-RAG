# Load .env (if present) so DATABASE_URL and friends reach the commands below.
-include .env
export

.PHONY: up down logs test lint

up: ## Start the stack in the background and wait until it is healthy
	docker compose up --detach --wait
	@echo "Stack is up. Jaeger UI: http://localhost:$${JAEGER_UI_PORT:-16686}"

down: ## Stop the stack (data in the postgres volume is kept)
	docker compose down

logs: ## Follow logs from all services
	docker compose logs --follow

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy core tests
