# Load .env (if present) so DATABASE_URL and friends reach the commands below.
-include .env
export

.PHONY: up down logs migrate test lint

up: ## Build the app image, start the stack, apply migrations, wait until healthy
	docker compose up --build --detach --wait
	@echo "Stack is up. Jaeger UI: http://localhost:$${JAEGER_UI_PORT:-16686}"

down: ## Stop the stack (data in the postgres volume is kept)
	docker compose down

logs: ## Follow logs from all services
	docker compose logs --follow

migrate: ## Apply database migrations from this machine
	uv run alembic upgrade head

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy core tests migrations
