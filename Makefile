.PHONY: help setup validate install lock format lint start-local start-server health-check chat stop-server clean

# Default target
help:
	@echo "SGLang Deployment Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  setup          - Create .env from .env.example (first time setup)"
	@echo "  validate       - Validate environment configuration"
	@echo "  install        - Install dependencies from uv.lock"
	@echo "  lock           - Update uv.lock lockfile"
	@echo "  format         - Format code with ruff"
	@echo "  lint           - Lint code with ruff"
	@echo "  start-local    - Start SGLang server locally (dev mode)"
	@echo "  start-server   - Start SGLang server with custom parameters"
	@echo "  health-check   - Check server health"
	@echo "  chat           - Interactive chat with server (test inference speed)"
	@echo "  stop-server    - Stop running SGLang server"
	@echo "  clean          - Clean cache and temporary files"

# Setup: Create .env from .env.example
setup:
	@if [ -f .env ]; then \
		echo ".env already exists. Skipping setup."; \
		echo "Delete .env first if you want to recreate it from .env.example"; \
	else \
		cp .env.example .env; \
		echo "Created .env from .env.example"; \
		echo ""; \
		echo "IMPORTANT: Edit .env and set your HuggingFace token:"; \
		echo "  https://huggingface.co/settings/tokens"; \
		echo ""; \
		echo "Then run: make validate"; \
	fi

# Validate environment configuration
validate:
	@echo "Validating environment configuration..."
	@uv run python scripts/validate_env.py

# Install dependencies from lockfile
install:
	uv sync

# Update lockfile
lock:
	uv lock

# Format code with ruff
format:
	uv run ruff format .

# Lint code with ruff
lint:
	uv run ruff check .

# Start SGLang server locally (for development/testing)
# Note: This will use CPU mode on macOS without CUDA
start-local:
	@echo "Starting SGLang server in local mode..."
	@echo "Note: Running on CPU. Use start-server for GPU deployment."
	uv run python -m sglang.launch_server \
		--model-path $(or $(MODEL_PATH),meta-llama/Llama-3.2-1B-Instruct) \
		--host $(or $(SERVER_HOST),127.0.0.1) \
		--port $(or $(SERVER_PORT),30000) \
		--log-level info

# Start SGLang server with configurable parameters
# Usage: make start-server MODEL_PATH=<path> TP=4
start-server:
	@echo "Starting SGLang server..."
	uv run python -m sglang.launch_server \
		--model-path $(or $(MODEL_PATH),meta-llama/Meta-Llama-3-70B-Instruct) \
		$(if $(TP),--tp $(TP),) \
		$(if $(DP),--dp $(DP),) \
		$(if $(PP),--pp $(PP),) \
		--host $(or $(SERVER_HOST),0.0.0.0) \
		--port $(or $(SERVER_PORT),30000) \
		$(if $(MEM_FRACTION),--mem-fraction-static $(MEM_FRACTION),--mem-fraction-static 0.85) \
		$(if $(QUANTIZATION),--quantization $(QUANTIZATION),) \
		--trust-remote-code \
		$(EXTRA_ARGS)

# Check server health
health-check:
	@echo "Checking SGLang server health..."
	@curl -s http://$(or $(SERVER_HOST),localhost):$(or $(SERVER_PORT),30000)/health || echo "Server not responding"

# Interactive chat with server
# Usage: make chat [HOST=localhost] [PORT=30000]
chat:
	@echo "Starting interactive chat..."
	@echo "Make sure the server is running (make start-server or sbatch)"
	@echo ""
	uv run python scripts/chat.py \
		$(if $(HOST),--host $(HOST),) \
		$(if $(PORT),--port $(PORT),) \
		$(if $(MODEL_PATH),--model $(MODEL_PATH),)

# Stop running SGLang server (finds process by port)
stop-server:
	@echo "Stopping SGLang server on port $(or $(SERVER_PORT),30000)..."
	@lsof -ti:$(or $(SERVER_PORT),30000) | xargs kill -9 2>/dev/null || echo "No server running on port $(or $(SERVER_PORT),30000)"

# Clean cache and temporary files
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .uv_cache
	rm -rf *.log
	rm -rf logs/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.log" -delete
	@echo "Cleaned cache and temporary files"
