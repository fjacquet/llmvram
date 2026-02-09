.PHONY: help install dev build typecheck lint lint-fix format test test-coverage refresh clean all

# Default target
help:
	@echo "LLM VRAM Calculator"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install        Install dependencies"
	@echo "  dev            Start development server"
	@echo "  build          Build for production"
	@echo "  typecheck      Run TypeScript type checking"
	@echo "  lint           Run Biome linter"
	@echo "  lint-fix       Run Biome linter with auto-fix"
	@echo "  format         Format code with Biome"
	@echo "  test           Run tests"
	@echo "  test-coverage  Run tests with coverage report"
	@echo "  refresh        Refresh model and GPU data from sources"
	@echo "  clean          Remove build artifacts"
	@echo "  all            Run lint, typecheck, and build"

# Install dependencies
install:
	npm install

# Development server
dev:
	npm run dev

# Production build
build:
	npm run build

# Type checking
typecheck:
	npm run typecheck

# Linting
lint:
	npm run lint

lint-fix:
	npm run lint:fix

# Formatting
format:
	npm run format

# Testing
test:
	npm test

test-coverage:
	npm run test:coverage

# Data refresh
refresh:
	npm run refresh:all

# Clean build artifacts
clean:
	rm -rf dist
	rm -rf node_modules/.vite
	rm -rf coverage

# Run all checks and build
all: lint typecheck build
