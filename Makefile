.PHONY: all build run-web run-benchmark clean test

# Build all binaries
all: build

build:
	@echo "Building chess experiments..."
	@go build -o bin/web cmd/web/main.go
	@go build -o bin/evaluator cmd/evaluator/main.go
	@go build -o bin/benchmark cmd/benchmark/main.go
	@echo "Build complete!"

# Run the web UI
run-web: build
	@echo "Starting web UI on http://localhost:8080"
	@./bin/web

# Run the benchmark
run-benchmark: build
	@echo "Running engine benchmark..."
	@./bin/benchmark

# Run the simple evaluator
run-evaluator: build
	@./bin/evaluator

# Run tests
test:
	@go test ./...

# Clean build artifacts
clean:
	@rm -rf bin/
	@echo "Clean complete!"

# Install dependencies
deps:
	@go mod download
	@go mod tidy

# Format code
fmt:
	@go fmt ./...

# Run linter (requires golangci-lint)
lint:
	@golangci-lint run

# Quick development server (with hot reload if you have air installed)
dev:
	@if command -v air > /dev/null; then \
		air -c .air.toml; \
	else \
		echo "Installing air for hot reload..."; \
		go install github.com/cosmtrek/air@latest; \
		air -c .air.toml; \
	fi