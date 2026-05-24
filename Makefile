# goml Makefile
# Targets:
#   make              — build CUDA libs + Go binaries
#   make cuda         — build CUDA libs only (fat binary, all supported SMs)
#   make cuda-blackwell — build CUDA libs targeting Blackwell only (faster)
#   make cuda-ampere  — build CUDA libs targeting Ampere/Ada only (faster)
#   make clean        — remove built artifacts
#   make test         — go test ./...

.DEFAULT_GOAL := all

GO       ?= go
LIBS_DIR ?= libs

# ─── CUDA libraries ─────────────────────────────────────────────────────────

.PHONY: cuda
cuda:
	@bash scripts/build_cuda.sh

.PHONY: cuda-blackwell
cuda-blackwell:
	@echo "=== Building for Blackwell only (sm_120, fastest build) ==="
	@GOML_GENCODE_EXTRA="-gencode arch=compute_120,code=compute_120" \
	GOML_TARGET_ONLY="-gencode arch=compute_120,code=sm_120" \
	bash scripts/build_cuda.sh

.PHONY: cuda-ampere
cuda-ampere:
	@echo "=== Building for Ampere/Ada only ==="
	@GOML_TARGET_ONLY="-gencode arch=compute_80,code=sm_80 -gencode arch=compute_89,code=sm_89" \
	bash scripts/build_cuda.sh

# ─── Go ─────────────────────────────────────────────────────────────────────

.PHONY: build
build:
	@$(GO) build ./...

.PHONY: test
test:
	@$(GO) test ./...

.PHONY: race
race:
	@$(GO) test -race ./...

.PHONY: bench
bench:
	@$(GO) test -bench=. -benchmem ./...

.PHONY: all
all: cuda build

# ─── Maintenance ────────────────────────────────────────────────────────────

.PHONY: clean
clean:
	@rm -f $(LIBS_DIR)/*.so
	@$(GO) clean ./...

.PHONY: info
info:
	@echo "GPU info:"
	@nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv 2>/dev/null || echo "  (no GPU detected)"
	@echo ""
	@echo "CUDA toolkit:"
	@nvcc --version 2>/dev/null | head -4 || echo "  (nvcc not in PATH)"
	@echo ""
	@echo "Built libs:"
	@ls -lh $(LIBS_DIR)/*.so 2>/dev/null || echo "  (none — run 'make cuda')"
