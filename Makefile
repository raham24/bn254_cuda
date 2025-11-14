# Simple Makefile for BN254 CUDA Implementation
# Alternative to CMake for quick builds

# Compiler settings
NVCC = nvcc
CXX = g++

# Auto-detect GPU architecture
GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.')
ifeq ($(GPU_ARCH),)
    # Default to common architectures if detection fails
    GPU_ARCH = 70
    $(info Could not detect GPU, using default architecture: $(GPU_ARCH))
else
    $(info Detected GPU architecture: $(GPU_ARCH))
endif

# Directories
INCLUDE_DIR = include
SRC_DIR = src
TEST_DIR = tests
BUILD_DIR = build
BIN_DIR = bin

# Compiler flags
NVCC_FLAGS = -std=c++14 --expt-relaxed-constexpr -I$(INCLUDE_DIR)
NVCC_ARCH = -arch=sm_$(GPU_ARCH)

# Debug/Release flags
ifeq ($(DEBUG),1)
    NVCC_FLAGS += -g -G -O0
    BUILD_TYPE = debug
else
    NVCC_FLAGS += -O3 -DNDEBUG
    BUILD_TYPE = release
endif

# Source files
FIELD_OPS_SRC = $(SRC_DIR)/field_ops.cu
API_SRC = $(SRC_DIR)/bn254_api.cpp

# Object files
FIELD_OPS_OBJ = $(BUILD_DIR)/field_ops.o
API_OBJ = $(BUILD_DIR)/bn254_api.o

# Test executables
TEST_BASIC = $(BIN_DIR)/test_basic
TEST_VECTORS = $(BIN_DIR)/test_vectors
TEST_PERFORMANCE = $(BIN_DIR)/test_performance

# All tests
ALL_TESTS = $(TEST_BASIC) $(TEST_VECTORS) $(TEST_PERFORMANCE)

# Default target
.PHONY: all
all: $(ALL_TESTS)
	@echo ""
	@echo "Build complete ($(BUILD_TYPE) mode)!"
	@echo "Executables are in $(BIN_DIR)/"
	@echo ""
	@echo "Run tests with:"
	@echo "  make test          - Run all tests"
	@echo "  make test_basic    - Run basic tests"
	@echo "  make test_vectors  - Run vector tests"
	@echo "  make test_perf     - Run performance benchmarks"

# Create directories
$(BUILD_DIR) $(BIN_DIR):
	mkdir -p $@

# Compile field operations
$(FIELD_OPS_OBJ): $(FIELD_OPS_SRC) | $(BUILD_DIR)
	@echo "Compiling field operations..."
	$(NVCC) $(NVCC_FLAGS) $(NVCC_ARCH) -dc $< -o $@

# Compile API wrapper
$(API_OBJ): $(API_SRC) | $(BUILD_DIR)
	@echo "Compiling API wrapper..."
	$(NVCC) $(NVCC_FLAGS) $(NVCC_ARCH) -dc $< -o $@

# Link test_basic
$(TEST_BASIC): $(TEST_DIR)/test_basic.cu $(FIELD_OPS_OBJ) $(API_OBJ) | $(BIN_DIR)
	@echo "Building test_basic..."
	$(NVCC) $(NVCC_FLAGS) $(NVCC_ARCH) $^ -o $@

# Link test_vectors
$(TEST_VECTORS): $(TEST_DIR)/test_vectors.cu $(FIELD_OPS_OBJ) $(API_OBJ) | $(BIN_DIR)
	@echo "Building test_vectors..."
	$(NVCC) $(NVCC_FLAGS) $(NVCC_ARCH) $^ -o $@

# Link test_performance
$(TEST_PERFORMANCE): $(TEST_DIR)/test_performance.cu $(FIELD_OPS_OBJ) $(API_OBJ) | $(BIN_DIR)
	@echo "Building test_performance..."
	$(NVCC) $(NVCC_FLAGS) $(NVCC_ARCH) $^ -o $@

# Run all tests
.PHONY: test
test: $(ALL_TESTS)
	@echo ""
	@echo "=========================================="
	@echo "Running All Tests"
	@echo "=========================================="
	@echo ""
	@echo ">>> Running Basic Tests..."
	@$(TEST_BASIC)
	@echo ""
	@echo ">>> Running Vector Tests..."
	@$(TEST_VECTORS)
	@echo ""
	@echo ">>> Running Performance Benchmarks..."
	@$(TEST_PERFORMANCE)
	@echo ""
	@echo "All tests completed!"

# Run individual tests
.PHONY: test_basic
test_basic: $(TEST_BASIC)
	@echo "Running basic tests..."
	@$(TEST_BASIC)

.PHONY: test_vectors
test_vectors: $(TEST_VECTORS)
	@echo "Running vector tests..."
	@$(TEST_VECTORS)

.PHONY: test_perf
test_perf: $(TEST_PERFORMANCE)
	@echo "Running performance benchmarks..."
	@$(TEST_PERFORMANCE)

# Debug build
.PHONY: debug
debug:
	@$(MAKE) DEBUG=1 all

# Release build (default)
.PHONY: release
release:
	@$(MAKE) DEBUG=0 all

# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR) $(BIN_DIR)
	@echo "Clean complete!"

# Print configuration
.PHONY: info
info:
	@echo ""
	@echo "Build Configuration:"
	@echo "  NVCC: $(NVCC)"
	@echo "  GPU Architecture: sm_$(GPU_ARCH)"
	@echo "  Build Type: $(BUILD_TYPE)"
	@echo "  Flags: $(NVCC_FLAGS) $(NVCC_ARCH)"
	@echo ""

# Help
.PHONY: help
help:
	@echo ""
	@echo "BN254 CUDA Implementation - Makefile Help"
	@echo ""
	@echo "Targets:"
	@echo "  all            - Build all tests (default)"
	@echo "  test           - Build and run all tests"
	@echo "  test_basic     - Run basic functionality tests"
	@echo "  test_vectors   - Run vector operation tests"
	@echo "  test_perf      - Run performance benchmarks"
	@echo "  debug          - Build with debug flags"
	@echo "  release        - Build with optimization flags"
	@echo "  clean          - Remove build artifacts"
	@echo "  info           - Print build configuration"
	@echo "  help           - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make              # Build all (release mode)"
	@echo "  make DEBUG=1      # Build with debug symbols"
	@echo "  make test         # Build and run all tests"
	@echo "  make clean all    # Clean rebuild"
	@echo ""
