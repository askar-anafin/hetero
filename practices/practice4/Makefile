# Makefile for CUDA Reduction and Sorting
# Adjust CUDA_ARCH based on your GPU (use sm_75 for RTX 20xx, sm_86 for RTX 30xx, etc.)

NVCC = nvcc
CUDA_ARCH = sm_75
NVCC_FLAGS = -O3 -arch=$(CUDA_ARCH) -Xcompiler -Wall
CURAND_LIB = -lcurand

# Target executable
TARGET = benchmark

# Source files
SOURCES = main.cu

# Default target
all: $(TARGET)

# Build the main executable
$(TARGET): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(SOURCES) -o $(TARGET) $(CURAND_LIB)
	@echo "Build complete! Run with: ./$(TARGET)"

# Run the benchmark
run: $(TARGET)
	./$(TARGET)

# Run and generate plots
plot: run
	python plot_results.py

# Clean build artifacts
clean:
	rm -f $(TARGET) $(TARGET).exe results.csv performance_analysis.png
	@echo "Cleaned build artifacts"

# Help
help:
	@echo "Available targets:"
	@echo "  make         - Build the benchmark executable"
	@echo "  make run     - Build and run the benchmark"
	@echo "  make plot    - Run benchmark and generate plots"
	@echo "  make clean   - Remove build artifacts"
	@echo "  make help    - Show this help message"

.PHONY: all run plot clean help
