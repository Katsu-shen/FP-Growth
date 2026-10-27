CXX := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -pedantic -O2
TARGET := fp_growth
SOURCES := main.cpp data_loader.cpp preprocessing.cpp fp_tree.cpp classifier.cpp output.cpp
NVCC ?= nvcc
CUDA_ARCH ?= 87
CUDA_TARGET := fp_growth_cuda
CUDA_SOURCES := $(SOURCES) classifier_cuda.cu
CUDAFLAGS := -std=c++17 -O2 -DFP_GROWTH_CUDA \
	-gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
	-Xcompiler -Wall,-Wextra,-pedantic

.PHONY: all cuda clean run-iris

all: $(TARGET)

$(TARGET): $(SOURCES) common.hpp data_loader.hpp preprocessing.hpp fp_tree.hpp classifier.hpp output.hpp
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)

cuda:
	@command -v $(NVCC) >/dev/null || { echo "Error: nvcc is required to build fp_growth_cuda."; exit 1; }
	$(NVCC) $(CUDAFLAGS) $(CUDA_SOURCES) -o $(CUDA_TARGET)

run-iris: $(TARGET)
	./$(TARGET) iris/iris.data 0.2 0.7

clean:
	rm -f $(TARGET) $(CUDA_TARGET)
