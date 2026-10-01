# ---------------------------------------------------------------------------
# Gradientes en paralelo: CPU secuencial + OpenMP + CUDA
#
#   make                  compila ./gradientes
#   make CUDA_ARCH=86     compila para otra arquitectura de GPU
#   make benchmark        compila y ejecuta la batería de pruebas
#   make clean            borra objetos y ejecutable
# ---------------------------------------------------------------------------

# Rutas de CUDA (se pueden sobrescribir: make CUDA_PATH=/opt/cuda)
CUDA_PATH     ?= /usr/local/cuda
CUDA_INC_PATH ?= $(CUDA_PATH)/include
CUDA_LIB_PATH ?= $(CUDA_PATH)/lib64

NVCC ?= $(CUDA_PATH)/bin/nvcc
CXX  ?= g++

# Arquitectura de la GPU (61 = GTX 10xx, 75 = RTX 20xx/T4, 80 = A100, 86 = RTX 30xx, 89 = RTX 40xx)
CUDA_ARCH ?= 75
GENCODE   := -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)

CXXFLAGS  := -m64 -O3 -fopenmp -Wall -Wextra
NVCCFLAGS := -m64 -O3 $(GENCODE) -Xptxas -dlcm=ca
INCLUDES  := -I$(CUDA_INC_PATH)
LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart -fopenmp

EXEC := gradientes
OBJS := main.o wtime.o fichero_cpu.o fichero_omp.o fichero_gpu.o

.PHONY: all clean benchmark

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

main.o: main.cpp definitions.h fichero_cpu.h fichero_omp.h fichero_gpu.h wtime.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

wtime.o: wtime.cpp wtime.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

fichero_cpu.o: fichero_cpu.cpp fichero_cpu.h definitions.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

fichero_omp.o: fichero_omp.cpp fichero_omp.h definitions.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

fichero_gpu.o: fichero_gpu.cu fichero_gpu.h definitions.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

benchmark: $(EXEC)
	./benchmark.sh

clean:
	rm -f $(EXEC) $(OBJS)
