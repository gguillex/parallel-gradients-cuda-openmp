# ---------------------------------------------------------------------------
# Gradientes en paralelo: variante solo CPU / OpenMP
#
#   make          compila ./gradientes_omp
#   make clean    borra objetos y ejecutable
# ---------------------------------------------------------------------------

CXX      ?= g++
CXXFLAGS := -O3 -fopenmp -Wall -Wextra
LDFLAGS  := -lm -fopenmp

EXEC := gradientes_omp
OBJS := main.o wtime.o fichero_cpu.o fichero_omp.o

.PHONY: all clean

all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

%.o: %.cpp definitions.h type.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(EXEC)
