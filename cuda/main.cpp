#include "definitions.h"
#include "fichero_cpu.h"
#include "fichero_omp.h"
#include "wtime.h"
#include "fichero_gpu.h"
#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main (int argc, char **argv){
    int mode = 0;
    int num_threads = -1; // -1 = valor por defecto según el modo
    int num_vars = 1000;
    int num_terms = 1000; // Por defecto 1000
    int nested = 0;

    // --- PARSEO DE ARGUMENTOS ---
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i+1 < argc) mode = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i+1 < argc) num_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0 && i+1 < argc) num_vars = atoi(argv[++i]);
        else if (strcmp(argv[i], "-n") == 0 && i+1 < argc) nested = atoi(argv[++i]);
        // Opción para cambiar términos (-terms o -l)
        else if ((strcmp(argv[i], "-terms") == 0 || strcmp(argv[i], "-l") == 0) && i+1 < argc)
            num_terms = atoi(argv[++i]);
    }

    if (mode < 0 || mode > 4) {
        fprintf(stderr, "Modo no valido: %d (usa 0-4)\n", mode);
        return 1;
    }
    if (num_vars < 1) {
        fprintf(stderr, "Error: -v debe ser mayor que 0\n");
        return 1;
    }

    int is_cuda = (mode == 2 || mode == 4);
    if (num_threads < 1) {
        // CUDA: 256 hilos por bloque | CPU/OpenMP: todos los núcleos disponibles
        num_threads = is_cuda ? 256 : omp_get_max_threads();
    }

    size_t size_in, size_out;
    int is_system = (mode == 3 || mode == 4);

    if (!is_system) {
        size_in = (size_t)num_vars;
        size_out = (size_t)num_vars;
    } else {
        size_in = (size_t)num_vars * 3;
        size_out = (size_t)num_vars * 9;
    }

    printf("\n=== GRADIENTES EN PARALELO ===\n");
    printf("Modo: %d | Hilos: %d | Nested: %d\n", mode, num_threads, nested);

    if(is_system) {
        printf(">> SISTEMA: Jacobiana 3x3 (%d Puntos)\n", num_vars);
        printf("   [Info] Sistema de ecuaciones:\n");
        printf("   -> f1(x,y,z) = 3x^2 + yz + z + 3\n");
        printf("   -> f2(x,y,z) = 4x + yz^3 + 2\n");
        printf("   -> f3(x,y,z) = y - 3xz + 8yz + z^2 - 3\n");
        printf("   --------------------------------------\n");
    }
    else {
        printf(">> GENERICO: Gradiente (%d Variables, %d Terminos)\n", num_vars, num_terms);
        printf("   [Info] Funcion a derivar (Carga de trabajo):\n");
        printf("   -> f(x) = x + Sumatoria[k=1..%d] ( sen(x*k) * cos(x*k) )\n", num_terms);
        printf("   --------------------------------------------\n");
    }

    type_data *data = (type_data*) malloc(size_in * sizeof(type_data));
    type_data *res = (type_data*) malloc(size_out * sizeof(type_data));
    if (!data || !res) {
        fprintf(stderr, "Error: no hay memoria suficiente\n");
        free(data); free(res);
        return 1;
    }

    srand(1234);
    for(size_t i = 0; i < size_in; i++) data[i] = ((type_data)rand()/RAND_MAX)*2.0f;

    double t1 = wtime();

    switch(mode) {
        case 0: ejecucion_CPU(data, res, num_vars, num_terms); break;
        case 1: ejecucion_OMP(data, res, num_vars, num_terms, num_threads, nested); break;
        case 2: ejecucion_CUDA_General(2, data, res, num_vars, num_terms, num_threads); break;
        case 3: ejecucion_CPU_Sistema(data, res, num_vars); break;
        case 4: ejecucion_CUDA_General(4, data, res, num_vars, 0, num_threads); break;
    }

    double total_time = wtime() - t1;
    if (!is_cuda) printf("Tiempo Host: %.4f s\n", total_time);

    printf("\n--- RESULTADOS (Muestra de los 5 primeros) ---\n");

    if (!is_system) {
        // Imprime los 5 primeros (o menos si hay pocas variables)
        int limit = (num_vars < 5) ? num_vars : 5;
        for(int i = 0; i < limit; i++) {
            printf("Gradiente[%d]: %.4f\n", i, res[i]);
        }
    } else {
        printf("Punto 0 (Jacobiana 3x3 aplanada):\n");
        printf("[ %.2f %.2f %.2f ]\n", res[0], res[1], res[2]);
        printf("[ %.2f %.2f %.2f ]\n", res[3], res[4], res[5]);
        printf("[ %.2f %.2f %.2f ]\n", res[6], res[7], res[8]);
    }

    free(data); free(res);
    return 0;
}
