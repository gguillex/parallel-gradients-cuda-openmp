#include "definitions.h"
#include "type.h"
#include "fichero_cpu.h"
#include "fichero_omp.h"
#include "wtime.h"

int main (int argc, char **argv){
    int mode = 0; int num_threads = 4; int num_vars = 10; int nested = 0;
    TYPE_DATA num_points = 100000;

    // Parseo de argumentos
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i+1 < argc) mode = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i+1 < argc) num_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "-s") == 0 && i+1 < argc) num_points = strtoull(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "-v") == 0 && i+1 < argc) num_vars = atoi(argv[++i]);
        else if (strcmp(argv[i], "-n") == 0 && i+1 < argc) nested = atoi(argv[++i]);
    }

    if (mode != 0 && mode != 1) {
        fprintf(stderr, "Modo no soportado: %d (solo 0 y 1)\n", mode);
        return 1;
    }
    if (num_vars < 1 || num_points < 1 || num_threads < 1) {
        fprintf(stderr, "Error: -v, -s y -t deben ser mayores que 0\n");
        return 1;
    }

    size_t total_elements = (size_t)num_points * (size_t)num_vars;

    printf("\n=== GRADIENTES EN PARALELO (OpenMP) ===\n");
    printf("Modo: %d | Puntos: %llu | Vars: %d | Hilos: %d\n", mode, num_points, num_vars, num_threads);
    if(nested) printf("Paralelismo Anidado: ACTIVADO\n");

    type_data *data = (type_data*) malloc(total_elements * sizeof(type_data));
    type_data *res = (type_data*) malloc(total_elements * sizeof(type_data));

    if (!data || !res) { fprintf(stderr, "Error memoria\n"); free(data); free(res); return 1; }

    // Usamos OpenMP para inicializar
    #pragma omp parallel for
    for(size_t i = 0; i < total_elements; i++) data[i] = (type_data)i * 0.001f + 1.0f;

    double t1 = wtime();

    if (mode == 0) ejecucion_CPU(data, res, num_points, num_vars);
    else ejecucion_OMP(data, res, num_points, num_vars, num_threads, nested);

    double t2 = wtime();
    printf("Tiempo Global: %f s\n", t2 - t1);

    printf("\n--- RESULTADO PUNTO 0 ---\n");
    printf("Gradiente: [ ");
    int limit = (num_vars > 5) ? 5 : num_vars;
    for(int j=0; j<limit; j++) printf("%.2f ", res[j]);
    if(num_vars > 5) printf("... ");
    printf("]\n");

    free(data); free(res);
    return 0;
}
