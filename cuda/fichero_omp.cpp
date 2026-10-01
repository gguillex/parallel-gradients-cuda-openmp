#include "fichero_omp.h"
#include <omp.h>
#include <math.h>

static type_data funcion_pesada_omp(type_data val, int terms) {
    type_data acumulado = val;
    for (int k = 1; k <= terms; k++) {
        acumulado += sin(val * k) * cos(val * k);
    }
    return acumulado;
}

void ejecucion_OMP(type_data *data, type_data *gradients, int num_vars, int num_terms, int num_threads, int nested)
{
    const type_data h = 1e-4;
    omp_set_num_threads(num_threads);

    // ESTUDIO DE ANIDAMIENTO
    if (nested) {
        omp_set_nested(1); // Activar anidamiento

        #pragma omp parallel for default(none) shared(data, gradients, num_vars, num_terms, h) schedule(static)
        for (int j = 0; j < num_vars; j++) {
            type_data x = data[j];

            // Región paralela interna de grano fino para medir el coste del anidamiento
            type_data f_minus, f_plus;

            #pragma omp parallel sections
            {
                #pragma omp section
                { f_minus = funcion_pesada_omp(x - h, num_terms); }

                #pragma omp section
                { f_plus = funcion_pesada_omp(x + h, num_terms); }
            }
            gradients[j] = (f_plus - f_minus) / (2 * h);
        }
    } else {
        // MODO ESTÁNDAR (Eficiente)
        omp_set_nested(0);

        #pragma omp parallel for default(none) shared(data, gradients, num_vars, num_terms, h) schedule(static)
        for (int j = 0; j < num_vars; j++) {
            type_data x = data[j];
            type_data f_minus = funcion_pesada_omp(x - h, num_terms);
            type_data f_plus = funcion_pesada_omp(x + h, num_terms);
            gradients[j] = (f_plus - f_minus) / (2 * h);
        }
    }
}
