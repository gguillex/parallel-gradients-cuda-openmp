#include "fichero_cpu.h"
#include <math.h>

// --- PARTE 1: GRADIENTE GENÉRICO (Fuerza Bruta) ---
static type_data funcion_pesada(type_data val, int terms) {
    type_data acumulado = val;
    for (int k = 1; k <= terms; k++) {
        acumulado += sin(val * k) * cos(val * k);
    }
    return acumulado;
}

void ejecucion_CPU(type_data *data, type_data *gradients, int num_vars, int num_terms) {
    const type_data h = 1e-4;
    // Bucle sobre variables (diferencia central)
    for (int j = 0; j < num_vars; j++) {
        type_data x = data[j];
        type_data f_minus = funcion_pesada(x - h, num_terms);
        type_data f_plus = funcion_pesada(x + h, num_terms);
        gradients[j] = (f_plus - f_minus) / (2 * h);
    }
}

// --- PARTE 2: SISTEMA DE ECUACIONES ---
// Sistema de 3 ecuaciones con 3 incógnitas (x,y,z)
// f1 = 3x^2 + yz + z + 3
// f2 = 4x + yz^3 + 2
// f3 = y - 3xz + 8yz + z^2 - 3

static void evaluar_sistema(type_data x, type_data y, type_data z, type_data* res) {
    res[0] = 3.0f*x*x + y*z + z + 3.0f;
    res[1] = 4.0f*x + y*z*z*z + 2.0f;
    res[2] = y - 3.0f*x*z + 8.0f*y*z + z*z - 3.0f;
}

void ejecucion_CPU_Sistema(type_data *data, type_data *results, int num_points) {
    const type_data h = 1e-3;

    for (int i = 0; i < num_points; i++) {
        // Leemos 3 variables por punto
        type_data x = data[i*3 + 0];
        type_data y = data[i*3 + 1];
        type_data z = data[i*3 + 2];

        type_data f_xp[3], f_xm[3], f_yp[3], f_ym[3], f_zp[3], f_zm[3];

        evaluar_sistema(x+h, y, z, f_xp); evaluar_sistema(x-h, y, z, f_xm); // Perturbar x
        evaluar_sistema(x, y+h, z, f_yp); evaluar_sistema(x, y-h, z, f_ym); // Perturbar y
        evaluar_sistema(x, y, z+h, f_zp); evaluar_sistema(x, y, z-h, f_zm); // Perturbar z

        size_t base_idx = (size_t)i * 9;
        // Rellenar Jacobiana (3x3 aplanada), fila e = derivadas de f(e+1)
        for (int e = 0; e < 3; e++) {
            results[base_idx + e*3 + 0] = (f_xp[e] - f_xm[e]) / (2 * h);
            results[base_idx + e*3 + 1] = (f_yp[e] - f_ym[e]) / (2 * h);
            results[base_idx + e*3 + 2] = (f_zp[e] - f_zm[e]) / (2 * h);
        }
    }
}
