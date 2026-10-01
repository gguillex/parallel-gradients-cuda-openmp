#include "definitions.h"
#include "fichero_omp.h"
#include <omp.h>

// Evalúa f con vars[pert_idx] desplazada 'delta' sin modificar el array,
// así los hilos internos (modo anidado) pueden leer el mismo punto sin carreras.
static double funcion_objetivo_omp(const type_data *vars, int n, int pert_idx, double delta) {
    double res = 5.0;
    for (int i = 0; i < n; i++) {
        double xi = vars[i] + (i == pert_idx ? delta : 0.0);
        res += 2.0 * xi;
        if (i < n - 1) {
            double xn = vars[i+1] + (i + 1 == pert_idx ? delta : 0.0);
            res += 3.0 * xi * xn;
        }
    }
    return res;
}

void ejecucion_OMP(type_data *data, type_data *gradients, TYPE_DATA num_points, int num_vars, int num_threads, int nested)
{
    const double h = 1e-3;
    omp_set_num_threads(num_threads);

    if (nested) omp_set_nested(1);
    else omp_set_nested(0);

    #pragma omp parallel for default(none) shared(data, gradients, num_points, num_vars, h, nested) schedule(static)
    for (TYPE_DATA i = 0; i < num_points; i++) {
        const type_data *punto_actual = &data[i * num_vars];
        type_data *grad_actual = &gradients[i * num_vars];

        if (nested) {
            #pragma omp parallel for default(none) shared(punto_actual, grad_actual, num_vars, h) schedule(static)
            for (int j = 0; j < num_vars; j++) {
                double f_plus = funcion_objetivo_omp(punto_actual, num_vars, j, h);
                double f_minus = funcion_objetivo_omp(punto_actual, num_vars, j, -h);
                grad_actual[j] = (type_data)((f_plus - f_minus) / (2.0 * h));
            }
        } else {
            for (int j = 0; j < num_vars; j++) {
                double f_plus = funcion_objetivo_omp(punto_actual, num_vars, j, h);
                double f_minus = funcion_objetivo_omp(punto_actual, num_vars, j, -h);
                grad_actual[j] = (type_data)((f_plus - f_minus) / (2.0 * h));
            }
        }
    }
}
