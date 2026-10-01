#include "definitions.h"
#include "fichero_cpu.h"

// Evalúa f con vars[pert_idx] desplazada 'delta' sin modificar el array.
// Se acumula en double: f crece con el número de variables y en float la
// resta de la diferencia finita perdía toda la precisión.
static double funcion_objetivo(const type_data *vars, int n, int pert_idx, double delta) {
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

void ejecucion_CPU(type_data *data, type_data *gradients, TYPE_DATA num_points, int num_vars) {
    const double h = 1e-3;
    for (TYPE_DATA i = 0; i < num_points; i++) {
        const type_data *p = &data[i * num_vars];
        type_data *g = &gradients[i * num_vars];

        // Diferencia central
        for (int j = 0; j < num_vars; j++) {
            double f_plus = funcion_objetivo(p, num_vars, j, h);
            double f_minus = funcion_objetivo(p, num_vars, j, -h);
            g[j] = (type_data)((f_plus - f_minus) / (2.0 * h));
        }
    }
}
