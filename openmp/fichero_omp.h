#ifndef FICHERO_OMP_H_INCLUDED
#define FICHERO_OMP_H_INCLUDED
#include "type.h"

void ejecucion_OMP(type_data *data, type_data *gradients, TYPE_DATA num_points, int num_vars, int num_threads, int nested);

#endif
