#ifndef FICHERO_OMP_H_INCLUDED
#define FICHERO_OMP_H_INCLUDED
#include "definitions.h"

void ejecucion_OMP(type_data *data, type_data *gradients, int num_vars, int num_terms, int num_threads, int nested);

#endif
