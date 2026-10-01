#ifndef FICHERO_CPU_H_INCLUDED
#define FICHERO_CPU_H_INCLUDED
#include "definitions.h"

// Modo Gradiente Genérico (Fuerza Bruta)
void ejecucion_CPU(type_data *data, type_data *gradients, int num_vars, int num_terms);

// Modo Sistema de Ecuaciones (Jacobiana 3x3 por punto)
void ejecucion_CPU_Sistema(type_data *data, type_data *results, int num_points);

#endif
