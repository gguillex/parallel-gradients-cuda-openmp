#ifndef FICHERO_GPU_H
#define FICHERO_GPU_H
#include "definitions.h"

#define CUDA_MAX_THREADS_PER_BLOCK 1024

#ifdef __cplusplus
extern "C" {
#endif
    // mode 2: gradiente genérico | mode 4: sistema de ecuaciones
    void ejecucion_CUDA_General(int mode, type_data *h_data, type_data *h_out,
                                int size_n, int load_terms, int threads);
#ifdef __cplusplus
}
#endif

#endif
