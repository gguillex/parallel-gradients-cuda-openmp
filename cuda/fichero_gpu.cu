#include <stdio.h>
#include <cuda_runtime.h>
#include "definitions.h"
#include "fichero_gpu.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s linea %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

// --- KERNEL 1: GRADIENTE GENÉRICO (Compute-Bound) ---
__device__ type_data funcion_pesada_gpu(type_data val, int terms) {
    type_data acumulado = val;
    for (int k = 1; k <= terms; k++) {
        acumulado += sin(val * k) * cos(val * k);
    }
    return acumulado;
}

__global__ void gradiente_variable_kernel(type_data *d_data, type_data *d_gradients, int num_vars, int num_terms) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    type_data h = 1e-4;

    if (idx < num_vars) {
        type_data x = d_data[idx];
        type_data f_minus = funcion_pesada_gpu(x - h, num_terms);
        type_data f_plus = funcion_pesada_gpu(x + h, num_terms);
        d_gradients[idx] = (f_plus - f_minus) / (2 * h);
    }
}

// --- KERNEL 2: SISTEMA DE ECUACIONES (Memory-Bound) ---
__device__ void evaluar_sistema_gpu(type_data x, type_data y, type_data z, type_data* r) {
    r[0] = 3.0f*x*x + y*z + z + 3.0f;
    r[1] = 4.0f*x + y*z*z*z + 2.0f;
    r[2] = y - 3.0f*x*z + 8.0f*y*z + z*z - 3.0f;
}

__global__ void sistema_kernel(type_data *d_data, type_data *d_res, int num_points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    type_data h = 1e-3;

    if (idx < num_points) {
        // Lectura coalescente (idx*3)
        type_data x = d_data[idx*3+0];
        type_data y = d_data[idx*3+1];
        type_data z = d_data[idx*3+2];

        type_data f_xp[3], f_xm[3], f_yp[3], f_ym[3], f_zp[3], f_zm[3];

        evaluar_sistema_gpu(x+h,y,z, f_xp); evaluar_sistema_gpu(x-h,y,z, f_xm);
        evaluar_sistema_gpu(x,y+h,z, f_yp); evaluar_sistema_gpu(x,y-h,z, f_ym);
        evaluar_sistema_gpu(x,y,z+h, f_zp); evaluar_sistema_gpu(x,y,z-h, f_zm);

        size_t base = (size_t)idx*9;
        // Escribir resultados (diferencia central), fila e = derivadas de f(e+1)
        for (int e = 0; e < 3; e++) {
            d_res[base+e*3+0] = (f_xp[e]-f_xm[e])/(2*h);
            d_res[base+e*3+1] = (f_yp[e]-f_ym[e])/(2*h);
            d_res[base+e*3+2] = (f_zp[e]-f_zm[e])/(2*h);
        }
    }
}

// Función Host Única
extern "C" void ejecucion_CUDA_General(int mode, type_data *h_data, type_data *h_out,
                                       int size_n, int load_terms, int threads)
{
    type_data *d_data, *d_out;
    size_t size_in_bytes, size_out_bytes;
    int num_items;

    if (threads < 1 || threads > CUDA_MAX_THREADS_PER_BLOCK) {
        fprintf(stderr, "Error: hilos por bloque (-t) debe estar entre 1 y %d (recibido %d)\n",
                CUDA_MAX_THREADS_PER_BLOCK, threads);
        exit(1);
    }

    // Configurar memoria según el modo
    if (mode == 2) { // Gradiente Genérico
        num_items = size_n; // Items = Variables
        size_in_bytes = (size_t)size_n * sizeof(type_data);
        size_out_bytes = (size_t)size_n * sizeof(type_data);
    } else { // Sistema (Modo 4)
        num_items = size_n; // Items = Puntos
        size_in_bytes = (size_t)size_n * 3 * sizeof(type_data);
        size_out_bytes = (size_t)size_n * 9 * sizeof(type_data);
    }

    cudaEvent_t start, stop_h2d, stop_kernel, stop_d2h;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop_h2d));
    CUDA_CHECK(cudaEventCreate(&stop_kernel)); CUDA_CHECK(cudaEventCreate(&stop_d2h));

    CUDA_CHECK(cudaMalloc((void**)&d_data, size_in_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_out, size_out_bytes));

    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size_in_bytes, cudaMemcpyHostToDevice));
    cudaEventRecord(stop_h2d);

    dim3 dimBlock(threads);
    dim3 dimGrid((num_items + threads - 1) / threads);

    // Selección de Kernel
    if (mode == 2) {
        gradiente_variable_kernel<<<dimGrid, dimBlock>>>(d_data, d_out, num_items, load_terms);
    } else {
        sistema_kernel<<<dimGrid, dimBlock>>>(d_data, d_out, num_items);
    }
    CUDA_CHECK(cudaGetLastError()); // Errores de configuración del lanzamiento

    cudaEventRecord(stop_kernel);
    CUDA_CHECK(cudaMemcpy(h_out, d_out, size_out_bytes, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop_d2h);
    CUDA_CHECK(cudaEventSynchronize(stop_d2h));

    float t_h2d, t_k, t_d2h;
    cudaEventElapsedTime(&t_h2d, start, stop_h2d);
    cudaEventElapsedTime(&t_k, stop_h2d, stop_kernel);
    cudaEventElapsedTime(&t_d2h, stop_kernel, stop_d2h);

    printf("\n   === CUDA SUMMARY (Modo %d) ===\n", mode);
    printf("   [H2D]:    %.4f ms\n", t_h2d);
    printf("   [Kernel]: %.4f ms\n", t_k);
    printf("   [D2H]:    %.4f ms\n", t_d2h);
    printf("   [TOTAL]:  %.4f ms\n", t_h2d + t_k + t_d2h);

    cudaEventDestroy(start); cudaEventDestroy(stop_h2d);
    cudaEventDestroy(stop_kernel); cudaEventDestroy(stop_d2h);
    cudaFree(d_data); cudaFree(d_out);
}
