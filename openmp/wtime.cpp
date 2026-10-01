#include <stdlib.h>
#include <omp.h> // Usamos OpenMP para medir el tiempo (Portable)
#include "wtime.h"

extern double wtime()
{
   // Devuelve el tiempo en segundos con alta precisión
   return omp_get_wtime();
}


    
