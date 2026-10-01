#!/bin/bash
# ---------------------------------------------------------------------------
# Batería de pruebas: compilación, validación numérica entre backends,
# escalabilidad OpenMP, paralelismo anidado, tamaño de bloque CUDA y
# sistema de ecuaciones. Necesita una GPU NVIDIA.
#
#   ./benchmark.sh    (o: make benchmark)
#
# Devuelve un código distinto de 0 si falla alguna validación.
# ---------------------------------------------------------------------------

cd "$(dirname "$0")" || exit 1
FAIL=0

# Colores
GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${CYAN}=================================================================${NC}"
echo -e "${CYAN}   GRADIENTES EN PARALELO: PRUEBAS                                ${NC}"
echo -e "${CYAN}=================================================================${NC}"

# 1. COMPILACIÓN
echo -e "\n${YELLOW}[FASE 1] COMPILACIÓN...${NC}"
make clean > /dev/null
make > /dev/null
if [ $? -eq 0 ]; then echo -e "${GREEN}✔ Compilación Exitosa.${NC}"; else echo -e "${RED}✘ ERROR DE COMPILACIÓN.${NC}"; exit 1; fi

# 2. VALIDACIÓN NUMÉRICA
echo -e "\n${YELLOW}[FASE 2] VALIDACIÓN DE RESULTADOS (Correctitud)${NC}"
V=10
T=100
echo -e "${BLUE}   -> Configuración: Variables=$V | Términos=$T (Carga Ligera)${NC}"

# CPU
echo -n "   [CPU]    -m 0: "
VAL_CPU=$(./gradientes -m 0 -v $V -terms $T | grep "Gradiente\[0\]" | awk '{print $2}')
echo "$VAL_CPU"

# OpenMP
echo -n "   [OpenMP] -m 1 -t 4: "
VAL_OMP=$(./gradientes -m 1 -v $V -terms $T -t 4 | grep "Gradiente\[0\]" | awk '{print $2}')
echo "$VAL_OMP"

# CUDA
echo -n "   [CUDA]   -m 2 -t 256: "
VAL_CUDA=$(./gradientes -m 2 -v $V -terms $T -t 256 | grep "Gradiente\[0\]" | awk '{print $2}')
echo "$VAL_CUDA"

if [ "$VAL_CPU" == "$VAL_OMP" ]; then echo -e "${GREEN}✔ TEST PASADO: CPU == OpenMP${NC}"; else echo -e "${RED}✘ ERROR: Resultados distintos${NC}"; FAIL=1; fi

# La GPU redondea sin/cos de forma ligeramente distinta: se compara con tolerancia relativa del 1%
if [ -n "$VAL_CUDA" ] && LC_ALL=C awk -v a="$VAL_CPU" -v b="$VAL_CUDA" 'BEGIN { d = a - b; if (d < 0) d = -d; m = (a < 0) ? -a : a; exit !(d <= 0.01 * m + 1e-3) }'; then
    echo -e "${GREEN}✔ TEST PASADO: CPU ≈ CUDA (tolerancia 1%)${NC}"
else
    echo -e "${RED}✘ ERROR: CUDA difiere de CPU${NC}"
    FAIL=1
fi


# 3. ESCALABILIDAD OPENMP
echo -e "\n${YELLOW}[FASE 3] ESCALABILIDAD OPENMP (Rendimiento)${NC}"
V=1000000
T=200
echo -e "${BLUE}   -> Configuración: Variables=$V | Términos=$T (Carga Pesada)${NC}"

echo "   Ejecutando con 1 Hilo (-t 1)..."
./gradientes -m 1 -v $V -terms $T -t 1 | grep "Tiempo Host"

echo "   Ejecutando con 4 Hilos (-t 4)..."
./gradientes -m 1 -v $V -terms $T -t 4 | grep "Tiempo Host"

echo "   Ejecutando con 8 Hilos (-t 8)..."
./gradientes -m 1 -v $V -terms $T -t 8 | grep "Tiempo Host"


# 4. ANIDAMIENTO (NESTED)
echo -e "\n${YELLOW}[FASE 4] ESTUDIO ANIDAMIENTO (Nested Parallelism)${NC}"
V=1000000
T=50
TH=4
echo -e "${BLUE}   -> Configuración: Vars=$V | Terms=$T | Threads=$TH${NC}"

echo -n "   Sin Nested (-n 0): "
./gradientes -m 1 -v $V -terms $T -t $TH -n 0 | grep "Tiempo Host"

echo -n "   Con Nested (-n 1): "
./gradientes -m 1 -v $V -terms $T -t $TH -n 1 | grep "Tiempo Host"


# 5. TUNING CUDA
echo -e "\n${YELLOW}[FASE 5] TUNING CUDA (Tamaño de Bloque)${NC}"
V=1000000
T=200
echo -e "${BLUE}   -> Configuración: Variables=$V | Términos=$T${NC}"

echo -n "   Bloque 32  (-t 32):  "
./gradientes -m 2 -v $V -terms $T -t 32 | grep "\[TOTAL\]"

echo -n "   Bloque 256 (-t 256): "
./gradientes -m 2 -v $V -terms $T -t 256 | grep "\[TOTAL\]"


# 6. EXTENSIÓN SISTEMAS
echo -e "\n${YELLOW}[FASE 6] SISTEMAS DE ECUACIONES (Jacobiana 3x3)${NC}"
echo -e "${BLUE}   -> NOTA: Aquí '-v' significa PUNTOS (no variables).${NC}"

# Prueba Visual
P=10
echo -e "${BLUE}   -> Prueba Visual: Puntos=$P (Verificar Matrices)${NC}"
MAT_CPU=$(./gradientes -m 3 -v $P | grep -A 3 "Jacobiana 3x3 aplanada" | tail -n 3)
MAT_CUDA=$(./gradientes -m 4 -v $P -t 256 | grep -A 3 "Jacobiana 3x3 aplanada" | tail -n 3)
echo "   [CPU Sistema - Modo 3]:"
echo "$MAT_CPU" | sed 's/^/      /'
echo "   [CUDA Sistema - Modo 4]:"
echo "$MAT_CUDA" | sed 's/^/      /'
if [ -n "$MAT_CUDA" ] && [ "$MAT_CPU" == "$MAT_CUDA" ]; then echo -e "${GREEN}✔ TEST PASADO: Jacobiana CPU == CUDA${NC}"; else echo -e "${RED}✘ ERROR: Jacobianas distintas${NC}"; FAIL=1; fi

# Prueba Rendimiento
P=100000
echo -e "\n${BLUE}   -> Prueba Rendimiento: Puntos=$P (Verificar Tiempos)${NC}"
echo -n "   [CUDA Sistema]: "
./gradientes -m 4 -v $P -t 256 | grep "\[TOTAL\]"

echo -e "\n${CYAN}=================================================================${NC}"
echo -e "${CYAN}   FIN DE LAS PRUEBAS                                            ${NC}"
exit $FAIL
