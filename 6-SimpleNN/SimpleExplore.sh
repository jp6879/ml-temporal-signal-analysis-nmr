#!/bin/bash

# Script para correr trabajo serial

# Directorio actual es el raiz
#$ -cwd

# Nombre del proceso
#$ -N NN_Explore

# stdout y stderr al mismo archivo de salida
#$ -j y

# Usar bash como shell para los comandos que se ejecutaran
#$ -S /bin/bash

# Pido la cola a usar
#$ -q cpuINgpu

# Pido 6GB RAM para el proceso (obligatorio)
#$ -l mem=6G

# Voy a usar solo CPU
#$ -l gpu=0

# Las variables de entorno actual son incluidas en el trabajo
#$ -V

# Opciones SGE

#$ -cwd # Cambia al directorio actual
#$ -V # Exporta las variables de entorno
#$ -t 1-72 # Setea el array de indices

# Llamamos al modulo que tiene julia, y ejecutamos el programa tal cual lo llamar�amos desde la linea
# de comandos con los argumentos correspondientes al array de indices
module load julia-1.9.0

julia ExploracionVersionSimplificada.jl $SGE_TASK_ID


