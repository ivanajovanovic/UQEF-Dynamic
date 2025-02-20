#!/bin/bash

#module load python/3.6_intel
module unload python
module load anaconda3
conda activate my_uq_env 
#conda activate uq_env
#source activate uq_env
salloc --clusters=inter --partition=cm4_inter -N 1 -t 00:30:00
# salloc --clusters=inter --partition=cm4_inter -N 1 -t 00:20:00
#--nodes=2 --ntasks-per-node=12 -t 120