#!/bin/bash

#module load python/3.6_intel
module unload python
module load anaconda3
conda activate uq_env
#source activate uq_env
salloc --nodes=2 --ntasks-per-node=12 -t 120 --partition=cm2_inter
