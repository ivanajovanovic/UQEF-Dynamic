#!/bin/bash

#module load python/3.6_intel
if [ -z "$1"]; then
	conda_env=uq_env
else 
	conda_env=$1	
fi 
source ~/.conda_init
conda activate $conda_env
export OMP_NUM_THREADS=1
#mpiexec -n 10 python3 "$HOME/Repositories/UQEFPP/uq_simulation_uqsim.py"
mpiexec -n 12 /dss/dsshome1/lxc0C/ga45met2/.conda/envs/$conda_env/bin/python "$HOME/Repositories/UQEF-Dynamic/uqef_dynamic/scientific_pipelines/uq_simulation_uqsim.py"
