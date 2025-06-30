#!/bin/bash

salloc --clusters=inter --partition=cm4_inter -N 1 -n 112 -c 1 -t 02:00:00
# salloc -M inter -p cm4_inter -N 1 -n 112 -c 1 -t 06:00:00
#module unload python
#module load anaconda3
#module load python/3.6_intel
if [ -z "$1"]; then
	conda_env=my_uq_env  #uq_env
else 
	conda_env=$1	
fi 
source ~/.conda_init
conda activate $conda_env
export OMP_NUM_THREADS=1
#mpiexec -n 10 python3 "$HOME/Repositories/UQEFPP/uq_simulation_uqsim.py"
#mpiexec -n 12 /dss/dsshome1/lxc0C/ga45met2/.conda/envs/$conda_env/bin/python "$HOME/Repositories/UQEF-Dynamic/uqef_dynamic/scientific_pipelines/uq_simulation_uqsim.py"
# mpiexec -n 100 /dss/dsshome1/lxc0C/ga45met2/.conda/envs/$conda_env/bin/python "$HOME/Repositories/UQEF-Dynamic/uqef_dynamic/scientific_pipelines/uq_simulation_uqsim_debug_cluster.py"
# mpiexec -n 110 /dss/dsshome1/lxc0C/ga45met2/.conda/envs/$conda_env/bin/python "$HOME/Repositories/UQEF-Dynamic/uqef_dynamic/scientific_pipelines/uq_simulation_uqsim_debug_cluster.py"
nohup mpiexec -n 112 /dss/dsshome1/lxc0C/ga45met2/.conda/envs/$conda_env/bin/python "$HOME/Repositories/UQEF-Dynamic/uqef_dynamic/scientific_pipelines/uq_simulation_uqsim_debug_cluster.py" > output.log 2>&1 &