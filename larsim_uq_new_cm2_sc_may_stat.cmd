#!/bin/bash

# config

#SBATCH -e /dss/dsshome1/lxc0C/ga45met2/Repositories/Larsim-UQ/uq_simulation.0066.job.%j.%N.out
#SBATCH -D /dss/dsshome1/lxc0C/ga45met2/Repositories/Larsim-UQ
#SBATCH -J larsim.0066
#SBATCH --get-user-env
#SBATCH --clusters=cm2
#SBATCH --partition=cm2_std
#SBATCH --qos=cm2_std
#SBATCH --nodes=4
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=28
#SBATCH --mail-type=end
#SBATCH --mail-user=ivana.jovanovic@tum.de
#SBATCH --export=NONE
#SBATCH --time=24:00:00
#SBATCH --exclusive
###--mem=55G

# load modules and activate the conda env
module load slurm_setup
source /etc/profile.d/modules.sh
module load python/3.6_intel
if [[ cm2login3  == mpp3* ]]; then
  module unload mpi.intel/2019
  module load mpi.intel/2020
elif [[ cm2login3  == cm2* ]]; then
  module unload intel-mpi
  module load intel-mpi/2018-intel
fi
source /dss/dsshome1/lxc0C/ga45met2/.conda/envs/uq_env/bin/activate uq_env

# export num threads for OMP
export OMP_NUM_THREADS=1

# start simulation
echo ---- start Larsim sim: `date`

    mpiexec -n $SLURM_NTASKS python3 /dss/dsshome1/lxc0C/ga45met2/Repositories/Larsim-UQ/uq_simulation_uqsim.py                             --outputResultDir /gpfs/scratch/pr63so/ga45met2/Larsim_runs/larsim_uq_new_cm2.0066                             --inputModelDir /dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/Larsim-data                             --sourceDir /dss/dsshome1/lxc0C/ga45met2/Repositories/Larsim-UQ                             --config_file /dss/dsshome1/lxc0C/ga45met2/Repositories/Larsim-UQ/configurations_Larsim/configurations_larsim_4_may_direct.json                             --model larsim                             --uncertain all                             --opt_strategy DYNAMIC --opt_algorithm FCFS                             --chunksize 1                             --num_cores 1                             --mpi                             --mpi_method MpiPoolSolver                             --uq_method sc                             --sc_q_order 10                             --sc_p_order 6                             --mc_numevaluations 1000                             --sampling_rule S                              --transformToStandardDist

echo ---- end Larsim sim: `date`


