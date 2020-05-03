#!/bin/bash

# config

#SBATCH -o /dss/dsshome1/lxc01/di73wal/work/simulation/diss.larsim/Larsim-UQ.git/st.0121.job.%j.%N.out
#SBATCH -D /dss/dsshome1/lxc01/di73wal/work/simulation/diss.larsim/Larsim-UQ.git
#SBATCH -J st.0121
#SBATCH --get-user-env
#SBATCH --cluster=cm2
#SBATCH --partition=cm2_std
#SBATCH --qos=cm2_std
#SBATCH --nodes=4
#SBATCH --cpus-per-task=28
#SBATCH --ntasks-per-node=1
#SBATCH --mem=55G
#SBATCH --exclusive
#SBATCH --mail-type=end
#SBATCH --mail-user=florian.kuenzner@tum.de
#SBATCH --export=NONE
#SBATCH --time=0:30:00

#module load slurm_setup
#module load python/3.6_intel
#module unload mpi.intel
#module load mpi.intel/2018

# load modules
module load slurm_setup
module load python/3.6_intel
module unload intel-mpi
module load intel-mpi/2018.4.274

export OMP_NUM_THREADS=28

# load python conda env
source activate larsim_uq
conda info -e
conda list

export CLASSPATH=/dss/dsshome1/lrz/sys/spack/staging/20.1.1/opt/x86_64/intel-mpi/2018.4.274-gcc-hoyrotl/compilers_and_libraries_2018.5.274/linux/mpi/intel64/lib/mpi.jar:/dss/dsshome1/lrz/sys/spack/staging/20.1.1/opt/x86_64/intel/19.1.1-gcc-zvv5bax/compilers_and_libraries_2020.1.217/linux/mpi/intel64/lib/mpi.jar
export PYTHONPATH=/dss/dsshome1/lxc01/di73wal/work/simulation/diss.larsim/uqef.git/src:/dss/dsshome1/lxc01/di73wal/work/simulation/diss.larsim/uqef.git/src:
#export PYTHONPATH=/dss/dsshome1/lxc01/di73wal/software/python/mpi4py.mpp2.git/build/lib.linux-x86_64-3.5:/dss/dsshome1/lxc01/di73wal/work/simulation/diss.larsim/uqef.git/src:
#export PYTHONPATH=/dss/dsshome1/lxc01/di73wal/software/python/chaospy_python3.git:/dss/dsshome1/lxc01/di73wal/work/simulation/diss.larsim/uqef.git/src:

# start simulation
echo ---- start sim: `date`

#python /dss/dsshome1/lxc01/di73wal/work/simulation/diss.larsim/Larsim-UQ.git/update_plots_uqsim.py --dir /dss/dsshome1/lxc01/di73wal/work/simulation/diss.larsim/Larsim-UQ.git/../larsim_runs/0121_sim

echo ---- end sim: `date`


