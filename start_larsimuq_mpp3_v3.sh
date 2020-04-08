#!/bin/bash

#module load python/3.6_intel

#export PYTHONPATH=$HOME/software/python/mpi4py.mpp2.git/build/lib.linux-x86_64-3.5:$PYTHONPATH

start_larsim_uq_sim(){
    local strategy="$1"
    local algorithm="$2"
    local q_order="$3"
    local p_order="$4"
    local model="$5"
    local opt="$6"
    local num_times="$7"
    local mpi_method="$8"
    local cluster_nodes="$9"
    local time_limit="${10}"

    #get counter
    counter=$((`cat counter` +1))
    echo $counter > counter

    if [[ ${#counter} < 2 ]]; then
        counter="000${counter}"
    elif [[ ${#counter} < 3 ]]; then
        counter="00${counter}"
    elif [[ ${#counter} < 4 ]]; then
        counter="0${counter}"
    fi
    counter="${counter: -4}"

    #print to the command line!
    #echo "$counter:mpp2: $@"
    echo "$counter:mpp3: $@" >> started_jobs.txt

    #create env
    # init vars
    basePath=`pwd`
    baseSourcePath=$basePath
    baseExecutionPath=$basePath
    #baseResultsPath=$basePath
    baseResultsPath=$WORK
    modelMasterPath=$basePath

    executionPath=$baseExecutionPath
    resultsPath=$baseResultsPath/Repositories/larsim_runs

    # init the stuff
    if [ "$strategy" == "FIXED_LINEAR" ] ; then
        cpus=28
        threads=$cpus
        tasks=1
    else
        cpus=1
        threads=$cpus
        tasks=64
        total_num_cores=128
    fi

#create batch file
echo "#!/bin/bash

# config

#SBATCH -o $baseSourcePath/uq_simulation.$counter.job.%j.%N.out
#SBATCH -D $baseSourcePath
#SBATCH -J larsim.$counter
#SBATCH --get-user-env
#SBATCH --clusters=mpp3
#SBATCH --nodes=$cluster_nodes
#SBATCH --cpus-per-task=$cpus
#SBATCH --ntasks-per-node=$tasks
#SBATCH --exclusive
#SBATCH --mail-type=end
#SBATCH --mail-user=ivana.jovanovic@tum.de
#SBATCH --export=NONE
#SBATCH --time=$time_limit

source /etc/profile.d/modules.sh
module load python/3.6_intel
module unload intel-mpi
module load intel-mpi/2018-intel
source /dss/dsshome1/lxc0C/ga45met2/.conda/envs/larsimuq/bin/activate larsimuq

export OMP_NUM_THREADS=$threads


# start simulation
echo "---- start sim:"

    mpiexec -genv I_MPI_DEBUG=+5 -print-rank-map python3 $executionPath/uq_simulation_uqsim.py \
                            --outputResultDir "/gpfs/scratch/pr63so/ga45met2/Larsim_runs" \
                            --model "larsim" \
                            --chunksize 1 \
                            --num_cores $threads \
                            --mpi \
                            --mpi_method "MpiPoolSolver" \
                            --config_file "/dss/dsshome1/lxc0C/ga45met2/Repositories/Larsim-UQ/configuration_larsim_uqsim_cm2.json" \
                            --uq_method "saltelli"  \
                            --mc_numevaluations 2000 \
                            --sampling_rule "S" \
                            --opt_algorithm "FCFS" \
                            --transformToStandardDist

echo "---- end \$i:"

" > uq_larsim_mpp3_saltelli_v3.cmd

    #execute batch file
    sbatch uq_larsim_mpp3_saltelli_v3.cmd
}

model="larsim"
opt_add=""
#runtimesim -> no wait!
nodes=2
low_time="2:30:00"
mid_time="5:45:00"
max_time="12:00:00"
uq_method="saltelli"

start_larsim_uq_sim "DYNAMIC" "FCFS"  12  6 "$model" "$opt_add" 1 "MpiPoolSolver" "$nodes" "$max_time"
