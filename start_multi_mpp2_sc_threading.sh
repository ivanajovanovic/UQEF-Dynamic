#!/bin/bash

module load python/3.5_intel

export PYTHONPATH=$HOME/software/python/mpi4py.mpp2.git/build/lib.linux-x86_64-3.5:$PYTHONPATH

start_uq_sim(){
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
    echo "$counter:mpp2: $@"
    echo "$counter:mpp2: $@" >> started_jobs.txt

    #create env
    # init vars
    basePath=`pwd`
    baseSourcePath=$basePath
    baseExecutionPath=$basePath
    baseResultsPath=$basePath
    modelMasterPath=$basePath

    executionPath=$baseExecutionPath
    resultsPath=$baseResultsPath/saves/

    # init the stuff
    if [ "$strategy" == "FIXED_LINEAR" ] ; then
        cpus=20
        threads=$cpus
        tasks=1
    else
        cpus=1
        threads=$cpus
        tasks=20
    fi

#create batch file
echo "#!/bin/bash

# config

#SBATCH -o $baseSourcePath/larsim_uq.$counter.job.%j.%N.out
#SBATCH -D $baseSourcePath
#SBATCH -J larsim.$counter
#SBATCH --get-user-env
#SBATCH --cluster=mpp2
#SBATCH --nodes=$cluster_nodes
#SBATCH --cpus-per-task=$cpus
#SBATCH --ntasks-per-node=$tasks
#SBATCH --mem=55G
#SBATCH --exclusive
#SBATCH --mail-type=end
#SBATCH --mail-user=frank.schraufstetter@gmail.com
#SBATCH --export=NONE
#SBATCH --time=$time_limit

source /etc/profile.d/modules.sh
#module unload python
#module unload mpi4py

#module load java/1.8
#module load python/2.7_intel
module load python/3.5_intel

export OMP_NUM_THREADS=$threads


# start simulation
echo "---- start sim: \`date\`"

    python3 $executionPath/simulation.py \
                            -or $resultsPath \
                            --uq_method sc --sc_q_order $q_order --sc_p_order $p_order \
                            --model "$model" --uncertain "all" \
                            --chunksize 1 \
                            --num_cores=$threads --parallel

echo "---- end \$i: \`date\`"

" > uq_sc_mpp2_gen.cmd

    #execute batch file
    sbatch uq_sc_mpp2_gen.cmd
}

model="larsim"
opt_add=""

#runtimesim -> no wait!
nodes=1
low_time="2:50:00"
mid_time="5:45:00"
max_time="70:00:00"

start_uq_sim "FIXED_LINEAR"   " "  10  6 "$model" "$opt_add" 1 "new" "$nodes" "$max_time"
