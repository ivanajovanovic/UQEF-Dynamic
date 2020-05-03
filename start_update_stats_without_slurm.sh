#!/bin/bash

# load modules
module load slurm_setup
module load python/3.6_intel
module unload intel-mpi
module load intel-mpi/2018.4.274

# uqef
export PYTHONPATH=$HOME/work/simulation/diss.larsim/uqef.git/src:$PYTHONPATH

# load python conda env
source activate larsim_uq
conda info -e
conda list

start_update_stats(){
    local id="$1"
    local cluster_nodes="$2"
    local time_limit="$3"

    #create env
    # init vars
    baseSourcePath="$(pwd)"
    baseExecutionPath="$baseSourcePath"
    resultsPath="$baseSourcePath/../larsim_runs"
    
    resultIdPath="$resultsPath/${id}_sim"
    executionPath="$baseExecutionPath"

    if [ ! -d "$resultIdPath" ]; then
        echo "$resultIdPath doesn't exist."
        exit 1
    fi

    if [ ! -d "$executionPath" ]; then
        echo "$executionPath doesn't exist."
        exit 1
    fi

    echo "---- start sim: \`date\`"

    python "$executionPath/update_plots_uqsim.py" --dir "$resultIdPath"

    echo "---- end sim: \`date\`"
}

#daily output
#start_update_stats 0121 1 "0:30:00"
#start_update_stats 0122 1 "0:10:00"
#start_update_stats 0123 1 "1:00:00"
#start_update_stats 0124 1 "2:00:00"
#start_update_stats 0125 1 "6:30:00"
#start_update_stats 0127 1 "24:00:00"
start_update_stats 0133 1 "48:00:00"
#start_update_stats 0134 1 "48:00:00"
#start_update_stats 0100 1 "5:40:00"
#start_update_stats 0101 1 "7:30:00"
#start_update_stats 0102 1 "10:00:00"
#start_update_stats 0009 1 "3:00:00"

#hourly output
#start_update_stats 0121 1 "0:30:00"
#start_update_stats 0122 1 "3:00:00"
#start_update_stats 0123 1 "9:00:00"
#start_update_stats 0124 1 "6:00:00"
#start_update_stats 0125 1 "12:30:00"
#start_update_stats 0127 1 "24:00:00"
#start_update_stats 0133 1 "48:00:00"
#start_update_stats 0134 1 "48:00:00"
#start_update_stats 0100 1 "5:40:00"
#start_update_stats 0101 1 "7:30:00"
#start_update_stats 0102 1 "10:00:00"
#start_update_stats 0009 1 "3:00:00"

