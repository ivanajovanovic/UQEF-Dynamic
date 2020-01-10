#!/bin/bash

module load python/3.6_intel

export PYTHONPATH=$HOME/work/simulation/diss.larsim/uqef.git/src:$PYTHONPATH

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

    cpus=28
    threads=$cpus
    tasks=1

#create batch file
echo "#!/bin/bash

# config

#SBATCH -o $baseSourcePath/st.$id.job.%j.%N.out
#SBATCH -D $baseSourcePath
#SBATCH -J st.$id
#SBATCH --get-user-env
#SBATCH --cluster=mpp2
#SBATCH --nodes=$cluster_nodes
#SBATCH --cpus-per-task=$cpus
#SBATCH --ntasks-per-node=$tasks
#SBATCH --mem=55G
#SBATCH --exclusive
#SBATCH --mail-type=end
#SBATCH --mail-user=florian.kuenzner@tum.de
#SBATCH --export=NONE
#SBATCH --time=$time_limit

source /etc/profile.d/modules.sh
module load python/3.6_intel
module unload mpi.intel
module load mpi.intel/2018

export OMP_NUM_THREADS=$threads

export CLASSPATH=$CLASSPATH
export PYTHONPATH=$HOME/software/python/mpi4py.mpp2.git/build/lib.linux-x86_64-3.5:$PYTHONPATH
export PYTHONPATH=$HOME/software/python/chaospy_python3.git:$PYTHONPATH

# start simulation
echo "---- start sim: \`date\`"

python "$executionPath/update_plots_uqsim.py" --dir "$resultIdPath"

echo "---- end sim: \`date\`"

" > update_stats_gen.cmd

    #execute batch file
    sbatch update_stats_gen.cmd
}

#DWP
#start_update_stats 0121 1 "0:05:00"
#start_update_stats 0122 1 "0:10:00"
#start_update_stats 0123 1 "1:00:00"
#start_update_stats 0124 1 "2:00:00"
#start_update_stats 0125 1 "6:30:00"
#start_update_stats 0127 1 "24:00:00"
start_update_stats 0133 1 "48:00:00"
start_update_stats 0134 1 "48:00:00"
#start_update_stats 0100 1 "5:40:00"
#start_update_stats 0101 1 "7:30:00"
#start_update_stats 0102 1 "10:00:00"
#start_update_stats 0009 1 "3:00:00"


