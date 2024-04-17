#!/bin/bash

#export PYTHONPATH=$HOME/software/python/mpi4py.mpp2.git/build/lib.linux-x86_64-3.5:$PYTHONPATH
#load modules
module load python/3.6_intel
#cm2
if [[ $HOSTNAME  == "mpp3"* ]]; then
  module unload mpi.intel/2019
  module load mpi.intel/2020
elif [[ $HOSTNAME  == "cm2"* ]]; then
  module unload intel-mpi
  module load intel-mpi/2019-intel
fi

start_uq_sim(){
    local sched_strut="$1"
    local strategy="$2"
    local algorithm="$3"
    local uq_method="$4"
    local q_order="$5"
    local p_order="$6"
    local mc_numevaluations="$7"
    local model="$8"
    local opt="$9"
    local mpi_method="${10}"
    local cluster_nodes="${11}"
    local tasks_per_node="${12}"
    local time_limit="${13}"
    local uncertain="${14}"
    local sampling_rule="${15}"
    local sc_poly_rule="${16}"
    local sc_quadrature_rule="${17}"
    local parameters_file="${18}"
    local parameters_setup_file="${19}"

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
    echo "$counter:cm2: $@"
    echo "$counter:cm2: $@" >> started_jobs.txt

    # define paths
    basePath=$HOME/Repositories #'pwd'
    baseSourcePath=$basePath/UQEF-Dynamic
    baseExecutionPath=$basePath/UQEF-Dynamic-on-cluster
    baseResultsPath=$SCRATCH/hbvsask_runs
    modelMasterPath=$WORK/HBV-SASK-data
    executionPath=$baseExecutionPath/hbv_uq_cm2.$counter
    resultsPath=$baseResultsPath/hbv_uq_cm2.$counter

    if [ "$sched_strut" = "SWPT" -o "$sched_strut" = "SWPT_OPT" ] ; then
        cpus=28
        threads=$cpus
        tasks=1
    else
        cpus=1
        threads=$cpus
        tasks=$tasks_per_node
    fi

    #let ntasks=$tasks*$cluster_nodes
    ntasks=$(($tasks * $cluster_nodes))
    echo $ntasks

    # TODO
    if [ $cluster_nodes -lt 4 ]; then
        partition="cm2_std" #"cm2_std"
    else
        partition="cm2_std" # cm2_large
    fi

#create batch file
echo "#!/bin/bash

# config

#SBATCH -e $baseExecutionPath/uq_simulation.$counter.job.%j.%N.out
#SBATCH -D $baseSourcePath
#SBATCH -J hbv.$counter
#SBATCH --get-user-env
#SBATCH --clusters=cm2
#SBATCH --partition=$partition
#SBATCH --qos=$partition
#SBATCH --nodes=$cluster_nodes
#SBATCH --cpus-per-task=$cpus
#SBATCH --ntasks-per-node=$tasks
#SBATCH --mail-type=end
#SBATCH --mail-user=ivana.jovanovic@tum.de
#SBATCH --export=NONE
#SBATCH --time=$time_limit
#SBATCH --exclusive
#SBATCH --mem=55G ###$MaxMemPerNode

# load modules and activate the conda env
module load slurm_setup
source /etc/profile.d/modules.sh
# module load python/3.6_intel
module load anaconda3
if [[ $HOSTNAME  == "mpp3"* ]]; then
  module unload mpi.intel/2019
  module load mpi.intel/2020
elif [[ $HOSTNAME  == "cm2"* ]]; then
  module unload intel-mpi
  module load intel-mpi/2019.8.254 #intel-mpi/2018-intel
fi
source /dss/dsshome1/lxc0C/ga45met2/.conda/envs/uq_env/bin/activate uq_env

# export num threads for OMP
export OMP_NUM_THREADS=$threads

# start simulation
# start simulation
echo "---- start HBV sim: \`date\`"

    mpiexec -n \$SLURM_NTASKS python $baseSourcePath/uq_simulation_uqsim.py \
                            --outputResultDir $resultsPath \
                            --inputModelDir $modelMasterPath \
                            --sourceDir $baseSourcePath \
                            --config_file $baseSourcePath/configurations/configuration_hbv_7D.json \
                            --model "$model" \
                            --uncertain "$uncertain" \
                            --opt_strategy "$strategy" --opt_algorithm "$algorithm" \
                            --chunksize 1 \
                            --num_cores $threads \
                            --mpi \
                            --mpi_method "$mpi_method" \
                            --uq_method "$uq_method" \
                            --sc_q_order $q_order \
                            --sc_p_order $p_order \
                            --mc_numevaluations $mc_numevaluations \
                            --sampling_rule "$sampling_rule" \
                            --sc_poly_rule "$sc_poly_rule" \
                            --sc_quadrature_rule "$sc_quadrature_rule" \
                            --parameters_file "$parameters_file" \
                            --parameters_setup_file "$parameters_setup_file" \
                            $opt

echo "---- end HBV sim: \`date\`"

" > $baseSourcePath/hbv_uq_cm2_sc_sparse_kpu_l_7_d_7_p_3.cmd

    #execute batch file
    sbatch $baseSourcePath/hbv_uq_cm2_sc_sparse_kpu_l_7_d_7_p_3.cmd

}

model="hbvsask"
opt_add="--sc_poly_normed --store_gpce_surrogate_in_stat_dict --sc_sparse_quadrature --parallel_statistics --compute_Sobol_t --compute_Sobol_m --sampleFromStandardDist --read_nodes_from_file --instantly_save_results_for_each_time_step"  # "--disable_statistics" "--sc_poly_normed"
nodes=4
tasks_per_node=10
low_time="2:30:00"
mid_time="6:00:00"
max_time="48:00:00"
uq_method="sc"
q_order=7
p_order=3
mc_numevaluations=10000
uc="all"
sampling_rule="latin_hypercube"
sc_poly_rule="three_terms_recurrence"
sc_quadrature_rule="p" # "clenshaw_curtis" "genz_keister_24" "p"
mpi_method="MpiPoolSolver"
parameters_file="/dss/dsshome1/lxc0C/ga45met2/Repositories/sparse_grid_nodes_weights/KPU_d7_l7.asc"
parameters_setup_file="/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic/configurations/KPU_HBV_d7.json"

# start_uq_sim "DWP" "DYNAMIC" "FCFS" saltelli 0 0 50 "$model" "$opt_add" "MpiPoolSolver" "$nodes" "$max_time" "$uc" "$sampling_rule"
# start_uq_sim "DWP" "DYNAMIC" "FCFS" "$uq_method" 20 10 50 "$model" "$opt_add" "MpiPoolSolver" "$nodes" "$max_time" "$uc" "$sampling_rule"
start_uq_sim "DWP" "DYNAMIC" "FCFS" "$uq_method" $q_order $p_order $mc_numevaluations "$model" "$opt_add" "$mpi_method" "$nodes" "$tasks_per_node" "$mid_time" "$uc" "$sampling_rule" "$sc_poly_rule" "$sc_quadrature_rule" "$parameters_file" "$parameters_setup_file"
