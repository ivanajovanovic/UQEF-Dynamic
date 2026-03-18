#!/bin/bash

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
    local regression_model_type="${19}"
    local kl_expansion_order="${20}"

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
    echo "$counter:cm4: $@"
    echo "$counter:cm4: $@" >> started_jobs.txt

    conda_env=my_uq_env

    # define paths
    basePath=$HOME/Repositories #'pwd'
    baseSourcePath=$basePath/UQEF-Dynamic
    baseExecutionPath=$basePath/UQEF-Dynamic
    baseResultsPath=$WORK/battery_runs #$SCRATCH/battery_runs
    modelMasterPath=/dss/dsshome1/lxc0C/ga45met2/.conda/envs/$conda_env/lib/python3.11/site-packages/pybamm/input/drive_cycles
    resultsPath=$baseResultsPath/battery_uq_cm4.$counter

    if [ "$sched_strut" = "SWPT" -o "$sched_strut" = "SWPT_OPT" ] ; then
        cpus=112
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

    partition="cm4_std" #"cm4_tiny" "cm4_inter" "teramem_inter"
    clusters="cm4"  #"inter"

#create batch file
echo "#!/bin/bash

# config

#SBATCH -e $baseExecutionPath/uq_simulation.$counter.job.%j.%N.out
#SBATCH -D $baseSourcePath
#SBATCH -J battery.$counter
#SBATCH --get-user-env
#SBATCH --clusters=$clusters
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
###--mem=55G

# load modules and activate the conda env
module load slurm_setup
module load stack/24.4.0
source /etc/profile.d/modules.sh
#module load intel/2025.2.0
#module load intel-mpi/2021.16.0
module load intel/2024.1.0
module load intel-mpi/2021.12.0
module load mpi.intel/2018

# Initialize conda and activate environment
# For older Python versions
# source /dss/dsshome1/lxc0C/ga45met2/.conda/envs/$conda_env/bin/activate $conda_env
# For newer Python versions
eval \"\$(/dss/lrzsys/sys/spack/release/23.1.0/opt/x86_64/anaconda3/2022.10-gcc-2f4y4xz/condabin/conda shell.bash hook)\"
conda activate $conda_env

# export num threads for OMP
export OMP_NUM_THREADS=$threads

# start simulation
echo "---- start Battery sim: \`date\`"

    mpiexec -n \$SLURM_NTASKS /dss/dsshome1/lxc0C/ga45met2/.conda/envs/$conda_env/bin/python $baseSourcePath/uqef_dynamic/scientific_pipelines/uq_simulation_uqsim.py \
                            --outputResultDir $resultsPath \
                            --inputModelDir "$modelMasterPath" \
                            --sourceDir $baseSourcePath \
                            --config_file $baseSourcePath/uqef_dynamic/models/pybamm/configuration_battery_24_shot_names.json \
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
                            --cross_truncation 0.7 \
                            --regression_model_type "$regression_model_type" \
                            --compute_kl_expansion_of_qoi \
                            --kl_expansion_order $kl_expansion_order \
                            $opt

echo "---- end Battery sim: \`date\`"

" > $baseSourcePath/battery_uq_mc_24d_10000_random_pce5_ct07_kl10.cmd

    #execute batch file
    sbatch $baseSourcePath/battery_uq_mc_24d_10000_random_pce5_ct07_kl10.cmd

}

model="battery"
opt_add="--regression --parallel_statistics --save_all_simulations --sampleFromStandardDist --compute_Sobol_m --compute_Sobol_t --sc_poly_normed --store_gpce_surrogate_in_stat_dict --save_gpce_surrogate --compute_other_stat_besides_pce_surrogate --compute_generalized_sobol_indices"
nodes=4
tasks_per_node=112  #22  112
low_time="2:30:00"
mid_time="24:00:00"
max_time="24:00:00"
uq_method="mc"
q_order=6
p_order=5
mc_numevaluations=10000
uc="all"
sampling_rule="random"
sc_poly_rule="three_terms_recurrence"
sc_quadrature_rule="p" # "clenshaw_curtis" "genz_keister_24" "p"
mpi_method="MpiPoolSolver"
parameters_file="/dss/dsshome1/lxc0C/ga45met2/Repositories/sparse_grid_nodes_weights/KPU_d24_l4.asc"
regression_model_type="LARS"
kl_expansion_order=10

start_uq_sim "DWP" "DYNAMIC" "FCFS" "$uq_method" $q_order $p_order $mc_numevaluations "$model" "$opt_add" "$mpi_method" "$nodes" "$tasks_per_node" "$max_time" "$uc" "$sampling_rule" "$sc_poly_rule" "$sc_quadrature_rule" "$parameters_file" "$regression_model_type" "$kl_expansion_order"
