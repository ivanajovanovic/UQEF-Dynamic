#!/bin/bash

#export PYTHONPATH=$PYTHONPATH:.

#export PYTHONPATH=/data/repos/repos_tum/chaospy_run.git:$PYTHONPATH
#export PYTHONPATH=/data/repos/repos_tum/uqef.git/src:$PYTHONPATH

#python_cmd = python3
#python_cmd=python2
#python_cmd=python3
#module unload python
#module load python/3.5_intel
#pip install --user -r

# Linear Solver - SC
#python3 uq_simulation.py \
#                                    --model "larsim" \
#                                    --uq_method "sc" --sc_q_order 10 --sc_p_order 6 \
#                                    --uncertain "all"

#python3 uq_simulation.py \
#                                     --model "larsim" \
#                                     --uq_method "sc" --sc_q_order 3 --sc_p_order 1 \
#                                     --uncertain "all" \
#                                     --regression

#python3 uq_simulation.py \
#                                     --model "larsim" \
#                                     --uq_method "mc" --mc_numevaluations 1000 --sc_p_order 3 \
#                                     --uncertain "all" \
#                                     --regression

#Parallel Solver - SC
#python3 uq_simulation.py \
#                                   --model "larsim" \
#                                   --uq_method "sc" --sc_q_order 3 --sc_p_order 2 \
#                                   --uncertain "all" \
#                                   --parallel


# MpiPoolSolver - SC
#mpiexec -n 4 python3 uq_simulation.py \
#                                     --model "larsim" \
#                                     --uq_method "sc" --sc_q_order 3 --sc_p_order 1 \
#                                     --uncertain "all" \
#                                     --mpi \
#                                     --regression


###########################################################################################

#mpiexec -n 4 python3 uq_simulation.py \
#                                     --model "larsim" \
#                                     --uq_method "mc" --mc_numevaluations 100 --sc_p_order 8 \
#                                     --uncertain "all" \
#                                     --mpi \
#                                     --regression

#pyfile="uq_simulation.py"
pyfile="uq_simulation_uqsim.py"

model="larsim"
#model="oscillator"
#model="ishigami"
#model="productFunction"

config_file="configuration_larsim.json"
#config_file="configuration_oscillator.json"
#config_file="configuration_ishigami.json"
#config_file="configuration_product_function.json"

uq_method="mc"
#uq_method="sc"
#uq_method="saltelli"

mpiexec -n 4 python3 $pyfile \
                                     --model "$model" \
                                     --uq_method "$uq_method" --mc_numevaluations 10 --sc_q_order 1 --sc_p_order 1\
                                     --outputResultDir "./larsim_runs/" \
                                     --mpi \
                                     --config_file "$config_file"

#mpiexec -n 4 python3 $pyfile \
#                                     --model "oscillator" \
#                                     --uq_method "mc" --mc_numevaluations 10 \
#                                     --outputResultDir "./larsim_runs/" \
#                                     --mpi \
#                                     --config_file "configuration_larsim.json"


#mpiexec -n 4 python3 uq_simulation.py \
#                                     --model "larsim" \
#                                     --uq_method "mc" --mc_numevaluations 10 --sc_p_order 8 \
#                                     --outputResultDir "./larsim_runs/" \
#                                     --mpi \
#                                     --configurationsFile "configuration_larsim.json" \
#                                     --saltelli \
#                                     --run_statistics
