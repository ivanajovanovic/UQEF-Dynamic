#!/bin/bash

#export PYTHONPATH=$PYTHONPATH:.

#python_cmd = python3
#python_cmd=python2
#python_cmd=python3
#module unload python
#module load python/3.5_intel
#pip install --user -r

# Linear Solver - SC
#pytho3 uq_simulation.py \
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

mpiexec -n 4 python3 uq_simulation.py \
                                     --model "larsim" \
                                     --uq_method "mc" --mc_numevaluations 100 --sc_p_order 8 \
                                     --uncertain "all" \
                                     --mpi \
                                     --regression
