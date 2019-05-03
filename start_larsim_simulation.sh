#!/bin/bash

#export PYTHONPATH=$PYTHONPATH:.

python_cmd = python3
#python_cmd=python2
#python_cmd=python3
#module unload python
#module load python/3.5_intel
#pip install --user -r

# Linear Solver - SC
#pytho3 simulation.py \
#                                    --model "larsim" \
#                                    --uq_method "sc" --sc_q_order 10 --sc_p_order 6 \
#                                    --uncertain "all"


#Parallel Solver - SC
#python3 simulation.py \
#                                   --model "larsim" \
#                                   --uq_method "sc" --sc_q_order 3 --sc_p_order 2 \
#                                   --uncertain "all" \
#                                   --parallel


# MpiPoolSolver - SC
 mpiexec -n 4 python3 uq_simulation.py \
                                     --model "larsim" \
                                     #--uq_method "sc" --sc_q_order 24 --sc_p_order 6 \
                                     --uq_method "sc" --sc_q_order 3 --sc_p_order 2 \
                                     --uncertain "all" \
                                     --mpi

