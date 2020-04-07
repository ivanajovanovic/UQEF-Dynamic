#!/bin/bash

#salloc --nodes=2 --cpus-per-task=1 --time=02:00:00 \
##or
#salloc -n 28 -t 120 \
##or
#salloc --ntasks=16 --cpus-per-task=1 --time=01:00:00 \
#mpiexec -genv I_MPI_DEBUG=+5 -print-rank-map python3 uq_simulation.py \
#                            --uq_method "mc" --mc_numevaluations 100 \
#                            --model "larsim" \
#                            --chunksize 1 \
#                            --mpi --mpi_method "new" \
#                            --num_cores 28 \
#                            --configurationsFile "./configurations/configuration_larsim_v4.json" \
#                            --saltelli \
#                            --run_statistics \
#                            --outputResultDir "/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/Repositories/larsim_runs" \
#2>&1 | tee log_larims_trial_run.txt \
#exit

##I've put --num_cores 16 \

#mpiexec -genv I_MPI_DEBUG=+5 -print-rank-map python3 uq_simulation.py --uq_method "mc" --mc_numevaluations 100 \
#                        --model "larsim" --chunksize 1 --mpi --mpi_method "new" --num_cores 16 --configurationsFile "./configurations/configuration_larsim_v4.json" \
#                        --saltelli --run_statistics --outputResultDir "/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/Repositories/larsim_runs" \
#2>&1 | tee log_larims_trial_run.txt

#salloc -n 28 -t 120
#mpiexec -genv I_MPI_DEBUG=+5 -print-rank-map python3 uq_simulation.py \
#                            --model "larsim" --chunksize 1 --mpi --mpi_method "new" --num_cores 28 \
#                            --uq_method "sc" --sc_q_order 5 --sc_p_order 3 \
#                            --run_statistics \
#                            --configurationsFile "./configurations/configuration_larsim_v4.json" --outputResultDir "/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/Repositories/larsim_runs" \
#2>&1 | tee log_larims_trial_run.txt \


#salloc -n 28 -t 120
#mpiexec -genv I_MPI_DEBUG=+5 -print-rank-map python3 uq_simulation.py \
#                            --model "larsim" --chunksize 1 --mpi --mpi_method "new" --num_cores 28 \
#                            --run_statistics \
#                            --configurationsFile "./configurations/configuration_larsim_v4.json" --outputResultDir "/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/Repositories/larsim_runs" \
#                            --uq_method "sc" --sc_q_order 3 --sc_p_order 2 \
#                            --transformToStandardDist \
#2>&1 | tee log_larims_trial_run.txt

salloc -n 28 -t 120
mpiexec -genv I_MPI_DEBUG=+5 -print-rank-map python uq_simulation_uqsim.py \
                        --outputResultDir "/gpfs/scratch/pr63so/ga45met2/Larsim_runs" \
                        --model "larsim" \
                        --chunksize 1 \
                        --mpi --mpi_method "MpiPoolSolver" \
                        --config_file "/dss/dsshome1/lxc0C/ga45met2/Repositories/Larsim-UQ/configuration_larsim_uqsim_cm2.json" \
                        --uq_method "saltelli"  \
                        --mc_numevaluations 50 \
                        --sampling_rule "S" \
                        --transformToStandardDist \
                        2>&1 | tee log_larims_trial_run.txt
