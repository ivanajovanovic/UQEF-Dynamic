#!/bin/bash

#mpiexec -n 4 python3 uq_simulation.py --model "oscillator" --outputResultDir "./oscilator/" --uq_method "mc" --mc_numevaluations 1000 --mpi --configurationsFile "./configurations/configuration_oscillator.json" --saltelli --run_statistics

#mpiexec -n 4 python3 uq_simulation.py --model "oscillator" --outputResultDir "./oscilator/" --uq_method "mc" --mc_numevaluations 1000 --mpi --configurationsFile "./configurations/configuration_oscillator.json" --saltelli --run_statistics

#python3 uq_simulation.py --model "oscillator" --outputResultDir "./oscilator/" --uq_method "mc" --mc_numevaluations 1000 --configurationsFile "./configurations/configuration_oscillator.json" --saltelli --run_statistics

#mpiexec -n 4 python3 uq_simulation.py --model "ishigami" --outputResultDir "./ishigami/" --uq_method "mc" --mc_numevaluations 16384 --sc_p_order 8 --mpi --configurationsFile "./configurations/configuration_ishigami.json" --saltelli --run_statistics


#Product Function
#mpiexec -n 4 python3 uq_simulation.py --model "productFunction" --outputResultDir "./productFunction/" --uq_method "mc" --mc_numevaluations 16384 --mpi --configurationsFile "./configurations/configuration_product_function.json" --saltelli --run_statistics

#mpiexec -n 4 python3 uq_simulation.py --model "productFunction" --outputResultDir "./productFunction/" --uq_method "sc" --sc_q_order 20 --sc_p_order 7 --mpi --configurationsFile "./configurations/configuration_product_function.json"--run_statistics

mpiexec -n 4 python3 "./uq_simulation_uqsim.py" \
                        --outputResultDir "/import/home/ga45met/Repositories/Larsim/UQEF-Dynamic/trial_larsim_run" \
                        --inputModelDir "/import/home/ga45met/Repositories/Larsim/Larsim-data" \
                        --sourceDir "." \
                        --model "larsim" \
                        --uncertain "all" \
                        --chunksize 1 \
                        --num_cores=1 \
                        --opt_strategy "DYNAMIC" \
                        --opt_algorithm "FCFS" \
                        --mpi \
                        --mpi_method "MpiPoolSolver" \
                        --config_file "./configurations_Larsim/configuration_larsim_uqsim_cm2_v4.json" \
                        --uq_method "saltelli" \
                        --mc_numevaluations 5 \
                        --sampling_rule "S" \
                        --transformToStandardDist \
                        --uqsim_store_to_file
                        #2>&1 | tee log_larims_trial_run.txt
