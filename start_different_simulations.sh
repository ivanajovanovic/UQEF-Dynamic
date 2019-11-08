#!/bin/bash

#mpiexec -n 4 python3 uq_simulation.py --model "oscillator" --outputResultDir "./oscilator/" --uq_method "mc" --mc_numevaluations 1000 --mpi --configurationsFile "./configurations/configuration_oscillator.json" --saltelli --run_statistics

mpiexec -n 4 python3 uq_simulation.py --model "oscillator" --outputResultDir "./oscilator/" --uq_method "mc" --mc_numevaluations 1000 --mpi --configurationsFile "./configurations/configuration_oscillator.json" --saltelli --run_statistics

#python3 uq_simulation.py --model "oscillator" --outputResultDir "./oscilator/" --uq_method "mc" --mc_numevaluations 1000 --configurationsFile "./configurations/configuration_oscillator.json" --saltelli --run_statistics

#mpiexec -n 4 python3 uq_simulation.py --model "ishigami" --outputResultDir "./ishigami/" --uq_method "mc" --mc_numevaluations 16384 --sc_p_order 8 --mpi --configurationsFile "./configurations/configuration_ishigami.json" --saltelli --run_statistics


#Product Function
#mpiexec -n 4 python3 uq_simulation.py --model "productFunction" --outputResultDir "./productFunction/" --uq_method "mc" --mc_numevaluations 16384 --mpi --configurationsFile "./configurations/configuration_product_function.json" --saltelli --run_statistics

#mpiexec -n 4 python3 uq_simulation.py --model "productFunction" --outputResultDir "./productFunction/" --uq_method "sc" --sc_q_order 20 --sc_p_order 7 --mpi --configurationsFile "./configurations/configuration_product_function.json"--run_statistics


