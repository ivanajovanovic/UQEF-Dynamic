salloc --nodes=2 --cpus-per-task=1 --time=02:00:00\
salloc -n 28 -t 120
salloc --ntasks=16 --cpus-per-task=1 --time=01:00:00\
mpiexec -genv I_MPI_DEBUG=+5 -print-rank-map python3 uq_simulation.py \
                            --uq_method "mc" --mc_numevaluations 100 \
                            --model "larsim" \
                            --chunksize 1 \
                            --mpi --mpi_method "new" \
                            --num_cores 28 \
                            --configurationsFile "configuration_larsim.json" \
                            --saltelli \
                            --run_statistics \
                            --outputResultDir "/naslx/projects/pr63so/ga45met2/Repositories/larsim_runs" \
2>&1 | tee log_larims_trial_run.txt \
exit

#I've put put --num_cores 16 \

mpiexec -genv I_MPI_DEBUG=+5 -print-rank-map python3 uq_simulation.py --uq_method "mc" --mc_numevaluations 100 \
                        --model "larsim" --chunksize 1 --mpi --mpi_method "new" --num_cores 16 --configurationsFile "configuration_larsim.json" \
                        --saltelli --run_statistics --outputResultDir "/naslx/projects/pr63so/ga45met2/Repositories/larsim_runs" \
2>&1 | tee log_larims_trial_run.txt

salloc -n 28 -t 120
mpiexec -genv I_MPI_DEBUG=+5 -print-rank-map python3 uq_simulation.py \
                            --model "larsim" --chunksize 1 --mpi --mpi_method "new" --num_cores 28 \
                            --uq_method "sc" --sc_q_order 5 --sc_p_order 3 \
                            --run_statistics \
                            --configurationsFile "configuration_larsim_sc.json" --outputResultDir "/naslx/projects/pr63so/ga45met2/Repositories/larsim_runs" \
2>&1 | tee log_larims_trial_run.txt \


salloc -n 28 -t 120
mpiexec -genv I_MPI_DEBUG=+5 -print-rank-map python3 uq_simulation.py \
                            --model "larsim" --chunksize 1 --mpi --mpi_method "new" --num_cores 28 \
                            --uq_method "sc" --sc_q_order 18 --sc_p_order 6 \
                            --run_statistics \
                            --configurationsFile "configuration_larsim_v4.json" --outputResultDir "/naslx/projects/pr63so/ga45met2/Repositories/larsim_runs" \
2>&1 | tee log_larims_trial_run.txt
