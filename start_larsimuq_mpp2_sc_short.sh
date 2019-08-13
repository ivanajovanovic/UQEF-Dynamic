salloc --ntasks=16 --cpus-per-task=1 \
mpiexec -genv I_MPI_DEBUG=+5 -print-rank-map python3 uq_simulation.py \
                            --uq_method "mc" --mc_numevaluations 1000  \
                            --model "larsim" \
                            --chunksize 1 \
                            --mpi --mpi_method "new" \
                            --num_cores 16 \
                            --configurationsFile "configuration_larsim.json" \
                            --saltelli \
                            --run_statistics \
                            --outputResultDir "/naslx/projects/pr63so/ga45met2/Repositories/larsim_runs/" \

exit


salloc --ntasks=8 --cpus-per-task=1 \
mpiexec python3 uq_simulation.py --uq_method "mc" --mc_numevaluations 10 --sc_q_order 4 --sc_p_order 3 --model "larsim" --chunksize 1 --mpi --configurationsFile "configuration_larsim.json" --saltelli --run_statistics --outputResultDir "/naslx/projects/pr63so/ga45met2/Repositories/larsim_runs" --num_cores 8 \
 2>&1 | tee log_larims_trial_run.txt
