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
