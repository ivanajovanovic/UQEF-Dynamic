salloc --ntasks=16 --cpus-per-task=1 \
mpiexec -genv I_MPI_DEBUG=+5 -print-rank-map python3 uq_simulation.py \
                            --uq_method "sc" --sc_q_order 4 --sc_p_order 3 \
                            --model "larsim" --uncertain "all" \
                            --chunksize 1 \
                            --mpi --mpi_method "new" \
                            --num_cores 16
exit
