import sys
from collections import defaultdict
from collections import defaultdict
import more_itertools
from mpi4py import MPI
import mpi4py.futures as futures
import pickle
import numpy as np
import scipy
import os

def _my_parallel_calc_stats_for_mc_saltelli(keyIter_chunk, qoi_values_chunk, numEvaluations, dim, compute_Sobol_t=False,
                                            compute_Sobol_m=False, store_qoi_data_in_stat_dict=False):
    results = []
    for ip in range(0, len(keyIter_chunk)):  # for each peace of work
        key = keyIter_chunk[ip]
        qoi_values = qoi_values_chunk[ip]
        local_result_dict = dict()

        qoi_values_saltelli = qoi_values[:, np.newaxis]
        standard_qoi_values = qoi_values_saltelli[:numEvaluations, :]
        extended_standard_qoi_values = qoi_values_saltelli[:(2 * numEvaluations), :]

        if store_qoi_data_in_stat_dict:
            local_result_dict["qoi_values"] = standard_qoi_values

        local_result_dict["E"] = np.mean(qoi_values[:numEvaluations], 0)
        local_result_dict["Var"] = np.sum((standard_qoi_values - local_result_dict["E"]) ** 2,
                                          axis=0, dtype=np.float64) / (numEvaluations - 1)
        # local_result_dict["Var"] = np.sum((qoi_values[:numEvaluations] - local_result_dict["E"]) ** 2,
        #                                    axis=0, dtype=np.float64)/(numEvaluations - 1)
        local_result_dict["Skew"] = scipy.stats.skew(qoi_values[:numEvaluations], axis=0, bias=True)
        local_result_dict["Kurt"] = scipy.stats.kurtosis(qoi_values[:numEvaluations], axis=0, bias=True)

        local_result_dict["StdDev"] = np.std(qoi_values[:numEvaluations], 0, ddof=1)

        local_result_dict["P10"] = np.percentile(qoi_values[:numEvaluations], 10, axis=0)
        local_result_dict["P90"] = np.percentile(qoi_values[:numEvaluations], 90, axis=0)

        if isinstance(local_result_dict["P10"], list) and len(local_result_dict["P10"]) == 1:
            local_result_dict["P10"] = local_result_dict["P10"][0]
            local_result_dict["P90"] = local_result_dict["P90"][0]

        if compute_Sobol_t:
            local_result_dict["Sobol_t"] = saltelliSobolIndicesHelpingFunctions._Sens_t_sample(
                qoi_values_saltelli, dim, numEvaluations, code=4)
        if compute_Sobol_m:
            local_result_dict["Sobol_m"] = saltelliSobolIndicesHelpingFunctions._Sens_m_sample(
                qoi_values_saltelli, dim, numEvaluations, code=4)

        results.append([key, local_result_dict])
    return results


def _process_chunk_result_single_qoi_single_time_step(self, single_qoi_column, timestamp, result_dict):
    result_dict.update({'qoi': single_qoi_column})
    if self.instantly_save_results_for_each_time_step:
        fileName = f"statistics_dictionary_{single_qoi_column}_{timestamp}.pkl"
        fullFileName = os.path.abspath(os.path.join(str(self.workingDir), fileName))
        with open(fullFileName, 'wb') as handle:
            pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        self.result_dict[single_qoi_column][timestamp] = result_dict


def _save_plot_and_clear_result_dict_single_qoi(self, single_qoi_column):
    if self.instantly_save_results_for_each_time_step:
        return

    # In this case the results where collected in the self.result_dict dict and can be saved and plotted
    # Saving Stat Dict for a single qoi as soon as it is computed/ all time steps are processed
    fileName = "statistics_dictionary_qoi_" + single_qoi_column + ".pkl"
    fullFileName = os.path.abspath(os.path.join(str(self.workingDir), fileName))
    with open(fullFileName, 'wb') as handle:
        pickle.dump(self.result_dict[single_qoi_column], handle, protocol=pickle.HIGHEST_PROTOCOL)

    self.plotResults_single_qoi(
        single_qoi_column=single_qoi_column,
        dict_what_to_plot=self.dict_what_to_plot
    )
    self.result_dict[single_qoi_column].clear()


def compute_chunk_results(chunk):
    key, simulations_df, numEvaluations, dim, compute_Sobol_t, compute_Sobol_m, store_qoi_data_in_stat_dict = chunk

    # Calculate statistics for the chunk and return the result
    results = []
    for timestamp, single_qoi_column in zip(key, simulations_df):
        result_dict = {}  # Create an empty result_dict for each iteration
        # Perform your calculations here and populate result_dict
        _my_parallel_calc_stats_for_mc_saltelli()
        # Append the timestamp and result_dict to the results list
        results.append((timestamp, result_dict))

    return results

def calcStatisticsForSaltelliParallel(chunksize=1, regression=False, *args, **kwargs):
    self.result_dict = defaultdict(dict)

    if self.rank == 0:
        grouped = self.samples.df_simulation_result.groupby([self.time_column_name, ])
        groups = grouped.groups
        keyIter = list(groups.keys())

    for single_qoi_column in self.list_qoi_column:
        if self.rank == 0:
            list_of_simulations_df = [
                self.samples.df_simulation_result.loc[groups[key].values][single_qoi_column].values
                for key in keyIter
            ]

            keyIter_chunk = list(more_itertools.chunked(keyIter, chunksize))
            list_of_simulations_df_chunk = list(more_itertools.chunked(list_of_simulations_df, chunksize))

            numEvaluations_chunk = [self.numEvaluations] * len(keyIter_chunk)
            dimChunks = [self.dim] * len(keyIter_chunk)

            compute_Sobol_t_Chunks = [self._compute_Sobol_t] * len(keyIter_chunk)
            compute_Sobol_m_Chunks = [self._compute_Sobol_m] * len(keyIter_chunk)
            store_qoi_data_in_stat_dict_Chunks = [self.store_qoi_data_in_stat_dict] * len(keyIter_chunk)

        # Create a pool of worker processes
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            chunk_results = pool.map(
                _my_parallel_calc_stats_for_mc_saltelli,
                zip(keyIter_chunk, list_of_simulations_df_chunk, numEvaluations_chunk, dimChunks,
                    compute_Sobol_t_Chunks, compute_Sobol_m_Chunks, store_qoi_data_in_stat_dict_Chunks)
            )

        # Process and save the results as before
        for chunk_result in chunk_results:
            for result in chunk_result:
                self._process_chunk_result_single_qoi_single_time_step(
                    single_qoi_column, timestamp=result[0], result_dict=result[1])
                if self.instantly_save_results_for_each_time_step:
                    del result[1]

        self._save_plot_and_clear_result_dict_single_qoi(single_qoi_column)

        # with futures.MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
        #     if executor is not None:  # master process
        #         chunk_results_it = executor.map(_parallel_calc_stats_for_mc_saltelli,
        #                                         keyIter_chunk,
        #                                         list_of_simulations_df_chunk,
        #                                         numEvaluations_chunk,
        #                                         dimChunks,
        #                                         compute_Sobol_t_Chunks,
        #                                         compute_Sobol_m_Chunks,
        #                                         store_qoi_data_in_stat_dict_Chunks,
        #                                         chunksize=self.mpi_chunksize,
        #                                         unordered=self.unordered)
        #         sys.stdout.flush()
        #         executor.shutdown(wait=True)
        #         sys.stdout.flush()
        #
        #         chunk_results = list(chunk_results_it)
        #         for chunk_result in chunk_results:
        #             for result in chunk_result:
        #                 self._process_chunk_result_single_qoi_single_time_step(
        #                     single_qoi_column, timestamp=result[0], result_dict=result[1])
        #                 if self.instantly_save_results_for_each_time_step:
        #                     del result[1]
        #         del chunk_results_it
        #         del chunk_results
        #
                # if self.rank == 0:
        #         self._save_plot_and_clear_result_dict_single_qoi(single_qoi_column)

        # def calcStatisticsForSc(self, regression=False, *args, **kwargs):
        #     self._groupby_df_simulation_results()
        #     # keyIter = list(self.groups.keys())
        #
        #     single_qoi_column = "Q"
        #     self.result_dict = defaultdict(dict)
        #
        #     print(f"computation of statistics for qoi {single_qoi_column} started...")
        #     solver_time_start = time.time()
        #     print(f"computation for qoi {single_qoi_column} - waits for shutdown...")
        #     sys.stdout.flush()
        #     print(f"computation for qoi {single_qoi_column} - shut down...")
        #     sys.stdout.flush()
        #
        #     for key, val_indices in self.groups.items():
        #         qoi_values = self.samples.df_simulation_result.loc[val_indices.values][single_qoi_column].values
        #         self.result_dict[key] = dict()
        #         if self.store_qoi_data_in_stat_dict:
        #             self.result_dict[key]["qoi_values"] = qoi_values
        #         if regression:
        #             qoi_gPCE = cp.fit_regression(self.polynomial_expansion, self.nodes, qoi_values)
        #         else:
        #             qoi_gPCE = cp.fit_quadrature(self.polynomial_expansion, self.nodes, self.weights, qoi_values)
        #         self._calc_stats_for_gPCE_single_qoi(self.dist, single_qoi_column, key, qoi_gPCE)
        #         if self.instantly_save_results_for_each_time_step:
        #             self._save_statistics_dictionary_single_qoi_single_timestamp(
        #                 single_qoi_column=single_qoi_column, timestamp=key,
        #                 result_dict=self.result_dict[key]
        #             )
        #     solver_time_end = time.time()
        #     solver_time = solver_time_end - solver_time_start
        #     print(f"solver_time for qoi {single_qoi_column}: {solver_time}")
        #
        #     self._save_plot_and_clear_result_dict_single_qoi(single_qoi_column)


# from mpi4py import MPI
# import mpi4py.futures as futures
# from common import parallelStatistics
#
# def calcStatisticsForScParallel(self, chunksize=1, regression=False, *args, **kwargs):
#     self.result_dict = defaultdict(dict)
#
#     if self.rank == 0:
#         self._groupby_df_simulation_results()
#         keyIter = list(self.groups.keys())
#
#     single_qoi_column = "QoI"
#     if self.rank == 0:
#         list_of_simulations_df = [
#             self.samples.df_simulation_result.loc[self.groups[key].values][single_qoi_column].values
#             for key in keyIter
#         ]
#         keyIter_chunk = list(more_itertools.chunked(keyIter, chunksize))
#         list_of_simulations_df_chunk = list(more_itertools.chunked(list_of_simulations_df, chunksize))
#
#         nodesChunks = [self.nodes] * len(keyIter_chunk)
#         distChunks = [self.dist] * len(keyIter_chunk)
#         weightsChunks = [self.weights] * len(keyIter_chunk)
#         polynomial_expansionChunks = [self.polynomial_expansion] * len(keyIter_chunk)
#
#         regressionChunks = [regression] * len(keyIter_chunk)
#         compute_Sobol_t_Chunks = [self._compute_Sobol_t] * len(keyIter_chunk)
#         compute_Sobol_m_Chunks = [self._compute_Sobol_m] * len(keyIter_chunk)
#         store_qoi_data_in_stat_dict_Chunks = [self.store_qoi_data_in_stat_dict] * len(keyIter_chunk)
#         store_gpce_surrogate_Chunks = [self.store_gpce_surrogate] * len(keyIter_chunk)
#         save_gpce_surrogate_Chunks = [self.save_gpce_surrogate] * len(keyIter_chunk)
#
#     with futures.MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
#         if executor is not None:
#             print(f"{self.rank}: computation of statistics for qoi {single_qoi_column} started...")
#             solver_time_start = time.time()
#             chunk_results_it = executor.map(
#                 parallelStatistics._parallel_calc_stats_for_gPCE,
#                 keyIter_chunk,
#                 list_of_simulations_df_chunk,
#                 distChunks,
#                 polynomial_expansionChunks,
#                 nodesChunks,
#                 weightsChunks,
#                 regressionChunks,
#                 compute_Sobol_t_Chunks,
#                 compute_Sobol_m_Chunks,
#                 store_qoi_data_in_stat_dict_Chunks,
#                 store_gpce_surrogate_Chunks,
#                 save_gpce_surrogate_Chunks,
#                 chunksize=self.mpi_chunksize,
#                 unordered=self.unordered
#             )
#             print(f"{self.rank}: computation for qoi {single_qoi_column} - waits for shutdown...")
#             sys.stdout.flush()
#             executor.shutdown(wait=True)
#             print(f"{self.rank}: computation for qoi {single_qoi_column} - shut down...")
#             sys.stdout.flush()
#
#             solver_time_end = time.time()
#             solver_time = solver_time_end - solver_time_start
#             print(f"solver_time for qoi {single_qoi_column}: {solver_time}")
#
#             chunk_results = list(chunk_results_it)
#             for chunk_result in chunk_results:
#                 for result in chunk_result:
#                     self._process_chunk_result_single_qoi_single_time_step(
#                         single_qoi_column, timestamp=result[0], result_dict=result[1])
#                     if self.instantly_save_results_for_each_time_step:
#                         del result[1]
#             del chunk_results_it
#             del chunk_results
#
#             self._save_plot_and_clear_result_dict_single_qoi(single_qoi_column)