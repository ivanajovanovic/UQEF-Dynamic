import os
import chaospy as cp
import numpy as np
import pandas as pd
import pickle
import scipy

# from saltelliSobolIndicesHelpingFunctions import *
from . import saltelliSobolIndicesHelpingFunctions

def _my_parallel_calc_stats_for_MC(keyIter_chunk, qoi_values_chunk, numEvaluations, store_qoi_data_in_stat_dict=False):
    results = []
    for ip in range(0, len(keyIter_chunk)):  # for each peace of work
        key = keyIter_chunk[ip]
        qoi_values = qoi_values_chunk[ip]
        local_result_dict = dict()
        if store_qoi_data_in_stat_dict:
            local_result_dict["qoi_values"] = qoi_values

        numEvaluations = len(qoi_values)

        # local_result_dict["E"] = np.sum(qoi_values, axis=0, dtype=np.float64) / numEvaluations
        local_result_dict["E"] = np.mean(qoi_values, 0)
        local_result_dict["Var"] = np.sum((qoi_values - local_result_dict["E"]) ** 2, axis=0,
                                          dtype=np.float64) / (numEvaluations - 1)
        # local_result_dict["StdDev"] = np.sqrt(local_result_dict["Var"], dtype=np.float64)
        local_result_dict["StdDev"] = np.std(qoi_values, 0, ddof=1)
        local_result_dict["Skew"] = scipy.stats.skew(qoi_values, axis=0, bias=True)
        local_result_dict["Kurt"] = scipy.stats.kurtosis(qoi_values, axis=0, bias=True)

        local_result_dict["P10"] = np.percentile(qoi_values, 10, axis=0)
        local_result_dict["P90"] = np.percentile(qoi_values, 90, axis=0)
        if isinstance(local_result_dict["P10"], list) and len(local_result_dict["P10"]) == 1:
            local_result_dict["P10"] = local_result_dict["P10"][0]
            local_result_dict["P90"] = local_result_dict["P90"][0]

        results.append([key, local_result_dict])
    return results


def _my_parallel_calc_stats_for_SC(keyIter_chunk, qoi_values_chunk, dist, polynomial_expansion, nodes,
                                   compute_Sobol_t=False, compute_Sobol_m=False, store_qoi_data_in_stat_dict=False):
    pass


def _my_parallel_calc_stats_for_gPCE(keyIter_chunk, qoi_values_chunk, dist, polynomial_expansion, nodes, weights=None,
                                     regression=False, compute_Sobol_t=False, compute_Sobol_m=False,
                                     store_qoi_data_in_stat_dict=False, store_gpce_surrogate=False,
                                     save_gpce_surrogate=False):
    results = []
    for ip in range(0, len(keyIter_chunk)):  # for each peace of work
        key = keyIter_chunk[ip]
        qoi_values = qoi_values_chunk[ip]
        local_result_dict = dict()
        if store_qoi_data_in_stat_dict:
            local_result_dict["qoi_values"] = qoi_values
        if regression:
            qoi_gPCE = cp.fit_regression(polynomial_expansion, nodes, qoi_values)
        else:
            qoi_gPCE = cp.fit_quadrature(polynomial_expansion, nodes, weights, qoi_values)

        numPercSamples = 10 ** 5

        if store_gpce_surrogate:
            local_result_dict["gPCE"] = qoi_gPCE

        if save_gpce_surrogate:
            # TODO create a unique file with key and save it in a working directoryself.result_dict
            # TODO add workingDir single_qoi_column
            # timestamp = pd.Timestamp(key).strftime('%Y-%m-%d %X')
            # fileName = f"gpce_surrogate_{single_qoi_column}_{timestamp}.pkl"
            # fullFileName = os.path.abspath(os.path.join(str(workingDir), fileName))
            # with open(fullFileName, 'wb') as handle:
            #     pickle.dump(qoi_gPCE, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pass

        local_result_dict["E"] = float(cp.E(qoi_gPCE, dist))
        local_result_dict["Var"] = float(cp.Var(qoi_gPCE, dist))
        local_result_dict["StdDev"] = float(cp.Std(qoi_gPCE, dist))

        local_result_dict["Skew"] = cp.Skew(qoi_gPCE, dist).round(4)
        local_result_dict["Kurt"] = cp.Kurt(qoi_gPCE, dist)
        local_result_dict["qoi_dist"] = cp.QoI_Dist(qoi_gPCE, dist)

        local_result_dict["P10"] = float(cp.Perc(qoi_gPCE, 10, dist, numPercSamples))
        local_result_dict["P90"] = float(cp.Perc(qoi_gPCE, 90, dist, numPercSamples))
        if isinstance(local_result_dict["P10"], list) and len(local_result_dict["P10"]) == 1:
            local_result_dict["P10"] = local_result_dict["P10"][0]
            local_result_dict["P90"] = local_result_dict["P90"][0]

        if compute_Sobol_t:
            local_result_dict["Sobol_t"] = cp.Sens_t(qoi_gPCE, dist)
        if compute_Sobol_m:
            local_result_dict["Sobol_m"] = cp.Sens_m(qoi_gPCE, dist)
            #local_result_dict["Sobol_m2"] = cp.Sens_m2(qoi_gPCE, dist) # second order sensitivity indices

        results.append([key, local_result_dict])
    return results


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
        # local_result_dict["Var"] = np.sum((qoi_values[:(2 * numEvaluations)] - local_result_dict["E"]) ** 2,
        #                                    axis=0, dtype=np.float64)/(2 * numEvaluations - 1)
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