import chaospy as cp
import numpy as np
import scipy
import time

# from saltelliSobolIndicesHelpingFunctions import *
from uqef_dynamic.utils import saltelliSobolIndicesHelpingFunctions

def parallel_calc_stats_for_MC(
        keyIter_chunk, qoi_values_chunk, numEvaluations, dim, compute_Sobol_t=False, store_qoi_data_in_stat_dict=False,
        compute_sobol_total_indices_with_samples=False, samples=None):
    results = []
    # for timestamp, qoi_values in zip(keyIter_chunk, qoi_values_chunk):
    for ip in range(0, len(keyIter_chunk)):  # for each peace of work
        timestamp = keyIter_chunk[ip]
        qoi_values = qoi_values_chunk[ip]

        if isinstance(numEvaluations, list):
            numEvaluations = numEvaluations[ip]
        numEvaluations = len(qoi_values)
        if isinstance(dim, list):
            dim = dim[ip]
        if isinstance(compute_Sobol_t, list):
            compute_Sobol_t = compute_Sobol_t[ip]
        if isinstance(store_qoi_data_in_stat_dict, list):
            store_qoi_data_in_stat_dict = store_qoi_data_in_stat_dict[ip]
        if isinstance(compute_sobol_total_indices_with_samples, list):
            compute_sobol_total_indices_with_samples = compute_sobol_total_indices_with_samples[ip]
        if isinstance(samples, list):
            samples = samples[ip]

        local_result_dict = dict()

        if store_qoi_data_in_stat_dict:
            local_result_dict["qoi_values"] = qoi_values

        # local_result_dict["E"] = np.sum(qoi_values, axis=0, dtype=np.float64) / numEvaluations
        local_result_dict["E"] = np.mean(qoi_values, 0)
        local_result_dict["Var"] = np.var(qoi_values, ddof=1)
        # local_result_dict["Var"] = np.sum((qoi_values - local_result_dict["E"]) ** 2, axis=0,
        #                                   dtype=np.float64) / (numEvaluations - 1)
        # local_result_dict["StdDev"] = np.sqrt(local_result_dict["Var"], dtype=np.float64)
        local_result_dict["StdDev"] = np.std(qoi_values, 0, ddof=1)
        local_result_dict["Skew"] = scipy.stats.skew(qoi_values, axis=0, bias=True)
        local_result_dict["Kurt"] = scipy.stats.kurtosis(qoi_values, axis=0, bias=True)

        local_result_dict["P10"] = np.percentile(qoi_values, 10, axis=0)
        local_result_dict["P90"] = np.percentile(qoi_values, 90, axis=0)
        if isinstance(local_result_dict["P10"], list) and len(local_result_dict["P10"]) == 1:
            local_result_dict["P10"] = local_result_dict["P10"][0]
            local_result_dict["P90"] = local_result_dict["P90"][0]

        # # TODO - What about this? - propagate
        # #  _instantly_save_results_for_each_time_step; single_qoi_column; workingDir
        # if _instantly_save_results_for_each_time_step:
        #     fileName = f"statistics_dictionary_{single_qoi_column}_{timestamp}.pkl"
        #     fullFileName = os.path.abspath(os.path.join(str(workingDir), fileName))
        #     with open(fullFileName, 'wb') as handle:
        #         pickle.dump(local_result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if compute_Sobol_t and compute_sobol_total_indices_with_samples and samples is not None:
            local_result_dict["Sobol_t"] = saltelliSobolIndicesHelpingFunctions.compute_total_sobol_indices_with_n_samples(
                samples=samples, Y=qoi_values[:numEvaluations, np.newaxis], D=dim, N=numEvaluations)

        results.append([timestamp, local_result_dict])
    return results


def parallel_calc_stats_for_SC(keyIter_chunk, qoi_values_chunk, dist, polynomial_expansion, nodes,
                                compute_Sobol_t=False, compute_Sobol_m=False, store_qoi_data_in_stat_dict=False):
    pass


def parallel_calc_stats_for_gPCE(keyIter_chunk, qoi_values_chunk, dist, polynomial_expansion, nodes, weights=None,
                                  regression=False, compute_Sobol_t=False, compute_Sobol_m=False,
                                  store_qoi_data_in_stat_dict=False, store_gpce_surrogate_in_stat_dict=False,
                                  save_gpce_surrogate=False, compute_other_stat_besides_pce_surrogate=True):
    results = []
    for ip in range(0, len(keyIter_chunk)):  # for each piece of work
        key = keyIter_chunk[ip]
        qoi_values = qoi_values_chunk[ip]
        local_result_dict = dict()
        if store_qoi_data_in_stat_dict:
            local_result_dict["qoi_values"] = qoi_values
        if regression:
            qoi_gPCE, goi_coeff = cp.fit_regression(polynomial_expansion, nodes, qoi_values, retall=True)
        else:
            qoi_gPCE, goi_coeff = cp.fit_quadrature(polynomial_expansion, nodes, weights, qoi_values, retall=True)

        if store_gpce_surrogate_in_stat_dict:
            local_result_dict["gPCE"] = qoi_gPCE
        local_result_dict['gpce_coeff'] = goi_coeff

        if save_gpce_surrogate:
            # TODO create a unique file with key and save it in a working directory
            # TODO add workingDir single_qoi_column
            # timestamp = pd.Timestamp(key).strftime('%Y-%m-%d %X')
            # fileName = f"gpce_surrogate_{single_qoi_column}_{key}.pkl"
            # fullFileName = os.path.abspath(os.path.join(str(workingDir), fileName))
            # with open(fullFileName, 'wb') as handle:
            #     pickle.dump(qoi_gPCE, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pass

        calculate_stats_gpce(
            local_result_dict, qoi_gPCE, dist, compute_other_stat_besides_pce_surrogate,
            compute_Sobol_t, compute_Sobol_m)

        results.append([key, local_result_dict])
    return results


def calculate_stats_gpce(
    local_result_dict, qoi_gPCE, dist, compute_other_stat_besides_pce_surrogate=True,
    compute_Sobol_t=False, compute_Sobol_m=False, compute_Sobol_m2=False):
    start = time.time()
    time_info_dict = {}

    local_result_dict["E"] = float(cp.E(qoi_gPCE, dist))
    end = time.time()
    time_info_dict["duration_E"] = end - start

    if compute_other_stat_besides_pce_surrogate:
        # local_result_dict["E"] = float(cp.E(qoi_gPCE, dist))
        start = time.time()
        local_result_dict["Var"] = float(cp.Var(qoi_gPCE, dist))
        end = time.time()
        time_info_dict["duration_Var"] = end - start

        start = time.time()
        local_result_dict["StdDev"] = float(cp.Std(qoi_gPCE, dist))
        end = time.time()
        time_info_dict["duration_StdDev"] = end - start

        start = time.time()
        local_result_dict["Skew"] = cp.Skew(qoi_gPCE, dist).round(4)
        end = time.time()
        time_info_dict["duration_Skew"] = end - start

        start = time.time()
        local_result_dict["Kurt"] = cp.Kurt(qoi_gPCE, dist)
        end = time.time()
        time_info_dict["duration_Kurt"] = end - start

        start = time.time()
        local_result_dict["qoi_dist"] = cp.QoI_Dist(qoi_gPCE, dist)
        end = time.time()
        time_info_dict["duration_qoi_dist"] = end - start

        numPercSamples = 10 ** 5

        start = time.time()
        local_result_dict["P10"] = float(cp.Perc(qoi_gPCE, 10, dist, numPercSamples))
        end = time.time()
        time_info_dict["duration_P10"] = end - start

        start = time.time()
        local_result_dict["P90"] = float(cp.Perc(qoi_gPCE, 90, dist, numPercSamples))
        end = time.time()
        time_info_dict["duration_P90"] = end - start

        if isinstance(local_result_dict["P10"], list) and len(local_result_dict["P10"]) == 1:
            local_result_dict["P10"] = local_result_dict["P10"][0]
            local_result_dict["P90"] = local_result_dict["P90"][0]

        if compute_Sobol_t:
            start = time.time()
            local_result_dict["Sobol_t"] = cp.Sens_t(qoi_gPCE, dist)
            end = time.time()
            time_info_dict["duration_Sobol_t"] = end - start
        if compute_Sobol_m:
            start = time.time()
            local_result_dict["Sobol_m"] = cp.Sens_m(qoi_gPCE, dist)
            end = time.time()
            time_info_dict["duration_Sobol_m"] = end - start
        if compute_Sobol_m2:
            start = time.time()
            local_result_dict["Sobol_m2"] = cp.Sens_m2(qoi_gPCE, dist) # second order sensitivity indices
            end = time.time()
            time_info_dict["duration_Sobol_m2"] = end - start

    print(f"DEBUGGING - time_info_dict - {time_info_dict}")



def parallel_calc_stats_for_mc_saltelli(
        keyIter_chunk, qoi_values_chunk, numEvaluations, dim, compute_Sobol_t=False,
        compute_Sobol_m=False, store_qoi_data_in_stat_dict=False, compute_sobol_total_indices_with_samples=False,
        samples=None):
    results = []
    for ip in range(0, len(keyIter_chunk)):  # for each peace of work
        key = keyIter_chunk[ip]
        qoi_values = qoi_values_chunk[ip]

        if isinstance(numEvaluations, list):
            numEvaluations = numEvaluations[ip]
        if isinstance(dim, list):
            dim = dim[ip]
        if isinstance(compute_Sobol_t, list):
            compute_Sobol_t = compute_Sobol_t[ip]
        if isinstance(compute_Sobol_m, list):
            compute_Sobol_m = compute_Sobol_m[ip]
        if isinstance(store_qoi_data_in_stat_dict, list):
            store_qoi_data_in_stat_dict = store_qoi_data_in_stat_dict[ip]

        local_result_dict = dict()

        qoi_values_saltelli = qoi_values[:, np.newaxis]
        # standard_qoi_values = qoi_values_saltelli[:numEvaluations, :]
        standard_qoi_values = qoi_values[:numEvaluations]
        # extended_standard_qoi_values = qoi_values_saltelli[:(2 * numEvaluations), :]

        if store_qoi_data_in_stat_dict:
            local_result_dict["qoi_values"] = qoi_values
            # local_result_dict["qoi_values"] = standard_qoi_values

        local_result_dict["E"] = np.mean(standard_qoi_values, 0)
        local_result_dict["Var"] = np.sum((standard_qoi_values - local_result_dict["E"]) ** 2,
                                          axis=0, dtype=np.float64) / (numEvaluations - 1)
        # local_result_dict["Var"] = np.sum((qoi_values[:numEvaluations] - local_result_dict["E"]) ** 2,
        #                                    axis=0, dtype=np.float64)/(numEvaluations - 1)
        local_result_dict["Skew"] = scipy.stats.skew(standard_qoi_values, axis=0, bias=True)
        local_result_dict["Kurt"] = scipy.stats.kurtosis(standard_qoi_values, axis=0, bias=True)

        local_result_dict["StdDev"] = np.std(standard_qoi_values, 0, ddof=1)

        local_result_dict["P10"] = np.percentile(standard_qoi_values, 10, axis=0)
        local_result_dict["P90"] = np.percentile(standard_qoi_values, 90, axis=0)

        if isinstance(local_result_dict["P10"], list) and len(local_result_dict["P10"]) == 1:
            local_result_dict["P10"] = local_result_dict["P10"][0]
            local_result_dict["P90"] = local_result_dict["P90"][0]

        if compute_sobol_total_indices_with_samples and samples is not None:
            if compute_Sobol_t:
                # print(f"DEBUGGING - standard_qoi_values.shape - {standard_qoi_values.shape}")
                # print(f"DEBUGGING - qoi_values_saltelli.shape - {qoi_values_saltelli.shape}")
                # print(f"DEBUGGING - samples.shape - {samples.shape}")
                local_result_dict["Sobol_t"] = saltelliSobolIndicesHelpingFunctions.compute_total_sobol_indices_with_n_samples(
                    samples=samples, Y=qoi_values_saltelli[:numEvaluations,:], D=dim, N=numEvaluations)
        else:
            if compute_Sobol_t:
                local_result_dict["Sobol_t"] = saltelliSobolIndicesHelpingFunctions._Sens_t_sample(
                    qoi_values_saltelli, dim, numEvaluations, code=4)
            if compute_Sobol_m:
                local_result_dict["Sobol_m"] = saltelliSobolIndicesHelpingFunctions._Sens_m_sample(
                    qoi_values_saltelli, dim, numEvaluations, code=4)

        results.append([key, local_result_dict])
    return results