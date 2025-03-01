from collections import defaultdict
import dill
import json
import inspect
import math
from mpi4py import MPI
import multiprocessing
import mpi4py
import numpy as np
import os
import sys
import time
import pathlib
import pandas as pd
import pickle

cwd = pathlib.Path(os.getcwd())
parent = cwd.parent.absolute()
sys.path.insert(0, os.getcwd())

import chaospy as cp
import uqef

linux_cluster_run = False
if linux_cluster_run:
    sys.path.insert(0, '/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic')
else:
    sys.path.insert(0, '/work/ga45met/mnt/linux_cluster_2/UQEF-Dynamic')

import sparseSpACE.Function as sparseSpACE_Function
from uqef_dynamic.models.sparsespace import sparsespace_functions
from uqef_dynamic.utils import utility
from uqef_dynamic.utils import uqef_dynamic_utils
from uqef_dynamic.utils import create_stat_object
from uqef_dynamic.utils import sparsespace_utils

from uqef_dynamic.models.time_dependent_baseclass import time_dependent_statistics

from uqef_dynamic.scientific_pipelines.list_of_simulation_runs import list_of_simulation_runs_ishigami
# TODO Save reevaluations of all the models as dataframes; maybe for plotting later on...
# TODO Add simple pce-part -> produce PCE surrogate
# TODO Evaluate PCE surrogate (from simple statistics, from combiinstance)
# TODO HBV Gof inverse logic
# TODO PCE part from SparseSpace - almost done
# TODO Things will get more complicated when using the combiinstance to produce PCE surrogate!!!
# TODO Add comparing final E, Var, Sobol indices with analytical values / MC

local_debugging = False
mpi = True

if mpi:
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()
    version = MPI.Get_library_version()
    version2 = MPI.Get_version()
else:
    rank = None

def is_master(mpi, rank=None):
    return mpi is False or (mpi is True and rank == 0)


def setup_nodes_via_config_file_or_parameters_file(
    configuration_object, uq_method="sc", read_nodes_from_file=False, parameters_file=None, 
    sampleFromStandardDist=False, regression=False):
    """
    Setup the nodes for the simulation based on the configuration file or parameters file.
    This logic is extracted from the UQEF
    """
    # node names
    node_names = []
    for parameter_config in configuration_object["parameters"]:
        node_names.append(parameter_config["name"])
    simulationNodes = uqef.nodes.Nodes(node_names)

    if uq_method == "ensemble" and read_nodes_from_file and parameters_file:
        print(f"INFO: Reading nodes values from parameters file {parameters_file}")
        simulationNodes.generateNodesFromListOfValues(
            read_nodes_from_file=read_nodes_from_file,
            parameters_file_name=parameters_file)
    else:
        if sampleFromStandardDist:
            simulationNodes.setTransformation()

        for parameter_config in configuration_object["parameters"]:
            if uq_method == "ensemble":
                if "values_list" in parameter_config:
                    simulationNodes.setValue(parameter_config["name"], parameter_config["values_list"])
                elif "default" in parameter_config:
                    simulationNodes.setValue(parameter_config["name"], parameter_config["default"])
                else:
                    raise Exception(f"Error in UQSim.setup_nodes_via_config_file_or_parameters_file() : "
                                    f" an ensemble simulation should be run, "
                                    f"but values_list or default entries for parameter values are missing")
            elif parameter_config["distribution"] == "None":
                # take default value(s) from config file
                if "values_list" in parameter_config:
                    simulationNodes.setValue(parameter_config["name"], parameter_config["values_list"])
                elif 'default' in parameter_config:
                    simulationNodes.setValue(parameter_config["name"], parameter_config["default"])
                else:
                    raise Exception(f"Error in UQSim.setup_nodes_via_config_file_or_parameters_file() : "
                                    f" distribution of a parameter is None, "
                                    f"but values_list or default entries are missing")
            else:
                # node values and distributions -> automatically maps dists and their parameters by reflection mechanisms
                cp_dist_signature = inspect.signature(getattr(cp, parameter_config["distribution"]))
                dist_parameters_values = []
                for p in cp_dist_signature.parameters:
                    dist_parameters_values.append(parameter_config[p])

                simulationNodes.setDist(parameter_config["name"], \
                    getattr(cp, parameter_config["distribution"])(*dist_parameters_values))

                if sampleFromStandardDist:
                    # for numerical stability work with nodes from 'standard' distributions,
                    # and use parameters for forcing the model
                    if parameter_config["distribution"] == "Uniform":
                        if (uq_method == "sc") or (uq_method == "mc" and regression):  # Gauss–Legendre quadrature
                            simulationNodes.setStandardDist(parameter_config["name"], 
                            getattr(cp, parameter_config["distribution"])(lower=-1, upper=1))
                        else:
                            simulationNodes.setStandardDist(parameter_config["name"],
                            getattr(cp, parameter_config["distribution"])(lower=0, upper=1))
                    else:
                        simulationNodes.setStandardDist(parameter_config["name"], getattr(cp, parameter_config["distribution"])())

        if uq_method == "ensemble":
            # in case of an ensemble method, when parameters_file is not specified,
            # take a cross product of values_list of all parameters
            simulationNodes.generateNodesFromListOfValues()
    return simulationNodes


def run_simplified_uqef_dynamic_simulation(
    problem_function, configurationObject, uqsim_args_dict,  
    workingDir, model, simulationNodes=None,
    list_of_surrogate_models=["larsim", "hbvsask", "ishigami", "oscillator",],
    surrogate_object=None, **kwargs):
    dictionary_with_inf_about_the_run_uqef = {}

    run_for_single_qoi = kwargs.get("run_for_single_qoi", False)
    compute_pce_from_uq = kwargs.get("compute_pce_from_uq", None) 
    if compute_pce_from_uq is None:
        compute_pce_from_uq = uqsim_args_dict["compute_pce_from_uq"] 

    uq_method = uqsim_args_dict["uq_method"]

    if is_master(mpi, rank):
   
        # ============================
        # Generate Nodes, Parameters, and Weights
        # ============================
        nodes = None
        parameters = None
        weights = None

        if simulationNodes is None:
            simulationNodes = setup_nodes_via_config_file_or_parameters_file(
                configurationObject=configurationObject, 
                uq_method=uq_method, 
                read_nodes_from_file=uqsim_args_dict["read_nodes_from_file"], 
                parameters_file=uqsim_args_dict["parameters_file"], 
                sampleFromStandardDist=uqsim_args_dict["sampleFromStandardDist"], 
                regression=uqsim_args_dict["regression"]
            )

        if uq_method == "ensemble":
            nodes = simulationNodes.nodes
            parameters = simulationNodes.parameters
        elif uq_method == "mc":
            nodes, parameters = simulationNodes.generateNodesForMC(
                numSamples=uqsim_args_dict["mc_numevaluations"], 
                rule=uqsim_args_dict["sampling_rule"], 
                read_nodes_from_file=uqsim_args_dict["read_nodes_from_file"], 
                parameters_file_name=uqsim_args_dict["parameters_file"]
            )
        elif uq_method == "sc":
            nodes, weights, parameters = simulationNodes.generateNodesForSC(
                numCollocationPointsPerDim=uqsim_args_dict["sc_q_order"], 
                rule=uqsim_args_dict["sc_quadrature_rule"], 
                sparse=uqsim_args_dict["sc_sparse_quadrature"],
                read_nodes_from_file=uqsim_args_dict["read_nodes_from_file"], 
                parameters_file_name=uqsim_args_dict["parameters_file"]
            )
        elif uq_method == "saltelli":
            raise NotImplementedError("Yet not implemented for the Saltelli method")

        # ============================
        # Save simulationNodes, nodes, parameters, and weights
        # ============================
        simulationNodes.saveToFile(str(workingDir) + "/" + "nodes")
        
        if parameters is not None:
            df = pd.DataFrame({'parameters': [row for row in parameters.T]})
            df.to_pickle(os.path.abspath(os.path.join(str(workingDir), utility.DF_UQSIM_SIMULATION_PARAMETERS_FILE)), compression="gzip")

        if nodes is not None:
            df = pd.DataFrame({'nodes': [row for row in nodes.T]})
            df.to_pickle(os.path.abspath(os.path.join(str(workingDir), utility.DF_UQSIM_SIMULATION_NODES_FILE)), compression="gzip")

        if weights is not None:
            df = pd.DataFrame({'weights': [row for row in weights]})
            df.to_pickle(os.path.abspath(os.path.join(str(workingDir), utility.DF_UQSIM_SIMULATION_WEIGHTS_FILE)), compression="gzip")

        # ============================
        # Simulate
        # ============================
        start_time_evaluationg_surrogate = time.time()
        problem_function.uq_method = uq_method

        single_qoi = None
        # This is the current logic - surrogate can be evaluated only for a single qoi at the time...
        if surrogate_object is not None:
            run_for_single_qoi = True
        if run_for_single_qoi:
            single_qoi = problem_function.single_qoi
            if single_qoi is None:
                single_qoi = problem_function.qoi_column
            if isinstance(single_qoi, list):
                single_qoi = single_qoi[0]
            if single_qoi not in problem_function.list_qoi_column:
                raise ValueError(f"Single qoi {single_qoi} is not in the list of qoi columns {problem_function.list_qoi_column}")
            problem_function.single_qoi = single_qoi
            problem_function.list_qoi_column = [single_qoi,]

        ## multiprocessing
        # list_of_unique_index_runs =range(parameters.T.shape[0])
        # num_cores = multiprocessing.cpu_count()
        # parameter_chunks = np.array_split(parameters.T, num_cores)
        # list_of_unique_index_runs_chunks = np.array_split(list_of_unique_index_runs, num_cores)
        # list_of_unique_index_runs_chunks = [chunk.tolist() for chunk in list_of_unique_index_runs_chunks]
        # surrogate_object_chunks = [surrogate_object]*num_cores
        # results_list = []
        # def process_nodes_concurrently(parameter_chunks):
        #     with multiprocessing.Pool(processes=num_cores) as pool:
        #         for result in pool.starmap(evaluate_chunk_model_run_function, \
        #             [(problem_function, parameter, i_s, surrogate_object, problem_function.single_qoi) for (parameter, i_s) in zip(parameter_chunks, list_of_unique_index_runs_chunks)]):
        #             yield result
        # for result in process_nodes_concurrently(parameter_chunks):
        #     results_list.append(result)
        ## version withour parallelization
        if surrogate_object is None:
            print(f"\n====[PCE PART INFO] - building the gPCE of the original model====")
            results_list = problem_function.run(
                i_s=range(parameters.T.shape[0]), 
                parameters=parameters.T, 
                raise_exception_on_model_break=True,
                evaluate_surrogate=False,
                surrogate_model=None,
                single_qoi_column_surrogate=None,
            )
            dictionary_with_inf_about_the_run_uqef["number_full_model_evaluations"] = parameters.T.shape[0]
        else:
            print(f"\n====[PCE PART INFO] - building the gPCE with the surrogate model====")
            results_list = problem_function.run(
                i_s=range(parameters.T.shape[0]), 
                parameters=parameters.T, 
                raise_exception_on_model_break=True,
                evaluate_surrogate=True, 
                surrogate_model=surrogate_object,
                single_qoi_column_surrogate=problem_function.single_qoi,
            )
            dictionary_with_inf_about_the_run_uqef["number_surrogate_model_evaluations"] = parameters.T.shape[0]

        end_time_evaluationg_surrogate = time.time()
        time_evaluationg_intermediate_surrogate  = end_time_evaluationg_surrogate - start_time_evaluationg_surrogate
        dictionary_with_inf_about_the_run_uqef["time_evaluationg_model_when_building_gpce"] = time_evaluationg_intermediate_surrogate
        print(f"INFO: time_evaluationg_model_when_building_gpce={dictionary_with_inf_about_the_run_uqef['time_evaluationg_model_when_building_gpce']}\n")

        if kwargs.get("run_original_model", False):
            start_time_evalauting_original_model_uqef = time.time()
            results_list_original_model = problem_function.run(
                i_s=range(parameters.T.shape[0]), 
                parameters=parameters.T, 
                raise_exception_on_model_break=True,
            )
            # print(f"DEBUGGING - results_list_original_model evaluating combiinstance surrogate model model={results_list_original_model}")
            end_time_evalauting_original_model_uqef = time.time()
            time_evalauting_original_model_uqef = end_time_evalauting_original_model_uqef - start_time_evalauting_original_model_uqef
            dictionary_with_inf_about_the_run_uqef["time_evalauting_original_model_uqef"] = time_evalauting_original_model_uqef
            print(f"INFO: time_evalauting_original_model_uqef={dictionary_with_inf_about_the_run_uqef['time_evalauting_original_model_uqef']}\n")

    # ============================
    # Statistics
    # ============================
    # Note: One has to make sure that statistics object get the single qoi information
    problem_statistics = None
    if model.lower() in list_of_surrogate_models:
        problem_statistics = create_stat_object.create_statistics_object(
            configuration_object=configurationObject, 
            uqsim_args_dict=uqsim_args_dict, 
            workingDir=workingDir, model=model, free_result_dict_memory=False)
        # Note: One has to make sure that statistics object get the single qoi information
        # Or maybe this is not necessary when only doing uqef analysis - since multiple qois are supported!
        print(f"DEBUGGING: problem_statistics.list_qoi_column - {problem_statistics.list_qoi_column}")
        if run_for_single_qoi:
            if problem_function.single_qoi not in problem_statistics.list_qoi_column:
                raise ValueError(f"Single qoi {problem_function.single_qoi} is not in the list of qoi columns {problem_statistics.list_qoi_column}")
            else:
                problem_statistics.list_qoi_column = [problem_function.single_qoi,]
                problem_statistics.qoi_column = problem_function.single_qoi
    else:
        problem_statistics = None
        raise ValueError(f"Model {model} is not yet supported for generating statistics object!")
    
    gPCE_surrogate = None
    problem_statistics_result_dict = None
    df_statistics = None
    if is_master(mpi, rank):
        start_time_computing_statistics = time.time()
        rawSamples = [single_results_dict for (single_results_dict, _) in results_list]
    if problem_statistics is not None:
        if is_master(mpi, rank):
            problem_statistics.prepare(rawSamples)
        if uq_method == "mc":
            if is_master(mpi, rank):
                numEvaluations = nodes.shape[1] # simulationNodes.numSamples
                problem_statistics.prepareForMcStatistics(
                    simulationNodes, numEvaluations=numEvaluations, 
                    regression=uqsim_args_dict["regression"], 
                    order=uqsim_args_dict["sc_p_order"],
                    poly_normed=uqsim_args_dict["sc_poly_normed"], 
                    poly_rule=uqsim_args_dict["sc_poly_rule"], 
                    cross_truncation=uqsim_args_dict["cross_truncation"])
            if uqsim_args_dict["parallel_statistics"] and uqsim_args_dict["mpi"]:
                problem_statistics.calcStatisticsForMcParallel()
            else:
                problem_statistics.calcStatisticsForMc()
        elif uq_method == "sc":
            if is_master(mpi, rank):
                problem_statistics.prepareForScStatistics(
                    simulationNodes,
                    order=uqsim_args_dict["sc_p_order"],
                    poly_normed=uqsim_args_dict["sc_poly_normed"], 
                    poly_rule=uqsim_args_dict["sc_poly_rule"], 
                    regression=uqsim_args_dict["regression"], 
                    cross_truncation=uqsim_args_dict["cross_truncation"])
            if uqsim_args_dict["parallel_statistics"] and uqsim_args_dict["mpi"]:
                problem_statistics.calcStatisticsForScParallel()
            else:
                problem_statistics.calcStatisticsForSc()
        elif uq_method == "saltelli":
            if is_master(mpi, rank):
                problem_statistics.prepareForMcSaltelliStatistics(
                    simulationNodes,
                    order=uqsim_args_dict["sc_p_order"],
                    poly_normed=uqsim_args_dict["sc_poly_normed"], 
                    poly_rule=uqsim_args_dict["sc_poly_rule"], 
                    regression=uqsim_args_dict["regression"], 
                    cross_truncation=uqsim_args_dict["cross_truncation"])
            if uqsim_args_dict["parallel_statistics"] and uqsim_args_dict["mpi"]:
                problem_statistics.calcStatisticsForMcSaltelliParallel()
            else:
                problem_statistics.calcStatisticsForMcSaltelli()
        if is_master(mpi, rank):
            problem_statistics.create_df_from_statistics_data()
            problem_statistics.saveToFile()
            problem_statistics_result_dict = problem_statistics.get_result_dict()
            df_statistics = problem_statistics.df_statistics
            # print(f"DEBUGGING problem_statistics.df_statistics - {problem_statistics.df_statistics}")
            # print(f"DEBUGGING problem_statistics.result_dict - {problem_statistics.result_dict}")
            # print(f"DEBUGGING problem_statistics.result_dict.keys()- {problem_statistics.result_dict.keys()}")
            if compute_pce_from_uq:
                single_timestamp_single_file = uqsim_args_dict.get("instantly_save_results_for_each_time_step", False) 
                surrogate_type="pce"
                gpce_surrogate_dict_over_qois, gpce_coeff_dict_over_qois, kl_coeff_dict_over_qois, kl_surrogate_df_dict_over_qois = uqef_dynamic_utils.read_surrogate_model_single_working_dir(
                        workingDir=workingDir, statisticsObject=problem_statistics, single_timestamp_single_file=single_timestamp_single_file,
                        surrogate_type=surrogate_type, recompute_generalized_sobol_indices=False, 
                        polynomial_expansion=problem_statistics.polynomial_expansion, polynomial_norms=problem_statistics.polynomial_norms
                )
                gPCE_surrogate = gpce_surrogate_dict_over_qois
                if run_for_single_qoi:
                    if isinstance(gpce_surrogate_dict_over_qois, dict):
                        if single_qoi in gpce_surrogate_dict_over_qois:
                            gPCE_surrogate = gpce_surrogate_dict_over_qois[single_qoi]
            dictionary_with_inf_about_the_run_uqef["statistics_pdTimesteps"] = problem_statistics.pdTimesteps
                        
    if is_master(mpi, rank):
        end_time_computing_statistics = time.time()
        time_computing_statistics = end_time_computing_statistics - start_time_computing_statistics
        dictionary_with_inf_about_the_run_uqef["time_computing_statistics"] = time_computing_statistics
        print(f"INFO: time_computing_statistics={dictionary_with_inf_about_the_run_uqef['time_computing_statistics']}\n")
    
    dictionary_with_inf_about_the_run_uqef['gPCE_surrogate'] = gPCE_surrogate
    dictionary_with_inf_about_the_run_uqef['problem_statistics_result_dict'] = problem_statistics_result_dict
    print(f"\n===Done with the UQEF-Dynamic Part===")

    return dictionary_with_inf_about_the_run_uqef

    # ============================
    # uqsim = uqef.UQsim()
    # uqsim.args.model = model # "combiinstance" 
    # uqsim.args.uncertain = "all"
    # uqsim.args.chunksize = 1
    # uqsim.setup_configuration_object()
    # uqsim.args.workingDir = str(uqsim.args.outputResultDir)
    # # TODO Register the model (SG Integral) and the statistics
    # uqsim.models.update({"combiinstance"         : (lambda: CombiinstanceModelUQEF(
    # combiinstance=combiObject,))})
    # uqsim.setup()  #here is where model_generator and setup_solver are taking place... not sure if i need this
    # if uqsim.is_master():
    #     simulationNodes_save_file = "nodes"
    #     uqsim.save_simulationNodes(fileName=simulationNodes_save_file)
    #     number_full_model_evaluations = uqsim.get_simulation_parameters_shape()[0]
    #     argsFileName = os.path.abspath(os.path.join(uqsim.args.outputResultDir, utility.ARGS_FILE))
    #     with open(argsFileName, 'wb') as handle:
    #         pickle.dump(uqsim.args, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     fileName = pathlib.Path(uqsim.args.outputResultDir) / utility.CONFIGURATION_OBJECT_FILE
    #     with open(fileName, 'wb') as f:
    #         dill.dump(uqsim.configuration_object, f)
    # # simulate
    # start_time_model_simulations = time.time()
    # uqsim.simulate()
    # end_time_model_simulations = time.time()
    # time_model_simulations = end_time_model_simulations - start_time_model_simulations
    # if uqsim.is_master():
    #     if hasattr(uqsim.simulation, 'parameters') and uqsim.simulation.parameters is not None:
    #         df = pd.DataFrame({'parameters': [row for row in uqsim.simulation.parameters]})
    #         df.to_pickle(os.path.abspath(os.path.join(uqsim.args.outputResultDir, utility.DF_UQSIM_SIMULATION_PARAMETERS_FILE)), compression="gzip")
    #     if hasattr(uqsim.simulation, 'nodes') and uqsim.simulation.nodes is not None:
    #         df = pd.DataFrame({'nodes': [row for row in uqsim.simulation.nodes]})
    #         df.to_pickle(os.path.abspath(os.path.join(uqsim.args.outputResultDir, utility.DF_UQSIM_SIMULATION_NODES_FILE)), compression="gzip")
    #     if hasattr(uqsim.simulation, 'weights') and uqsim.simulation.weights is not None:
    #         df = pd.DataFrame({'weights': [row for row in uqsim.simulation.weights]})
    #         df.to_pickle(os.path.abspath(os.path.join(uqsim.args.outputResultDir, utility.DF_UQSIM_SIMULATION_WEIGHTS_FILE)), compression="gzip")
    #     fileName = pathlib.Path(uqsim.args.outputResultDir) / utility.CONFIGURATION_OBJECT_FILE
    #     with open(fileName, 'wb') as f:
    #         dill.dump(uqsim.configuration_object, f)
    # # statistics
    # start_time_computing_statistics = time.time()
    # uqsim.prepare_statistics()
    # uqsim.calc_statistics()
    # end_time_computing_statistics = time.time()
    # time_computing_statistics = end_time_computing_statistics - start_time_computing_statistics
    # uqsim.save_statistics()
    # uqsim.tear_down()
    # # ============================
# ============================


def compute_numpy_array_errors(results_array_surrogate_model, results_array_original_model, printing=False):
    """
    Compute various error metrics between the results of a surrogate model and the results of an original model.

    Parameters:
    results_array_surrogate_model (numpy.ndarray): Array of results from the surrogate model. Expected to be of shape n_samples x n_timestamps
    results_array_original_model (numpy.ndarray): Array of results from the original model.

    Returns:
    dict: A dictionary containing various error metrics including RMSE, 1-norm, 2-norm, L2 error, L1 error, and mean L1 error.
    """
    error = results_array_surrogate_model - results_array_original_model
    abs_error = np.abs(error)
    squared_error = error**2
    # For each timestamp (column), compute RMSE over samples (rows)
    rmse_over_time = np.sqrt(np.mean(squared_error, axis=0))
    # Compute overall RMSE across all time steps
    overall_rmse = np.sqrt(np.mean(squared_error))

    # overall_linf = squared_error.max()
    overall_linf = np.max(abs_error)
    overall_l2_scaled = np.sqrt(np.sum(squared_error)) / math.sqrt(len(results_array_surrogate_model))
    l2_error = np.linalg.norm(error)
    l1_error = np.sum(np.abs(error))
    # average L1 error per element
    mean_l1_error = np.mean(np.abs(error))

    if printing:
        print(f"INFO: Overall errors: RMSE={overall_rmse}; linf={overall_linf};"
        f" L2_scaled={overall_l2_scaled}; L2_error={l2_error}; L1_error={l1_error}; mean_L1_error={mean_l1_error}")
        
    result_dict = {}
    result_dict['rmse_over_time'] = rmse_over_time
    result_dict['overall_rmse'] = overall_rmse
    result_dict['overall_linf'] = overall_linf
    result_dict['overall_l2_scaled'] = overall_l2_scaled
    result_dict['l2_error'] = l2_error
    result_dict['l1_error'] = l1_error
    result_dict['mean_l1_error'] = mean_l1_error
    return result_dict


def evaluate_chunk_model(problem_function, chunk):
    # return np.array([problem_function(parameter) for parameter in chunk])
    return np.array(problem_function(coordinates=chunk))


def evaluate_simple_gpce_model(gPCE_model, nodes):
    return np.array(gPCE_model(*nodes))


def evaluate_chunk_model_run_function(problem_function, parameter, i_s, surrogate_object=None, single_qoi=None):
    print(f"DEBUGGING evaluate_chunk_model_run_function:parameter.shape-{parameter.shape}")
    print(f"DEBUGGING evaluate_chunk_model_run_function:i_s-{i_s}")
    if surrogate_object is None:
        results_list = problem_function.run(
            i_s=i_s, 
            parameters=parameter, 
            raise_exception_on_model_break=True,
            evaluate_surrogate=False,
            surrogate_model=None,
            single_qoi_column_surrogate=single_qoi,
        )
    else:
        results_list = problem_function.run(
            i_s=i_s, 
            parameters=parameter, 
            raise_exception_on_model_break=True,
            evaluate_surrogate=True, 
            surrogate_model=surrogate_object,
            single_qoi_column_surrogate=single_qoi,
        )
    return results_list


def evaluate_over_time_gpce_model(gPCE_model, nodes, list_of_indices, single_qoi):
    surrogate_evaluated_dict_over_time = defaultdict(list)
    for date, gpce_model in gPCE_model.items():
        surrogate_evaluated_dict_over_time[date] = evaluate_simple_gpce_model(gpce_model, nodes)
    rows = [(time, index, val) for time, values in surrogate_evaluated_dict_over_time.items() for index, val in zip(list_of_indices, values)]
    df_surrogate_evaluated_dict_over_time = pd.DataFrame(rows, columns=[utility.TIME_COLUMN_NAME, utility.INDEX_COLUMN_NAME, single_qoi])
    return df_surrogate_evaluated_dict_over_time

# ============================
# Main routine
# ============================

def main_routine(model, current_output_folder, **kwargs):
    # ============================
    # Initialization
    # ============================

    dictionary_with_inf_about_the_run = dict()
    uqsim_args_dict = dict()
    dict_with_time_info = None
    dictionary_with_inf_about_the_run["model"] = uqsim_args_dict["model"] = model
    scratch_dir = cwd

    # mpi = kwargs.get("mpi", False)

    compute_sparsespace= kwargs.get("compute_sparsespace", False)
    compute_pce_from_sparsespace = kwargs.get("compute_pce_from_sparsespace", False)
    compute_uq = kwargs.get("compute_uq", False)
    compute_pce_from_uq = kwargs.get("compute_pce_from_uq", False)

    if not compute_sparsespace and not compute_uq:
        raise ValueError("The current implementation requires both ct and pce surrogates to be computed!")
    if not compute_sparsespace:
        compute_pce_from_sparsespace = False
    if not compute_uq:
        compute_pce_from_uq = False
    uqsim_args_dict["compute_sparsespace"] = compute_sparsespace
    uqsim_args_dict["compute_uq"] = compute_uq
    uqsim_args_dict["compute_pce_from_sparsespace"] = compute_pce_from_sparsespace
    uqsim_args_dict["compute_pce_from_uq"] = compute_pce_from_uq

    can_model_evaluate_all_vector_nodes = kwargs.get("can_model_evaluate_all_vector_nodes", False)  # set to True if eval_vectorized is implemented,
    inputModelDir = kwargs.get("inputModelDir", None)
    outputModelDir = kwargs.get("outputModelDir", None)
    outputResultDir = kwargs.get("outputResultDir", None)
    config_file = kwargs.get("config_file", None)
    single_qoi = kwargs.get("single_qoi", None)
    sourceDir = scratch_dir
    parameters_setup_file_name = None
    parameters_file_name = None

    surrogate_type = kwargs.get("surrogate_type", kwargs.get("surrogate_model_of_interest", None))
    variant = kwargs.get('variant', 1)

    if model.lower() == "ishigami":
        workingDir = outputModelDir / current_output_folder
    elif model.lower() == "hbvsask":
        workingDir = outputModelDir / current_output_folder
    elif model.lower() in sparsespace_functions.LIST_OF_GENZ_FUNCTIONS:
        workingDir = pathlib.Path(f'{outputModelDir}/{model}/{current_output_folder}')
    else:
        workingDir = pathlib.Path(outputModelDir / current_output_folder)

    if is_master(mpi, rank):
        if workingDir is not None and not workingDir.exists():
            workingDir.mkdir(parents=True, exist_ok=True)
    outputResultDir = workingDir

    uqsim_args_dict['mpi'] = mpi
    uqsim_args_dict["config_file"] = str(config_file)
    uqsim_args_dict["outputModelDir"] = str(outputModelDir)
    uqsim_args_dict["workingDir"] = str(workingDir)
    uqsim_args_dict["inputModelDir"] = str(inputModelDir)
    uqsim_args_dict["single_qoi"] = single_qoi
    # uqsim_args_dict["surrogate_type"] = surrogate_type
    uqsim_args_dict['can_model_evaluate_all_vector_nodes'] = can_model_evaluate_all_vector_nodes
    # uqsim_args_dict['variant'] = variant

    dictionary_with_inf_about_the_run["config_file"] = str(config_file)
    dictionary_with_inf_about_the_run["workingDir"] = str(workingDir)
    dictionary_with_inf_about_the_run["inputModelDir"] = str(inputModelDir)
    dictionary_with_inf_about_the_run["single_qoi"] = single_qoi
    # dictionary_with_inf_about_the_run["surrogate_type"] = surrogate_type
    # dictionary_with_inf_about_the_run['variant'] = variant

    # ============================
    # Setting the 'stochastic part'
    # ============================

    # default values, most likely will be overwritten later on based on settings for each model
    dim = 0
    param_names = []
    distributions_list_of_dicts = []
    distributionsForSparseSpace = []
    a = []
    b = []
    anisotropic = kwargs.get("anisotropic", True)

    if config_file is not None:
        with open(config_file) as f:
            configurationObject = json.load(f)
    else:
        configurationObject = None

    # in case the set-up of the model is done via some configuration_file
    if configurationObject is not None \
            and isinstance(configurationObject, dict) and "parameters" in configurationObject:
        for single_param in configurationObject["parameters"]:
            if single_param["distribution"] != "None":
                if model.lower() == "larsim":
                    param_names.append((single_param["type"], single_param["name"]))
                else:
                    param_names.append(single_param["name"])
                dim += 1
                distributions_list_of_dicts.append(single_param)
                distributionsForSparseSpace.append((single_param["distribution"], single_param["lower"], single_param["upper"]))
                a.append(single_param["lower"])
                b.append(single_param["upper"])
    else:
        # TODO manual setup of params, param_names, dim, distributions, a, b; change this eventually.
        #  Hard-coded to Uniform dist!
        if model.lower() == "ishigami":
            param_names = ["x1", "x2", "x3"]
            a = [-math.pi, -math.pi, -math.pi]
            b = [math.pi, math.pi, math.pi]
            can_model_evaluate_all_vector_nodes = True
        elif model.lower() == "gfunction":
            param_names = ["x0", "x1", "x2"]
            a = [0.0, 0.0, 0.0]
            b = [1.0, 1.0, 1.0]
            can_model_evaluate_all_vector_nodes = True
        elif model.lower() == "zabarras2d":
            param_names = ["x0", "x1"]
            a = [-1.0, -1.0]  # [-2.5, -2]
            b = [1.0, 1.0]  # [2.5, 2]
            can_model_evaluate_all_vector_nodes = True
        elif model.lower() == "zabarras3d":
            param_names = ["x0", "x1", "x2"]
            a = [-1.0, -1.0, -1.0]  # [-2.5, -2, 5]
            b = [1.0, 1.0, 1.0]  # [2.5, 2, 15]
        elif model.lower() in sparsespace_functions.LIST_OF_GENZ_FUNCTIONS:
            # param_names = ["x0", "x1", "x2"]
            # a = [0.0, 0.0, 0.0]
            # b = [1.0, 1.0, 1.0]
            # dim = 3
            # coeffs, _ = sparsespace_functions.generate_and_scale_coeff_and_weights_genz(dim, b=sparsespace_functions.GENZ_DICT[model.lower()])
            param_names = ["x0", "x1", "x2", "x3", "x4"]
            a = [0.0, 0.0, 0.0, 0.0, 0.0]
            b = [1.0, 1.0, 1.0, 1.0, 1.0]
            dim = 5
            if "coeffs" in kwargs:
                coeffs = kwargs["coeffs"]
                weights = None
                if "weights" in kwargs:
                    weights = kwargs["weights"]
            else:
                coeffs, weights = sparsespace_functions.generate_and_scale_coeff_and_weights_genz(
                    dim=dim, b=sparsespace_functions.GENZ_DICT[model.lower()], anisotropic=anisotropic)
            # coeffs = [float(1) for _ in range(dim)]
            can_model_evaluate_all_vector_nodes = True
            uqsim_args_dict["coeffs"] = coeffs
            uqsim_args_dict["weights"] = weights

        dim = len(param_names)
        distributions_list_of_dicts = [{"distribution": "Uniform", "lower": a[i], "upper": b[i]} for i in range(dim)]
        distributionsForSparseSpace = [("Uniform", a[i], b[i]) for i in range(dim)]

    a = np.array(a)
    b = np.array(b)
    
    uqsim_args_dict["dim"] = dim
    uqsim_args_dict["param_names"] = param_names
    uqsim_args_dict["a"] = a
    uqsim_args_dict["b"] = b
    uqsim_args_dict["anisotropic"] = anisotropic

    # ============================
    # Creating the model object
    # ============================

    problem_function = None
    intermediate_surrogate_object = None
    surrogate_object = None

    if model.lower() == "ishigami":
        # problem_function = sparsespace_functions.IshigamiFunction()
        problem_function = sparsespace_functions.IshigamiFunctionUQEF(
            configurationObject=configurationObject,
        )
    elif model.lower() == "hbvsask":
        # problem_function = sparsespace_functions.HBVFunction(
        #     configurationObject=configurationObject,
        #     inputModelDir=inputModelDir,
        #     workingDir=workingDir,
        # )
        problem_function = sparsespace_functions.HBVSASKFunctionUQEF(
            configurationObject=configurationObject,
            inputModelDir=inputModelDir,
            workingDir=workingDir,
            single_qoi=single_qoi,
        )
    elif model.lower() == "corner_peak":
        problem_function = sparseSpACE_Function.GenzCornerPeak(coeffs=coeffs) #sparsespace_functions.CornerPeakFunction(coeffs=coeffs)
    elif model.lower() == "product_peak":
        problem_function = sparseSpACE_Function.GenzProductPeak(coefficients=coeffs, midpoint=weights)
    elif model.lower() == "oscillatory":
        problem_function = sparseSpACE_Function.GenzOszillatory(coeffs=coeffs, offset=weights[0])
    elif model.lower() == "gaussian":
        problem_function = sparsespace_functions.GenzGaussian(midpoint=weights, coefficients=coeffs)  # TODO ubiquitous
    elif model.lower() == "discontinuous":
        problem_function = sparsespace_functions.GenzDiscontinious(coeffs=coeffs, border=weights)  # TODO ubiquitous
    else:
        raise ValueError(f"Model {model} is not yet supported!")
   
    single_qoi = problem_function.single_qoi
    if single_qoi is None:
        single_qoi = problem_function.qoi_column
    if single_qoi is None:
        raise ValueError(f"Single qoi is not set for the model {model}!")
    # ============================
    # I
    # ============================
    # TODO add a question to see if combiinstance is used as (intermediate) surrogate model
    if compute_sparsespace:
        # params for SparseSpACE
        grid_type = kwargs.get("grid_type", "trapezoidal")  # 'trapezoidal', 'chebyshev', 'leja', 'bspline_p3'; For spetical adaptive single dimensions algorithm: 'globa_trapezoidal', 'trapezoidal' and 'bspline_p3'
        method = kwargs.get("method", "standard_combi")  # 'standard_combi', 'dim_adaptive_combi', 'dim_wise_spat_adaptive_combi'
        operation_str = kwargs.get("operation", kwargs.get("operation_str", 'integration'))  # 'uncertaintyquantification', 'interpolation', 'integration'
        # optional parameters - their default values match ones from sparsespace_utils
        minimum_level = kwargs.get("minimum_level", kwargs.get("lmin", 1))  # used to be lmin
        maximum_level = kwargs.get("maximum_level", kwargs.get("lmax", 3))  # used to be lmax
        max_evaluations = kwargs.get("max_evaluations", kwargs.get("max_evals", 100)) # 0, 22, used to be max_evals
        tol = kwargs.get("tol", kwargs.get("tolerance", 10**-6))   # 0.3*10**-1, 10**-4  # used to be tolerance
        modified_basis = kwargs.get("modified_basis", False)
        boundary = kwargs.get("boundary", kwargs.get("boundary_points", True))  # used to be boundary_points
        norm = kwargs.get("norm", 2)  #np.inf
        p_bsplines = kwargs.get("p_bsplines", 3)
        rebalancing = kwargs.get("rebalancing", True)
        version = kwargs.get("version", 6)
        margin = kwargs.get("margin", 0.9)
        grid_surplusses = kwargs.get("grid_surplusses", 'grid')
        distributions = distributionsForSparseSpace
        uq_optimization = kwargs.get("uq_optimization", 'mean')  # 'mean' | 'mean_and_var' | 'pce'; Default: 'mean'
        sc_p_order  = kwargs.get("sc_p_order", 2)
        scale_weights = kwargs.get("scale_weights", False)
        if operation_str.lower() not in ["uq", 'uncertaintyquantification', 'uncertainty quantification']:  # uq_optimization.lower()!='pce'
            compute_pce_from_sparsespace = False
        elif uq_optimization.lower()=='pce':
            compute_pce_from_sparsespace = True
        uqsim_args_dict["compute_pce_from_sparsespace"] = compute_pce_from_sparsespace
        # Collect all the above local variables into kwargs
        kwargs_sparsespace_pipeline = {
            key: value for key, value in locals().items() if key in \
                ['operation_str', 'grid_type', 'method', 'minimum_level', 'maximum_level', 'max_evaluations', 'tol', \
                    'modified_basis', 'boundary', 'norm', 'p_bsplines', 'rebalancing', 'version', 'margin', 'grid_surplusses', \
                        'distributions', 'sc_p_order', 'uq_optimization', 'compute_pce_from_sparsespace', 'scale_weights']}

        uqsim_args_dict.update(kwargs_sparsespace_pipeline)
        
        if is_master(mpi, rank):
            result_dict_sparsespace = sparsespace_utils.sparsespace_pipeline(
                a=a, b=b, model=problem_function,
                dim=dim, 
                directory_for_saving_plots=workingDir,
                do_plot=True,  # TODO Check if you want plotting always!!!
                **kwargs_sparsespace_pipeline
            )
            # total_points, total_weights = combiObject.get_points_and_weights()
            # total_surplusses = combiObject.get_surplusses()
            combiObject = result_dict_sparsespace.pop('combiObject', None)
            for key, value in result_dict_sparsespace.items():
                print(f"{key} - {value}")
            number_full_model_evaluations = result_dict_sparsespace.get('number_full_model_evaluations', None)
            print(f"INFO: combiObject: {combiObject},\n number_full_model_evaluations: {number_full_model_evaluations},\n result_dict_sparsespace: {result_dict_sparsespace}")
            
            E_sparsespace = result_dict_sparsespace.get("E", None)
            Var_sparsespace = result_dict_sparsespace.get("Var", None)
            gPCE_sparsespace = result_dict_sparsespace.get("gPCE", None)
            poly_expansion_sparsespace = result_dict_sparsespace.get("pce_polys", None)
            E_gpce_sparsespace = result_dict_sparsespace.get("E_gpce", None)
            Var_gpce_sparsespace = result_dict_sparsespace.get("Var_gpce", None)
            first_order_sobol_indices_sparsespace = result_dict_sparsespace.get("first_order_sobol_indices", None)
            total_order_sobol_indices_sparsespace = result_dict_sparsespace.get("total_order_sobol_indices", None)

            dictionary_with_inf_about_the_run.update(result_dict_sparsespace)

            # TODO Try to save combiObject in some other way
            # dictionary_with_inf_about_the_run["combiObject"] = combiObject
    
            # TODO Things will get more complicated when using the combiinstance to produce PCE surrogate!!!
            surrogate_object = combiObject
            uqsim_args_dict["surrogate_type"] = dictionary_with_inf_about_the_run["surrogate_type"] = "combiinstance"

            if gPCE_sparsespace is not None and compute_pce_from_sparsespace:
                intermediate_surrogate_object = surrogate_object
                uqsim_args_dict["intermediate_surrogate_type"] = dictionary_with_inf_about_the_run["intermediate_surrogate_type"] = "combiinstance"
                surrogate_object = gPCE_sparsespace
                uqsim_args_dict["surrogate_type"] = dictionary_with_inf_about_the_run["surrogate_type"] = "pce_sparsespace"
                print(f"DEBUGGING gPCE_sparsespace - {type(gPCE_sparsespace)}")

    # ============================
    # II - cut...
    # ============================

    # ============================
    # Building the PCE-surrogate model (based on the UQEF-Dynamic)
    # ============================
    if compute_uq or compute_pce_from_uq:
        uq_method = kwargs.get("uq_method", "mc")
        uqsim_args_dict["uq_method"] = uq_method
        uqsim_args_dict["mc_numevaluations"] = kwargs.get("mc_numevaluations", 100)
        uqsim_args_dict["sampling_rule"] = kwargs.get("sampling_rule", "random")
        uqsim_args_dict["sc_q_order"] = kwargs.get("sc_q_order", 5)
        uqsim_args_dict["sc_p_order"] = kwargs.get("sc_p_order", 2)
        uqsim_args_dict["sc_quadrature_rule"] = kwargs.get("sc_quadrature_rule",  "g")
        uqsim_args_dict["parameters_setup_file"] = kwargs.get("parameters_setup_file", None)
        uqsim_args_dict["sc_poly_rule"] = kwargs.get("sc_poly_rule", "three_terms_recurrence")   # "gram_schmidt" | "three_terms_recurrence" | "cholesky"
        uqsim_args_dict["sc_poly_normed"] = kwargs.get("sc_poly_normed", True)
        uqsim_args_dict["sc_sparse_quadrature"] = kwargs.get("sc_sparse_quadrature", False)
        uqsim_args_dict["regression"] = kwargs.get("regression", False)
        uqsim_args_dict["cross_truncation"] = kwargs.get("cross_truncation", 0.7)
        uqsim_args_dict["read_nodes_from_file"] = kwargs.get("read_nodes_from_file", False)
        uqsim_args_dict["l_sg"] = kwargs.get("l_sg", kwargs.get("l", 5))
        parameters_file = kwargs.get("parameters_file", None)
        # path_to_file = pathlib.Path("/dss/dsshome1/lxc0C/ga45met2/Repositories/sparse_grid_nodes_weights")
        if parameters_file is not None:
            uqsim_args_dict["parameters_file"] = pathlib.Path(parameters_file) / f"KPU_d{dim}_l{uqsim_args_dict['l_sg']}.asc"
            parameters_file = uqsim_args_dict["parameters_file"]
        else:
            uqsim_args_dict["parameters_file"] = None
        uqsim_args_dict["sampleFromStandardDist"] = kwargs.get("sampleFromStandardDist", True)
        uqsim_args_dict["disable_statistics"] = kwargs.get("disable_statistics", False)
        uqsim_args_dict["disable_calc_statistics"] = kwargs.get("disable_calc_statistics", False)
        uqsim_args_dict["parallel_statistics"] = mpi #kwargs.get("parallel_statistics", False)
        uqsim_args_dict["instantly_save_results_for_each_time_step"] = kwargs.get("instantly_save_results_for_each_time_step", False)
        uqsim_args_dict["compute_Sobol_m"] = kwargs.get("compute_Sobol_m", True)
        uqsim_args_dict["compute_Sobol_t"] = kwargs.get("compute_Sobol_t", True)
        uqsim_args_dict["num_cores"] = kwargs.get("num_cores", False)
        uqsim_args_dict["store_qoi_data_in_stat_dict"] = kwargs.get("store_qoi_data_in_stat_dict", False)
        uqsim_args_dict["store_gpce_surrogate_in_stat_dict"] = kwargs.get("store_gpce_surrogate_in_stat_dict", False)

        if uqsim_args_dict["uq_method"] != 'sc' and not uqsim_args_dict["regression"]:
            compute_pce_from_uq = False
            uqsim_args_dict["compute_pce_from_uq"] = compute_pce_from_uq

    # ============================
    # Setiiing up the nodes / simulationNodes / distributions
    # ============================
    if configurationObject is not None:
        simulationNodes = setup_nodes_via_config_file_or_parameters_file(
            configuration_object=configurationObject, 
            uq_method=kwargs.get("uq_method", "mc"), 
            read_nodes_from_file=kwargs.get("read_nodes_from_file", False),
            parameters_file=kwargs.get("parameters_file", None),
            sampleFromStandardDist=kwargs.get("sampleFromStandardDist", True), 
            regression=kwargs.get("regression", False)
        )
        simulationNodes.set_joined_dists()
        jointDists = simulationNodes.joinedDists
        jointStandardDists = simulationNodes.joinedStandardDists
    else:
        # setup via the chaospy
        simulationNodes = None
        dists = []
        standardDists = []
        standardDists_min_one_one = []
        standardDists_zero_one = []
        jointDists = None
        jointStandardDists = None
        for single_param_dist_config_dict in distributions_list_of_dicts:
            cp_dist_signature = inspect.signature(getattr(cp, single_param_dist_config_dict["distribution"]))
            dist_parameters_values = []
            for p in cp_dist_signature.parameters:
                dist_parameters_values.append(single_param_dist_config_dict[p])
            dists.append(getattr(cp, single_param_dist_config_dict["distribution"])(*dist_parameters_values))
            standardDists.append(getattr(cp, single_param_dist_config_dict["distribution"])())
            standardDists_min_one_one.append(getattr(cp, single_param_dist_config_dict["distribution"])(lower=-1, upper=1))
            standardDists_zero_one.append(getattr(cp, single_param_dist_config_dict["distribution"])(lower=0, upper=1))
        jointDists = cp.J(*dists)
        jointStandardDists = cp.J(*standardDists)
        jointDistsStandardDists_min_one_one = cp.J(*standardDists_min_one_one)

    # ============================
    # Debugging the model run...
    # ============================

    if local_debugging and is_master(mpi, rank):
        coordinates = jointDists.sample(size=1, rule='random')
        coordinates = np.array(coordinates)
        results = problem_function.eval(coordinates=coordinates.T)
        print(f"INFO: eval results={results}")

        coordinates = jointDists.sample(size=10, rule='random')
        coordinates = np.array(coordinates)
        results_vector = problem_function.eval_vectorized(coordinates=coordinates.T)
        print(f"INFO: eval_vectorized results_vector={results_vector}")
        
        # results_vector = problem_function(coordinates=coordinates.T)
        # print(f"__call__ with coordinates results_vector={results_vector}")

        # results_list = problem_function.run(
        #     i_s=range(coordinates.T.shape[0]), parameters=coordinates.T, raise_exception_on_model_break=True)
        # print(f"run results_list={results_list}")

        # results_list = problem_function(
        #     i_s=range(coordinates.T.shape[0]), parameters=coordinates.T, raise_exception_on_model_break=True)
        # print(f"run with i_s and parameters results_list={results_list}")

    # ============================

    if compute_uq or compute_pce_from_uq:
        gPCE_surrogate = None

        # NOTE: this is needed; so uqef never uses PCE from sparsespace as a surrogate
        if surrogate_object is not None and uqsim_args_dict.get("surrogate_type", None) == "pce_sparsespace":
            # NOTE: this is a combiinstance in this setup
            surrogate_to_build_pce_uqef_on = intermediate_surrogate_object 
        else:
            surrogate_to_build_pce_uqef_on = surrogate_object

        result_dict_uqef = run_simplified_uqef_dynamic_simulation(
            problem_function=problem_function, configurationObject=configurationObject, uqsim_args_dict=uqsim_args_dict,
            workingDir=workingDir, model=model,
            simulationNodes=simulationNodes, 
            list_of_surrogate_models=["larsim", "hbvsask", "ishigami", "oscillator",],
            surrogate_object=surrogate_to_build_pce_uqef_on,)

        gPCE_surrogate = result_dict_uqef.pop("gPCE_surrogate", None)
        problem_statistics_result_dict = result_dict_uqef.pop("problem_statistics_result_dict", None)
        # approximated_mean = result_dict_uqef.get("E", None)
        # approximated_var = result_dict_uqef.get("Var", None)
        # first_order_sobol_indices = result_dict_uqef.get("Sobol_m", None)
        # total_order_sobol_indices = result_dict_uqef.get("Sobol_t", None)
        # print(f"DEBUGGING - problem_statistics_result_dict-{problem_statistics_result_dict}")
        
        if gPCE_surrogate is not None and compute_pce_from_uq:
            intermediate_surrogate_object = surrogate_object
            if intermediate_surrogate_object is not None:
                uqsim_args_dict["intermediate_surrogate_type"] = dictionary_with_inf_about_the_run["intermediate_surrogate_type"] = uqsim_args_dict.get("surrogate_type", "combiinstance")  # 
            if isinstance(gPCE_surrogate, dict) and single_qoi in gPCE_surrogate:
                gPCE_surrogate = gPCE_surrogate[single_qoi]
            # print(f"DEBUGGING - gPCE_surrogate-{gPCE_surrogate}")
            surrogate_object = gPCE_surrogate #gPCE
            uqsim_args_dict["surrogate_type"] = dictionary_with_inf_about_the_run["surrogate_type"] = "pce"
        else:
            uqsim_args_dict["compute_pce_from_uq"] = compute_pce_from_uq = False

        dictionary_with_inf_about_the_run.update(result_dict_uqef)

    # ============================
    # III
    # # ============================
    # # TODO Add extraction - PCE, E, Mean, Sobol... simple, complex-vector...

    if is_master(mpi, rank):
        reevaluate_surrogate = kwargs.get("reevaluate_surrogate", False)
        reevaluate_original_model = kwargs.get("reevaluate_original_model", False)
        reevaluate_intermediate_surrogate = kwargs.get("reevaluate_intermediate_surrogate", False)
    
        if surrogate_object is None:
            reevaluate_surrogate = False
        if intermediate_surrogate_object is None:
            reevaluate_intermediate_surrogate = False

        if reevaluate_surrogate or reevaluate_original_model or reevaluate_intermediate_surrogate:
            number_of_samples_model_comparison = kwargs.get('number_of_samples_model_comparison', 1000)
            sampling_rule_model_comparison = kwargs.get('sampling_rule_model_comparison', "random")
            sample_nodes_from_standard_dist_model_comparison = kwargs.get('sample_nodes_from_standard_dist_model_comparison', True)
            read_nodes_from_file_model_comparison = kwargs.get('read_nodes_from_file_model_comparison', False)
            parameters_file_name_model_comparison = kwargs.get('parameters_file_name_model_comparison', None)
            rounding_model_comparison = kwargs.get('rounding_model_comparison', False)
            round_dec_model_comparison = kwargs.get('round_dec_model_comparison', 4)

            evaluateSurrogateAtStandardDist = False
            if uqsim_args_dict.get("surrogate_type", None) == "combiinstance":
                evaluateSurrogateAtStandardDist = False
            elif uqsim_args_dict.get("surrogate_type", None) == "pce":
                evaluateSurrogateAtStandardDist = True
            elif uqsim_args_dict.get("surrogate_type", None) == "pce_sparsespace":
                evaluateSurrogateAtStandardDist = False

            evaluateIntermediateSurrogateAtStandardDist = False
            if uqsim_args_dict.get("intermediate_surrogate_type", None) == "combiinstance":
                evaluateIntermediateSurrogateAtStandardDist = False
            elif uqsim_args_dict.get("intermediate_surrogate_type", None) == "pce":
                evaluateIntermediateSurrogateAtStandardDist = True
            elif uqsim_args_dict.get("surrogate_type", None) == "pce_sparsespace":
                evaluateIntermediateSurrogateAtStandardDist = False

            set_lower_predictions_to_zero = kwargs.get('set_lower_predictions_to_zero', False)
            
            # TODO Check this
            parameters_model_comparison = uqef_dynamic_utils.generate_parameters_for_mc_simulation(
                jointDists=jointDists, jointStandard=jointStandardDists, 
                numSamples=number_of_samples_model_comparison, 
                rule=sampling_rule_model_comparison,
                sampleFromStandardDist=sample_nodes_from_standard_dist_model_comparison, 
                read_nodes_from_file=read_nodes_from_file_model_comparison, 
                parameters_file_name=parameters_file_name_model_comparison,
                rounding=rounding_model_comparison, round_dec=round_dec_model_comparison,
            )
            if evaluateSurrogateAtStandardDist and jointStandardDists is not None:
                nodes_model_comparison = utility.transformation_of_parameters(
                    parameters_model_comparison, jointDists, jointStandardDists)
            else:
                nodes_model_comparison = parameters_model_comparison
            
            list_unique_generate_new_samples = range(parameters_model_comparison.shape[1]) # unique index for each sample
            number_of_samples_model_comparison = parameters_model_comparison.shape[1]
            # print(f"DEBUGGING: nodes_model_comparison.shape={nodes_model_comparison.shape}")
            uqsim_args_dict["reevaluate_surrogate"] = reevaluate_surrogate
            uqsim_args_dict["reevaluate_original_model"] = reevaluate_original_model
            uqsim_args_dict["reevaluate_intermediate_surrogate"] = reevaluate_intermediate_surrogate
            uqsim_args_dict["number_of_samples_model_comparison"] = number_of_samples_model_comparison
            uqsim_args_dict["sampling_rule_model_comparison"] = sampling_rule_model_comparison
            uqsim_args_dict["read_nodes_from_file_model_comparison"] = read_nodes_from_file_model_comparison
            dictionary_with_inf_about_the_run["nodes_model_comparison"] = nodes_model_comparison
            dictionary_with_inf_about_the_run["parameters_model_comparison"] = parameters_model_comparison

        # ============================

        results_array_original_model = None
        if reevaluate_original_model:
            start_time_reevaluating_original_model = time.time()
            print(f"\n===Reevaluating the original model...===")
            ## Var 0
            # modelObject = create_stat_object.create_model_object(
            #             configuration_object=configurationObject, uqsim_args_dict=uqsim_args_dict, workingDir=directory_for_saving_plots, model=None, 
            #             time_column_name=utility.TIME_COLUMN_NAME, index_column_name=utility.INDEX_COLUMN_NAME
            #         )
            # modelObject = problem_function
            # results = modelObject(i_s=range(parameters_model_comparison.T.shape[0]), parameters=parameters_model_comparison.T, raise_exception_on_model_break=False)
            # df_model_reevaluated, df_index_parameter_reevaluated, _, _, _, _ =  uqef_dynamic_utils.uqef_dynamic_model_run_results_array_to_dataframe(\
            #     results_array=results, extract_only_qoi_columns=False, qoi_columns=modelObject.list_qoi_column, 
            #     time_column_name=utility.TIME_COLUMN_NAME, index_column_name=utility.INDEX_COLUMN_NAME)
            ## Var 1
            num_cores = multiprocessing.cpu_count()
            parameter_chunks = np.array_split(parameters_model_comparison.T, num_cores)
            results_array_original_model = []
            def process_nodes_concurrently(parameter_chunks):
                with multiprocessing.Pool(processes=num_cores) as pool:
                    for result in pool.starmap(evaluate_chunk_model, \
                                            [(problem_function, parameter) for parameter in parameter_chunks]):
                        yield result
            for result in process_nodes_concurrently(parameter_chunks):
                results_array_original_model.append(result)
            results_array_original_model = np.vstack(np.array(results_array_original_model))
            ## Var 2
            # def evaluate_chunk_original_model(chunk):
            #     return np.array(problem_function(chunk))
            # with multiprocessing.Pool(processes=num_cores) as pool:
            #     results = pool.map(evaluate_chunk_original_model, parameter_chunks)
            # results_array_original_model = np.concatenate(results)
            ## Var 3
            # results_array_original_model = problem_function(coordinates=parameters_model_comparison.T)
            if results_array_original_model is not None:
                print(f"DEBUGGING results_array_original_model.shape-{results_array_original_model.shape}")
                end_time_reevaluating_original_model = time.time()
                dictionary_with_inf_about_the_run["number_original_model_reevaluations"] = len(results_array_original_model)
                dictionary_with_inf_about_the_run["time_parallel_original_model_reevaluations"] = end_time_reevaluating_original_model - start_time_reevaluating_original_model
                print(f"INFO: Time for evaluation original model {len(results_array_original_model)} time with {num_cores} processes is: {dictionary_with_inf_about_the_run['time_parallel_original_model_reevaluations']}")
                if set_lower_predictions_to_zero:
                    pass

        # ============================

        # TODO make sure this works when gPCE is a surrogate model
        results_array_surrogate_model = None
        if surrogate_object is not None and reevaluate_surrogate:
            start_time_reevaluating_surrogate_model = time.time()
            print(f"\n===Reevaluating the surroget model of type {uqsim_args_dict.get('surrogate_type', None)}...===")
            num_cores = multiprocessing.cpu_count()

            # with multiprocessing.Pool(processes=num_cores) as pool:
            #     results = pool.starmap(evaluate_chunk_surrogate, [(surrogate_object, parameter_values) for parameter_values in parameter_chunks])
            # results_array_surrogate_model = np.concatenate(results)
            if uqsim_args_dict.get("surrogate_type", None) == "pce":
                # DEBUGGING for simple surrogate...
                # timestamp = 0.0
                # gpce_surrogate = surrogate_object[timestamp]
                # results_array_surrogate_model = gpce_surrogate(*nodes_model_comparison)

                if evaluateSurrogateAtStandardDist:
                    parameter_chunks = np.array_split(nodes_model_comparison.T, num_cores)
                else:
                    parameter_chunks = np.array_split(parameters_model_comparison.T, num_cores)
                indices_chunks = np.array_split(list_unique_generate_new_samples, num_cores)
                if isinstance(surrogate_object, dict):
                    # statistics_pdTimesteps = dictionary_with_inf_about_the_run.get("statistics_pdTimesteps", None)
                    list_of_df_surrogate_evaluations = []
                    def process_nodes_concurrently(parameter_chunks, indices_chunks):
                        with multiprocessing.Pool(processes=num_cores) as pool:
                            # NOTE: for chaospy pce surrogate parameters should be of size dim x num_samples
                            # NOTE: in complext set-up (i.e., vector / time dependent result is a dataframe)
                            for result in pool.starmap(evaluate_over_time_gpce_model, \
                                                    [(surrogate_object, parameter.T, indices_list, single_qoi) for parameter, indices_list in zip(parameter_chunks, indices_chunks)]):
                                yield result
                    for result in process_nodes_concurrently(parameter_chunks, indices_chunks):
                        list_of_df_surrogate_evaluations.append(result)
                    df_surrogate_reevaluated = pd.concat(list_of_df_surrogate_evaluations, ignore_index=True, sort=False, axis=0)
                    df_surrogate_reevaluated.sort_values(
                        by=[utility.INDEX_COLUMN_NAME, utility.TIME_COLUMN_NAME], ascending=[True, True], 
                        inplace=True, kind='quicksort', na_position='last'
                    )
                    if df_surrogate_reevaluated is not None:
                        df_surrogate_reevaluated.to_pickle(
                            os.path.abspath(os.path.join(str(workingDir), "df_surrogate_reevaluated.pkl")), compression="gzip")
                        results_array_surrogate_model = np.array(df_surrogate_reevaluated[single_qoi].values)
                    else:
                        results_array_surrogate_model = None
                else:
                    # NOTE: This non parallel version works for simple pce surrogates
                    # if evaluateSurrogateAtStandardDist:
                    #     results_array_surrogate_model = surrogate_object(*nodes_model_comparison)
                    # else:
                    #     results_array_surrogate_model = surrogate_object(*parameters_model_comparison)
                    results_array_surrogate_model = []
                    def process_nodes_concurrently(parameter_chunks):
                        with multiprocessing.Pool(processes=num_cores) as pool:
                            # NOTE: for chaospy pce surrogate parameters should be of size dim x num_samples
                            for result in pool.starmap(evaluate_simple_gpce_model, \
                                                    [(surrogate_object, parameter.T) for parameter in parameter_chunks]):
                                yield result
                    for result in process_nodes_concurrently(parameter_chunks):
                        results_array_surrogate_model.append(result)
                    results_array_surrogate_model = np.vstack(np.array(results_array_surrogate_model))
            elif uqsim_args_dict.get("surrogate_type", None) == "pce_sparsespace":
                # NOTE: This should work for pce produced by the sparsespace
                # TODO this assumes simple pce from sparsespace surrogate for now; not over time...
                if evaluateSurrogateAtStandardDist:
                    results_array_surrogate_model = surrogate_object(*nodes_model_comparison)
                else:
                    results_array_surrogate_model = surrogate_object(*parameters_model_comparison)
            else:
                # NOTE: This non parallel version works for sparse grid surrogates
                if evaluateSurrogateAtStandardDist:
                    results_array_surrogate_model = surrogate_object(nodes_model_comparison.T)
                else:
                    results_array_surrogate_model = surrogate_object(parameters_model_comparison.T)
            
            if results_array_surrogate_model is not None:
                results_array_surrogate_model = results_array_surrogate_model.reshape(number_of_samples_model_comparison, problem_function.output_length())
                print(f"DEBUGGING results_array_surrogate_model.shape-{results_array_surrogate_model.shape}")
                end_time_reevaluating_surrogate_model = time.time()
                dictionary_with_inf_about_the_run["number_surrogate_model_reevaluations"] = len(results_array_surrogate_model)
                dictionary_with_inf_about_the_run["time_parallel_surrogate_model_reevaluations"] = end_time_reevaluating_surrogate_model - start_time_reevaluating_surrogate_model
                print(f"INFO: Time for evaluation surrogate model {len(results_array_surrogate_model)} time (with one process) is: {dictionary_with_inf_about_the_run['time_parallel_surrogate_model_reevaluations']}")
                if set_lower_predictions_to_zero:
                    pass

        # ============================

        # TODO: See what to do when ther is both combiinstance and pce_sparsespace as intermediate surrogates...
        results_array_intermediate_surrogate_model = None
        if intermediate_surrogate_object is not None and reevaluate_intermediate_surrogate:
            start_time_reevaluating_intermediate_surrogate_model = time.time()
            print(f"\n===Reevaluating intermediate the surroget model of type {uqsim_args_dict.get('intermediate_surrogate_type', None)}...===")
            # NOTE: These are only options for intermediate surrogates...
            if uqsim_args_dict.get('intermediate_surrogate_type', None) == "pce_sparsespace":
                if evaluateIntermediateSurrogateAtStandardDist:
                    results_array_intermediate_surrogate_model = intermediate_surrogate_object(*nodes_model_comparison)
                else:
                    results_array_intermediate_surrogate_model = intermediate_surrogate_object(*parameters_model_comparison)
            else:
                if evaluateIntermediateSurrogateAtStandardDist:
                    results_array_intermediate_surrogate_model = intermediate_surrogate_object(nodes_model_comparison.T)
                else:
                    results_array_intermediate_surrogate_model = intermediate_surrogate_object(parameters_model_comparison.T)
            # num_cores = multiprocessing.cpu_count()
            # parameter_chunks = np.array_split(parameters_model_comparison.T, num_cores)
            # def evaluate_chunk(chunk):
            #     return np.array([intermediate_surrogate_object(parameter) for parameter in chunk])
            # with multiprocessing.Pool(processes=num_cores) as pool:
            #     results = pool.map(evaluate_chunk, parameter_chunks)
            # results_array_intermediate_surrogate_model = np.concatenate(results)
            if results_array_intermediate_surrogate_model is not None:
                results_array_intermediate_surrogate_model = results_array_intermediate_surrogate_model.reshape(number_of_samples_model_comparison, problem_function.output_length())
                end_time_reevaluating_intermediate__surrogate_model = time.time()
                dictionary_with_inf_about_the_run["number_intermediate_surrogate_model_reevaluations"] = len(results_array_intermediate_surrogate_model)
                dictionary_with_inf_about_the_run["time_parallel_intermediate_surrogate_model_reevaluations"] = end_time_reevaluating_intermediate__surrogate_model - start_time_reevaluating_intermediate_surrogate_model
                print(f"INFO: Time for evaluation (intermediate) surrogate model {len(results_array_intermediate_surrogate_model)} time is {dictionary_with_inf_about_the_run['time_parallel_intermediate_surrogate_model_reevaluations']}")
                if set_lower_predictions_to_zero:
                    pass

        # ============================

        compare_surrogate_and_original_model_runs = False
        if reevaluate_original_model and reevaluate_surrogate and results_array_original_model is not None and results_array_surrogate_model is not None:
            compare_surrogate_and_original_model_runs = True

        compare_intermediate_surrogate_and_original_model_runs = False
        if reevaluate_original_model and reevaluate_intermediate_surrogate and results_array_original_model is not None and results_array_intermediate_surrogate_model is not None:
            compare_intermediate_surrogate_and_original_model_runs = True

        if compare_surrogate_and_original_model_runs:
            print(f"\n===Comparing Original and ({uqsim_args_dict.get('surrogate_type', None)}) Surrogate===")
            resul_dict_comparing_mode_and_surrogate = compute_numpy_array_errors(results_array_surrogate_model, results_array_original_model, printing=True)
            dictionary_with_inf_about_the_run.update(resul_dict_comparing_mode_and_surrogate)

        if compare_intermediate_surrogate_and_original_model_runs:
            print(f"\n===Comparing Original and Intermediate({uqsim_args_dict.get('intermediate_surrogate_type', None)}) Surrogate===")
            resul_dict_comparing_mode_and_intermediate_surrogate = compute_numpy_array_errors(results_array_intermediate_surrogate_model, results_array_original_model, printing=True)
            resul_dict_comparing_mode_and_intermediate_surrogate_update = {f"intermediate_{key}": value for key, value in resul_dict_comparing_mode_and_intermediate_surrogate.items()}
            dictionary_with_inf_about_the_run.update(resul_dict_comparing_mode_and_intermediate_surrogate_update)
    
    # ============================
    # Some final saving
    # ============================

    if is_master(mpi, rank):
        
        if configurationObject is not None:
            fileName = pathlib.Path(workingDir) / utility.CONFIGURATION_OBJECT_FILE
            with open(fileName, 'wb') as handle:
                dill.dump(configurationObject, handle)

        argsFileName = os.path.abspath(os.path.join(str(workingDir), utility.ARGS_FILE))
        with open(argsFileName, 'wb') as handle:
            pickle.dump(uqsim_args_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        fileName = pathlib.Path(workingDir) / utility.DICT_INFO_FILE
        with open(fileName, 'wb') as handle:
            pickle.dump(dictionary_with_inf_about_the_run, handle, protocol=pickle.HIGHEST_PROTOCOL)
            # dill.dump(dictionary_with_inf_about_the_run, handle)

    return dictionary_with_inf_about_the_run


# if __name__ == "__main__":

# ============================
# Simulation setup
# ============================

list_of_dict_run_setups = [
    {"model": "corner_peak", "list_of_function_ids": [1, ], 
    "current_output_folder": "ct_trapez_lmax2_tol_10-6_maxeval1000_bound_nomodify_norm2",
    "grid_type": 'trapezoidal', "method": 'standard_combi', "minimum_level": 1, "maximum_level": 2, 
    "max_evaluations":100, "tol":10**-6, "modified_basis":False, "boundary":True, "norm":2, "p_bsplines":3, 
    "rebalancing":True, "version":6, "margin":0.8, "grid_surplusses":'grid'},
]


can_model_evaluate_all_vector_nodes = True
config_file = pathlib.Path('/work/ga45met/mnt/linux_cluster_2/UQEF-Dynamic/data/configurations/configuration_ishigami.json')
outputModelDir = pathlib.Path('/work/ga45met/uqef_dynamic_runs/ishigami_runs/sg_anaysis_feb_25')
list_of_dict_run_setups = [
    {"model":'ishigami',
    "current_output_folder":'sact_trapez_uq_mean_lmax4_maxeval1000_bound_nomodify_norm2_157mc_random_p5_regression_ct10', 
    "config_file": config_file,
    "outputModelDir": outputModelDir,
    "can_model_evaluate_all_vector_nodes": can_model_evaluate_all_vector_nodes,
    "single_qoi":None,
    "compute_sparsespace": True, 
    "compute_pce_from_sparsespace": False,
    "compute_uq": True, 
    "compute_pce_from_uq": True,
    "grid_type": 'trapezoidal', # try trapezoidal vs gauss_legendre
    "method": 'dim_wise_spat_adaptive_combi',  #  'standard_combi', 'dim_adaptive_combi', 'dim_wise_spat_adaptive_combi'
    "operation_str": "uq", # try out "uq" | 'integration'
    "uq_optimization": 'mean',  # try 'mean' | 'mean_and_var' | 'pce'; Default: 'mean'
    "grid_surplusses":'grid',  # Try with None
    "minimum_level": 1, "maximum_level": 4, 
    "max_evaluations":1000, "tol":10**-6, "modified_basis":False, "boundary":True, "norm":2, "p_bsplines":3, 
    "rebalancing":True, "version":6, "margin":0.8,
    "scale_weights": False,
    "uq_method": "mc", "regression": True,
    "read_nodes_from_file": False, "parameters_file": None, "sampleFromStandardDist": True,
    "mc_numevaluations": 157, "sampling_rule":"random",
    "sc_p_order": 5,  "sc_q_order": 6, "cross_truncation": 1.0,
    "reevaluate_surrogate":True, 
    "reevaluate_original_model":True,
    "reevaluate_intermediate_surrogate": True, 
    "number_of_samples_model_comparison": 1000, "sampling_rule_model_comparison": "random",
    "sample_nodes_from_standard_dist_model_comparison": True, 
    },
]

# list_of_dict_run_setups = [
#     {"model":'ishigami',
#     "current_output_folder":'157mc_random_p7_regression', 
#     "config_file": config_file,
#     "outputModelDir": outputModelDir,
#     "can_model_evaluate_all_vector_nodes": can_model_evaluate_all_vector_nodes,
#     "single_qoi":None,
#     "compute_sparsespace": False, 
#     "compute_pce_from_sparsespace": False,
#     "compute_uq": True, 
#     "compute_pce_from_uq": True,
    # "uq_method": "mc", "regression": True,
    # "read_nodes_from_file": False, "parameters_file": None, "sampleFromStandardDist": True,
    # "mc_numevaluations": 157, "sampling_rule":"random",
    # "sc_p_order": 7,  "sc_q_order": 6, "cross_truncation": 1.0,
#     "reevaluate_surrogate":True, 
#     "reevaluate_original_model":True,
#     "reevaluate_intermediate_surrogate": True, 
#     "number_of_samples_model_comparison": 1000, "sampling_rule_model_comparison": "random",
#     "sample_nodes_from_standard_dist_model_comparison": True, 
#     },
# ]


# #sact_trapez_lmax2_tol_10_6_maxeval1000_nobound_nomodify_norm2
# can_model_evaluate_all_vector_nodes = True
# inputModelDir = pathlib.Path("/work/ga45met/Hydro_Models/HBV-SASK-data")
# outputModelDir = pathlib.Path('/work/ga45met/uqef_dynamic_runs/hbv_sask_runs/Oldman_Basin')
# config_file = pathlib.Path('/work/ga45met/mnt/linux_cluster_2/UQEF-Dynamic/data/configurations/configuration_hbv_10D_single_qoi.json')
# single_qoi={'qoi':'Q_cms', 'gof':'RMSE'}  # 'Q_cms', 'RMSE' | None
# surrogate_type='combiinstance' # 'pce' | 'kl+pce' | 'sg' | 'combiinstance' | 'sgi+pce' | 'kl+sg' | 'kl+combiinstance' | 'kl+sgi+pce'
# list_of_dict_run_setups = [
#     {"model": "hbvsask", 
#     "current_output_folder": "ct_trapez_lmax2_tol_10_6_maxeval100_bound_nomodify_norm2_1000mc_random",
#     "inputModelDir":inputModelDir,
#     "outputModelDir": outputModelDir,
#     "config_file": config_file,
#     "can_model_evaluate_all_vector_nodes": can_model_evaluate_all_vector_nodes,
#     "single_qoi":None,  #single_qoi
#     "compute_sparsespace": True, 
#     "compute_pce_from_sparsespace": False,
#     "compute_uq": True, 
#     "compute_pce_from_uq": True,
#     "grid_type": 'trapezoidal', # try trapezoidal vs gauss_legendre
#     "method": 'standard_combi',  #  'standard_combi', 'dim_adaptive_combi', 'dim_wise_spat_adaptive_combi'
#     "operation_str": "uq", # try out "uq" | 'integration'
#     "uq_optimization": 'mean',  # try 'mean' | 'mean_and_var' | 'pce'; Default: 'mean'
#     "grid_surplusses":'grid', 
#     "method": 'standard_combi', 
#     "minimum_level": 1, "maximum_level": 2, 
#     "max_evaluations":100, "tol":10**-6, "modified_basis":False, "boundary":True, "norm":2, "p_bsplines":3, 
#     "rebalancing":True, "version":6, "margin":0.8,
#     "uq_method": "mc", "regression": False,
#     "read_nodes_from_file": False, "parameters_file": "/work/ga45met/sparseSpACE/sparse_grid_nodes_weights", "l_sg":4,
#     "sampleFromStandardDist": True,
#     "mc_numevaluations": 1000, "sampling_rule":"random",
#     "sc_p_order": 3,  "sc_q_order": 6, "cross_truncation": 0.7,
#     "reevaluate_surrogate":True, 
#     "reevaluate_original_model":True,
#     "reevaluate_intermediate_surrogate": True, 
#     "number_of_samples_model_comparison": 1000, "sampling_rule_model_comparison": "random",
#     "sample_nodes_from_standard_dist_model_comparison": True, 
#     "set_lower_predictions_to_zero" : True,
#     },
# ]

# ============================
# Initial Model Setup
# ============================
list_of_models = ["hbvsask", "larsim", "ishigami", "gfunction", "zabarras2d", "zabarras3d",
                    "oscillatory", "product_peak", "corner_peak", "gaussian", "discontinuous"]

# Hard-coded for Genz functions
# Additional Genz Options: GenzOszillatory, GenzDiscontinious2, GenzC0, GenzGaussian
path_to_saved_all_genz_functions = pathlib.Path("/work/ga45met/sg_anaysis/genz_functions")
read_saved_genz_functions = True
anisotropic = True

# ============================
def run_single_model_setup(single_setup_dict, model, current_output_folder, read_saved_genz_functions, anisotropic, path_to_saved_all_genz_functions):
    if model in sparsespace_functions.LIST_OF_GENZ_FUNCTIONS and single_setup_dict.get("list_of_function_ids") is not None:
        # Hard-coded
        dim = 5
        dictionary_with_inf_about_the_run = defaultdict(dict)
        list_of_function_ids = single_setup_dict["list_of_function_ids"]
        base_output_folder = current_output_folder
        for i in list_of_function_ids:
            if read_saved_genz_functions:
                if anisotropic:
                    path_to_saved_genz_functions = str(path_to_saved_all_genz_functions / model / f"coeffs_weights_anisotropic_{dim}d_{i}.npy")
                else:
                    path_to_saved_genz_functions = str(path_to_saved_all_genz_functions / model / f"coeffs_weights_{dim}d_{i}.npy")
                with open(path_to_saved_genz_functions, 'rb') as f:
                    coeffs_weights = np.load(f)
                    single_coeffs = coeffs_weights[0]
                    single_weights = coeffs_weights[1]
            else:
                single_coeffs, single_weights = sparsespace_functions.generate_and_scale_coeff_and_weights_genz(
                    dim=dim, b=sparsespace_functions.GENZ_DICT[model.lower()], anisotropic=anisotropic)
            current_output_folder_single_model = f"{current_output_folder}_model_{i}"
            single_setup_dict["current_output_folder"] = current_output_folder_single_model
            single_setup_dict["coeffs"] = single_coeffs
            single_setup_dict["weights"] = single_weights

            dictionary_with_inf_about_the_run_single_model = main_routine(**single_setup_dict,)
            dictionary_with_inf_about_the_run[i] = dictionary_with_inf_about_the_run_single_model
    else:
        dictionary_with_inf_about_the_run = main_routine(**single_setup_dict)
    return dictionary_with_inf_about_the_run

# for single_setup_dict in list_of_simulation_runs_ishigami:
for single_setup_dict in list_of_dict_run_setups:
    if is_master(mpi, rank):
        start_time = time.time()

    temp = single_setup_dict.get("list_of_function_ids", None)
    number_of_functions = 1 if temp is None else len(temp)
    model = single_setup_dict["model"]
    assert(model in list_of_models)
    current_output_folder = single_setup_dict['current_output_folder']
    dictionary_with_inf_about_the_run = run_single_model_setup(single_setup_dict, model, current_output_folder, read_saved_genz_functions, anisotropic, path_to_saved_all_genz_functions)
        # model='ishigami', current_output_folder='ct_trapez_lmax2_tol_10-6_maxeval1000_bound_nomodify_norm2', \
        # grid_type='trapezoidal', method='dim_adaptive_combi', minimum_level=1, maximum_level=2, 
        # max_evaluations=1000, tol=10**-6, modified_basis=False, boundary=True, norm=2, p_bsplines=3, 
        # rebalancing=True, version=6, margin=0.8, grid_surplusses='grid'
    
    if is_master(mpi, rank):
        print(f"\n===Final Output - Single Set-up ===")
        print(f"INFO: dictionary_with_inf_about_the_run-{dictionary_with_inf_about_the_run}")
        end_time = time.time()
        duration = end_time - start_time
        print(f"INFO: The single setup run took {duration} (for examing in total {number_of_functions} different functions)")

# ============================
# Observations  - max_evaluations is dominant! 
# CT with the same max level as SACT leades to many more runs
# DACT allowes lmax to be max 2
# DACT needs reference_solution/real_integral!!!

# TODO Add PCE Part and computation of Stat

# # Example
# a = np.zeros(2)
# b = np.ones(2)
# model = FunctionExpVar()
# result_dict_sparsespace= sparsespace_utils.sparsespace_pipeline(
#     a, b, model=model, dim=2, 
#     grid_type='trapezoidal', method='standard_combi',
#     directory_for_saving_plots='./', do_plot=True)
