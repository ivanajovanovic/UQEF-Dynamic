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

# TODO Add simple pce-part -> produce PCE surrogate
# TODO Evaluate PCE surrogate (from UQEF-Dynamic, from simple statistics, from combiinstance)
# TODO HBV Gof inverse logic
# TODO PCE part from SparseSpace - partially
# TODO Things will get more complicated when using the combiinstance to produce PCE surrogate!!!
# TODO Add comparing final E, Var, Sobol indices with analytical values / MC
# TODO Plotting Issue with SparseSpace

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
        print(f"Reading nodes values from parameters file {parameters_file}")
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

    # From this point on distinguish between the two setup 
    # when configurationObject and simulationNodes are not None - Complex model
    # vs. simple model when configurationObject is None and simulationNodes is None
    
    if is_master(mpi, rank):

        uq_method = uqsim_args_dict["uq_method"]

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
            print(f"====[PCE PART INFO] - building the gPCE of the original model====")
            results_list = problem_function.run(
                i_s=range(parameters.T.shape[0]), 
                parameters=parameters.T, 
                raise_exception_on_model_break=True,
                evaluate_surrogate=False,
                surrogate_model=None,
                single_qoi_column_surrogate=problem_function.single_qoi,
            )
            dictionary_with_inf_about_the_run_uqef["number_full_model_evaluations"] = parameters.T.shape[0]
        eLse:
            print(f"====[PCE PART INFO] - building the gPCE without the surrogate model====")
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
        print(f"time_evaluationg_model_when_building_gpce={dictionary_with_inf_about_the_run_uqef['time_evaluationg_model_when_building_gpce']}\n")

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
            print(f"time_evalauting_original_model_uqef={dictionary_with_inf_about_the_run_uqef['time_evalauting_original_model_uqef']}\n")

    # ============================
    # Statistics
    # ============================
    problem_statistics = None
    if model.lower() in list_of_surrogate_models:
        problem_statistics = create_stat_object.create_statistics_object(
            configuration_object=configurationObject, 
            uqsim_args_dict=uqsim_args_dict, 
            workingDir=workingDir, model=model)
    else:
        problem_statistics = None
        raise ValueError(f"Model {model} is not yet supported for generating statistics object!")
    
    gPCE_surrogate = None
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
            problem_statistics.saveToFile()

    # TODO Extract gPCE surrogate (if available) and use it below for the comparison with the original model
    
    if is_master(mpi, rank):
        end_time_computing_statistics = time.time()
        time_computing_statistics = end_time_computing_statistics - start_time_computing_statistics
        dictionary_with_inf_about_the_run_uqef["time_computing_statistics"] = time_computing_statistics
        print(f"time_computing_statistics={dictionary_with_inf_about_the_run_uqef['time_computing_statistics']}\n")
    
    dictionary_with_inf_about_the_run_uqef['gPCE_surrogate'] = gPCE_surrogate

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
        print(f"Overall errors: RMSE={overall_rmse}; linf={overall_linf};"
        f"l2_scaled={overall_l2_scaled}; L2_error={l2_error}; L1_error={l1_error}; mean_L1_error={mean_l1_error}")
        
    resul_dict = {}
    resul_dict['rmse_over_time'] = rmse_over_time
    resul_dict['overall_rmse'] = overall_rmse
    resul_dict['overall_linf'] = overall_linf
    resul_dict['overall_l2_scaled'] = overall_l2_scaled
    resul_dict['l2_error'] = l2_error
    resul_dict['l1_error'] = l1_error
    resul_dict['mean_l1_error'] = mean_l1_error
    return resul_dict


def evaluate_chunk_model(problem_function, chunk):
    # return np.array([problem_function(parameter) for parameter in chunk])
    return np.array(problem_function(coordinates=chunk))


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

    compute_ct_surrogate= kwargs.get("compute_ct_surrogate", True)
    compute_pce_from_ct_surrogate = kwargs.get("compute_pce_from_ct_surrogate", False)
    compute_pce_surrogate = kwargs.get("compute_pce_surrogate", True)
    if not compute_ct_surrogate or not compute_pce_surrogate:
        raise ValueError("The current implementation requires both ct and pce surrogates to be computed!")
    uqsim_args_dict["compute_ct_surrogate"] = compute_ct_surrogate
    uqsim_args_dict["compute_pce_surrogate"] = compute_pce_surrogate

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
   
    # ============================
    # I
    # ============================
    # TODO add a question to see if combiinstance is used as (intermediate) surrogate model
    if compute_ct_surrogate:
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
        # Collect all the above local variables into kwargs
        kwargs_sparsespace_pipeline = {
            key: value for key, value in locals().items() if key in \
                ['operation_str', 'grid_type', 'method', 'minimum_level', 'maximum_level', 'max_evaluations', 'tol', \
                    'modified_basis', 'boundary', 'norm', 'p_bsplines', 'rebalancing', 'version', 'margin', 'grid_surplusses', \
                        'distributions', 'sc_p_order', 'uq_optimization']}

        if operation_str.lower() not in ["uq", 'uncertaintyquantification', 'uncertainty quantification'] or uq_optimization.lower()!='pce':
            compute_pce_from_ct_surrogate = False
        uqsim_args_dict["compute_pce_from_ct_surrogate"] = compute_pce_from_ct_surrogate

        uqsim_args_dict.update(kwargs_sparsespace_pipeline)
        
        if is_master(mpi, rank):
            result_dict_sparsespace = sparsespace_utils.sparsespace_pipeline(
                a=a, b=b, model=problem_function,
                dim=dim, 
                directory_for_saving_plots=workingDir,
                do_plot=True,
                **kwargs_sparsespace_pipeline
            )
            # total_points, total_weights = combiObject.get_points_and_weights()
            # total_surplusses = combiObject.get_surplusses()
            combiObject = result_dict_sparsespace.pop('combiObject', None)
            number_full_model_evaluations = result_dict_sparsespace.get('number_full_model_evaluations', None)
            print(f"combiObject: {combiObject},\n number_full_model_evaluations: {number_full_model_evaluations},\n result_dict_sparsespace: {result_dict_sparsespace}")
            
            # gPCE = result_dict_sparsespace.get("gPCE", None)
            # approximated_mean = result_dict_sparsespace.get("E", None)
            # approximated_var = result_dict_sparsespace.get("Var", None)
            # first_order_sobol_indices = result_dict_sparsespace.get("Sobol_m", None)
            # total_order_sobol_indices = result_dict_sparsespace.get("Sobol_t", None)

            dictionary_with_inf_about_the_run.update(result_dict_sparsespace)

            # TODO Try to save combiObject in some other way
            # dictionary_with_inf_about_the_run["combiObject"] = combiObject
    
            # TODO Things will get more complicated when using the combiinstance to produce PCE surrogate!!!
            surrogate_object = combiObject
            uqsim_args_dict["surrogate_type"] = "combiinstance"
            dictionary_with_inf_about_the_run["surrogate_type"] = "combiinstance"
    
    # ============================
    # II
    # ============================

    # ============================
    # Building the PCE-surrogate model
    # ============================
    if compute_pce_surrogate:
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
            uqsim_args_dict["parameters_file"] = parameters_file / f"KPU_d{dim}_l{uqsim_args_dict['l_sg']}.asc"
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

    # ============================
    # Setiiing up the nodes / simulationNodes / distributions
    # ============================
    if configurationObject is not None:
        simulationNodes = setup_nodes_via_config_file_or_parameters_file(
            configuration_object=configurationObject, 
            uq_method=uqsim_args_dict.get("uq_method", "mc"), 
            read_nodes_from_file=uqsim_args_dict.get("read_nodes_from_file", False),
            parameters_file=uuqsim_args_dict.get("parameters_file", None),
            sampleFromStandardDist=uqsim_args_dict.get("sampleFromStandardDist", True), 
            regression=uqsim_args_dict.get("regression", False)
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
        print(f"eval results={results}")

        coordinates = jointDists.sample(size=10, rule='random')
        coordinates = np.array(coordinates)
        results_vector = problem_function.eval_vectorized(coordinates=coordinates.T)
        print(f"eval_vectorized results_vector={results_vector}")
        
        # results_vector = problem_function(coordinates=coordinates.T)
        # print(f"__call__ with coordinates results_vector={results_vector}")

        # results_list = problem_function.run(
        #     i_s=range(coordinates.T.shape[0]), parameters=coordinates.T, raise_exception_on_model_break=True)
        # print(f"run results_list={results_list}")

        # results_list = problem_function(
        #     i_s=range(coordinates.T.shape[0]), parameters=coordinates.T, raise_exception_on_model_break=True)
        # print(f"run with i_s and parameters results_list={results_list}")

    # ============================
    if compute_pce_surrogate:
        result_dict_uqef = run_simplified_uqef_dynamic_simulation(
            problem_function=problem_function, configurationObject=configurationObject, uqsim_args_dict=uqsim_args_dict,
            workingDir=workingDir, mode=model,
            simulationNodes=simulationNodes, 
            list_of_surrogate_models=["larsim", "hbvsask", "ishigami", "oscillator",],
            surrogate_object=surrogate_object, **kwargs)

        gPCE_surrogate = result_dict_uqef.get("gPCE_surrogate", None)
        # approximated_mean = result_dict_uqef.get("E", None)
        # approximated_var = result_dict_uqef.get("Var", None)
        # first_order_sobol_indices = result_dict_uqef.get("Sobol_m", None)
        # total_order_sobol_indices = result_dict_uqef.get("Sobol_t", None)

        intermediate_surrogate_object = surrogate_object
        surrogate_object = gPCE_surrogate #gPCE
        uqsim_args_dict["surrogate_type"] = dictionary_with_inf_about_the_run["surrogate_type"] = "pce"
        if intermediate_surrogate_object is not None:
            uqsim_args_dict["intermediate_surrogate_type"] = "combiinstance"
            dictionary_with_inf_about_the_run["intermediate_surrogate_type"] = "combiinstance"
    
        dictionary_with_inf_about_the_run.update(result_dict_uqef)

    # ============================
    # III
    # ============================
    # TODO Add extraction - PCE, E, Mean, Sobol... simple, complex-vector...

    if is_master(mpi, rank):
        reevaluate_surrogate = kwargs.get("reevaluate_surrogate", False)
        reevaluate_original_model = kwargs.get("reevaluate_original_model", False)

        if surrogate_object is None and intermediate_surrogate_object is None:
            reevaluate_surrogate = False

        if reevaluate_surrogate or reevaluate_original_model:
            # evaluateSurrogateAtStandardDist = sampleFromStandardDist_when_evaluating_surrogate
            evaluateSurrogateAtStandardDist = False
            if surrogate_type == "combiinstance":
                evaluateSurrogateAtStandardDist = False
            # TODO rename
            number_of_samples = kwargs.get('number_of_samples', 1000)
            sampling_rule = kwargs.get('sampling_rule', "random")
            # TODO Rename this so it is clear this file is used for reading nodes for re-evaulating the surrogate and original model
            sample_new_nodes_from_standard_dist = kwargs.get('sample_new_nodes_from_standard_dist', False)
            # TODO Rename this so it is clear this file is used for reading nodes for re-evaulating the surrogate and original model
            read_new_nodes_from_file = kwargs.get('read_new_nodes_from_file', False)
            new_parameters_file_name = kwargs.get('new_parameters_file_name', None)
            rounding = kwargs.get('rounding', False)
            round_dec = kwargs.get('round_dec', 4)

            set_lower_predictions_to_zero = kwargs.get('set_lower_predictions_to_zero', False)
            
            # TODO Check this
            parameters = uqef_dynamic_utils.generate_parameters_for_mc_simulation(
                jointDists=jointDists, jointStandard=jointStandardDists, 
                numSamples=number_of_samples, rule=sampling_rule,
                sampleFromStandardDist=sample_new_nodes_from_standard_dist, 
                read_nodes_from_file=read_new_nodes_from_file, 
                parameters_file_name=new_parameters_file_name,
                rounding=rounding, round_dec=round_dec,
            )
            if evaluateSurrogateAtStandardDist and jointStandardDists is not None:
                nodes = utility.transformation_of_parameters(
                    parameters, jointDists, jointStandardDists)
            else:
                nodes = parameters
            
            list_unique_generate_new_samples = range(parameters.shape[1]) # unique index for each sample
            number_of_samples_generated = parameters.shape[1]
            print(f"nodes.shape={nodes.shape}")
            uqsim_args_dict["reevaluate_surrogate"] = reevaluate_surrogate
            uqsim_args_dict["reevaluate_original_model"] = reevaluate_original_model
            uqsim_args_dict["number_of_samples_generated"] = number_of_samples_generated
            uqsim_args_dict["sampling_rule"] = sampling_rule
            uqsim_args_dict["read_new_nodes_from_file"] = read_new_nodes_from_file
             # TODO Rename
            dictionary_with_inf_about_the_run["nodes"] = nodes
            dictionary_with_inf_about_the_run["parameters"] = parameters

        intermediate_surrogate_evaluations = None
        results_array_intermediate_surrogate_model = None
        if intermediate_surrogate is not None:
            # Intermediate surrogate can only be combiinstance
            start_time_reevaluating_intermediate_surrogate_model = time.time()
            print(f"Reevaluating the surroget model...")
            results_array_intermediate_surrogate_model = intermediate_surrogate_object(parameters.T)
            # num_cores = multiprocessing.cpu_count()
            # parameter_chunks = np.array_split(parameters.T, num_cores)
            # def evaluate_chunk(chunk):
            #     return np.array([intermediate_surrogate(parameter) for parameter in chunk])
            # with multiprocessing.Pool(processes=num_cores) as pool:
            #     results = pool.map(evaluate_chunk, parameter_chunks)
            # results_array_intermediate_surrogate_model = np.concatenate(results)
            end_time_reevaluating_intermediate__surrogate_model = time.time()
            dictionary_with_inf_about_the_run["number_intermediate_surrogate_model_reevaluations"] = len(results_array_intermediate_surrogate_model)
            dictionary_with_inf_about_the_run["time_parallel_intermediate_surrogate_model_reevaluations"] = end_time_reevaluating_intermediate__surrogate_model - start_time_reevaluating_intermediate_surrogate_model
            print(f"Time for evaluation (intermediate) surrogate model {len(results_array_intermediate_surrogate_model)} time is {dictionary_with_inf_about_the_run['time_parallel_intermediate_surrogate_model_reevaluations']}")
            if set_lower_predictions_to_zero:
                pass

        results_array_original_model = None
        if reevaluate_original_model:
            start_time_reevaluating_original_model = time.time()
            print(f"Reevaluating the original model...")
            ## Var 0
            # modelObject = create_stat_object.create_model_object(
            #             configuration_object=configurationObject, uqsim_args_dict=uqsim_args_dict, workingDir=directory_for_saving_plots, model=None, 
            #             time_column_name=utility.TIME_COLUMN_NAME, index_column_name=utility.INDEX_COLUMN_NAME
            #         )
            # modelObject = problem_function
            # results = modelObject(i_s=range(parameters.T.shape[0]), parameters=parameters.T, raise_exception_on_model_break=False)
            # df_model_reevaluated, df_index_parameter_reevaluated, _, _, _, _ =  uqef_dynamic_utils.uqef_dynamic_model_run_results_array_to_dataframe(\
            #     results_array=results, extract_only_qoi_columns=False, qoi_columns=modelObject.list_qoi_column, 
            #     time_column_name=utility.TIME_COLUMN_NAME, index_column_name=utility.INDEX_COLUMN_NAME)
            ## Var 1
            num_cores = multiprocessing.cpu_count()
            parameter_chunks = np.array_split(parameters.T, num_cores)
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
            # results_array_original_model = problem_function(coordinates=parameters.T)
            end_time_reevaluating_original_model = time.time()
            dictionary_with_inf_about_the_run["number_original_model_reevaluations"] = len(results_array_original_model)
            dictionary_with_inf_about_the_run["time_parallel_original_model_reevaluations"] = end_time_reevaluating_original_model - start_time_reevaluating_original_model
            print(f"Time for evaluation original model {len(results_array_original_model)} time with {num_cores} processes is: {dictionary_with_inf_about_the_run['time_parallel_original_model_reevaluations']}")
            if set_lower_predictions_to_zero:
                pass

        results_array_surrogate_model = None
        if reevaluate_surrogate:
            start_time_reevaluating_surrogate_model = time.time()
            print(f"Reevaluating the surroget model...")
            num_cores = multiprocessing.cpu_count()
            parameter_chunks = np.array_split(nodes.T, num_cores)
            ## Var 1
            # results_array_surrogate_model = []
            # def process_nodes_concurrently(parameter_chunks):
            #     with multiprocessing.Pool(processes=num_cores) as pool:
            #         for result in pool.starmap(evaluate_chunk_model, \
            #                                 [(surrogate_object, parameter) for parameter in parameter_chunks]):
            #             yield result
            # for result in process_nodes_concurrently(parameter_chunks):
            #     results_array_surrogate_model.append(result)
            # results_array_surrogate_model = np.vstack(np.array(results_array_surrogate_model))
            # with multiprocessing.Pool(processes=num_cores) as pool:
            #     results = pool.starmap(evaluate_chunk_surrogate, [(surrogate_object, parameter_values) for parameter_values in parameter_chunks])
            # results_array_surrogate_model = np.concatenate(results)
            ## Var 2
            if surrogate_type == "combiinstance":
                results_array_surrogate_model = surrogate_object(nodes.T)
            else:
                results_array_surrogate_model = Nones
            print(f"DEBUGGING results_array_surrogate_model.shape-{results_array_surrogate_model.shape}")
            end_time_reevaluating_surrogate_model = time.time()
            dictionary_with_inf_about_the_run["number_surrogate_model_reevaluations"] = len(results_array_surrogate_model)
            dictionary_with_inf_about_the_run["time_parallel_surrogate_model_reevaluations"] = end_time_reevaluating_surrogate_model - start_time_reevaluating_surrogate_model
            print(f"Time for evaluation surrogate model {len(results_array_surrogate_model)} time (with one process) is: {dictionary_with_inf_about_the_run['time_parallel_surrogate_model_reevaluations']}")
            if set_lower_predictions_to_zero:
                pass

        compare_surrogate_and_original_model_runs = False
        if reevaluate_original_model and reevaluate_surrogate and results_array_original_model is not None and results_array_surrogate_model is not None:
            compare_surrogate_and_original_model_runs = True

        if compare_surrogate_and_original_model_runs:
            # Compute the element-wise error
            resul_dict = compute_numpy_array_errors(results_array_surrogate_model, results_array_original_model, printing=True)
            dictionary_with_inf_about_the_run.update(resul_dict)
            if results_array_intermediate_surrogate_model is not None:
                resul_dict = compute_numpy_array_errors(results_array_intermediate_surrogate_model, results_array_original_model, printing=True)
                resul_dict_update = {f"intermediate_{key}": value for key, value in resul_dict.items()}
                dictionary_with_inf_about_the_run.update(resul_dict_update)
   
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

#sact_trapez_lmax2_tol_10_6_maxeval1000_nobound_nomodify_norm2
can_model_evaluate_all_vector_nodes = True
inputModelDir = pathlib.Path("/work/ga45met/Hydro_Models/HBV-SASK-data")
outputModelDir = pathlib.Path('/work/ga45met/uqef_dynamic_runs/hbv_sask_runs/Oldman_Basin')
config_file = pathlib.Path('/work/ga45met/mnt/linux_cluster_2/UQEF-Dynamic/data/configurations/configuration_hbv_10D.json')
single_qoi={'qoi':'Q_cms', 'gof':'RMSE'}  # 'Q_cms', 'RMSE' | None
surrogate_type='combiinstance' # 'pce' | 'kl+pce' | 'sg' | 'combiinstance' | 'sgi+pce' | 'kl+sg' | 'kl+combiinstance' | 'kl+sgi+pce'
list_of_dict_run_setups = [
    {"model": "hbvsask", 
    "current_output_folder": "ct_trapez_lmax2_tol_10_6_maxeval1000_bound_nomodify_norm2",
    "inputModelDir":inputModelDir,
    "outputModelDir": outputModelDir,
    "config_file": config_file,
    "can_model_evaluate_all_vector_nodes": can_model_evaluate_all_vector_nodes,
    "grid_type": 'trapezoidal', "method": 'standard_combi', "minimum_level": 1, "maximum_level": 2, 
    "max_evaluations":1000, "tol":10**-6, "modified_basis":False, "boundary":True, "norm":2, "p_bsplines":3, 
    "rebalancing":True, "version":6, "margin":0.8, "grid_surplusses":'grid',
    "surrogate_type":'combiinstance', "single_qoi":None,
    "reevaluate_surrogate":True, "reevaluate_original_model":True,
    "number_of_samples":1000, "sampling_rule":"random", "sample_new_nodes_from_standard_dist":False,
    "set_lower_predictions_to_zero" : True,
    },
]

can_model_evaluate_all_vector_nodes = True
config_file = pathlib.Path('/work/ga45met/mnt/linux_cluster_2/UQEF-Dynamic/data/configurations/configuration_ishigami.json')
outputModelDir = pathlib.Path('/work/ga45met/uqef_dynamic_runs/ishigami_runs/sg_anaysis_feb_25')
list_of_dict_run_setups = [
    {"model":'ishigami',
    "current_output_folder":'sact_trapez_uq__lmax4_tol_10-6_maxeval1000_bound_nomodify_norm2_157mc_random_p7_regression', 
    "config_file": config_file,
    "outputModelDir": outputModelDir,
    "can_model_evaluate_all_vector_nodes": can_model_evaluate_all_vector_nodes,
    "grid_type": 'trapezoidal', # try trapezoidal vs gauss_legendre
    "method": 'dim_wise_spat_adaptive_combi',  #  'standard_combi', 'dim_adaptive_combi', 'dim_wise_spat_adaptive_combi'
    "operation_str": "uq", # try out "uq" | 'integration'
    "uq_optimization": 'mean',  # try 'mean' | 'mean_and_var' | 'pce'; Default: 'mean'
    "grid_surplusses":'grid',  # Try with None
    "minimum_level": 1, "maximum_level": 4, 
    "max_evaluations":1000, "tol":10**-6, "modified_basis":False, "boundary":True, "norm":2, "p_bsplines":3, 
    "rebalancing":True, "version":6, "margin":0.8,
    "surrogate_type":'combiinstance', "single_qoi":None,
    "reevaluate_surrogate":True, "reevaluate_original_model":True,
    "number_of_samples":1000, "sampling_rule":"random", "sample_new_nodes_from_standard_dist":False,
    "uq_method": "mc", "read_nodes_from_file": False, "parameters_file": None, "sampleFromStandardDist": True,
    "regression": True,
    "mc_numevaluations": 157, "sc_p_order": 7,  "cross_truncation": 1.0,
    },
]
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
        print(f"dictionary_with_inf_about_the_run-{dictionary_with_inf_about_the_run}")
        end_time = time.time()
        duration = end_time - start_time
        print(f"The single setup run took {duration} (for examing in total {number_of_functions} different functions)")

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
