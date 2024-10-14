
import dill
import pickle
import pathlib
import pandas as pd

from uqef_dynamic.models.larsim import LarsimStatistics
from uqef_dynamic.models.linearDampedOscillator import LinearDampedOscillatorStatistics
from uqef_dynamic.models.ishigami import IshigamiStatistics
from uqef_dynamic.models.productFunction import ProductFunctionStatistics
from uqef_dynamic.models.hbv_sask import HBVSASKStatistics
from uqef_dynamic.models.pybamm import pybammStatistics
from uqef_dynamic.utils import utility
from uqef_dynamic.utils import uqef_dynamic_utils
# from uqef_dynamic.models.time_dependent_baseclass import time_dependent_statistics


def create_statistics_object(configuration_object, uqsim_args_dict, workingDir, model="hbvsask"):
    """
    Note: hardcoded for a couple of currently supported models
    :param configuration_object:
    :param uqsim_args_dict:
    :param workingDir:
    :param model: "larsim" | "hbvsask"
    :return:
    """
    # TODO make this function more general or move it somewhere else
    if model == "larsim":
        statisticsObject = LarsimStatistics.LarsimStatistics(configuration_object, workingDir=workingDir,
                                                                   parallel_statistics=uqsim_args_dict[
                                                                       "parallel_statistics"],
                                                                   mpi_chunksize=uqsim_args_dict["mpi_chunksize"],
                                                                   unordered=False,
                                                                   uq_method=uqsim_args_dict["uq_method"],
                                                                   compute_Sobol_t=uqsim_args_dict["compute_Sobol_t"],
                                                                   compute_Sobol_m=uqsim_args_dict["compute_Sobol_m"])
    elif model == "hbvsask":
        statisticsObject = HBVSASKStatistics.HBVSASKStatistics(
            configurationObject=configuration_object,
            workingDir=workingDir,
            inputModelDir=uqsim_args_dict["inputModelDir"],
            sampleFromStandardDist=uqsim_args_dict["sampleFromStandardDist"],
            parallel_statistics=uqsim_args_dict["parallel_statistics"],
            mpi_chunksize=uqsim_args_dict["mpi_chunksize"],
            uq_method=uqsim_args_dict["uq_method"],
            compute_Sobol_t=uqsim_args_dict["compute_Sobol_t"],
            compute_Sobol_m=uqsim_args_dict["compute_Sobol_m"],
            compute_Sobol_m2=uqsim_args_dict["compute_Sobol_m2"]
        )
    elif model == "battery":
        statisticsObject = pybammStatistics.pybammStatistics(
            configurationObject=configuration_object,
            workingDir=workingDir,
            inputModelDir=uqsim_args_dict["inputModelDir"],
            sampleFromStandardDist=uqsim_args_dict["sampleFromStandardDist"],
            parallel_statistics=uqsim_args_dict["parallel_statistics"],
            mpi_chunksize=uqsim_args_dict["mpi_chunksize"],
            unordered=False,
            uq_method=uqsim_args_dict["uq_method"],
            compute_Sobol_t=uqsim_args_dict["compute_Sobol_t"],
            compute_Sobol_m=uqsim_args_dict["compute_Sobol_m"],
            compute_Sobol_m2=uqsim_args_dict["compute_Sobol_m2"],
            save_all_simulations=uqsim_args_dict["save_all_simulations"],
            collect_and_save_state_data=uqsim_args_dict["collect_and_save_state_data"],
            store_qoi_data_in_stat_dict=uqsim_args_dict["store_qoi_data_in_stat_dict"],
            store_gpce_surrogate_in_stat_dict=uqsim_args_dict["store_gpce_surrogate_in_stat_dict"],
            instantly_save_results_for_each_time_step=uqsim_args_dict["instantly_save_results_for_each_time_step"]
        )
    else:
        raise ValueError("Model not supported")
        # statisticsObject = time_dependent_statistics.TimeDependentStatistics(
        #     configurationObject=configuration_object,
        #     workingDir=workingDir,
        #     inputModelDir=uqsim_args_dict["inputModelDir"],
        #     sampleFromStandardDist=uqsim_args_dict["sampleFromStandardDist"],
        #     parallel_statistics=uqsim_args_dict["parallel_statistics"],
        #     mpi_chunksize=uqsim_args_dict["mpi_chunksize"],
        #     uq_method=uqsim_args_dict["uq_method"],
        #     compute_Sobol_t=uqsim_args_dict["compute_Sobol_t"],
        #     compute_Sobol_m=uqsim_args_dict["compute_Sobol_m"],
        #     compute_Sobol_m2=uqsim_args_dict["compute_Sobol_m2"]
        # )

    return statisticsObject

# ==============================================================================================================
# Functions for reading all saved output files from UQ and SA simulations and creating a DataFrame
# ==============================================================================================================


def get_df_statistics_and_df_si_from_saved_files(workingDir, inputModelDir=None):
    """
    Retrieves the statistics and sensitivity indices data from saved files.

    Args:
        workingDir (str): The working directory where the saved files are located.
        inputModelDir (str, optional): The input model directory. Defaults to None.

    Returns:
        tuple: A tuple containing the following:
            - statisticsObject: The statistics object.
            - df_statistics_and_measured: The DataFrame containing the statistics and measured data.
            - si_t_df: The DataFrame containing the sensitivity indices.

    Raises:
        FileNotFoundError: If any of the required files are not found.
    """
    dict_output_file_paths = utility.get_dict_with_output_file_paths_based_on_workingDir(workingDir)
    args_file = dict_output_file_paths.get("args_file")
    configuration_object_file = dict_output_file_paths.get("configuration_object_file")
    nodes_file = dict_output_file_paths.get("nodes_file")
    df_all_index_parameter_file = dict_output_file_paths.get("df_all_index_parameter_file")
    df_all_index_parameter_gof_file = dict_output_file_paths.get("df_all_index_parameter_gof_file")
    df_all_simulations_file = dict_output_file_paths.get("df_all_simulations_file")
    time_info_file = dict_output_file_paths.get("time_info_file")

    with open(configuration_object_file, 'rb') as f:
        configurationObject = dill.load(f)
    simulation_settings_dict = utility.read_simulation_settings_from_configuration_object(configurationObject)
    with open(args_file, 'rb') as f:
        uqsim_args = pickle.load(f)
    uqsim_args_dict = vars(uqsim_args)
    model = uqsim_args_dict["model"]
    if inputModelDir is None:
        inputModelDir = pathlib.Path(uqsim_args_dict["inputModelDir"])
    else:
        inputModelDir = pathlib.Path(inputModelDir) 

    # with open(nodes_file, 'rb') as f:
    #     simulationNodes = pickle.load(f)
        
    if df_all_index_parameter_file.is_file():
        df_index_parameter = pd.read_pickle(df_all_index_parameter_file, compression="gzip")
    else:
        df_index_parameter = None
    if df_index_parameter is not None:
        params_list = utility._get_parameter_columns_df_index_parameter_gof(
            df_index_parameter)
    else:
        params_list = []
        for single_param in configurationObject["parameters"]:
            params_list.append(single_param["name"])
    if df_all_index_parameter_gof_file.is_file():
        df_index_parameter_gof = pd.read_pickle(df_all_index_parameter_gof_file, compression="gzip")
        df_index_parameter_gof
    else:
        print(f"Be careful - {df_all_index_parameter_gof_file} does not exist!")
        df_index_parameter_gof = None
    if df_index_parameter_gof is not None:
        gof_list = utility._get_gof_columns_df_index_parameter_gof(
            df_index_parameter_gof)
    else:
        gof_list = None
        print(f"Be careful - {df_all_index_parameter_gof_file} does not exist - therefore gof_list is not populated!")

    # df_nodes = utility.get_df_from_simulationNodes(simulationNodes, nodes_or_paramters="nodes", params_list=params_list)
    # df_nodes_params = utility.get_df_from_simulationNodes(simulationNodes, nodes_or_paramters="parameters",  params_list=params_list)

    read_all_saved_simulations_file = False
    if read_all_saved_simulations_file and df_all_simulations_file.is_file():
        # Reading Saved Simulations - Note: This might be a huge file,
        # especially for MC/Saltelli kind of simulations
        df_simulation_result = pd.read_pickle(df_all_simulations_file, compression="gzip")
    else:
        df_simulation_result = None

    statisticsObject = create_statistics_object(
        configurationObject, uqsim_args_dict, workingDir, model=model)
    statistics_dictionary = uqef_dynamic_utils.read_all_saved_statistics_dict(\
        workingDir, statisticsObject.list_qoi_column, uqsim_args_dict.get("instantly_save_results_for_each_time_step", False), throw_error=True)

    uqef_dynamic_utils.extend_statistics_object(
        statisticsObject=statisticsObject, 
        statistics_dictionary=statistics_dictionary, 
        df_simulation_result=df_simulation_result,  # df_simulation_result=None,
        get_measured_data=False, 
        get_unaltered_data=False
    )

    # This is hardcoded for HBV fro now
    # if model == "HBV" or model == "hbvsask" or model == "hbv" or model == "HBV-SASK" or model == "hbv-sask":
    #     basis = configurationObject['model_settings']['basis']
    #     statisticsObject.inputModelDir_basis = inputModelDir / basis

    # Add measured Data
    statisticsObject.get_measured_data(
        timestepRange=(statisticsObject.timesteps_min, statisticsObject.timesteps_max), 
        transforme_mesured_data_as_original_model="False")

    # Create a Pandas.DataFrame
    statisticsObject.create_df_from_statistics_data()

    # Add forcing Data
    statisticsObject.get_forcing_data(time_column_name="TimeStamp")

    # Merge Everything
    df_statistics_and_measured = pd.merge(
        statisticsObject.df_statistics, 
        statisticsObject.forcing_df, left_on=statisticsObject.time_column_name, right_index=True)

    df_statistics_and_measured['E_minus_std'] = df_statistics_and_measured['E_minus_std'].apply(lambda x: max(0, x))
    df_statistics_and_measured['P10'] = df_statistics_and_measured['P10'].apply(lambda x: max(0, x))

    si_t_df = statisticsObject.create_df_from_sensitivity_indices(si_type="Sobol_t")
    si_m_df = statisticsObject.create_df_from_sensitivity_indices(si_type="Sobol_m")

    return statisticsObject, df_statistics_and_measured, si_t_df, si_m_df