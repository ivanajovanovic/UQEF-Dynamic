
import dill
import pickle
import pathlib
import pandas as pd

from uqef_dynamic.models.larsim import LarsimModelUQ
from uqef_dynamic.models.linearDampedOscillator import LinearDampedOscillatorModel
from uqef_dynamic.models.ishigami import IshigamiModel
from uqef_dynamic.models.productFunction import ProductFunctionModel
from uqef_dynamic.models.hbv_sask import HBVSASKModelUQ
# from uqef_dynamic.models.pybamm import pybammModelUQ as pybammmodel
from uqef_dynamic.models.simpleOscilator.simple_oscillator_model import simpleOscillatorUQ

from uqef_dynamic.models.larsim import LarsimStatistics
from uqef_dynamic.models.linearDampedOscillator import LinearDampedOscillatorStatistics
from uqef_dynamic.models.ishigami import IshigamiStatistics
from uqef_dynamic.models.productFunction import ProductFunctionStatistics
from uqef_dynamic.models.hbv_sask import HBVSASKStatistics
# from uqef_dynamic.models.pybamm import pybammStatistics
from uqef_dynamic.models.simpleOscilator.simple_oscillator_statistics import simpleOscillatorStatistics

from uqef_dynamic.utils import utility
from uqef_dynamic.utils import uqef_dynamic_utils
# from uqef_dynamic.models.time_dependent_baseclass import time_dependent_statistics

def create_model_object(configuration_object, uqsim_args_dict, workingDir, model=None, **kwargs):
    """
    Note: hardcoded for a couple of currently (more complex) supported models
    :param configuration_object:
    :param uqsim_args_dict:
    :param workingDir:
    :param model: "larsim" | "hbvsask" | "oscillator" | "battery"
    :return:
    """
    if model is None:
        if 'model' in uqsim_args_dict:
            model = uqsim_args_dict['model']
        else:
            raise ValueError("Model, for which the statistics object shoudl be re-created, is not provided!")
    if model.lower() == "larsim":
        modelObject = LarsimModelUQ.LarsimModelUQ(
            configurationObject=configuration_object,
            inputModelDir=uqsim_args_dict["inputModelDir"],
            workingDir=workingDir,
            sourceDir=uqsim_args_dict["sourceDir"],
            disable_statistics=uqsim_args_dict["disable_statistics"],
            uq_method=uqsim_args_dict["uq_method"],
            **kwargs)
    elif model.lower() == "ishigami":
        modelObject = IshigamiModel.IshigamiModel(
            configurationObject=configuration_object,
            workingDir=workingDir,**kwargs)
    elif model.lower() == "hbvsask":
        modelObject = HBVSASKModelUQ.HBVSASKModelUQ(
            configurationObject=configuration_object,
            inputModelDir=uqsim_args_dict["inputModelDir"],
            workingDir=workingDir,
            disable_statistics=uqsim_args_dict["disable_statistics"],
            uq_method=uqsim_args_dict["uq_method"],**kwargs)
    # elif model.lower() == "battery":
    #     modelObject = pybammmodel.pybammModelUQ(
    #         configurationObject=configuration_object,
    #         inputModelDir=uqsim_args_dict["inputModelDir"],
    #         workingDir=workingDir,**kwargs)
    elif model.lower() == "simple_oscillator":
        modelObject = simpleOscillatorUQ(
            configurationObject=configuration_object,
            workingDir=workingDir,**kwargs)
    else:
        raise ValueError("Model not supported")
    return modelObject

def create_statistics_object(configuration_object, uqsim_args_dict, workingDir, model=None, **kwargs):
    """
    Note: hardcoded for a couple of currently (more complex) supported models
    :param configuration_object:
    :param uqsim_args_dict:
    :param workingDir:
    :param model: "larsim" | "hbvsask" | "ishigami" | "oscillator" | "battery"
    :return:
    """
    if model is None:
        if 'model' in uqsim_args_dict:
            model = uqsim_args_dict['model']
        else:
            raise ValueError("Model, for which the statistics object shoudl be re-created, is not provided!")
    # TODO make this function more general or move it somewhere else
    if model == "larsim":
        statisticsObject = LarsimStatistics.LarsimStatistics(configuration_object, workingDir=workingDir,
                                                                   parallel_statistics=uqsim_args_dict[
                                                                       "parallel_statistics"],
                                                                   mpi_chunksize=uqsim_args_dict.get("mpi_chunksize", 1),
                                                                   unordered=False,
                                                                   uq_method=uqsim_args_dict["uq_method"],
                                                                   compute_Sobol_t=uqsim_args_dict["compute_Sobol_t"],
                                                                   compute_Sobol_m=uqsim_args_dict["compute_Sobol_m"], **kwargs)
    elif model == "hbvsask":
        statisticsObject = HBVSASKStatistics.HBVSASKStatistics(
            configurationObject=configuration_object,
            workingDir=workingDir,
            inputModelDir=uqsim_args_dict["inputModelDir"],
            sampleFromStandardDist=uqsim_args_dict["sampleFromStandardDist"],
            parallel_statistics=uqsim_args_dict["parallel_statistics"],
            mpi_chunksize=uqsim_args_dict.get("mpi_chunksize", 1),
            uq_method=uqsim_args_dict["uq_method"],
            compute_Sobol_t=uqsim_args_dict.get("compute_Sobol_t", False),
            compute_Sobol_m=uqsim_args_dict.get("compute_Sobol_m", False),
            compute_Sobol_m2=uqsim_args_dict.get("compute_Sobol_m2", False),
            save_all_simulations=uqsim_args_dict.get("save_all_simulations", True),
            collect_and_save_state_data=uqsim_args_dict.get("collect_and_save_state_data", False),
            store_qoi_data_in_stat_dict=uqsim_args_dict.get("store_qoi_data_in_stat_dict", False),
            store_gpce_surrogate_in_stat_dict=uqsim_args_dict.get("store_gpce_surrogate_in_stat_dict", True),
            instantly_save_results_for_each_time_step=uqsim_args_dict.get("instantly_save_results_for_each_time_step", False),
            compute_sobol_indices_with_samples=kwargs.get('compute_sobol_indices_with_samples', False),
            save_gpce_surrogate=kwargs.get('save_gpce_surrogate', False),
            compute_other_stat_besides_pce_surrogate=kwargs.get('compute_other_stat_besides_pce_surrogate', True),
            compute_kl_expansion_of_qoi = kwargs.get('compute_kl_expansion_of_qoi', False),
            index_column_name = kwargs.get('index_column_name', utility.INDEX_COLUMN_NAME),
            allow_conditioning_results_based_on_metric=kwargs.get('allow_conditioning_results_based_on_metric', False),
            condition_results_based_on_metric = kwargs.get('condition_results_based_on_metric', 'NSE'),
            condition_results_based_on_metric_value = kwargs.get('condition_results_based_on_metric_value', 0.2),
            condition_results_based_on_metric_sign = kwargs.get('condition_results_based_on_metric_sign', "greater_or_equal"),
            compute_timewise_gpce_next_to_kl_expansion=kwargs.get('compute_timewise_gpce_next_to_kl_expansion', False),
            kl_expansion_order = kwargs.get("kl_expansion_order", 2),
            compute_generalized_sobol_indices = kwargs.get('compute_generalized_sobol_indices', False),
            compute_generalized_sobol_indices_over_time = kwargs.get('compute_generalized_sobol_indices_over_time', False),
            compute_covariance_matrix_in_time = kwargs.get('compute_covariance_matrix_in_time', False),
            dict_stat_to_compute=kwargs.get("dict_stat_to_compute", utility.DEFAULT_DICT_STAT_TO_COMPUTE),
            dict_what_to_plot=kwargs.get("dict_what_to_plot", utility.DEFAULT_DICT_WHAT_TO_PLOT),
            **kwargs
        )
    # elif model == "battery":
    #     statisticsObject = pybammStatistics.pybammStatistics(
    #         configurationObject=configuration_object,
    #         workingDir=workingDir,
    #         inputModelDir=uqsim_args_dict["inputModelDir"],
    #         sampleFromStandardDist=uqsim_args_dict["sampleFromStandardDist"],
    #         parallel_statistics=uqsim_args_dict["parallel_statistics"],
    #         mpi_chunksize=uqsim_args_dict.get("mpi_chunksize", 1),
    #         unordered=False,
    #         uq_method=uqsim_args_dict["uq_method"],
            # compute_Sobol_t=uqsim_args_dict.get("compute_Sobol_t", False),
            # compute_Sobol_m=uqsim_args_dict.get("compute_Sobol_m", False),
            # compute_Sobol_m2=uqsim_args_dict.get("compute_Sobol_m2", False),
            # save_all_simulations=uqsim_args_dict.get("save_all_simulations", True),
            # collect_and_save_state_data=uqsim_args_dict.get("collect_and_save_state_data", False),
            # store_qoi_data_in_stat_dict=uqsim_args_dict.get("store_qoi_data_in_stat_dict", False),
            # store_gpce_surrogate_in_stat_dict=uqsim_args_dict.get("store_gpce_surrogate_in_stat_dict", True),
            # instantly_save_results_for_each_time_step=uqsim_args_dict.get("instantly_save_results_for_each_time_step", False),
    #     )
    elif model == "ishigami":
        statisticsObject = IshigamiStatistics.IshigamiStatistics(
            configurationObject=configuration_object,
            workingDir=workingDir,
            sampleFromStandardDist=uqsim_args_dict["sampleFromStandardDist"],
            parallel_statistics=uqsim_args_dict["parallel_statistics"],
            mpi_chunksize=uqsim_args_dict.get("mpi_chunksize", 1),
            unordered=False,
            uq_method=uqsim_args_dict["uq_method"],
            compute_Sobol_t=uqsim_args_dict.get("compute_Sobol_t", False),
            compute_Sobol_m=uqsim_args_dict.get("compute_Sobol_m", False),
            compute_Sobol_m2=uqsim_args_dict.get("compute_Sobol_m2", False),
            save_all_simulations=uqsim_args_dict.get("save_all_simulations", True),
            collect_and_save_state_data=uqsim_args_dict.get("collect_and_save_state_data", False),
            store_qoi_data_in_stat_dict=uqsim_args_dict.get("store_qoi_data_in_stat_dict", False),
            store_gpce_surrogate_in_stat_dict=uqsim_args_dict.get("store_gpce_surrogate_in_stat_dict", True),
            instantly_save_results_for_each_time_step=uqsim_args_dict.get("instantly_save_results_for_each_time_step", False),
            compute_sobol_indices_with_samples=kwargs.get('compute_sobol_indices_with_samples', False),
            save_gpce_surrogate=kwargs.get('save_gpce_surrogate', False),
            compute_other_stat_besides_pce_surrogate=kwargs.get('compute_other_stat_besides_pce_surrogate', True),
            dict_stat_to_compute=kwargs.get("dict_stat_to_compute", utility.DEFAULT_DICT_STAT_TO_COMPUTE),
            dict_what_to_plot=kwargs.get("dict_what_to_plot", utility.DEFAULT_DICT_WHAT_TO_PLOT),
            index_column_name = kwargs.get('index_column_name', utility.INDEX_COLUMN_NAME),
            **kwargs
        )
    elif model == "oscillator":
        statisticsObject = LinearDampedOscillatorStatistics.LinearDampedOscillatorStatistics(
            configurationObject=configuration_object,
            workingDir=workingDir,
            inputModelDir=uqsim_args_dict["inputModelDir"],
            sampleFromStandardDist=uqsim_args_dict["sampleFromStandardDist"],
            parallel_statistics=uqsim_args_dict["parallel_statistics"],
            mpi_chunksize=uqsim_args_dict.get("mpi_chunksize", 1),
            unordered=False,
            uq_method=uqsim_args_dict["uq_method"],
            compute_Sobol_t=uqsim_args_dict.get("compute_Sobol_t", False),
            compute_Sobol_m=uqsim_args_dict.get("compute_Sobol_m", False),
            compute_Sobol_m2=uqsim_args_dict.get("compute_Sobol_m2", False),
            save_all_simulations=uqsim_args_dict.get("save_all_simulations", True),
            collect_and_save_state_data=uqsim_args_dict.get("collect_and_save_state_data", False),
            store_qoi_data_in_stat_dict=uqsim_args_dict.get("store_qoi_data_in_stat_dict", False),
            store_gpce_surrogate_in_stat_dict=uqsim_args_dict.get("store_gpce_surrogate_in_stat_dict", True),
            instantly_save_results_for_each_time_step=uqsim_args_dict.get("instantly_save_results_for_each_time_step", False),
            compute_sobol_indices_with_samples=kwargs.get('compute_sobol_indices_with_samples', False),
            save_gpce_surrogate=kwargs.get('save_gpce_surrogate', False),
            compute_other_stat_besides_pce_surrogate=kwargs.get('compute_other_stat_besides_pce_surrogate', True),
            compute_kl_expansion_of_qoi = kwargs.get('compute_kl_expansion_of_qoi', False),
            compute_timewise_gpce_next_to_kl_expansion=kwargs.get('compute_timewise_gpce_next_to_kl_expansion', False),
            kl_expansion_order = kwargs.get("kl_expansion_order", 2),
            compute_generalized_sobol_indices = kwargs.get('compute_generalized_sobol_indices', False),
            compute_generalized_sobol_indices_over_time = kwargs.get('compute_generalized_sobol_indices_over_time', False),
            compute_covariance_matrix_in_time = kwargs.get('compute_covariance_matrix_in_time', False),
            dict_stat_to_compute=kwargs.get("dict_stat_to_compute", utility.DEFAULT_DICT_STAT_TO_COMPUTE),
            dict_what_to_plot=kwargs.get("dict_what_to_plot", utility.DEFAULT_DICT_WHAT_TO_PLOT),
            **kwargs
)
    else:
        raise ValueError("Model not supported")
    return statisticsObject


def create_and_extend_statistics_object(configurationObject, uqsim_args_dict, workingDir, model, df_simulation_result=None, printing=False):
    """
    This function creates the statistics object (calles create_statistics_object)
    reads the statistics_dictionary and extend the statistics object with the statistics_dictionary.
    Args:
    - configurationObject: dict, configuration object
    - uqsim_args_dict: dict, dictionary with the UQEF simulation arguments
    - workingDir: pathlib.Path object, path to the working directory
    - model: str, name of the model
    - df_simulation_result: pandas DataFrame, simulation results; Default is None
    - printing: bool, if True, print the statistics dictionary
    Returns:
    - statisticsObject: object, UQEF-Dynamic time-dependent statistics object
    """
    # Load the statistics object
    # statistics_dictionary_file = utility.get_dict_with_qoi_name_specific_output_file_paths_based_on_workingDir(\
    # workingDir, qoi_string=qoi_string)
    statisticsObject = create_statistics_object(
        configuration_object=configurationObject, uqsim_args_dict=uqsim_args_dict, \
        workingDir=workingDir, model=model)
    statistics_dictionary = uqef_dynamic_utils.read_all_saved_statistics_dict(\
        workingDir=workingDir, list_qoi_column=statisticsObject.list_qoi_column, 
        single_timestamp_single_file=uqsim_args_dict.get("instantly_save_results_for_each_time_step", False), 
        throw_error=False
        )
    if printing:
        print(f"INFO: statistics_dictionary - {statistics_dictionary}")
    
    uqef_dynamic_utils.extend_statistics_object(
        statisticsObject=statisticsObject, 
        statistics_dictionary=statistics_dictionary, 
        df_simulation_result=df_simulation_result,
        get_measured_data=False, 
        get_unaltered_data=False
    )

    # Create a Pandas.DataFrame
    statisticsObject.create_df_from_statistics_data()

    return statisticsObject
# ==============================================================================================================
# Functions for reading all saved output files from UQ and SA simulations and creating a DataFrame
# ==============================================================================================================


def get_df_statistics_and_df_si_from_saved_files(workingDir, inputModelDir=None, **kwargs):
    """
    Retrieves the statistics and sensitivity indices data from saved files.

    Args:
        workingDir (str): The working directory where the saved files are located.
        inputModelDir (str, optional): The input model directory. Defaults to None.
    Keyword Args:
        read_saved_simulations (bool, optional): If True, reads the saved simulations. Defaults to False.
        read_saved_states (bool, optional): If True, reads the saved states. Defaults to False.
        set_lower_predictions_to_zero (bool, optional): If True, sets the lower predictions ('E', 'P10') to zero. Defaults to False.
        set_mean_prediction_to_zero(bool, optional): If True, sets mean predicted values to zero; Defaults to False.
        correct_sobol_indices (bool, optional): Set lower value to zero, defaults to False
        instantly_save_results_for_each_time_step(bool, optional): Overwrite the same entry in uqsim_args_dict is different than None;  defaults to None
        add_measured_data (bool, optional): If True, adds measured data to the DataFrame. Defaults to False.
        add_forcing_data (bool, optional): If True, adds forcing data to the DataFrame. Defaults to False.
        transform_measured_data_as_original_model (bool, optional): If True, transforms the measured data as the original model. Defaults to False.
    Returns:
        tuple: A tuple containing the following:
            - statisticsObject: The statistics object.
            - df_statistics_and_measured: The DataFrame containing the statistics ( and measured data and forcing).
            - si_t_df: The DataFrame containing the total order sensitivity indices.
            - si_m_df: The DataFrame containing the first order sensitivity indices.
            - df_simulation_result: The DataFrame containing the simulation results.
            - df_index_parameter: The DataFrame containing the index parameter file.
            - df_index_parameter_gof: The DataFrame containing the index parameter goodness-of-fit file.
            - uqsim_args_dict: The UQSim arguments dictionary.
            - simulationNodes: The simulation nodes.

    Raises:
        FileNotFoundError: If any of the required files are not found.
    """
    # dict_output_file_paths = utility.get_dict_with_output_file_paths_based_on_workingDir(workingDir)
    # args_file = dict_output_file_paths.get("args_file")
    # configuration_object_file = dict_output_file_paths.get("configuration_object_file")
    # nodes_file = dict_output_file_paths.get("nodes_file")
    # df_all_index_parameter_file = dict_output_file_paths.get("df_all_index_parameter_file")
    # df_all_index_parameter_gof_file = dict_output_file_paths.get("df_all_index_parameter_gof_file")
    # df_all_simulations_file = dict_output_file_paths.get("df_all_simulations_file")
    # time_info_file = dict_output_file_paths.get("time_info_file")
    if not workingDir.is_dir():
        raise Exception(f"Directory {workingDir} does not exist!")

    read_saved_simulations = kwargs.get('read_saved_simulations', False)
    read_saved_states = kwargs.get('read_saved_states', False)
    set_lower_predictions_to_zero = kwargs.get('set_lower_predictions_to_zero', False)
    set_mean_prediction_to_zero = kwargs.get('set_mean_prediction_to_zero', False)
    correct_sobol_indices = kwargs.get('correct_sobol_indices', False)
    instantly_save_results_for_each_time_step = kwargs.get('instantly_save_results_for_each_time_step', None)

    add_measured_data = kwargs.get('add_measured_data', False)
    add_forcing_data = kwargs.get('add_forcing_data', False)
    transform_measured_data_as_original_model = kwargs.get('transform_measured_data_as_original_model', False)

    args_files = utility.get_dict_with_output_file_paths_based_on_workingDir(workingDir)
    for key, value in args_files.items():
        globals()[key] = value

    # Load the UQSim args dictionary
    uqsim_args_dict = utility.load_uqsim_args_dict(args_file)
    model = uqsim_args_dict["model"]

    # Load the configuration object
    configurationObject = utility.load_configuration_object(configuration_object_file)
    simulation_settings_dict = utility.read_simulation_settings_from_configuration_object(configurationObject)

    if inputModelDir is not None:
        if inputModelDir != uqsim_args_dict["inputModelDir"]:
            uqsim_args_dict["inputModelDir"] = pathlib.Path(inputModelDir)
    else:
        inputModelDir = uqsim_args_dict["inputModelDir"]
    inputModelDir = pathlib.Path(inputModelDir)

    # just in certain situations it is necessary to update/overwrite instantly_save_results_for_each_time_step
    if instantly_save_results_for_each_time_step is not None:
        uqsim_args_dict["instantly_save_results_for_each_time_step"] = instantly_save_results_for_each_time_step
    
    with open(nodes_file, 'rb') as f:
        simulationNodes = pickle.load(f)
        
    if df_index_parameter_file.is_file():
        df_index_parameter = pd.read_pickle(df_index_parameter_file, compression="gzip")
        params_list = utility._get_parameter_columns_df_index_parameter_gof(
            df_index_parameter)
    else:
        df_index_parameter = None
        print(f"Be careful - {df_index_parameter_file} does not exist; df_index_parameter is None!")
        params_list = []
        for single_param in configurationObject["parameters"]:
            params_list.append(single_param["name"])

    if df_index_parameter_gof_file.is_file():
        df_index_parameter_gof = pd.read_pickle(df_index_parameter_gof_file, compression="gzip")
        gof_list = utility._get_gof_columns_df_index_parameter_gof(
            df_index_parameter_gof)
    else:
        print(f"Be careful - {df_index_parameter_gof_file} does not exist; df_index_parameter_gof is None; gof_list is not populated!")
        df_index_parameter_gof = None
        gof_list = None

    # df_nodes = utility.get_df_from_simulationNodes(simulationNodes, nodes_or_paramters="nodes", params_list=params_list)
    # df_nodes_params = utility.get_df_from_simulationNodes(simulationNodes, nodes_or_paramters="parameters",  params_list=params_list)

    if read_saved_simulations and df_simulations_file.is_file():
        df_simulation_result = pd.read_pickle(df_simulations_file, compression="gzip")
    else:
        df_simulation_result = None
    if read_saved_states and df_state_file.is_file():
        df_state = pd.read_pickle(df_state_file, compression="gzip")
    else:
        df_state = None

    # Load the statistics object

    # statisticsObject = create_statistics_object(
    #     configurationObject, uqsim_args_dict, workingDir, model=model)
    # statistics_dictionary = uqef_dynamic_utils.read_all_saved_statistics_dict(\
    #     workingDir, statisticsObject.list_qoi_column, uqsim_args_dict.get("instantly_save_results_for_each_time_step", False), throw_error=True)
    # uqef_dynamic_utils.extend_statistics_object(
    #     statisticsObject=statisticsObject, 
    #     statistics_dictionary=statistics_dictionary, 
    #     df_simulation_result=df_simulation_result,  # df_simulation_result=None,
    #     get_measured_data=False, 
    #     get_unaltered_data=False
    # )
    # # This is hardcoded for HBV fro now
    # if model == "HBV" or model == "hbvsask" or model == "hbv" or model == "HBV-SASK" or model == "hbv-sask":
    #     basis = configurationObject['model_settings']['basis']
    #     statisticsObject.inputModelDir_basis = inputModelDir / basis
    # # Add measured Data
    # statisticsObject.get_measured_data(
    #     timestepRange=(statisticsObject.timesteps_min, statisticsObject.timesteps_max), 
    #     transforme_mesured_data_as_original_model=True)

    # # Create a Pandas.DataFrame
    # statisticsObject.create_df_from_statistics_data()
    # # Add forcing Data
    # statisticsObject.get_forcing_data(time_column_name=statisticsObject.time_column_name)
    # # Merge Everything
    # df_statistics_and_measured = pd.merge(
    #     statisticsObject.df_statistics, 
    #     statisticsObject.forcing_df, left_on=statisticsObject.time_column_name, right_index=True)
    
    # or
    statisticsObject = create_and_extend_statistics_object(
        configurationObject, uqsim_args_dict, workingDir, model, 
        df_simulation_result=df_simulation_result, printing=False)
    df_statistics_and_measured = statisticsObject.merge_df_statistics_data_with_measured_and_forcing_data(
    add_measured_data=add_measured_data, add_forcing_data=add_forcing_data, transform_measured_data_as_original_model=transform_measured_data_as_original_model)

    if set_lower_predictions_to_zero:
        if 'E_minus_std' in df_statistics_and_measured:
            df_statistics_and_measured['E_minus_std'] = df_statistics_and_measured['E_minus_std'].apply(lambda x: max(0, x))        
        if 'E_minus_2std' in df_statistics_and_measured:
            df_statistics_and_measured['E_minus_2std'] = df_statistics_and_measured['E_minus_2std'].apply(lambda x: max(0, x))        
        if 'P10' in df_statistics_and_measured:
            df_statistics_and_measured['P10'] = df_statistics_and_measured['P10'].apply(lambda x: max(0, x))
    if set_mean_prediction_to_zero:
        df_statistics_and_measured['E'] = df_statistics_and_measured['E'].apply(lambda x: max(0, x)) 

    si_t_df = statisticsObject.create_df_from_sensitivity_indices(si_type="Sobol_t")
    si_m_df = statisticsObject.create_df_from_sensitivity_indices(si_type="Sobol_m")

    if si_m_df is not None:
        si_m_df.sort_values(by=statisticsObject.time_column_name, ascending=True, inplace=True)
        if correct_sobol_indices:
            si_columns_to_plot = [x for x in si_m_df.columns.tolist() if x != 'measured' \
                                        and x != 'measured_norm' and x != 'qoi' and x!= statisticsObject.time_column_name]
            for single_column in si_columns_to_plot: 
                si_m_df[single_column] = si_m_df[single_column].apply(lambda x: max(0, x))
    if si_t_df is not None:
        si_t_df.sort_values(by=statisticsObject.time_column_name, ascending=True, inplace=True)
        if correct_sobol_indices:
            si_columns_to_plot = [x for x in si_t_df.columns.tolist() if x != 'measured' \
                                        and x != 'measured_norm' and x != 'qoi' and x!= statisticsObject.time_column_name]
            for single_column in si_columns_to_plot: 
                si_t_df[single_column] = si_t_df[single_column].apply(lambda x: max(0, x))

    return statisticsObject, df_statistics_and_measured, si_t_df, si_m_df, df_simulation_result, df_index_parameter, df_index_parameter_gof, uqsim_args_dict, simulationNodes


def get_dict_with_dfs_from_statistics_and_other_relevant_data(
    workingDir, inputModelDir=None, **kwargs):
    """   
    This is one big function which served to read all the relevant data produced by the Statistics class after running the 
    UQEF-Dynamic simulations. It retursn one dictionary storying all the relevant data in the form of DataFrames, dictionaries and 
    relevant variables.

    This function is different from similar functions in uqef_dynamic_utils because it also creates
    a Statistics object and reads the statistics_dictionary and extends the statistics object with the statistics_dictionary.
    That is why this function is here and not in uqef_dynamic_utils.

    This function in essence combines the following function in one:
    * get_df_statistics_and_df_si_from_saved_files
    * uqef_dynamic_utils.read_output_files_uqef_dynamic
    * uqef_dynamic_utils.read_all_saved_uqef_dynamic_results_and_produce_dict_of_interest

    Args:
        workingDir (str): The working directory where the saved files are located.
        inputModelDir (str, optional): The input model directory. Defaults to None.
    Keyword Args:
        read_saved_simulations (bool, optional): If True, reads the saved simulations. Defaults to False.
        read_saved_states (bool, optional): If True, reads the saved states. Defaults to False.
        set_lower_predictions_to_zero (bool, optional): If True, sets the lower predictions ('E', 'P10') to zero. Defaults to False.
        set_mean_prediction_to_zero(bool, optional): If True, sets mean predicted values to zero; Defaults to False.
        correct_sobol_indices (bool, optional): Set lower value to zero, defaults to False
        instantly_save_results_for_each_time_step(bool, optional): Overwrite the same entry in uqsim_args_dict is different than None;  defaults to None

    Returns:
    - results_dict: dict, dictionary with the following keys / values:
        - workingDir: pathlib.Path object, path to the working directory
        - args_files: dict, dictionary sotrying the paths to the output files
           take a look at the function utility.get_dict_with_output_file_paths_based_on_workingDir
        - uqsim_args_dict: dict, dictionary with the UQEF simulation arguments
        - model: str, name of the model
        - inputModelDir: pathlib.Path object, path to the input model directory
        - configurationObject: dict, configuration object
        - simulation_settings_dict: dict, dictionary with the simulation settings
           take a look at the function utility.read_simulation_settings_from_configuration_object
        - simulationNodes: object, UQEF simulation nodes
        - time_info: str, time information
        - params_list: list, list of model uncertain parameters
        - df_index_parameter: pandas DataFrame, index parameter; None is missing
        - df_index_parameter_gof: pandas DataFrame, index parameter goodness-of-fit (GOF); None is missing
        - gof_list: list, list of goodness-of-fit (GOF) measures; None is missing
        - df_simulation_result: pandas DataFrame, simulation results; None is missing
        - df_state: pandas DataFrame, state; None is missing
        - statisticsObject: The statistics object.
        - df_statistics_and_measured: The DataFrame containing the statistics ( and measured data and forcing).
        - si_t_df: The DataFrame containing the total order sensitivity indices.
        - si_m_df: The DataFrame containing the first order sensitivity indices.

        - time_model_simulations: str, time for model simulations
        - time_computing_statistics: str, time for computing statistics
        - stochasticParameterNames: list, list of stochastic parameter names
        - number_full_model_evaluations: int, number of full model evaluations
        - full_number_quadrature_points
        - plus extra entries from uqef_dynamic_utils.update_dict_with_results_of_interest_based_on_uqsim_args_dict

    Raises:
        FileNotFoundError: If any of the required files are not found.
    """
    if not workingDir.is_dir():
        raise Exception(f"Directory {workingDir} does not exist!")

    results_dict = {}
    
    read_saved_simulations = kwargs.get('read_saved_simulations', False)
    read_saved_states = kwargs.get('read_saved_states', False)
    set_lower_predictions_to_zero = kwargs.get('set_lower_predictions_to_zero', False)
    set_mean_prediction_to_zero = kwargs.get('set_mean_prediction_to_zero', False)
    correct_sobol_indices = kwargs.get('correct_sobol_indices', False)
    instantly_save_results_for_each_time_step = kwargs.get('instantly_save_results_for_each_time_step', None)

    # get output paths based on workingDir
    args_files = utility.get_dict_with_output_file_paths_based_on_workingDir(workingDir)
    for key, value in args_files.items():
        globals()[key] = value

    # Load the UQSim args dictionary
    uqsim_args_dict = utility.load_uqsim_args_dict(args_file)
    model = uqsim_args_dict["model"]

    # Load the configuration object
    configurationObject = utility.load_configuration_object(configuration_object_file)
    simulation_settings_dict = utility.read_simulation_settings_from_configuration_object(configurationObject)

    if inputModelDir is not None:
        if inputModelDir != uqsim_args_dict["inputModelDir"]:
            uqsim_args_dict["inputModelDir"] = pathlib.Path(inputModelDir)
    else:
        inputModelDir = uqsim_args_dict["inputModelDir"]
    inputModelDir = pathlib.Path(inputModelDir)

    # Timing information
    with open(time_info_file, 'r') as f:
        time_info = f.read() #readlines()?

    # just in certain situations it is necessary to update/overwrite instantly_save_results_for_each_time_step
    if instantly_save_results_for_each_time_step is not None:
        uqsim_args_dict["instantly_save_results_for_each_time_step"] = instantly_save_results_for_each_time_step
    
    with open(nodes_file, 'rb') as f:
        simulationNodes = pickle.load(f)
    dim = simulationNodes.distNodes.shape[0]

    results_dict["workingDir"]=workingDir
    results_dict["args_files"]=args_files
    results_dict["uqsim_args_dict"]=uqsim_args_dict
    results_dict["inputModelDir"]=inputModelDir
    results_dict["model"]=model
    results_dict["configurationObject"]=configurationObject
    results_dict["simulation_settings_dict"]=simulation_settings_dict
    results_dict["time_info"]=time_info
    results_dict["simulationNodes"]=simulationNodes

    if df_index_parameter_file.is_file():
        df_index_parameter = pd.read_pickle(df_index_parameter_file, compression="gzip")
        params_list = utility._get_parameter_columns_df_index_parameter_gof(
            df_index_parameter)
    else:
        df_index_parameter = None
        print(f"Be careful - {df_index_parameter_file} does not exist; df_index_parameter is None!")
        params_list = []
        for single_param in configurationObject["parameters"]:
            params_list.append(single_param["name"])

    if df_index_parameter_gof_file.is_file():
        df_index_parameter_gof = pd.read_pickle(df_index_parameter_gof_file, compression="gzip")
        gof_list = utility._get_gof_columns_df_index_parameter_gof(
            df_index_parameter_gof)
    else:
        print(f"Be careful - {df_index_parameter_gof_file} does not exist; df_index_parameter_gof is None; gof_list is not populated!")
        df_index_parameter_gof = None
        gof_list = None

    # df_nodes = utility.get_df_from_simulationNodes(simulationNodes, nodes_or_paramters="nodes", params_list=params_list)
    # df_nodes_params = utility.get_df_from_simulationNodes(simulationNodes, nodes_or_paramters="parameters",  params_list=params_list)

    if read_saved_simulations and df_simulations_file.is_file():
        df_simulation_result = pd.read_pickle(df_simulations_file, compression="gzip")
    else:
        df_simulation_result = None
    if read_saved_states and df_state_file.is_file():
        df_state = pd.read_pickle(df_state_file, compression="gzip")
    else:
        df_state = None

    statisticsObject = create_and_extend_statistics_object(
        configurationObject, uqsim_args_dict, workingDir, model, 
        df_simulation_result=df_simulation_result, printing=False)
    df_statistics_and_measured = statisticsObject.merge_df_statistics_data_with_measured_and_forcing_data(
    add_measured_data=True, add_forcing_data=True, transform_measured_data_as_original_model=True)

    if set_lower_predictions_to_zero:
        if 'E_minus_std' in df_statistics_and_measured:
            df_statistics_and_measured['E_minus_std'] = df_statistics_and_measured['E_minus_std'].apply(lambda x: max(0, x))        
        if 'E_minus_2std' in df_statistics_and_measured:
            df_statistics_and_measured['E_minus_2std'] = df_statistics_and_measured['E_minus_2std'].apply(lambda x: max(0, x))        
        if 'P10' in df_statistics_and_measured:
            df_statistics_and_measured['P10'] = df_statistics_and_measured['P10'].apply(lambda x: max(0, x))
    if set_mean_prediction_to_zero:
        df_statistics_and_measured['E'] = df_statistics_and_measured['E'].apply(lambda x: max(0, x)) 

    si_t_df = statisticsObject.create_df_from_sensitivity_indices(si_type="Sobol_t")
    si_m_df = statisticsObject.create_df_from_sensitivity_indices(si_type="Sobol_m")

    if si_m_df is not None:
        si_m_df.sort_values(by=statisticsObject.time_column_name, ascending=True, inplace=True)
        if correct_sobol_indices:
            si_columns_to_plot = [x for x in si_m_df.columns.tolist() if x != 'measured' \
                                        and x != 'measured_norm' and x != 'qoi' and x!= statisticsObject.time_column_name]
            for single_column in si_columns_to_plot: 
                si_m_df[single_column] = si_m_df[single_column].apply(lambda x: max(0, x))
    if si_t_df is not None:
        si_t_df.sort_values(by=statisticsObject.time_column_name, ascending=True, inplace=True)
        if correct_sobol_indices:
            si_columns_to_plot = [x for x in si_t_df.columns.tolist() if x != 'measured' \
                                        and x != 'measured_norm' and x != 'qoi' and x!= statisticsObject.time_column_name]
            for single_column in si_columns_to_plot: 
                si_t_df[single_column] = si_t_df[single_column].apply(lambda x: max(0, x))

    # ========================================================
    # Etra stuff as from read_all_saved_uqef_dynamic_results_and_produce_dict_of_interest
    # Update dict with results of interest based on uqsim_args_dict - add variant, q_order, mc_numevaluations
    uqef_dynamic_utils.update_dict_with_results_of_interest_based_on_uqsim_args_dict(results_dict, uqsim_args_dict)
    
    # Extra Timing information
    for line in time_info:
        if line.startswith("time_model_simulations"):
            results_dict["time_model_simulations"] = line.split(':')[1].strip()
        elif line.startswith("time_computing_statistics"):
            results_dict["time_computing_statistics"] = line.split(':')[1].strip()

    # whatch-out this might be tricky when not all params are regarded as uncertain!
    param_labeles = utility.get_list_of_uncertain_parameters_from_configuration_dict(
        configurationObject, raise_error=True, uq_method=uqsim_args_dict["uq_method"])
    #print(f"Debugging - params_list: {params_list}; simulationNodes.nodeNames: {simulationNodes.nodeNames}; param_labeles: {param_labeles}")    
    # results_dict["parameterNames"] = params_list  #not simulationNodes.nodeNames, instead better simulationNodes.orderdDistsNames
    results_dict["stochasticParameterNames"] = param_labeles

    simulation_parameters_file = args_files["simulation_parameters_file"]
    if df_simulation_result is not None:
        results_dict["number_full_model_evaluations"] = len(df_simulation_result)
    elif simulation_parameters_file.is_file():
        simulation_parameters = np.load(simulation_parameters_file,  allow_pickle=True)
        #print(f"Debugging - simulation_parameters.shape: {simulation_parameters.shape}")
        results_dict["number_full_model_evaluations"] = simulation_parameters.shape[0]
    else:
        if uqsim_args_dict["uq_method"]!="saltelli":
            results_dict["number_full_model_evaluations"] = simulationNodes.nodes.shape[1]
        else:
            results_dict["number_full_model_evaluations"] = (uqsim_args_dict["mc_numevaluations"]) * (2 + dim)

    if results_dict["variant"] not in ["m1", "m2"]:
        results_dict["full_number_quadrature_points"] = \
        (results_dict["q_order"] + 1) ** dim
    
    # list_qoi_column = simulation_settings_dict.list_qoi_column
    # list_qoi_column = statisticsObject.list_qoi_column
    # ========================================================

    results_dict["df_index_parameter"]=df_index_parameter
    results_dict["params_list"]=params_list
    results_dict["df_index_parameter_gof"]=df_index_parameter_gof
    results_dict["gof_list"]=gof_list
    results_dict["df_simulation_result"]=df_simulation_result
    results_dict["df_state"]=df_state
    results_dict["statisticsObject"]=statisticsObject
    results_dict["df_statistics_and_measured"]=df_statistics_and_measured
    results_dict["si_t_df"]=si_t_df
    results_dict["si_m_df"]=si_m_df

    return results_dict