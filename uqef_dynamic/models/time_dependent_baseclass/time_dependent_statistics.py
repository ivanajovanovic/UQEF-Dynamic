from abc import ABC, abstractmethod
import chaospy as cp
from collections import defaultdict
from distutils.util import strtobool
from functools import reduce
import more_itertools
from mpi4py import MPI
import mpi4py.futures as futures
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as pyo
from plotly.subplots import make_subplots
import scipy
from scipy import stats
# from sklearn.preprocessing import MinMaxScaler
import sys
import time

from uqef.stat import Statistics

from uqef_dynamic.utils import parallel_statistics
from uqef_dynamic.utils import uqef_dynamic_utils
from uqef_dynamic.utils import utility
from uqef_dynamic.utils import colors
from uqef_dynamic.utils import sens_indices_sampling_based_utils

# TODO two cases - time_column_name is or is not an index column in returned data!?
# TODO Samples class - write in some log file runs which have returned None, in case of model break!
# TODO Samples class - eventually, add a subroutine that will check some metric in gof_df and conditioned on that potentially disregard the run
# TODO Samples class - Add 'qoi' column to self.df_simulation_result
# TODO Samples class - Move computation of dict_of_approx_matrix_c and dict_of_matrix_c_eigen_decomposition from Sample class to Statistics class!?
# TODO Samples class - Eventually, add options to set other dataframes as well
class Samples(object):
    def __init__(self, rawSamples, qoi_columns="Value", time_column_name=utility.TIME_COLUMN_NAME, index_column_name=utility.INDEX_COLUMN_NAME, 
    *args, **kwargs):
        """
        Samples class is a class which is used to collect and process the (raw) results of the model runs.
        Therefore it is highly dependent on the way the model results are returned and processed in the Model class.
        It expects that the results are returned either as:
        - a tuple where:
          the first element is a DataFrame containing the results 
          and the second element is a dictionary containing the parameters used in the model run and corresponding unique index model runs 
        - or a dictionary containing the following (key, value) pairs:
          "result_time_series": a DataFrame containing the results
          "parameters_dict": a dictionary containing the parameters used in the model run and corresponding unique index model runs, 
          "gof_df": a DataFrame containing the goodness of fit values over different model runs, 
          "grad_matrix": a dictionary containing the gradient matrix for each QoI and each parameter, 
          "state_df": a DataFrame containing the state variables over different model runs

        :param rawSamples: if None, then the class will try to fatch df_simulation_result from kwargs;
          otherwise it will be initialized with None values for all attributes
        :param qoi_columns: should be a list or a string containing column names from the result DF,
        :param time_column_name: default is "TimeStamp"
        :param index_column_name: default is utility.INDEX_COLUMN_NAME
        :param kwargs:
            "original_model_output_column" - default is "Value" 
            "qoi_is_a_single_number" - default is False
            "grad_columns" - default is []
            "collect_and_save_state_data" - default is False
            "extract_only_qoi_columns" - default is False
            "df_simulation_result" - default is None; if rawSamples is None, 
                then the class checks if this parameter containsed saved simulation results is a form of a DataFrame
            "df_state_results" - default is None; relevant only if rawSamples is None and df_simulation_result is not None
            "df_index_parameter_values" - default is None; relevant only if rawSamples is None and df_simulation_result is not None
            "df_index_parameter_gof_values" - default is None; relevant only if rawSamples is None and df_simulation_result is not None
            "dict_of_approx_matrix_c" - default is None; relevant only if rawSamples is None and df_simulation_result is not None
            "dict_of_matrix_c_eigen_decomposition" - default is None; relevant only if rawSamples is None and df_simulation_result is not None

        Note: logic in Samples (and therefore in Statistics class) is opposite the one in a Model, e.g., 
          it is assumed that time_column is not an index in all the received/processed DFs!
        """
        self.time_column_name = time_column_name
        self.index_column_name = index_column_name
        original_model_output_column = kwargs.get('original_model_output_column', "Value")
        qoi_is_a_single_number = kwargs.get('qoi_is_a_single_number', False)
        grad_columns = kwargs.get('grad_columns', [])
        collect_and_save_state_data =  kwargs.get('collect_and_save_state_data', False)
        extract_only_qoi_columns = kwargs.get('extract_only_qoi_columns', False)
        df_simulation_result = kwargs.get('df_simulation_result', None)

        if not isinstance(qoi_columns, list):
            qoi_columns = [qoi_columns, ]

        self.qoi_columns = qoi_columns + [self.time_column_name, self.index_column_name]

        if grad_columns:
            self.qoi_columns = self.qoi_columns + grad_columns

        if rawSamples is not None:
            self._process_raw_samples(rawSamples, collect_and_save_state_data, extract_only_qoi_columns)
        elif df_simulation_result is not None:
            # self.df_simulation_result = df_simulation_result
            self.df_simulation_result = self._process_df_simulation_result(df_simulation_result, extract_only_qoi_columns)
            self.df_state_results = kwargs.get('df_state_results', None)
            self._process_state_df()
            self.df_index_parameter_values = kwargs.get('df_index_parameter_values', None)
            self.df_index_parameter_gof_values = kwargs.get('df_index_parameter_gof_values', None)
            self.dict_of_approx_matrix_c = kwargs.get('dict_of_approx_matrix_c', None)
            self.dict_of_matrix_c_eigen_decomposition = kwargs.get('dict_of_matrix_c_eigen_decomposition', None)
            self.list_index_run_with_None = []
        else:
            self.df_simulation_result = None
            self.df_state_results = None
            self.df_index_parameter_values = None
            self.df_index_parameter_gof_values = None
            self.dict_of_approx_matrix_c = None
            self.dict_of_matrix_c_eigen_decomposition = None
            self.list_index_run_with_None = []
        
    def _process_raw_samples(self, rawSamples, collect_and_save_state_data, extract_only_qoi_columns):
        list_of_single_df = []
        list_index_parameters_dict = []
        list_of_single_index_parameter_gof_df = []
        list_of_gradient_matrix_dict = []
        list_of_single_state_df = []
        self.list_index_run_with_None = []
        for index_run, value in enumerate(rawSamples, ):
            df_result = None
            parameters_dict = None
            gof_df = None
            gradient_matrix_dict = None
            state_df = None
            # this branch is due to some legacy code
            if isinstance(value, tuple):
                df_result = value[0]
                parameters_dict = value[1]
                list_index_parameters_dict.append(parameters_dict)
            # the expected structure of value is dictionary with the following keys
            elif isinstance(value, dict):
                if "result_time_series" in value:
                    df_result = value["result_time_series"]
                if "parameters_dict" in value:
                    parameters_dict = value["parameters_dict"]
                if "gof_df" in value:
                    gof_df = value["gof_df"]
                if "grad_matrix" in value:
                    gradient_matrix_dict = value["grad_matrix"]
                if "state_df" in value:
                    state_df = value["state_df"]
            else:
                df_result = value

            # Processing collected results
            if df_result is not None:
                df_result = self._process_df_simulation_result(df_result, extract_only_qoi_columns)
                list_of_single_df.append(df_result)                    
            else:
                if parameters_dict is not None and "successful_run" in parameters_dict and parameters_dict["successful_run"]:
                    print(f"Samples INFO - run {index_run} returned None, but successful_run flag is True! Something might be wrong!")
                # any other case is considered to be a clear flag that model execution was unsuccessful
                else:
                    print(f"Samples INFO - run {index_run} returned None, and successful_run flag is False! This is expected!")    
                    self.list_index_run_with_None.append(index_run)
                    # continue
            
            if parameters_dict is not None:
                list_index_parameters_dict.append(parameters_dict)

            if gof_df is not None:
                list_of_single_index_parameter_gof_df.append(gof_df)

            if gradient_matrix_dict is not None:
                list_of_gradient_matrix_dict.append(gradient_matrix_dict)

            if collect_and_save_state_data and state_df is not None:
                if isinstance(state_df, pd.DataFrame) and state_df.index.name == self.time_column_name:
                    state_df = state_df.reset_index()
                    state_df.rename(columns={state_df.index.name: self.time_column_name}, inplace=True)
                list_of_single_state_df.append(state_df)

        if list_of_single_df:
            self.df_simulation_result = pd.concat(list_of_single_df, ignore_index=True, sort=False, axis=0)
        else:
            self.df_simulation_result = None

        if list_index_parameters_dict:
            self.df_index_parameter_values = pd.DataFrame(list_index_parameters_dict)
        else:
            self.df_index_parameter_values = None

        if list_of_single_index_parameter_gof_df:
            self.df_index_parameter_gof_values = pd.concat(list_of_single_index_parameter_gof_df,
                                                           ignore_index=True, sort=False, axis=0)
        else:
            self.df_index_parameter_gof_values = None

        # In case compute_gradients mode was turned on and compute_active_subspaces set to True
        if list_of_gradient_matrix_dict:
            self._process_list_of_gradient_matrix_dict(list_of_gradient_matrix_dict)
        else:
            self.dict_of_approx_matrix_c = None
            self.dict_of_matrix_c_eigen_decomposition = None

        if collect_and_save_state_data and list_of_single_state_df:
            self.df_state_results = pd.concat(list_of_single_state_df, ignore_index=True, sort=False, axis=0)
        else:
            self.df_state_results = None
        
    def _process_df_simulation_result(self, df_result, extract_only_qoi_columns=False):
        # Note: logic in Statistics is opposite the one in a Model, e.g., it is assumed that time_column is not an index in DFs
        if df_result is not None:
            if isinstance(df_result, pd.DataFrame) and df_result.index.name == self.time_column_name:
                df_result = df_result.reset_index()
                df_result.rename(columns={df_result.index.name: self.time_column_name}, inplace=True)
            if self.time_column_name not in list(df_result.columns):
                raise Exception(f"Error in Samples class - {self.time_column_name} is not in the "
                                f"columns of the result DataFrame")
            if self.index_column_name not in list(df_result.columns):
                print(f"Be careful - {self.index_column_name} is not in samples.df_result.columns")
            if extract_only_qoi_columns:
                df_result = df_result[self.qoi_columns]
            if df_result.empty:
                raise Exception(f"Error in Samples class - df_result is empty after processing done inside _process_df_simulation_result!")
        return df_result

    def _process_state_df(self):
        if self.df_state_results is not None:
            if self.df_state_results.index.name == self.time_column_name:
                self.df_state_results = self.df_state_results.reset_index()
                self.df_state_results.rename(columns={self.df_state_results.index.name: self.time_column_name}, inplace=True)

    def _process_list_of_gradient_matrix_dict(self, list_of_gradient_matrix_dict):
        """ 
        This function is used to process the list of gradient matrix dictionaries
        and compute the average gradient matrix and its eigen decomposition.
        """
        # self.list_of_gradient_matrix_dict = list_of_gradient_matrix_dict
        gradient_matrix_dict = defaultdict(list)
        self.dict_of_approx_matrix_c = defaultdict(list)
        self.dict_of_matrix_c_eigen_decomposition = defaultdict(list)

        for single_gradient_matrix_dict in list_of_gradient_matrix_dict:
            for key, value in single_gradient_matrix_dict.items():
                gradient_matrix_dict[key].append(np.array(value))

        for key in gradient_matrix_dict.keys():
            # for single_objective_function in self.list_objective_function_qoi:
            self.dict_of_approx_matrix_c[key] = \
                sum(gradient_matrix_dict[key]) / len(gradient_matrix_dict[key])
            self.dict_of_matrix_c_eigen_decomposition[key] = np.linalg.eigh(self.dict_of_approx_matrix_c[key])
            # np.linalg.eig(self.dict_of_approx_matrix_c[key])

    def set_df_simulation_result(self, df_simulation_result, extract_only_qoi_columns=False):
        self.df_simulation_result = self._process_df_simulation_result(df_simulation_result, extract_only_qoi_columns)

    def set_df_state_results(self, df_state_results):
        self.df_state_results = df_state_results
        self._process_state_df()

    def set_df_index_parameter_values(self, df_index_parameter_values):
        self.df_index_parameter_values = df_index_parameter_values

    def set_df_index_parameter_gof_values(self, df_index_parameter_gof_values):
        self.df_index_parameter_gof_values = df_index_parameter_gof_values

    def set_list_of_gradient_matrix_dict(self, list_of_gradient_matrix_dict):
        self._process_list_of_gradient_matrix_dict(list_of_gradient_matrix_dict)

    def set_dict_of_approx_matrix_c(self, dict_of_approx_matrix_c):
        self.dict_of_approx_matrix_c = dict_of_approx_matrix_c
    
    def set_dict_of_matrix_c_eigen_decomposition(self, dict_of_matrix_c_eigen_decomposition):
        self.dict_of_matrix_c_eigen_decomposition = dict_of_matrix_c_eigen_decomposition

    def set_list_index_run_with_None(self, list_index_run_with_None):
        self.list_index_run_with_None = list_index_run_with_None

    def get_df_simulation_result(self):
        return self.df_simulation_result

    def get_df_state_results(self):
        return self.df_state_results

    def get_df_index_parameter_values(self):
        return self.df_index_parameter_values

    def get_df_index_parameter_gof_values(self):
        return self.df_index_parameter_gof_values

    def get_dict_of_approx_matrix_c(self):
        return self.dict_of_approx_matrix_c

    def get_dict_of_matrix_c_eigen_decomposition(self):
        return self.dict_of_matrix_c_eigen_decomposition

    def get_list_index_run_with_None(self):
        return self.list_index_run_with_None

    def save_simulation_results_to_file(self, file_path='./', filename=utility.DF_SIMULATIONS_FILE):
        file_path = str(file_path)
        if self.df_simulation_result is not None:
            self.df_simulation_result.to_pickle(
                os.path.abspath(os.path.join(file_path, filename)), compression="gzip")
   
    def save_all_simulations_to_file(self, file_path='./'):
        self.save_simulation_results_to_file(file_path)

    def save_states_to_file(self, file_path='./', filename=utility.DF_STATE_FILE):
        file_path = str(file_path)
        if self.df_state_results is not None:
            self.df_state_results.to_pickle(
                os.path.abspath(os.path.join(file_path, filename)), compression="gzip")

    def save_index_parameter_values(self, file_path='./', filename=utility.DF_INDEX_PARAMETER_FILE):
        file_path = str(file_path)
        if self.df_index_parameter_values is not None:
            self.df_index_parameter_values.to_pickle(
                os.path.abspath(os.path.join(file_path, filename)), compression="gzip")

    def save_index_parameter_gof_values(self, file_path='./', filename=utility.DF_INDEX_PARAMETER_GOF_FILE):
        file_path = str(file_path)
        if self.df_index_parameter_gof_values is not None:
            self.df_index_parameter_gof_values.to_pickle(
                os.path.abspath(os.path.join(file_path, filename)), compression="gzip")

    def save_dict_of_approx_matrix_c(self, file_path='./', filename=utility.DICT_APPROX_MATRIX_C_FILE):
        file_path = str(file_path)
        if self.dict_of_matrix_c_eigen_decomposition is not None:
            fileName = os.path.abspath(os.path.join(file_path, filename))
            with open(fileName, 'wb') as handle:
                pickle.dump(self.dict_of_approx_matrix_c, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_dict_of_matrix_c_eigen_decomposition(self, file_path='./', filename=utility.DICT_MATRIX_C_EIGEN_DECOMPOSITION_FILE):
        file_path = str(file_path)
        if self.dict_of_matrix_c_eigen_decomposition is not None:
            fileName = os.path.abspath(os.path.join(file_path, filename))
            with open(fileName, 'wb') as handle:
                pickle.dump(self.dict_of_matrix_c_eigen_decomposition, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def get_number_of_runs(self):
        if self.df_simulation_result is not None:
            self.number_unique_model_runs = self.df_simulation_result[self.index_column_name].nunique()
        else:
            self.number_unique_model_runs = None
        return self.number_unique_model_runs

    def get_number_of_timesteps(self, time_stamp_column=utility.TIME_COLUMN_NAME):
        if self.df_simulation_result is not None:
            self.number_unique_timesteps = self.df_simulation_result[time_stamp_column].nunique()
        else:
            self.number_unique_timesteps = None
        return self.number_unique_timesteps

    def get_simulation_timesteps(self, time_stamp_column=utility.TIME_COLUMN_NAME):
        if self.df_simulation_result is not None:
            return list(self.df_simulation_result[time_stamp_column].unique())
        else:
            return None

    def get_timesteps_min(self, time_stamp_column=utility.TIME_COLUMN_NAME):
        if self.df_simulation_result is not None:
            return self.df_simulation_result[time_stamp_column].min()
        else:
            return None

    def get_timesteps_max(self, time_stamp_column=utility.TIME_COLUMN_NAME):
        if self.df_simulation_result is not None:
            return self.df_simulation_result[time_stamp_column].max()
        else:
            return None


class TimeDependentStatistics(ABC, Statistics):
    def __init__(self, configurationObject, workingDir=None, *args, **kwargs):
        Statistics.__init__(self)

        self.configurationObject = utility.check_if_configurationObject_is_in_right_format_and_return(
            configurationObject, raise_error=True)

        # in the statistics class specification of the workingDir is necessary
        self.workingDir = pathlib.Path(workingDir)
        self.dict_output_file_paths = utility.get_dict_with_output_file_paths_based_on_workingDir(self.workingDir)
        #####################################
        # Set of configuration variables propagated via UQsim.args and/or **kwargs
        # These are mainly UQ simulation - related configurations
        #####################################
        self.uq_method = kwargs.get('uq_method', None)

        self.raise_exception_on_model_break = kwargs.get('raise_exception_on_model_break', False)
        if self.uq_method is not None and self.uq_method == "sc":  # always break when running gPCE simulation
            self.raise_exception_on_model_break = True

        self.sampleFromStandardDist = kwargs.get('sampleFromStandardDist', False)

        # if set to True original qoi values will be saved in the
        # in the stat result dict under key "qoi_values"; note - this might take a lot of space
        self.store_qoi_data_in_stat_dict = kwargs.get('store_qoi_data_in_stat_dict', False)
        # if set to True the computed gPCE model will be saved in the stat result dict under key "gPCE"
        self.store_gpce_surrogate_in_stat_dict = kwargs.get('store_gpce_surrogate_in_stat_dict', False)
        # if set to True the computed gPCE model will be saved is a file for each qoi and each time-step;
        # Note: no need to have both store_gpce_surrogate_in_stat_dict and save_gpce_surrogate set to True; 
        # however current implamantion requires that store_gpce_surrogate_in_stat_dict is set to True if save_gpce_surrogate is set to True
        self.save_gpce_surrogate = kwargs.get('save_gpce_surrogate', False)
        if self.save_gpce_surrogate:
            self.store_gpce_surrogate_in_stat_dict = True
        self.compute_other_stat_besides_pce_surrogate = kwargs.get('compute_other_stat_besides_pce_surrogate', True)

        self.parallel_statistics = kwargs.get('parallel_statistics', False)
        if self.parallel_statistics:
            self.size = MPI.COMM_WORLD.Get_size()
            self.rank = MPI.COMM_WORLD.Get_rank()
            self.name = MPI.Get_processor_name()
            self.version = MPI.Get_library_version()
            self.mpi_chunksize = kwargs.get('mpi_chunksize', 1)
            self.unordered = kwargs.get('unordered', False)

        # if set to True, different arguments of the Samples class will be saved, e.g., df_simulation_result
        self.save_all_simulations = kwargs.get('save_all_simulations', False)
        self.collect_and_save_state_data = kwargs.get('collect_and_save_state_data', False)

        self.compute_stat_on_delta_qoi = kwargs.get('compute_stat_on_delta_qoi', False)

        self.compute_Sobol_t = kwargs.get('compute_Sobol_t', False)
        self.compute_Sobol_m = kwargs.get('compute_Sobol_m', False)
        self.compute_Sobol_m2 = kwargs.get('compute_Sobol_m2', False)

        self.instantly_save_results_for_each_time_step = kwargs.get(
            'instantly_save_results_for_each_time_step', False)

        self.compute_sobol_indices_with_samples = kwargs.get(
            'compute_sobol_indices_with_samples', False)
        # if self.uq_method == "mc" and not self.regression and self.compute_Sobol_m:
        if self.uq_method == "mc" and self.compute_Sobol_m:
            self.compute_sobol_indices_with_samples = True
        if self.uq_method == "saltelli":
            self.compute_sobol_indices_with_samples = False

        if 'nodes_file' in kwargs:
            self.nodes_file = kwargs['nodes_file']
        elif 'nodes_file' in self.dict_output_file_paths:
            self.nodes_file = self.dict_output_file_paths['nodes_file']
        else:
            self.nodes_file  = utility.NODES_FILE  # todo, think about adding as another option utility.DF_UQSIM_SIMULATION_NODES_FILE
        if not self.nodes_file.is_absolute():
            self.nodes_file = self.workingDir / self.nodes_file

        self.allow_conditioning_results_based_on_metric = kwargs.get(
            'allow_conditioning_results_based_on_metric', False)

        self.compute_kl_expansion_of_qoi = kwargs.get('compute_kl_expansion_of_qoi', False)
        self.compute_timewise_gpce_next_to_kl_expansion = kwargs.get('compute_timewise_gpce_next_to_kl_expansion', False)
        self.compute_generalized_sobol_indices = kwargs.get('compute_generalized_sobol_indices', False)
        self.compute_generalized_sobol_indices_over_time = kwargs.get('compute_generalized_sobol_indices_over_time', False)
        self.compute_covariance_matrix_in_time = kwargs.get('compute_covariance_matrix_in_time', False)
        if self.compute_covariance_matrix_in_time and self.instantly_save_results_for_each_time_step:
            print(f"[STAT INFO]: Covariance Matrix will not be computed since since the fleg \
            instantly_save_results_for_each_time_step is set to True")
        
        # These are set-up which have stronger priorty than instantly_save_results_for_each_time_step
        # What if someone set compute_generalized_sobol_indices or compute_kl_expansion_of_qoi to True by mistake when, for example, uq_method is saltelli
        # condition_for_turning_off_instantly_save_results_for_each_time_step = self.compute_generalized_sobol_indices or self.compute_kl_expansion_of_qoi
        # if condition_for_turning_off_instantly_save_results_for_each_time_step:
        #     print(f"Variable time_dependent_statistics.instantly_save_results_for_each_time_step will be set to False")
        #     self.instantly_save_results_for_each_time_step = False

        if self.compute_kl_expansion_of_qoi:
            self.kl_expansion_order = kwargs.get("kl_expansion_order", 2)
        #####################################
        # Set of configuration variables propagated via **kwargs or read from configurationObject
        # These are mainly model related configurations
        #####################################
        self.time_column_name = kwargs.get("time_column_name", utility.TIME_COLUMN_NAME)
        self.index_column_name = kwargs.get("index_column_name", utility.INDEX_COLUMN_NAME)
        self.forcing_data_column_names = kwargs.get("forcing_data_column_names", "precipitation")

        try:
            self.resolution = self.configurationObject["time_settings"]["resolution"]
        except KeyError as e:
            self.resolution = "integer"
        if self.resolution != "daily" and self.resolution != "hourly" and self.resolution != "minute" and self.resolution != "integer":
            raise Exception(f"Error in Statistics class - resolution is not daily, hourly, minute or integer")
        
        if "corrupt_forcing_data" in kwargs:
            self.corrupt_forcing_data = kwargs['corrupt_forcing_data']
        else:
            try:
                self.corrupt_forcing_data = strtobool(self.configurationObject["model_settings"]["corrupt_forcing_data"])
            except KeyError as e:
                self.corrupt_forcing_data = False

        self.dict_what_to_plot = kwargs.get("dict_what_to_plot", utility.DEFAULT_DICT_WHAT_TO_PLOT)
        self.dict_stat_to_compute = kwargs.get("dict_stat_to_compute", utility.DEFAULT_DICT_STAT_TO_COMPUTE)

        self.dict_stat_to_compute['Sobol_t'] = self.compute_Sobol_t
        self.dict_stat_to_compute['Sobol_m'] = self.compute_Sobol_m
        self.dict_stat_to_compute['Sobol_m2'] = self.compute_Sobol_m2

        self.free_result_dict_memory = kwargs.get("free_result_dict_memory", True)

        #####################################
        # Parameters related set-up part
        #####################################
        self.nodeNames = [] # list of uncertain parameter names
        try:
            list_of_parameters = self.configurationObject["parameters"]
        except KeyError as e:
            print(f"Statistics: parameters key does "
                  f"not exists in the configurationObject{e}")
            raise
        for i in list_of_parameters:
            if self.uq_method == "ensemble" or i["distribution"] != "None":
                self.nodeNames.append(i["name"])
        self.dim = len(self.nodeNames)  # this should be equal to number of uncertain parameters and dim = simulationNodes.distNodes.shape[0]
        self.labels = [nodeName.strip() for nodeName in self.nodeNames]

        #####################################
        # Initialize different variables of the Statistics class
        #####################################
        self.samples = None
        self.list_of_unsuccessful_runs = []
        # self.result_dict = dict()
        self.result_dict = None
        self.groups = None

        self.nodes = None

        self.dist = None
        self.weights = None

        self.df_unaltered = None
        self.df_measured = None
        self.forcing_df = None
        self.df_statistics = None
        self.unaltered_computed = False
        self.measured_fetched = False
        self.forcing_data_fetched = False

        self.precipitation_temperature_df = None
        self.time_series_measured_data_df = None

        self._is_Sobol_t_computed = False
        self._is_Sobol_m_computed = False
        self._is_Sobol_m2_computed = False

        self.timesteps = None
        self.timesteps_min = None
        self.timesteps_max = None
        self.numTimesteps = None
        self.N_quad = None
        self.pdTimesteps = None
        self.timestep_qoi = None

        self.weights_time = None

        self.number_of_unique_index_runs = None  # depricated; the same as self.number_unique_model_runs
        self.numEvaluations = None
        self.N = None
        self.number_unique_model_runs = None

        self.qoi_mean_df = None
        self.gof_mean_measured = None

        self.active_scores_dict = None
        self.df_time_varying_grad_analysis = None
        self.df_time_aggregated_grad_analysis = None
        self.solverTimes = None
        self.work_package_indexes = None

        self.polynomial_expansion = None
        self.polynomial_norms = None

        self.dict_processed_simulation_settings_from_config_file = None
        self.set_attributes_based_on_dict_processed_simulation_settings_from_config_file(**kwargs)

    def set_attributes_based_on_dict_processed_simulation_settings_from_config_file(self, **kwargs):
        if self.dict_processed_simulation_settings_from_config_file is None:
            self.set_dict_processed_simulation_settings_from_config_file(**kwargs)

        self.assign_values(self.dict_processed_simulation_settings_from_config_file)

        if self.autoregressive_model_first_order and (self.qoi == "GoF" or self.mode == "sliding_window"):
            print(f"Possible error in the configuration file - autoregressive_model_first_order is set to True, \
            but qoi is GoF or mode is sliding_window. Setting autoregressive_model_first_order to False!")
            self.autoregressive_model_first_order = False
        self.compute_stat_on_delta_qoi = self.autoregressive_model_first_order

        # additional assignments based on attributes set from dict_processed_simulation_settings_from_config_file
        self.list_original_model_output_columns = self.list_qoi_column.copy()
        self.dict_corresponding_original_qoi_column = defaultdict()
        self.additional_qoi_columns_besides_original_model_output = False
        self.qoi_is_a_single_number = False
        self.list_grad_columns = []

        self.infer_qoi_column_names(**kwargs)

    def set_dict_processed_simulation_settings_from_config_file(
            self, dict_processed_simulation_settings_from_config_file=None, **kwargs):
        if dict_processed_simulation_settings_from_config_file is None:
            self.dict_processed_simulation_settings_from_config_file = \
                utility.read_simulation_settings_from_configuration_object(
            self.configurationObject, **kwargs)
        else:
            self.dict_processed_simulation_settings_from_config_file = dict_processed_simulation_settings_from_config_file

    def assign_values(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

    def infer_qoi_column_names(self, **kwargs):
        """
        This is an important function when some other output or processed model output is regarded as final QoI.
        This function largely depends on the way different qoi columns were computed and named in the Model class!
        :param kwargs:
        :return:
        """
        # TODO Make one general function from this one in uqef_dynamic_utils or utilities...
        # TODO Think if this function should be moved to the Model class or utility and then info propagated!?
        # TODO Is this redundant with self.store_qoi_data_in_stat_dict
        always_process_original_model_output = kwargs.get("always_process_original_model_output", False)
        list_qoi_column_processed = []
        dict_corresponding_original_qoi_column = defaultdict()

        if self.mode == "continuous":
            if self.qoi == "GoF":
                for idx, single_qoi_column in enumerate(self.list_qoi_column):
                    if self.list_read_measured_data[idx]:
                        for single_objective_function_name_qoi in self.list_objective_function_names_qoi:
                            new_column_name = single_objective_function_name_qoi + "_" + single_qoi_column
                            list_qoi_column_processed.append(new_column_name)
                            dict_corresponding_original_qoi_column[new_column_name] = single_qoi_column
                            self.additional_qoi_columns_besides_original_model_output = True
                            self.qoi_is_a_single_number = True
                            # TODO in this case QoI is a single number, not a time-series!!!
            else:
                if self.autoregressive_model_first_order:
                    for single_qoi_column in self.list_qoi_column:
                        new_column_name = "delta_" + single_qoi_column
                        list_qoi_column_processed.append(new_column_name)
                        dict_corresponding_original_qoi_column[new_column_name] = single_qoi_column
                        self.additional_qoi_columns_besides_original_model_output = True
                else:
                    # here, model output itself is regarded as a QoI
                    pass
                # list_qoi_column_processed.append(self.list_qoi_column)
        elif self.mode == "sliding_window":
            if self.qoi == "GoF":
                for idx, single_qoi_column in enumerate(self.list_qoi_column):
                    if self.list_read_measured_data[idx]:
                        for single_objective_function_name_qoi in self.list_objective_function_names_qoi:
                            new_column_name = single_objective_function_name_qoi + "_" + single_qoi_column + \
                                              "_sliding_window"
                            list_qoi_column_processed.append(new_column_name)
                            dict_corresponding_original_qoi_column[new_column_name] = single_qoi_column
                            self.additional_qoi_columns_besides_original_model_output = True
            else:
                for idx, single_qoi_column in enumerate(self.list_qoi_column):
                    new_column_name = single_qoi_column + "_" + self.method + "_sliding_window"
                    list_qoi_column_processed.append(new_column_name)
                    dict_corresponding_original_qoi_column[new_column_name] = single_qoi_column
                    self.additional_qoi_columns_besides_original_model_output = True

        if self.compute_gradients:
            if self.gradient_analysis:
                always_process_original_model_output = True
                for single_param_name in self.nodeNames:
                    if self.mode == "continuous":
                        if self.qoi == "GoF":
                            for idx, single_qoi_column in enumerate(self.list_qoi_column):
                                if self.list_read_measured_data[idx]:
                                    for single_objective_function_name_qoi in self.list_objective_function_names_qoi:
                                        new_column_name = "d_" + single_objective_function_name_qoi + "_" + \
                                                          single_qoi_column + "_" + "_d_" + single_param_name
                                        # list_qoi_column_processed.append(new_column_name)
                                        self.list_grad_columns.append(new_column_name)
                                        # self.additional_qoi_columns_besides_original_model_output = True
                                        self.qoi_is_a_single_number = True
                                        # TODO in this case QoI is a single number, not a time-series!!!
                        else:
                            for idx, single_qoi_column in enumerate(self.list_qoi_column):
                                new_column_name = "d_" + single_qoi_column + "_d_" + single_param_name
                                # list_qoi_column_processed.append(new_column_name)
                                self.list_grad_columns.append(new_column_name)
                                # self.additional_qoi_columns_besides_original_model_output = True
                    elif self.mode == "sliding_window":
                        if self.qoi == "GoF":
                            for idx, single_qoi_column in enumerate(self.list_qoi_column):
                                if self.list_read_measured_data[idx]:
                                    for single_objective_function_name_qoi in self.list_objective_function_names_qoi:
                                        new_column_name = "d_" + single_objective_function_name_qoi + "_" + \
                                                          single_qoi_column + "_sliding_window" \
                                                          + "_d_" + single_param_name
                                        # list_qoi_column_processed.append(new_column_name)
                                        self.list_grad_columns.append(new_column_name)
                                        # self.additional_qoi_columns_besides_original_model_output = True
                        else:
                            for idx, single_qoi_column in enumerate(self.list_qoi_column):
                                new_column_name = "d_" + single_qoi_column + "_" + self.method + "_sliding_window" + \
                                                  "_d_" + single_param_name
                                # list_qoi_column_processed.append(new_column_name)
                                self.list_grad_columns.append(new_column_name)
                                # self.additional_qoi_columns_besides_original_model_output = True

            elif self.compute_active_subspaces:
                always_process_original_model_output = True
                pass

        if self.corrupt_forcing_data:
            always_process_original_model_output = True
            pass  # 'precipitation' column is in the results df

        wrong_computation_of_new_qoi_columns = self.additional_qoi_columns_besides_original_model_output and \
                              len(list_qoi_column_processed) == 0
        assert not wrong_computation_of_new_qoi_columns

        if self.additional_qoi_columns_besides_original_model_output and len(list_qoi_column_processed) != 0:
            if always_process_original_model_output:
                self.list_qoi_column = self.list_original_model_output_columns + list_qoi_column_processed
                for single_qoi_column in self.list_original_model_output_columns:
                    dict_corresponding_original_qoi_column[single_qoi_column] = single_qoi_column
            else:
                self.list_qoi_column = list_qoi_column_processed
        else:
            for single_qoi_column in self.list_original_model_output_columns:
                dict_corresponding_original_qoi_column[single_qoi_column] = single_qoi_column

        self.dict_corresponding_original_qoi_column = dict_corresponding_original_qoi_column

        # print(f"[STAT INFO] Statistics class will process the following QoIs:\n {self.list_qoi_column}\n"
        #       f"whereas the columns representing the model output itself are:{self.list_original_model_output_columns}"
        #       f"and additional QoI columns are { list_qoi_column_processed}. "
        #       f"Plus the following gradient columns exist in the result DF {self.list_grad_columns}")

    # =================================================================================================

    # TODO Write version of the function which reads already saved self.samples.df_simulation_result
    # from some file and continues...
    def _get_list_of_columns_to_filter_from_results(self):
        list_of_columns_to_filter_from_results = self.list_qoi_column + list(
            self.dict_corresponding_original_qoi_column.values())
        if self.corrupt_forcing_data:
            list_of_columns_to_filter_from_results = list_of_columns_to_filter_from_results + [
                self.forcing_data_column_names]
        list_of_columns_to_filter_from_results = list(set(list_of_columns_to_filter_from_results))
        return list_of_columns_to_filter_from_results

    def prepare(self, rawSamples, **kwargs):
        """
        Prepares the data for statistical analysis. This method is called before any statistical analysis is performed.

        Args:
            rawSamples (list or numpy.ndarray): The raw samples data.
            **kwargs: Additional keyword arguments.

        Keyword Args:
            timesteps (list or numpy.ndarray): The timesteps data. UQEF inherited. Will most likely be overwritten based on samples data.
            solverTimes (list or numpy.ndarray): The solver times data. UQEF inherited.
            work_package_indexes (list or numpy.ndarray): The work package indexes data. UQEF inherited.
            extract_only_qoi_columns (bool): Flag indicating whether to extract only QoI columns.
            read_saved_simulations (bool): Flag indicating whether to read saved simulations. Will be set to True if rawSamples is None.

        Raises:
            Exception: If the specified file for reading saved simulations does not exist. 
            Logic is that if read_saved_simulations is True or rawSamples is None
            then the class will try to read df_simulation_result from df_all_simulations_file; 
            otherwise it will throw an exception if df_all_simulations_file does not exist.

        Returns:
            None
        """
        self.timesteps = kwargs.get('timesteps', None)
        self.solverTimes = kwargs.get('solverTimes', None)
        self.work_package_indexes = kwargs.get('work_package_indexes', None)
        self.extract_only_qoi_columns = kwargs.get('extract_only_qoi_columns', False)
        self.read_saved_simulations = kwargs.get('read_saved_simulations', False)
        # TODO a couple of similar/redundant variables
        # self.store_qoi_data_in_stat_dict, self.extract_only_qoi_columns always_process_original_model_output

        list_of_columns_to_filter_from_results = self._get_list_of_columns_to_filter_from_results()

        self.samples = Samples(rawSamples,
                        qoi_columns=list_of_columns_to_filter_from_results,
                        index_column_name=self.index_column_name,
                        time_column_name=self.time_column_name,
                        extract_only_qoi_columns=self.extract_only_qoi_columns,
                        original_model_output_column=self.list_original_model_output_columns,
                        qoi_is_a_single_number=self.qoi_is_a_single_number,
                        grad_columns=self.list_grad_columns,
                        collect_and_save_state_data=self.collect_and_save_state_data,
                        )
        
        if rawSamples is None:
            self.read_saved_simulations = True

        if self.read_saved_simulations:
            # TODO Move to new function
            df_simulations_file = self.dict_output_file_paths.get("df_simulations_file")
            df_index_parameter_file = self.dict_output_file_paths.get("df_index_parameter_file")
            df_index_parameter_gof_file = self.dict_output_file_paths.get("df_index_parameter_gof_file")
            # df_uqsim_simulation_parameters_file = self.dict_output_file_paths.get("df_uqsim_simulation_parameters_file")
            # df_uqsim_simulation_nodes_file = self.dict_output_file_paths.get("df_uqsim_simulation_nodes_file")
            # df_uqsim_simulation_weights_file = self.dict_output_file_paths.get("df_uqsim_simulation_weights_file")
            if df_simulations_file.is_file():
                df_simulation_result = pd.read_pickle(df_simulations_file, compression="gzip")
                self.samples.set_df_simulation_result(df_simulation_result, extract_only_qoi_columns=self.extract_only_qoi_columns)
            else:
                raise Exception(f"Error in Statistics class - file {df_simulations_file} does not exist!")
            # TODO read other files if they exist, e.g., state_df and gradient based data!
            if df_index_parameter_file.is_file():
                df_index_parameter = pd.read_pickle(df_index_parameter_file, compression="gzip")
                self.samples.set_df_index_parameter_values(df_index_parameter)
            if df_index_parameter_gof_file.is_file():
                df_index_parameter_gof = pd.read_pickle(df_index_parameter_gof_file, compression="gzip")
                self.samples.set_df_index_parameter_gof_values(df_index_parameter_gof)

        if self.samples is not None:
            # Here, some functions are called to process the data from self.samples

            # this is a list of indexes of runs which returned None
            self.list_of_unsuccessful_runs = self.samples.get_list_index_run_with_None()

            if self.samples.df_index_parameter_gof_values is not None:
                # here, maybe you want to extend self.list_of_unsuccessful_runs with indexes of runs which do not satisfy GoF criteria
                # this is done in prepared methods for different uq methods, e.g., prepareForMcStatistics
                # or do this later in the prepare section!!!
                pass

            if self.samples.df_simulation_result is not None:
                self.samples.df_simulation_result.sort_values(
                    by=[self.index_column_name, self.time_column_name], ascending=[True, True], inplace=True, kind='quicksort',
                    na_position='last'
                )

            if self.samples.df_state_results is not None:
                self.samples.df_state_results.sort_values(
                    by=[self.index_column_name, self.time_column_name], ascending=[True, True], inplace=True, kind='quicksort',
                    na_position='last'
                )

            if self.compute_gradients and self.compute_active_subspaces:
                self.active_scores_dict = TimeDependentStatistics._compute_active_score(self.samples.dict_of_matrix_c_eigen_decomposition)

            if self.compute_gradients and self.gradient_analysis:
                self.param_grad_analysis()

            # saving the results of the model runs
            if self.save_all_simulations:
                self.samples.save_simulation_results_to_file(self.workingDir)
            if self.collect_and_save_state_data:
                self.samples.save_states_to_file(self.workingDir)
            # These other files should take much less memory, therefore, always save tham
            self.samples.save_index_parameter_values(self.workingDir)
            self.samples.save_index_parameter_gof_values(self.workingDir)

            if self.compute_gradients and self.compute_active_subspaces:
                self.samples.save_dict_of_approx_matrix_c(self.workingDir)
                self.samples.save_dict_of_matrix_c_eigen_decomposition(self.workingDir)

            # print(f"DEBUGGING self.samples.df_simulation_result\n{self.samples.df_simulation_result}")
            # print(f"DEBUGGING self.samples.df_simulation_result.columns\n{self.samples.df_simulation_result.columns}")
            # print(f"DEBUGGING self.samples.df_simulation_result.dtypes\n{self.samples.df_simulation_result.dtypes}")

            # Read info about the time and total number of model runs from propagated model runs, i.e., samples
            self.timesteps = self.samples.get_simulation_timesteps()
            self.timesteps_min = self.samples.get_timesteps_min()
            self.timesteps_min_minus_one = utility.compute_previous_timestamp(
                self.timesteps_min, resolution=self.resolution)
            self.timesteps_max = self.samples.get_timesteps_max()
            self.number_of_unique_index_runs = self.number_unique_model_runs = self.samples.get_number_of_runs()

        if self.timesteps is not None:
            self._set_pdTimesteps_based_on_timesteps_and_resolution()
            self.numTimesteps = self.N_quad = len(self.timesteps)
            if self.N_quad > 1:
                # for now uniform weight in time are default
                difference_between_two_time_stamps = utility.compute_difference_between_two_time_stamps(
                    end_timestamp=self.timesteps_max, start_timestamp=self.timesteps_min, resolution=self.resolution)
                h = (difference_between_two_time_stamps)/(self.N_quad-1) #1/(self.N_quad-1) #
                self.weights_time = [h for i in range(self.N_quad)]
                self.weights_time[0] /= 2
                self.weights_time[-1] /= 2
                # self.weights_time = np.asfarray(self.weights_time)
                self.weights_time = np.asarray(self.weights_time, dtype=np.float32)
            else:
                # self.weights_time = np.asfarray([1.0])
                self.weights_time = np.asarray([1.0], dtype=np.float32)
            assert len(self.timesteps)==len(self.weights_time)
        else:
            self.numTimesteps = self.N_quad = None  
            self.weights_time = None

        self.centered_output = defaultdict(np.ndarray, {key:[] for key in self.list_qoi_column})
        self.covariance_matrix = defaultdict(np.ndarray, {key:[] for key in self.list_qoi_column})

        if self.autoregressive_model_first_order:
            self.get_measured_data(
                timestepRange=[self.timesteps_min_minus_one, self.timesteps_max],
                qoi_column_name=self.list_original_model_output_columns  # or list(self.dict_corresponding_original_qoi_column.values)
            )
        
        # This, though, does not hold for Saltelli's approach
        # self.numEvaluations = self.number_of_unique_index_runs

        self._set_timestep_qoi()

        self.prepare_for_plotting(**kwargs)

    # =================================================================================================

    # TODO Write more getters and setters!

    def set_result_dict(self, result_dict):
        self.result_dict = result_dict
        # TODO Should I update self.timesteps self.pdTimesteps self.numTimesteps based on self.result_dict??/
        #self.set_timesteps()

    def get_result_dict(self):
        return self.result_dict

    # def _compute_previous_timestep(self, timestamp):
    #     if self.resolution == "daily":
    #         # pd.DateOffset(days=1)
    #         previous_timestamp = pd.to_datetime(timestamp) - pd.Timedelta(days=1)
    #     elif self.resolution == "hourly":
    #         previous_timestamp = pd.to_datetime(timestamp) - pd.Timedelta(h=1)
    #     elif self.resolution == "minute":
    #         previous_timestamp = pd.to_datetime(timestamp) - pd.Timedelta(m=1)
    #     return previous_timestamp

    def set_resolution(self, resolution: str=None):
        possible_resolutions = ["integer", "daily", "hourly", "minute"]
        if resolution is not None and resolution in possible_resolutions:
            self.resolution = resolution
        else:
            raise Exception(f"Error in Statistics.set_resolution - resolution is not in the list of possible resolutions")

    def _set_pdTimesteps_based_on_timesteps_and_resolution(self):
        if self.resolution == "integer":
            self.pdTimesteps = self.timesteps
        elif self.resolution == "daily" or self.resolution == "hourly" or self.resolution == "minute":
            self.pdTimesteps = [pd.Timestamp(timestep) for timestep in self.timesteps]
        else:
            self.pdTimesteps = self.timesteps
    
    def set_timesteps(self, timesteps=None):
        if timesteps is not None:
            self.timesteps = timesteps
        elif self.samples is not None:
            self.timesteps = self.samples.get_simulation_timesteps()
        elif self.result_dict is not None:
            try:
                self.timesteps = list(self.result_dict[self.list_qoi_column[0]].keys())
            except KeyError as e:
                print(f"Error in Statistics.set_timesteps - one is trying to infer timesteps from result_dict;"
                      f"however, entry {self.list_qoi_column[0]} does not exist in the dict")
                raise
        if self.timesteps is not None:
            self._set_pdTimesteps_based_on_timesteps_and_resolution()
            # self.pdTimesteps = [pd.Timestamp(timestep) for timestep in self.timesteps]
            # self.numTimesteps = self.N_quad = len(self.timesteps)
            self.set_numTimesteps()  # TODO check if this should be here
            self.set_weights_time()  # TODO check if this should be here

    def set_pdTimesteps(self, pdTimesteps=None):
        if pdTimesteps is not None:
            self.pdTimesteps = pdTimesteps
        elif self.timesteps is not None:
            self._set_pdTimesteps_based_on_timesteps_and_resolution()
            # self.pdTimesteps = [pd.Timestamp(timestep) for timestep in self.timesteps]

    def set_timesteps_min(self, timesteps_min=None):
        if timesteps_min is not None:
            self.timesteps_min = timesteps_min
        elif self.samples is not None:
            self.timesteps_min = self.samples.get_timesteps_min()
        elif self.timesteps is not None:
            self.timesteps_min = min(self.timesteps)
        self.timesteps_min_minus_one = utility.compute_previous_timestamp(
            self.timesteps_min, resolution=self.resolution)

    def set_timesteps_max(self, timesteps_max=None):
        if timesteps_max is not None:
            self.timesteps_max = timesteps_max
        elif self.samples is not None:
            self.timesteps_max = self.samples.get_timesteps_max()
        elif self.timesteps is not None:
            self.timesteps_max = max(self.timesteps)

    def set_number_of_unique_index_runs(self, number_of_unique_index_runs=None):
        if number_of_unique_index_runs is not None:
            self.number_of_unique_index_runs = self.number_unique_model_runs = number_of_unique_index_runs
        elif self.samples is not None:
            self.number_of_unique_index_runs = self.number_unique_model_runs = self.samples.get_number_of_runs()

        # if self.number_of_unique_index_runs is not None:
        #     self.numEvaluations = self.number_of_unique_index_runs

    def get_number_of_unique_index_runs(self):
        return self.number_unique_model_runs   # self.number_of_unique_index_runs

    def set_numTimesteps(self, numbTimesteps=None):
        if numbTimesteps is not None:
            self.numTimesteps = numbTimesteps
        elif self.timesteps is not None:
            self.numTimesteps = self.N_quad = len(self.timesteps)

    def set_weights_time(self, weights_time=None):
        if weights_time is not None:
            self.weights_time = weights_time
            #self.weights_time = np.asfarray(self.weights_time)
            self.weights_time = np.asarray(self.weights_time, dtype=np.float32)
            self.N_quad = len(self.weights_time)
        elif self.timesteps is not None:
            if self.timesteps_max is None:
                self.set_timesteps_max()
            if self.timesteps_min is None:
                self.set_timesteps_min()
            if self.timesteps_max is None or self.timesteps_min is None:
                raise Exception("[STAT ERROR] - in set_weights_time() - self.timesteps_max or self.timesteps_min are missing/None")
            self.N_quad = len(self.timesteps)
            if self.N_quad > 1:
                difference_between_two_time_stamps = utility.compute_difference_between_two_time_stamps(
                    end_timestamp=self.timesteps_max, start_timestamp=self.timesteps_min, resolution=self.resolution)
                h = (difference_between_two_time_stamps)/(self.N_quad-1) #1/(self.N_quad-1) #
                self.weights_time = [h for i in range(self.N_quad)]
                self.weights_time[0] /= 2
                self.weights_time[-1] /= 2
                #self.weights_time = np.asfarray(self.weights_time)
                self.weights_time = np.asarray(self.weights_time, dtype=np.float32)
            else:
                #self.weights_time = np.asfarray([1.0])
                self.weights_time = np.asarray([1.0], dtype=np.float32)
            assert len(self.timesteps)==len(self.weights_time)
        else:
            self.weights_time = None

    def _set_timestep_qoi(self):
        if self.qoi_is_a_single_number:
            if self.timesteps is not None and isinstance(self.timesteps, list):
                # TODO take a middle element
                self.timestep_qoi = self.timesteps[-1]
        else:
            self.timestep_qoi = self.timesteps

    def get_time_range(self):
        if self.resolution == "integer":
            return (self.timesteps_min, self.timesteps_max)
        elif self.resolution == "daily" or self.resolution == "hourly" or self.resolution == "minute":
            return (pd.Timestamp(self.timesteps_min), pd.Timestamp(self.timesteps_max))
        else:
            return (self.timesteps_min, self.timesteps_max)
    # =================================================================================================

    def param_grad_analysis(self):
        # In case compute_gradients mode was turned on and gradient_analysis set to True
        if self.samples.df_simulation_result is not None and self.compute_gradients \
                and self.gradient_analysis and self.list_grad_columns:
            list_of_single_dict_grad_analysis_time_aggregated = []
            list_of_single_df_grad_analysis_over_time = []
            if not isinstance(self.list_grad_columns, list):
                self.list_grad_columns = [self.list_grad_columns, ]
            for single_grad_columns in self.list_grad_columns:
                result_dict_time_aggregated, result_df_over_time = \
                    TimeDependentStatistics._single_qoi_single_param_grad_analysis(
                        self.samples.df_simulation_result, single_grad_columns, self.time_column_name)
                list_of_single_dict_grad_analysis_time_aggregated.append(result_dict_time_aggregated)
                list_of_single_df_grad_analysis_over_time.append(result_df_over_time)
            self.df_time_aggregated_grad_analysis = pd.DataFrame.from_dict(list_of_single_dict_grad_analysis_time_aggregated)
            self.df_time_varying_grad_analysis = pd.concat(list_of_single_df_grad_analysis_over_time,
                                                           ignore_index=False, axis=1)
        else:
            raise Exception("[STAT ERROR] - Calling method param_grad_analysis "
                            "when not all the requirements for the analysis are meet")

    def save_param_grad_analysis(self, file_path='./'):
        file_path = str(file_path)
        if self.df_time_varying_grad_analysis is not None:
            self.df_time_varying_grad_analysis.to_pickle(
                os.path.abspath(os.path.join(file_path, "df_time_varying_grad_analysis.pkl")), compression="gzip")
        if self.df_time_aggregated_grad_analysis is not None:
            self.df_time_aggregated_grad_analysis.to_pickle(
                os.path.abspath(os.path.join(file_path, "df_time_aggregated_grad_analysis.pkl")), compression="gzip")
    
    # =================================================================================================
    
    def ensure_nodes_are_loaded(self):
        """
        in case self.nodes are not provided (i.e., not set via simulationNodes), 
        but they are reuqired (i.e., PSP approach or MC/Saltelli when computing Sobol indices via nodes/samples) 
        then this function checks if self.nodes_file is provided, 
        and reads the nodes from the file. 

        As a side effect this function will update/modify 
        self.weights, self.dim, self.N and self.numEvaluations
        """
        if self.nodes is None:
            if self.nodes_file is not None and self.nodes_file.is_file():
                with open(self.nodes_file, 'rb') as f:
                    uqef_simulationNodes = pickle.load(f)
                    self.nodes = uqef_simulationNodes.distNodes
                    self.weights = uqef_simulationNodes.weights
                    if self.sampleFromStandardDist:
                        self.dist = uqef_simulationNodes.joinedStandardDists
                    else:
                        self.dist = uqef_simulationNodes.joinedDists
                    self.dim = self.nodes.shape[0]
                    self.N = self.numEvaluations = self.nodes.shape[1]
        if self.nodes.size == 0 or self.nodes is None:
            raise Exception("[STAT ERROR] - Nodes are not provided and they can not be read from the file!")
     
    def handle_expansion_generation(self, simulationNodes=None):
        self.polynomial_expansion, self.polynomial_norms = cp.generate_expansion(
            self.order, self.dist, rule=self.poly_rule, normed=self.poly_normed,
            graded=True, reverse=True, cross_truncation=self.cross_truncation, retall=True)

    def _check_if_parameters_are_set_for_regression_or_psp(self):
        if self.order is None:
            raise Exception("[STAT ERROR] - order for polynomial expansion is not provided")
        if self.dist is None:
            raise Exception("[STAT ERROR] - distribution for polynomial expansion is not provided")

    # =================================================================================================

    def handle_unsuccessful_runs(self):
        """
        This function should be called after the simulation results are fetched and before any statistical analysis
        is performed. It removes the nodes corresponding to the runs which returned None. 
        Actually these runs where not added to self.samples.df_simulation_result, at the first place. 
        This function also updated self.numEvaluations, self.N, self.dim, etc.

        self.list_of_unsuccessful_runs should contain list of indexes of runs which returned None
        """
        if self.list_of_unsuccessful_runs:
            if self.nodes is not None:
                self.nodes = np.delete(self.nodes, self.list_of_unsuccessful_runs, axis=1)
                self.dim = self.nodes.shape[0]
                assert self.nodes.shape[1] == self.numEvaluations - len(self.list_of_unsuccessful_runs)
                self.numEvaluations = self.N = self.nodes.shape[1]  # make sure this holds for Saltelli's approach

    def handle_unsuccessful_runs_psp_saltelli(self):
        if self.list_of_unsuccessful_runs is not None and self.list_of_unsuccessful_runs:
            raise Exception(f"[STAT ERROR] - it is not allowed to have any unsuccessful runs is \
            {self.uq_method} mode")

    # =================================================================================================
    # Section of methods for filtering dataframe based on some condition (i.e., value of some metric/GoF/likelihood function)
    # =================================================================================================

    def _update_df_index_parameter_gof_values_based_on_mask(self, mask):
        """
        This function filters based on mask and overwrites self.samples.df_index_parameter_gof_values
        """
        if self.samples.df_index_parameter_gof_values is not None:
            filtered_df = self.samples.df_index_parameter_gof_values[mask]
            if filtered_df.empty:
                raise ValueError("There is no condition ment - After filtering based on provided mask the DF \
                (df_index_parameter_gof_values) will be empty!")
            else:
                self.samples.df_index_parameter_gof_values = filtered_df
                self.samples.df_index_parameter_gof_values.sort_values(
                    by=self.index_column_name, ascending=True, inplace=True, kind='quicksort', na_position='last')
        else:
            raise ValueError("There is no condition ment - samples.df_index_parameter_gof_values is empty!")

    def _get_list_of_index_runs(self, df):
        if df is not None and not df.empty:
            # list_of_index_runs= df.index.tolist()
            list_of_index_runs = df[self.index_column_name].tolist()
            if not list_of_index_runs:
                raise ValueError("There is no condition ment - After filtering based on provided mask the DF \
                (df_index_parameter_gof_values) list_of_index_runs is empty!")
            return list_of_index_runs
        else:
            raise ValueError("There is no condition ment - samples.df_index_parameter_gof_values is empty!")

    def _update_nodes_based_on_index_runs_to_keep(self, list_of_index_runs_to_keep):
        if self.nodes is not None:
            # print(f"[DEBUGGING] self.nodes.shape BEFORE: {self.nodes.shape}")
            # print(f"[DEBUGGING] self.numEvaluations BEFORE: {self.numEvaluations}")
            new_set_of_nodes = np.empty((0, self.dim))
            for single_index_run_to_keep in list_of_index_runs_to_keep:
                row_to_add = np.array(self.nodes.T[single_index_run_to_keep])
                new_set_of_nodes = np.vstack((new_set_of_nodes, row_to_add))
            self.nodes = new_set_of_nodes.T
            self.numEvaluations = self.N = self.nodes.shape[1]  # make sure this holds for Saltelli's approach
            # print(f"[DEBUGGING] self.nodes.shape AFTER: {self.nodes.shape}")
            # print(f"[DEBUGGING] self.numEvaluations AFTER: {self.numEvaluations}")

    def _update_df_simulations_based_on_index_runs_to_keep(self, list_of_index_runs_to_keep):
        # print(f"[DEBUGGING] len(self.samples.df_simulation_result) BEFORE: {len(self.samples.df_simulation_result)}")
        if self.samples.df_simulation_result is not None:
            filtered_df = self.samples.df_simulation_result[\
            self.samples.df_simulation_result[self.index_column_name].isin(list_of_index_runs_to_keep)]
            if filtered_df.empty:
                raise ValueError("There is no condition ment - After filtering based on list_of_index_runs_to_keep the DF \
                (self.samples.df_simulation_result) is empty!")
            else:
                self.samples.df_simulation_result = filtered_df
                self.samples.df_simulation_result.sort_values(
                    by=[self.index_column_name, self.time_column_name], ascending=[True, True], inplace=True, kind='quicksort',
                    na_position='last'
                )
        else:
            raise ValueError("self.samples.df_simulation_result is None!")
        # print(f"[DEBUGGING] len(self.samples.df_simulation_result) AFTER: {len(self.samples.df_simulation_result)}")

    def _update_df_index_parameter_based_on_index_runs_to_keep(self, list_of_index_runs_to_keep):
        # print(f"[DEBUGGING] len(self.samples.df_index_parameter_values) BEFORE: {len(self.samples.df_index_parameter_values)}")
        if  self.samples.df_index_parameter_values is not None:
            filtered_df = self.samples.df_index_parameter_values[\
        self.samples.df_index_parameter_values[self.index_column_name].isin(list_of_index_runs_to_keep)]
            if filtered_df.empty:
                raise ValueError("There is no condition ment - After filtering based on list_of_index_runs_to_keep the DF \
                (self.samples.df_index_parameter_values) is empty!")
            else:
                self.samples.df_index_parameter_values  = filtered_df
        else:
            raise ValueError(" self.samples.df_index_parameter_values is None!")
        # print(f"[DEBUGGING] len(self.samples.df_index_parameter_values) AFTER: {len(self.samples.df_index_parameter_values)}")

    # def _validate_condition(self, df, condition_results_based_on_metric, condition_results_based_on_metric_value):
    #     if df is None or condition_results_based_on_metric is None or condition_results_based_on_metric_value is None:
    #         raise Exception(f"Error in Statistics.handle_condition - it is not possible to condition df-index-parameter-gof-values on the column {condition_results_based_on_metric}")
    #     if isinstance(condition_results_based_on_metric, str):
    #         if condition_results_based_on_metric not in df.columns:
    #             raise Exception(f"Error in Statistics.handle_condition - the column {condition_results_based_on_metric} is not in the df-index-parameter-gof-values")
    #     elif isinstance(condition_results_based_on_metric, list):
    #         for single_condition in condition_results_based_on_metric:
    #             if single_condition not in df.columns:
    #                 raise Exception(f"Error in Statistics.handle_condition - the column {single_condition} is not in the df-index-parameter-gof-values")

    def _apply_condition(self, condition_results_based_on_metric, condition_results_based_on_metric_value, condition_results_based_on_metric_sign):
        """
        Apply a condition to filter the data (model runs) based on a given column (metric/gof/likelihood) and the value.
        The analysis is performed based on self.samples.df_index_parameter_gof_values pd.DataFrame.

        As a side effect this function also updates other dataframe 
        (self.nodes, self.samples.df_simulation_result, self.samples.df_index_parameter_values)

        Args:
            condition_results_based_on_metric (str or list): The name(s) of the column(s) to compare.
            condition_results_based_on_metric_value (float or list): The threshold value(s) to compare against.
            condition_results_based_on_metric_sign (str or list): The comparison operator(s) to use for the comparison(
                e.g., '==', '!=', '<', '>', '<=', '>=', "smaller", "greater", "equal", "not_equal", "smaller_or_equal",  "greater_or_equal").
        Returns:
            None
        """
        try:
            mask = utility.generate_mask_based_on_multiple_column_comparison(
                df=self.samples.df_index_parameter_gof_values, column_name=condition_results_based_on_metric,
                threshold_value=condition_results_based_on_metric_value, comparison=condition_results_based_on_metric_sign
            )
            # mask = utility.generate_mask_based_on_column_comparison(
            #     df=self.samples.df_index_parameter_gof_values, column_name=condition_results_based_on_metric,
            #     threshold_value=condition_results_based_on_metric_value, comparison=condition_results_based_on_metric_sign)
        except Exception as e:
            print(f"Caught an exception: {e}; the execution will continue without any updated of nodes of dataframes \
            storing model runs!")

        # updating self.samples.df_index_parameter_gof_values
        try:
            self._update_df_index_parameter_gof_values_based_on_mask(mask)
        except ValueError as e:
            print(f"Caught an exception: {e}; the execution will continue without any updated of nodes of dataframes \
            storing model runs!")
            return

        try:
            list_of_index_runs_to_keep = self._get_list_of_index_runs(df=self.samples.df_index_parameter_gof_values)
        except ValueError as e:
            print(f"Caught an exception: {e}; the execution will continue without any updated of nodes of dataframes \
            storing model runs!")
            return
        # print(f"[DEBUGGING] list_of_index_runs_to_keep: {list_of_index_runs_to_keep}")

        self._update_nodes_based_on_index_runs_to_keep(list_of_index_runs_to_keep)

        try:
            self._update_df_simulations_based_on_index_runs_to_keep(list_of_index_runs_to_keep)
        except ValueError as e:
            print(f"Caught an exception: {e}")
            raise

        try:
            self._update_df_index_parameter_based_on_index_runs_to_keep(list_of_index_runs_to_keep)
        except ValueError as e:
            print(f"Caught an exception: {e}")

        # Save updated DataFrames
        self.samples.save_index_parameter_values(self.workingDir, filename=utility.DF_INDEX_PARAMETER_CONDITIONED_FILE)
        self.samples.save_index_parameter_gof_values(self.workingDir, filename=utility.DF_INDEX_PARAMETER_GOF_CONDITIONED_FILE)
        if self.save_all_simulations:
            self.samples.save_simulation_results_to_file(self.workingDir, filename=utility.DF_SIMULATIONS_CONDITIONED_FILE)

        self.set_number_of_unique_index_runs()

    def handle_conditioning_model_runs(self, kwargs):
        """
        This function handles the conditioning of model runs based on the provided conditions.

        Parameters:
        kwargs (dict): A dictionary containing the conditions for the model runs. It should have the following keys:
            - condition_results_based_on_metric: column(s) to condition on.
            - condition_results_based_on_metric_value: value(s) to condition on.
            - condition_results_based_on_metric_sign: comparison sign(s) string.

        Raises:
        Exception: If it is not possible to condition df_index_parameter_gof_values on the column.
        """
        condition_results_based_on_metric = kwargs.get("condition_results_based_on_metric", self.condition_results_based_on_metric)
        condition_results_based_on_metric_value = kwargs.get("condition_results_based_on_metric_value", self.condition_results_based_on_metric_value)
        condition_results_based_on_metric_sign = kwargs.get("condition_results_based_on_metric_sign", self.condition_results_based_on_metric_sign)

        if not self.allow_conditioning_results_based_on_metric:
            print(f"[STAT INFO] - The conditioning of model runs based on the metric {condition_results_based_on_metric} is not allowed \
            becuase the variable allow_conditioning_results_based_on_metric is set to False!")
            return

        if condition_results_based_on_metric is not None and condition_results_based_on_metric_value is not None:
            # self._validate_condition(self.samples.df_index_parameter_gof_values, condition_results_based_on_metric, condition_results_based_on_metric_value)
            uqef_dynamic_utils.validate_condition(self.samples.df_index_parameter_gof_values, condition_results_based_on_metric, condition_results_based_on_metric_value)
            self._apply_condition(condition_results_based_on_metric, condition_results_based_on_metric_value, condition_results_based_on_metric_sign)
        else:
            print(f"[STAT INFO] - The conditioning of model runs based on the metric {condition_results_based_on_metric} is not performed \
            becuase the variables condition_results_based_on_metric and/or condition_results_based_on_metric_value are not set (i.e., are None)!")

    # =================================================================================================

    def mc_set_initial_values(self, simulationNodes, numEvaluations, regression=False, order=None,
                              poly_normed=False, poly_rule='three_terms_recurrence', cross_truncation=1.0):
        self.numEvaluations = self.N = numEvaluations
        # TODO Think about this, tricky for saltelli, makes sense for mc
        # self.numEvaluations = self.number_of_unique_index_runs
        if simulationNodes is not None and simulationNodes:
            self.nodes = simulationNodes.distNodes
            self.dim = self.nodes.shape[0]
            # should it be - this does not hold for Saltelli's approach!
            # self.N = self.numEvaluations = self.nodes.shape[1]
            if self.sampleFromStandardDist:
                self.dist = simulationNodes.joinedStandardDists
            else:
                self.dist = simulationNodes.joinedDists
        else:
            self.nodes = None
            self.dist = None
        self.weights = None
        self.regression = regression
        if self.regression:
            self.order = order
            self.poly_normed = poly_normed
            self.poly_rule = poly_rule
            self.cross_truncation = cross_truncation
            self._check_if_parameters_are_set_for_regression_or_psp()
            # These are set-up which have stronger priorty than instantly_save_results_for_each_time_step
            if self.uq_method == "mc" and self.regression and (self.compute_generalized_sobol_indices or self.compute_kl_expansion_of_qoi):
                print(f"Variable time_dependent_statistics.instantly_save_results_for_each_time_step will be set to False")
                self.instantly_save_results_for_each_time_step = False

    def pce_set_initial_values(self, simulationNodes, order, poly_normed=False, poly_rule='three_terms_recurrence', regression=False, cross_truncation=1.0):
        # self.numEvaluations = self.number_of_unique_index_runs
        if simulationNodes is not None and simulationNodes:
            self.nodes = simulationNodes.distNodes
            self.dim = self.nodes.shape[0]
            self.weights = simulationNodes.weights
            self.N = self.numEvaluations = self.nodes.shape[1]  # should be equal to self.number_of_unique_index_runs
            if self.sampleFromStandardDist:
                self.dist = simulationNodes.joinedStandardDists
            else:
                self.dist = simulationNodes.joinedDists
        else:
            self.ensure_nodes_are_loaded()
            # TODO what to do with self.dist in this case? Read it from some file?
        self.order = order
        self.poly_normed = poly_normed
        self.poly_rule = poly_rule
        self.regression = regression
        self.cross_truncation = cross_truncation
        self._check_if_parameters_are_set_for_regression_or_psp()
        # These are set-up which have stronger priorty than instantly_save_results_for_each_time_step
        if self.compute_generalized_sobol_indices or self.compute_kl_expansion_of_qoi:
            print(f"Variable time_dependent_statistics.instantly_save_results_for_each_time_step will be set to False")
            self.instantly_save_results_for_each_time_step = False

    # =================================================================================================

    def prepareForMcStatistics(self, simulationNodes, numEvaluations, regression=False, order=None,
                              poly_normed=False, poly_rule='three_terms_recurrence', cross_truncation=1.0, *args, **kwargs):
        """
        This function should be called before any statistical analysis is performed. It sets the initial values
        for the Monte Carlo simulation.

        Parameters:
        - simulationNodes: UQEF Nodes object.
        - numEvaluations (int): Number of evaluations for the Monte Carlo simulation. Can be infered from simulationNodes as well.
        - regression (bool, optional): Flag indicating whether gPCE+regression method is used. Defaults to False.
        - order (int, optional): Order of the polynomial. Only relevant if regression is True. Defaults to None.
        - poly_normed (bool, optional): Flag indicating whether the polynomial is normalized.  Only relevant if regression is True. Defaults to False.
        - poly_rule (str, optional): Rule for generating the polynomial.  Only relevant if regression is True. Defaults to 'three_terms_recurrence'.
        - cross_truncation (float, optional): Cross truncation value.  Only relevant if regression is True. Defaults to 1.0.
        - *args, **kwargs: Additional arguments and keyword arguments.
            - condition_results_based_on_metric: column(s) to condition on.
            - condition_results_based_on_metric_value: value(s) to condition on.
            - condition_results_based_on_metric_sign: comparison sign(s) string.

        Returns:
        None

        """
        self.mc_set_initial_values(simulationNodes, numEvaluations, regression, order, poly_normed, poly_rule, cross_truncation)
        if self.regression:
            self.handle_expansion_generation()
        if self.compute_sobol_indices_with_samples:
            self.ensure_nodes_are_loaded()
        self.handle_unsuccessful_runs()
        if self.allow_conditioning_results_based_on_metric:
            self.handle_conditioning_model_runs(kwargs)

    def prepareForScStatistics(self, simulationNodes, order, poly_normed=False, poly_rule='three_terms_recurrence', regression=False, cross_truncation=1.0, *args, **kwargs):
        """
        Prepares the statistics for using methods based on gPCE (i.e., Stochastic Collocation or Pseudo-spectra-projection PSP method)

        Args:
            simulationNodes (list): UQEF Nodes object.
            order (int): Order of the polynomial chaos expansion.
            poly_normed (bool, optional): Whether to normalize the polynomial basis. Defaults to False.
            poly_rule (str, optional): The rule for generating the polynomial basis. Defaults to 'three_terms_recurrence'.
            regression (bool, optional): Whether to perform regression analysis (i.e., stochastic collocation). Defaults to False (i.e., then PSP method is performed).
            cross_truncation (float, optional): The cross truncation threshold for regression analysis. Defaults to 1.0.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
                - condition_results_based_on_metric: column(s) to condition on.
                - condition_results_based_on_metric_value: value(s) to condition on.
                - condition_results_based_on_metric_sign: comparison sign(s) string.

        Returns:
            None
        """
        self.pce_set_initial_values(simulationNodes, order, poly_normed, poly_rule, regression, cross_truncation)
        self.handle_expansion_generation()
        if self.regression:
            self.handle_unsuccessful_runs()
            if self.allow_conditioning_results_based_on_metric:
                self.handle_conditioning_model_runs(kwargs)
        else:
            self.handle_unsuccessful_runs_psp_saltelli()

    def prepareForMcSaltelliStatistics(self, simulationNodes, numEvaluations=None, regression=False, order=None,
                                    poly_normed=False, poly_rule='three_terms_recurrence', cross_truncation=1.0, *args, **kwargs):
        """
        Prepares the statistics for performing McSaltelli analysis.

        Args:
            simulationNodes: UQEF Nodes object.
            numEvaluations (int): Number of evaluations for the Monte Carlo simulation. Can be refered from simulationNodes as well.
            regression (bool, optional): Flag indicating whether gPCE+regression method is used. Defaults to False.
            order (int, optional): Order of the polynomial. Only relevant if regression is True. Defaults to None.
            poly_normed (bool, optional): Flag indicating whether the polynomial is normalized.  Only relevant if regression is True. Defaults to False.
            poly_rule (str, optional): Rule for generating the polynomial.  Only relevant if regression is True. Defaults to 'three_terms_recurrence'.
            cross_truncation (float, optional): Cross truncation value.  Only relevant if regression is True. Defaults to 1.0.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
                - condition_results_based_on_metric: column(s) to condition on.
                - condition_results_based_on_metric_value: value(s) to condition on.
                - condition_results_based_on_metric_sign: comparison sign(s) string.
        Returns:
            None
        """
        self.mc_set_initial_values(simulationNodes, numEvaluations, regression, order, poly_normed, poly_rule, cross_truncation)
        # if self.regression:
        #     self.handle_expansion_generation()
        # if self.compute_sobol_indices_with_samples:  # this is actually mc approach;
        #     self.ensure_nodes_are_loaded()
        #     self.handle_unsuccessful_runs()
        #     if self.allow_conditioning_results_based_on_metric:
        #         self.handle_conditioning_model_runs(kwargs)
        # else:
        self.handle_unsuccessful_runs()  # TODO - let's try this...
        # self.handle_unsuccessful_runs_psp_saltelli()
            
    # =================================================================================================

    def _get_measured_qoi_at_previous_timestamp_if_autoregressive_module_first_order(self, single_qoi_column, timestamp):
        df_measured_subset = None
        if self.measured_fetched and self.df_measured is not None:
            if single_qoi_column in list(self.df_measured["qoi"].unique()):
                df_measured_subset = self.df_measured.loc[self.df_measured["qoi"] == single_qoi_column][[
                    self.time_column_name, "measured"]]

            elif self.dict_corresponding_original_qoi_column[single_qoi_column] \
                    in list(self.df_measured["qoi"].unique()):
                df_measured_subset = self.df_measured.loc[
                    self.df_measured["qoi"] == self.dict_corresponding_original_qoi_column[single_qoi_column]][[
                    self.time_column_name, "measured"]]

        if df_measured_subset is not None:
            reset_index = False
            if not df_measured_subset.index.name == self.time_column_name:
                df_measured_subset.set_index(self.time_column_name, inplace=True)
                reset_index = True
            previous_timestamp = utility.compute_previous_timestamp(
                timestamp=timestamp, resolution=self.resolution)
            measured_qoi_at_previous_timestamp = df_measured_subset.loc[previous_timestamp]["measured"] #.values[0]
            if reset_index:
                df_measured_subset.reset_index(inplace=True)
                df_measured_subset.rename(columns={"index": self.time_column_name}, inplace=True)
        else:
            measured_qoi_at_previous_timestamp = None
        return measured_qoi_at_previous_timestamp

    def _if_autoregressive_model_first_order_do_modification(self,
                                single_qoi_column, timestamp, result_dict):
        """
        This function checks if the autoregressive_model_first_order is set to True and if so, modifies the result_dict,
        i.e.m mean computed value of the QoI, by adding to the value of the QoI at the previous time step.
        :param single_qoi_column:
        :param timestamp:
        :param result_dict:
        :return:
        """
        df_measured_subset = None
        if self.scale_factor_autoregressive_model_first_order is None:
            self.scale_factor_autoregressive_model_first_order = 1.0
        if self.measured_fetched and self.df_measured is not None:
            if single_qoi_column in list(self.df_measured["qoi"].unique()):
                df_measured_subset = self.df_measured.loc[self.df_measured["qoi"] == single_qoi_column][[
                    self.time_column_name, "measured"]]

            elif self.dict_corresponding_original_qoi_column[single_qoi_column] \
                    in list(self.df_measured["qoi"].unique()):
                df_measured_subset = self.df_measured.loc[
                    self.df_measured["qoi"] == self.dict_corresponding_original_qoi_column[single_qoi_column]][[
                    self.time_column_name, "measured"]]
            if df_measured_subset.empty:
                df_measured_subset = None
        else:
            df_measured_subset = None

        if df_measured_subset is not None:
            reset_index = False
            if not df_measured_subset.index.name == self.time_column_name:
                df_measured_subset.set_index(self.time_column_name, inplace=True)
                reset_index = True
            # previous_timestamp = self.pdTimesteps[self.pdTimesteps.index(timestamp) - 1]
            # Compute the previous timestamp
            previous_timestamp = utility.compute_previous_timestamp(
                timestamp=timestamp, resolution=self.resolution)
            result_dict["E"] = result_dict["E"] + self.scale_factor_autoregressive_model_first_order*df_measured_subset.loc[previous_timestamp]["measured"] #.values[0]
            # if result_dict["E"]<1e-10:
            #     result_dict["E"] = 0.0
            if reset_index:
                df_measured_subset.reset_index(inplace=True)
                df_measured_subset.rename(columns={"index": self.time_column_name}, inplace=True)
        # else:
        #     result_dict["E"] = result_dict["E"] + self.scale_factor_autoregressive_model_first_order*result_dict_previousr_timestamp["E"]

    def _save_statistics_dictionary_single_qoi_single_timestamp(self, single_qoi_column, timestamp, result_dict):
        fileName = f"statistics_dictionary_{single_qoi_column}_{timestamp}.pkl"
        fullFileName = os.path.abspath(os.path.join(str(self.workingDir), fileName))
        with open(fullFileName, 'wb') as handle:
            pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # TODO Remove this function
    # def _process_result_single_qoi_single_time_step(self, single_qoi_column, timestamp):
    #     self.result_dict[single_qoi_column][timestamp].update({'qoi': single_qoi_column})
    #     if self.autoregressive_model_first_order:
    #         self._if_autoregressive_model_first_order_do_modification(
    #             single_qoi_column, timestamp, self.result_dict[single_qoi_column][timestamp])
    #     if self.instantly_save_results_for_each_time_step:
    #         self._save_statistics_dictionary_single_qoi_single_timestamp(single_qoi_column, timestamp, self.result_dict[single_qoi_column][timestamp])
    #     if self.save_gpce_surrogate and "gPCE" in self.result_dict[single_qoi_column][timestamp]:
    #         utility.save_gpce_surrogate_model(
    #             workingDir=self.workingDir, gpce=self.result_dict[single_qoi_column][timestamp]["gPCE"], qoi=single_qoi_column, timestamp=timestamp)
    #     if self.save_gpce_surrogate and "gpce_coeff" in self.result_dict[single_qoi_column][timestamp]:
    #         utility.save_gpce_coeffs(
    #             workingDir=self.workingDir, coeff=self.result_dict[single_qoi_column][timestamp]["gpce_coeff"], qoi=single_qoi_column, timestamp=timestamp)

    def _process_chunk_result_single_qoi_single_time_step(self, single_qoi_column, timestamp, result_dict):
        """
        Process the result for a single quantity of interest (QoI) at a single time step.
        Ment to be run when parallel processing / statistics is used.
        Args:
            single_qoi_column (str): The name of the quantity of interest.
            timestamp (float /  pd.Timestamp): The timestamp of the result.
            result_dict (dict): The dictionary containing the result for the QoI at the given timestamp.

        Returns:
            None
        """
        result_dict.update({'qoi': single_qoi_column})
        if self.autoregressive_model_first_order:
            self._if_autoregressive_model_first_order_do_modification(
                single_qoi_column, timestamp, result_dict)
        if self.instantly_save_results_for_each_time_step:
            # TODO maybe comment out all this part, in case this is performed in _my_parallel_calc_stats_for...
            self._save_statistics_dictionary_single_qoi_single_timestamp(single_qoi_column, timestamp, result_dict)
        # else:
        #     self.result_dict[single_qoi_column][timestamp] = result_dict
        # TODO - think if this should be done here or in the parallel_calc_stats_for_gPCE...
        if self.save_gpce_surrogate and "gPCE" in result_dict:
            utility.save_gpce_surrogate_model(workingDir=self.workingDir, gpce=result_dict["gPCE"], qoi=single_qoi_column, timestamp=timestamp)
        if self.save_gpce_surrogate and "gpce_coeff" in result_dict:
            utility.save_gpce_coeffs(workingDir=self.workingDir, coeff=result_dict["gpce_coeff"], qoi=single_qoi_column, timestamp=timestamp)

    def save_print_plot_and_clear_result_dict_single_qoi(self, single_qoi_column):
        if self.instantly_save_results_for_each_time_step:
            # In this case self.result_dict has already been emptied
            return

        # In this case the results where collected in the self.result_dict dict and can be saved and plotted
        # Saving Stat Dict for a single qoi as soon as it is computed/ all time steps are processed
        fileName = f"statistics_dictionary_qoi_{single_qoi_column}.pkl" #+ single_qoi_column + ""
        fullFileName = os.path.abspath(os.path.join(str(self.workingDir), fileName))
        with open(fullFileName, 'wb') as handle:
            pickle.dump(self.result_dict[single_qoi_column], handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.plotResults_single_qoi(
            single_qoi_column=single_qoi_column,
            dict_time_vs_qoi_stat = self.result_dict[single_qoi_column],
            dict_what_to_plot=self.dict_what_to_plot,
            directory=self.workingDir
        )

        self.printResults_single_qoi(
            single_qoi_column=single_qoi_column, 
            dict_time_vs_qoi_stat=self.result_dict[single_qoi_column]
            )

        # Freeing up the memory
        # del self.result_dict[single_qoi_column]
        if self.free_result_dict_memory:
            self.result_dict[single_qoi_column].clear()

    def _groupby_df_simulation_results(self, columns_to_group_by: list=[]):
        if not columns_to_group_by:
            columns_to_group_by = [self.time_column_name,]
        grouped = self.samples.df_simulation_result.groupby(columns_to_group_by)
        self.groups = grouped.groups

    def _postprocess_kl_expansion_or_generalized_sobol_indices_computation_from_results_single_qoi(
        self, single_qoi_column):
        """
        Postprocesses the KL expansion or generalized Sobol indices (computer for the final timestamp or time-vise) 
        computation results for a single quantity of interest (QoI).

        Args:
            single_qoi_column (str): The name of the single quantity of interest (QoI) column.

        Returns:
            None
        """
        if self.compute_kl_expansion_of_qoi and not self.instantly_save_results_for_each_time_step:
            # Setting-up some variables
            variance_integral = None
            total_variance = None
            total_variance_based_on_pce_coefficients = None

            # TODO  - think how to allow over time computation of KL surrogate and generalized Sobol indices
            # Var_kl_approx is sum of the eigenvalues
            eigenvalues, eigenvectors, f_kl_surrogate_dict, f_kl_surrogate_coefficients, Var_kl_approx \
            = self.compute_kl_expansion_single_qoi(single_qoi_column)
            # Generalized Sobol Indices (for now just for the final time-stamp)
            last_time_step = max(self.result_dict[single_qoi_column].keys())  #last_time_step = list(self.result_dict[single_qoi_column].keys())[-1]

            if self.compute_generalized_sobol_indices:
                fileName = self.workingDir / f"generalized_sobol_indices_{single_qoi_column}.pkl"
                param_name_generalized_sobol_total_indices, total_variance, total_variance_based_on_pce_coefficients = utility.computing_generalized_sobol_total_indices_from_kl_expan(
                    f_kl_surrogate_coefficients=f_kl_surrogate_coefficients, 
                    polynomial_expansion=self.polynomial_expansion, 
                    weights=self.weights_time, 
                    param_names=self.labels, 
                    fileName=fileName, 
                    total_variance=Var_kl_approx,
                    compute_total_variance_based_on_pce_coefficients=True,
                    )
                print(f"INFO: computation of generalized S.S.I based on KL+gPCE(MC) finished...")
                for param_name in self.labels:
                    self.result_dict[single_qoi_column][last_time_step][f"generalized_sobol_total_index_{param_name}"] = \
                        param_name_generalized_sobol_total_indices[param_name]

            # Comparing different Variances
            # Comparing Var_kl_approx and time integral of self.result_dict[single_qoi_column][over_time_stamps]["Var"]
            if "Var" in self.result_dict[single_qoi_column][last_time_step]:
                variance_over_time_array = np.asarray([self.result_dict[single_qoi_column][time]["Var"] for time in self.result_dict[single_qoi_column].keys()], dtype=np.float64)
                # TODO - Play with the weights_time
                variance_integral = np.dot(variance_over_time_array, self.weights_time)
            
            print(f"INFO: Total Variance computed via eigenvalues: {Var_kl_approx} \
                Total Variance integral over time: {variance_integral}; \
                    Total Variance computed via PCE coefficients: {total_variance_based_on_pce_coefficients};\
                         Total Variance returned by the computing_generalized_sobol_total_indices_from_kl_expan: {total_variance}")
            fileName = self.workingDir / f"total_variance_{single_qoi_column}.txt"
            with open(fileName, 'w') as fp:
                fp.write(f'Total Variance computed via eigenvalues: {Var_kl_approx}\n')
                fp.write(f'Total Variance integral over time: {variance_integral}\n')
                fp.write(f'Total Variance computed via PCE coefficients: {total_variance_based_on_pce_coefficients}')

        elif self.compute_generalized_sobol_indices and not self.instantly_save_results_for_each_time_step: 
            fileName = self.workingDir / f"generalized_sobol_indices_{single_qoi_column}.pkl"
            if self.compute_generalized_sobol_indices_over_time:
                utility.computing_generalized_sobol_total_indices_from_poly_expan_over_time(
                    self.result_dict[single_qoi_column], 
                    self.polynomial_expansion, self.weights_time, self.labels,
                    fileName)
                print(f"INFO: computation of (over time) generalized S.S.I based on PCE finished...")
            else:
                # the computation of the generalized Sobol indices is done only for the last time step
                utility.computing_generalized_sobol_total_indices_from_poly_expan(
                    result_dict_statistics=self.result_dict[single_qoi_column], 
                    polynomial_expansion=self.polynomial_expansion, 
                    weights=self.weights_time, 
                    param_names=self.labels,
                    fileName=fileName)
                print(f"INFO: computation of (over time) generalized S.S.I based on PCE finished...")
        else:
            print(f"INFO: computation of KL expansion and/or generalized Sobol indices for {single_qoi_column} is not performed; \
            maybe the problem is that instantly_save_results_for_each_time_step variable is set to True, or the user does not want that...")

    # =================================================================================================

    def _initialize_chunks(self, keyIter, single_qoi_column, chunksize):
        """
        # TODO Memory problem - I should not store all the simulations in the memory!
        # TODO Should I as well transfer Index_run column
        #  in order to be sure that right values were multiplied with right polynomials?
        """
        keyIter_chunk = list(more_itertools.chunked(keyIter, chunksize))
        list_of_qoi_values = (
            self.samples.df_simulation_result.loc[self.groups[key].values][single_qoi_column].values
            for key in keyIter
        )
        list_of_qoi_values_chunk = list(more_itertools.chunked(list_of_qoi_values, chunksize))
        # list_of_qoi_values = []
        # for key in keyIter:
        #     timestamp = self.groups[key].values
        #     measured_qoi_at_previous_timestamp = self._get_measured_qoi_at_previous_timestamp_if_autoregressive_module_first_order(
        #         single_qoi_column, timestamp=timestamp)
        #     if measured_qoi_at_previous_timestamp is not None:
        #         self.samples.df_simulation_result.loc[timestamp][single_qoi_column] = \
        #             self.samples.df_simulation_result.loc[timestamp][single_qoi_column] + measured_qoi_at_previous_timestamp
        #         if self.samples.df_simulation_result.loc[timestamp][single_qoi_column] < 1e-10:
        #             self.samples.df_simulation_result.loc[timestamp][single_qoi_column] = 0.0
        #     list_of_qoi_values.append(
        #         self.samples.df_simulation_result.loc[timestamp][single_qoi_column].values
        #     )
        # generator_of_simulations_df = (
        #     self.samples.df_simulation_result.loc[self.groups[key].values][single_qoi_column].values
        #     for key in keyIter
        # )
        return keyIter_chunk, list_of_qoi_values_chunk

    # =================================================================================================

    def calcStatisticsForMcParallel(self, chunksize=1, *args, **kwargs):
        self.result_dict = defaultdict(dict)

        if self.rank == 0:
            self._groupby_df_simulation_results(columns_to_group_by=[self.time_column_name,])
            keyIter = list(self.groups.keys())

        compute_other_stat_besides_pce_surrogate = kwargs.get("compute_other_stat_besides_pce_surrogate", self.compute_other_stat_besides_pce_surrogate)

        for single_qoi_column in self.list_qoi_column:
            if self.rank == 0:
                keyIter_chunk, list_of_qoi_values_chunk = self._initialize_chunks(keyIter, single_qoi_column, chunksize)
                chunks = self._prepare_chunks_mc(keyIter_chunk, compute_other_stat_besides_pce_surrogate)

            with futures.MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
                if executor is not None:  # master proces; or .\executor.mpi_comm.rank == 0
                    self._compute_mc_statistic_single_qoi_parallel_in_time(
                        executor, single_qoi_column, keyIter_chunk, list_of_qoi_values_chunk, chunks)
                    self._postprocess_mc_chunk_results_single_qoi_after_parallel_in_time_analysis(single_qoi_column)

    def _prepare_chunks_mc(self, keyIter_chunk, compute_other_stat_besides_pce_surrogate=False):
        # TODO Probably all processes already have these data -
        #  I would not have to propagate those if parallel_calc_stats_for_gPCE would be a class method
    
        if self.compute_sobol_indices_with_samples and \
                self.compute_Sobol_m and self.nodes is not None:
            samples_chunks = [self.nodes.T[:self.numEvaluations]] * len(keyIter_chunk)
            # samples = self.uqef_simulationNodes.parameters.T[:self.numEvaluations]
        else:
            samples_chunks = [None] * len(keyIter_chunk)
            # samples = None
        
        chunks = {
            'compute_Sobol_m_Chunks': [self.compute_Sobol_m] * len(keyIter_chunk),
            'store_qoi_data_in_stat_dict_Chunks': [self.store_qoi_data_in_stat_dict] * len(keyIter_chunk),
            'dict_stat_to_compute_Chunks': [self.dict_stat_to_compute] * len(keyIter_chunk),
        }
        if not self.regression or (self.compute_kl_expansion_of_qoi and not self.compute_timewise_gpce_next_to_kl_expansion):
            chunks.update({
                'numEvaluations_chunk':  [self.numEvaluations] * len(keyIter_chunk),
                'dimChunks': [self.dim] * len(keyIter_chunk),
                'compute_sobol_indices_with_samples_chunks': [self.compute_sobol_indices_with_samples] * len(keyIter_chunk),
                'samples_chunks': samples_chunks,
            })
        else:
            chunks.update({
                'distChunks': [self.dist] * len(keyIter_chunk),
                'polynomial_expansionChunks': [self.polynomial_expansion] * len(keyIter_chunk),
                'nodesChunks': [self.nodes] * len(keyIter_chunk),
                'weightsChunks': [self.weights] * len(keyIter_chunk),
                'polynomial_norms_expansionChunks': [self.polynomial_norms] * len(keyIter_chunk),
                'regressionChunks': [self.regression] * len(keyIter_chunk),
                'compute_Sobol_t_Chunks': [self.compute_Sobol_t] * len(keyIter_chunk),
                'compute_Sobol_m2_Chunks': [self.compute_Sobol_m2] * len(keyIter_chunk),
                'store_gpce_surrogate_in_stat_dict_Chunks': [self.store_gpce_surrogate_in_stat_dict] * len(keyIter_chunk),
                'save_gpce_surrogate_Chunks': [self.save_gpce_surrogate] * len(keyIter_chunk),
                'compute_other_stat_besides_pce_surrogate_Chunks': [compute_other_stat_besides_pce_surrogate] * len(keyIter_chunk),
            })

        return chunks

    def _compute_mc_statistic_single_qoi_parallel_in_time(
        self, executor, single_qoi_column, keyIter_chunk, list_of_qoi_values_chunk, chunks):
        print(f"{self.rank}: computation of statistics for qoi {single_qoi_column} started...")
        solver_time_start = time.time()
        if not self.regression or (self.compute_kl_expansion_of_qoi and not self.compute_timewise_gpce_next_to_kl_expansion):
            chunk_results_it = executor.map(
                parallel_statistics.parallel_calc_stats_for_MC,
                keyIter_chunk,
                list_of_qoi_values_chunk, #generator_of_simulations_df
                chunks['numEvaluations_chunk'],
                chunks['dimChunks'],
                chunks['compute_Sobol_m_Chunks'],
                chunks['store_qoi_data_in_stat_dict_Chunks'],
                chunks['compute_sobol_indices_with_samples_chunks'],
                chunks['samples_chunks'],
                chunks['dict_stat_to_compute_Chunks'],
                chunksize=self.mpi_chunksize,
                unordered=self.unordered
            )
            # chunk_results_it = executor.map(
            #     parallel_statistics.parallel_calc_stats_for_MC,
            #     keyIter,
            #     generator_of_simulations_df,
            #     self.numEvaluations,
            #     self.dim,
            #     self.compute_Sobol_m,
            #     self.store_qoi_data_in_stat_dict,
            #     self.compute_sobol_indices_with_samples,
            #     samples,
            #      self.dict_stat_to_compute,
            #     chunksize=self.mpi_chunksize,
            #     unordered=self.unordered
            # )
        else:
            chunk_results_it = executor.map(
                parallel_statistics.parallel_calc_stats_for_gPCE,
                keyIter_chunk,
                list_of_qoi_values_chunk,
                chunks['distChunks'],
                chunks['polynomial_expansionChunks'],
                chunks['nodesChunks'],
                chunks['weightsChunks'],
                chunks['polynomial_norms_expansionChunks'],
                chunks['regressionChunks'],
                chunks['compute_Sobol_t_Chunks'],
                chunks['compute_Sobol_m_Chunks'],
                chunks['compute_Sobol_m2_Chunks'],
                chunks['store_qoi_data_in_stat_dict_Chunks'],
                chunks['store_gpce_surrogate_in_stat_dict_Chunks'],
                chunks['save_gpce_surrogate_Chunks'],
                chunks['compute_other_stat_besides_pce_surrogate_Chunks'],
                chunks['dict_stat_to_compute_Chunks'],
                chunksize=self.mpi_chunksize,
                unordered=self.unordered
            )

        print(f"{self.rank}: computation for qoi {single_qoi_column} - waits for shutdown...")
        sys.stdout.flush()
        executor.shutdown(wait=True)
        print(f"{self.rank}: computation for qoi {single_qoi_column} - shut down...")
        sys.stdout.flush()

        solver_time_end = time.time()
        solver_time = solver_time_end - solver_time_start
        print(f"solver_time for qoi {single_qoi_column}: {solver_time}")

        self.chunk_results = list(chunk_results_it)

    def _postprocess_mc_chunk_results_single_qoi_after_parallel_in_time_analysis(self, single_qoi_column):
        for chunk_result in self.chunk_results:
            for result in chunk_result:
                self._process_chunk_result_single_qoi_single_time_step(
                    single_qoi_column, timestamp=result[0], result_dict=result[1])
                if self.instantly_save_results_for_each_time_step:
                    del result[1]
                else:
                    self.result_dict[single_qoi_column][result[0]] = result[1]
        self._postprocess_mc_results_single_qoi(single_qoi_column)
        self.save_print_plot_and_clear_result_dict_single_qoi(single_qoi_column)
        del self.chunk_results
    
    def _postprocess_mc_results_single_qoi(self, single_qoi_column):
        if self.regression:
            self._postprocess_kl_expansion_or_generalized_sobol_indices_computation_from_results_single_qoi(single_qoi_column)
        if self.compute_covariance_matrix_in_time and not self.compute_kl_expansion_of_qoi and not self.instantly_save_results_for_each_time_step:
            covariance_matrix_loc = self.compute_covariance_matrix_in_time_single_qoi(single_qoi_column)
            utility.save_covariance_matrix(covariance_matrix_loc, self.workingDir, single_qoi_column)
            utility.plot_covariance_matrix(covariance_matrix_loc, self.workingDir, filname=f"covariance_matrix_{single_qoi_column}.png")
            print(f"covariance_matrix for {single_qoi_column} is computed: {covariance_matrix_loc}")

    # =================================================================================================
    def calcStatisticsForEnsembleParallel(self, chunksize=1, *args, **kwargs):
        self.calcStatisticsForMcParallel(chunksize=chunksize, *args, **kwargs)

    # =================================================================================================

    def calcStatisticsForMcSaltelliParallel(self, chunksize=1, *args, **kwargs):
        self.result_dict = defaultdict(dict)

        if self.rank == 0:
            self._groupby_df_simulation_results(columns_to_group_by=[self.time_column_name,])
            keyIter = list(self.groups.keys())

        for single_qoi_column in self.list_qoi_column:
            if self.rank == 0:
                keyIter_chunk, list_of_qoi_values_chunk = self._initialize_chunks(keyIter, single_qoi_column, chunksize)
                chunks = self._prepare_chunks_saltelli(keyIter_chunk)

            with futures.MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
                if executor is not None:  # master process
                    self._compute_saltelli_statistic_single_qoi_parallel_in_time(\
                    executor, single_qoi_column, keyIter_chunk, list_of_qoi_values_chunk, chunks
                    )
                    self._postprocess_saltelli_chunk_results_single_qoi_after_parallel_in_time_analysis(single_qoi_column)

    def _prepare_chunks_saltelli(self, keyIter_chunk):
        if self.compute_sobol_indices_with_samples and \
                        self.compute_Sobol_m and self.nodes is not None:
            samples_chunks = [self.nodes.T[:self.numEvaluations]] * len(keyIter_chunk)
        else:
            samples_chunks = [None] * len(keyIter_chunk)
        chunks = {
            'numEvaluations_chunk': [self.numEvaluations] * len(keyIter_chunk),
            'dimChunks': [self.dim] * len(keyIter_chunk),
            'compute_Sobol_t_Chunks': [self.compute_Sobol_t] * len(keyIter_chunk),
            'compute_Sobol_m_Chunks': [self.compute_Sobol_m] * len(keyIter_chunk),
            'store_qoi_data_in_stat_dict_Chunks': [self.store_qoi_data_in_stat_dict] * len(keyIter_chunk),
            'compute_sobol_indices_with_samples_chunks': [self.compute_sobol_indices_with_samples] * len(keyIter_chunk),
            'samples_chunks': samples_chunks,
            'dict_stat_to_compute_Chunks': [self.dict_stat_to_compute] * len(keyIter_chunk),
        }
        return chunks

    def _compute_saltelli_statistic_single_qoi_parallel_in_time(self, executor, single_qoi_column, keyIter_chunk, list_of_qoi_values_chunk, chunks):
        print(f"{self.rank}: computation of statistics for qoi {single_qoi_column} started...")
        solver_time_start = time.time()
        chunk_results_it = executor.map(
            parallel_statistics.parallel_calc_stats_for_mc_saltelli,
            keyIter_chunk,
            list_of_qoi_values_chunk,
            chunks['numEvaluations_chunk'],
            chunks['dimChunks'],
            chunks['compute_Sobol_t_Chunks'],
            chunks['compute_Sobol_m_Chunks'],
            chunks['store_qoi_data_in_stat_dict_Chunks'],
            chunks['compute_sobol_indices_with_samples_chunks'],
            chunks['samples_chunks'],
            chunks['dict_stat_to_compute_Chunks'],
            chunksize=self.mpi_chunksize,
            unordered=self.unordered
        )
        print(f"{self.rank}: computation for qoi {single_qoi_column} - waits for shutdown...")
        sys.stdout.flush()
        executor.shutdown(wait=True)
        print(f"{self.rank}: computation for qoi {single_qoi_column} - shut down...")
        sys.stdout.flush()

        solver_time_end = time.time()
        solver_time = solver_time_end - solver_time_start
        print(f"solver_time for qoi {single_qoi_column}: {solver_time}")

        self.chunk_results = list(chunk_results_it)
        
    def _postprocess_saltelli_chunk_results_single_qoi_after_parallel_in_time_analysis(self, single_qoi_column):
        for chunk_result in self.chunk_results:
            for result in chunk_result:
                self._process_chunk_result_single_qoi_single_time_step(
                    single_qoi_column, timestamp=result[0], result_dict=result[1])
                if self.instantly_save_results_for_each_time_step:
                    del result[1]
                else:
                    self.result_dict[single_qoi_column][result[0]] = result[1]
        self._postprocess_saltelli_results_single_qoi(single_qoi_column)
        self.save_print_plot_and_clear_result_dict_single_qoi(single_qoi_column)
        del self.chunk_results

    def _postprocess_saltelli_results_single_qoi(self, single_qoi_column):
        pass
# =================================================================================================

    def calcStatisticsForScParallel(self, chunksize=1, *args, **kwargs):
        self.result_dict = defaultdict(dict)

        if self.rank == 0:
            self._groupby_df_simulation_results(columns_to_group_by=[self.time_column_name,])
            keyIter = list(self.groups.keys())
        
        compute_other_stat_besides_pce_surrogate = kwargs.get("compute_other_stat_besides_pce_surrogate", self.compute_other_stat_besides_pce_surrogate)

        for single_qoi_column in self.list_qoi_column:
            if self.rank == 0:                            
                keyIter_chunk, list_of_qoi_values_chunk = self._initialize_chunks(keyIter, single_qoi_column, chunksize)
                chunks = self._prepare_chunks_pce(keyIter_chunk, compute_other_stat_besides_pce_surrogate)

            with futures.MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
                if executor is not None:  # master process
                    self._compute_pce_statistic_single_qoi_parallel_in_time(executor, single_qoi_column, keyIter_chunk, list_of_qoi_values_chunk, chunks)
                    self._postprocess_pce_chunk_results_single_qoi_after_parallel_in_time_analysis(single_qoi_column)

    def _prepare_chunks_pce(self, keyIter_chunk, compute_other_stat_besides_pce_surrogate):
        chunks = {
            'nodesChunks': [self.nodes] * len(keyIter_chunk),
            'distChunks': [self.dist] * len(keyIter_chunk),
            'weightsChunks': [self.weights] * len(keyIter_chunk),
            'polynomial_expansionChunks': [self.polynomial_expansion] * len(keyIter_chunk),
            'polynomial_norms_expansionChunks': [self.polynomial_norms] * len(keyIter_chunk),
            'regressionChunks': [self.regression] * len(keyIter_chunk),
            'compute_Sobol_t_Chunks': [self.compute_Sobol_t] * len(keyIter_chunk),
            'compute_Sobol_m_Chunks': [self.compute_Sobol_m] * len(keyIter_chunk),
            'compute_Sobol_m2_Chunks': [self.compute_Sobol_m2] * len(keyIter_chunk),
            'store_qoi_data_in_stat_dict_Chunks': [self.store_qoi_data_in_stat_dict] * len(keyIter_chunk),
            'store_gpce_surrogate_in_stat_dict_Chunks': [self.store_gpce_surrogate_in_stat_dict] * len(keyIter_chunk),
            'save_gpce_surrogate_Chunks': [self.save_gpce_surrogate] * len(keyIter_chunk),
            'compute_other_stat_besides_pce_surrogate_Chunks': [compute_other_stat_besides_pce_surrogate] * len(keyIter_chunk),
            'dict_stat_to_compute_Chunks': [self.dict_stat_to_compute] * len(keyIter_chunk)
        }
        return chunks

    def _compute_pce_statistic_single_qoi_parallel_in_time(
        self, executor, single_qoi_column, keyIter_chunk, list_of_qoi_values_chunk, chunks):
        print(f"{self.rank}: computation of statistics for qoi {single_qoi_column} started...")
        solver_time_start = time.time()
        if self.compute_kl_expansion_of_qoi and not self.compute_timewise_gpce_next_to_kl_expansion:
            chunk_results_it = executor.map(
                parallel_statistics.parallel_calc_stats_for_KL,
                keyIter_chunk,
                list_of_qoi_values_chunk,
                chunks['weightsChunks'],
                chunks['regressionChunks'],
                chunks['store_qoi_data_in_stat_dict_Chunks'],
                chunks['dict_stat_to_compute_Chunks'],
            )
        else:
            chunk_results_it = executor.map(
                parallel_statistics.parallel_calc_stats_for_gPCE,
                keyIter_chunk,
                list_of_qoi_values_chunk,
                chunks['distChunks'],
                chunks['polynomial_expansionChunks'],
                chunks['nodesChunks'],
                chunks['weightsChunks'],
                chunks['polynomial_norms_expansionChunks'],
                chunks['regressionChunks'],
                chunks['compute_Sobol_t_Chunks'],
                chunks['compute_Sobol_m_Chunks'],
                chunks['compute_Sobol_m2_Chunks'],
                chunks['store_qoi_data_in_stat_dict_Chunks'],
                chunks['store_gpce_surrogate_in_stat_dict_Chunks'],
                chunks['save_gpce_surrogate_Chunks'],
                chunks['compute_other_stat_besides_pce_surrogate_Chunks'],
                chunks['dict_stat_to_compute_Chunks'],
                chunksize=self.mpi_chunksize,
                unordered=self.unordered
            )
        print(f"{self.rank}: computation for qoi {single_qoi_column} - waits for shutdown...")
        sys.stdout.flush()
        executor.shutdown(wait=True)
        print(f"{self.rank}: computation for qoi {single_qoi_column} - shut down...")
        sys.stdout.flush()

        solver_time_end = time.time()
        solver_time = solver_time_end - solver_time_start
        print(f"solver_time for qoi {single_qoi_column}: {solver_time}")

        self.chunk_results = list(chunk_results_it)
    
    def _postprocess_pce_chunk_results_single_qoi_after_parallel_in_time_analysis(self, single_qoi_column):
        for chunk_result in self.chunk_results:
            for result in chunk_result:
                self._process_chunk_result_single_qoi_single_time_step(
                    single_qoi_column, timestamp=result[0], result_dict=result[1])
                if self.instantly_save_results_for_each_time_step:
                    del result[1]
                else:
                    self.result_dict[single_qoi_column][result[0]] = result[1]
        self._postprocess_pce_results_single_qoi(single_qoi_column)
        self.save_print_plot_and_clear_result_dict_single_qoi(single_qoi_column)
        del self.chunk_results

    def _postprocess_pce_results_single_qoi(self, single_qoi_column):
        self._postprocess_kl_expansion_or_generalized_sobol_indices_computation_from_results_single_qoi(single_qoi_column)
        if self.compute_covariance_matrix_in_time and not self.compute_kl_expansion_of_qoi and not self.instantly_save_results_for_each_time_step:
            covariance_matrix_loc = self.compute_covariance_matrix_in_time_single_qoi(single_qoi_column)
            utility.save_covariance_matrix(covariance_matrix_loc, self.workingDir, single_qoi_column)
            utility.plot_covariance_matrix(covariance_matrix_loc, self.workingDir, filname=f"covariance_matrix_{single_qoi_column}.png")
            print(f"covariance_matrix for {single_qoi_column} is computed: {covariance_matrix_loc}")

    # =================================================================================================
    def calcStatisticsForMc(self, rawSamples=None, timesteps=None,
                            simulationNodes=None, numEvaluations=None, order=None, regression=None, solverTimes=None,
                            work_package_indexes=None, original_runtime_estimator=None, 
                            poly_normed=None, poly_rule=None, cross_truncation=1.0,
                            *args, **kwargs):
        """
        This function groups results by time column and then iterates over all the qois of interest columns
        and updates the self.result_dict by adding the following entries:
        for each [single_qoi_column][key/single time step]
              qoi_values[optional], gPCE[optional]
              gpce_coeff[optional]
              E, Var, StdDev, Skew, Kurt, P10, P90, Sobol_t[optional]
        """
        self.result_dict = dict()
        self._groupby_df_simulation_results(columns_to_group_by=[self.time_column_name,])
        # keyIter = list(self.groups.keys())

        compute_other_stat_besides_pce_surrogate = kwargs.get("compute_other_stat_besides_pce_surrogate", self.compute_other_stat_besides_pce_surrogate)

        for single_qoi_column in self.list_qoi_column:
            self.result_dict[single_qoi_column] = defaultdict(dict)

            print(f"computation of statistics for qoi {single_qoi_column} started...")
            solver_time_start = time.time()
            print(f"computation for qoi {single_qoi_column} - waits for shutdown...")
            sys.stdout.flush()
            print(f"computation for qoi {single_qoi_column} - shut down...")
            sys.stdout.flush()

            for key, val_indices in self.groups.items():
                qoi_values = self.samples.df_simulation_result.loc[val_indices.values][single_qoi_column].values
                self.result_dict[single_qoi_column][key] = dict()

                if self.store_qoi_data_in_stat_dict:
                    self.result_dict[single_qoi_column][key]["qoi_values"] = qoi_values
                
                if not self.regression or (self.compute_kl_expansion_of_qoi and not self.compute_timewise_gpce_next_to_kl_expansion):
                    self.numEvaluations = len(qoi_values)
                    # local_result_dict["E"] = np.sum(qoi_values, axis=0, dtype=np.float64) / self.numEvaluations
                    self.result_dict[single_qoi_column][key]["E"] = np.mean(qoi_values, 0)

                    if self.dict_stat_to_compute.get("Var", False):
                        self.result_dict[single_qoi_column][key]["Var"] = np.var(qoi_values, ddof=1)
                        # self.result_dict[single_qoi_column][key]["Var"] = np.sum((qoi_values - self.result_dict[single_qoi_column][key]["E"]) ** 2, axis=0,
                        #                                   dtype=np.float64) / (self.numEvaluations - 1)
                    if self.dict_stat_to_compute.get("StdDev", False):
                        # local_result_dict["StdDev"] = np.sqrt(local_result_dict["Var"], dtype=np.float64)
                        self.result_dict[single_qoi_column][key]["StdDev"] = np.std(qoi_values, 0, ddof=1)
                    if self.dict_stat_to_compute.get("Skew", False):
                        self.result_dict[single_qoi_column][key]["Skew"] = scipy.stats.skew(qoi_values, axis=0, bias=True)
                    if self.dict_stat_to_compute.get("Kurt", False):
                        self.result_dict[single_qoi_column][key]["Kurt"] = scipy.stats.kurtosis(qoi_values, axis=0, bias=True)
                    
                    if self.dict_stat_to_compute.get("P10", False):
                        self.result_dict[single_qoi_column][key]["P10"] = np.percentile(qoi_values, 10, axis=0)
                        if isinstance(self.result_dict[single_qoi_column][key]["P10"], list) and len(self.result_dict[single_qoi_column][key]["P10"]) == 1:
                            self.result_dict[single_qoi_column][key]["P10"] = self.result_dict[single_qoi_column][key]["P10"][0]
                    if self.dict_stat_to_compute.get("P90", False):
                        self.result_dict[single_qoi_column][key]["P90"] = np.percentile(qoi_values, 90, axis=0)
                        if isinstance(self.result_dict[single_qoi_column][key]["P90"], list) and len(self.result_dict[single_qoi_column][key]["P90"]) == 1:
                            self.result_dict[single_qoi_column][key]["P90"] = self.result_dict[single_qoi_column][key]["P90"][0]

                    if self.compute_Sobol_m and self.compute_sobol_indices_with_samples \
                            and self.nodes is not None:
                            self.result_dict[single_qoi_column][key]["Sobol_m"] = \
                                sens_indices_sampling_based_utils.compute_sens_indices_based_on_samples_rank_based(
                                    samples=self.nodes.T[:self.numEvaluations],
                                    Y=qoi_values[:self.numEvaluations, np.newaxis], D=self.dim, N=self.numEvaluations)
                else:
                    qoi_gPCE, qoi_coeff = cp.fit_regression(
                        polynomials=self.polynomial_expansion, abscissas=self.nodes, evals=qoi_values, retall=True, 
                        model=None  # classical least-square; one can use as well sklearn.linear_model.LinearRegression(fit_intercept=False)
                    )
                    self.result_dict[single_qoi_column][key]['gpce_coeff'] = qoi_coeff
                    self._calc_stats_for_gPCE_single_qoi(
                        single_qoi_column, key, self.dist, qoi_gPCE, compute_other_stat_besides_pce_surrogate)
                
                self._process_chunk_result_single_qoi_single_time_step(
                    single_qoi_column=single_qoi_column, timestamp=key, result_dict=self.result_dict[single_qoi_column][key])

            self._postprocess_mc_results_single_qoi(single_qoi_column)
            self.save_print_plot_and_clear_result_dict_single_qoi(single_qoi_column)

            solver_time_end = time.time()
            solver_time = solver_time_end - solver_time_start
            print(f"solver_time for qoi {single_qoi_column}: {solver_time}")

    # =================================================================================================

    def calcStatisticsForMcSaltelli(self, rawSamples=None, timesteps=None,
                                    simulationNodes=None, numEvaluations=None, order=None, regression=None, solverTimes=None,
                                    work_package_indexes=None, original_runtime_estimator=None, 
                                    poly_normed=None, poly_rule=None, cross_truncation=1.0,
                                    *args, **kwargs):
        """
        This function groups resutl by time column and then iterates over all the qois of interest columns
        and updates the self.result_dict by adding the following entries:
        for each [single_qoi_column][key/single time step]
              qoi_values[optional],
              E, Var, StdDev, Skew, Kurt, P10, P90, Sobol_t[optional], Sobol_m[optional]

        Note: regression is still not implemented for Saltelli
        """
        self.result_dict = dict()
        self._groupby_df_simulation_results(columns_to_group_by=[self.time_column_name,])
        # keyIter = list(self.groups.keys())

        for single_qoi_column in self.list_qoi_column:
            self.result_dict[single_qoi_column] = defaultdict(dict)

            print(f"computation of statistics for qoi {single_qoi_column} started...")
            solver_time_start = time.time()
            print(f"computation for qoi {single_qoi_column} - waits for shutdown...")
            sys.stdout.flush()
            print(f"computation for qoi {single_qoi_column} - shut down...")
            sys.stdout.flush()

            for key, val_indices in self.groups.items():
                self.result_dict[single_qoi_column][key] = dict()

                qoi_values = self.samples.df_simulation_result.loc[val_indices.values][single_qoi_column].values
                qoi_values_saltelli = qoi_values[:, np.newaxis]
                # standard_qoi_values = qoi_values_saltelli[:self.numEvaluations, :]
                # standard_qoi_values = qoi_values_saltelli[:numEvaluations, :]
                standard_qoi_values = qoi_values[:self.numEvaluations]

                if self.store_qoi_data_in_stat_dict:
                    self.result_dict[single_qoi_column][key]["qoi_values"] = qoi_values # for Saltelli this is N(d+2)xt
                    # self.result_dict[single_qoi_column][key]["qoi_values"] = standard_qoi_values # for Saltelli this is Nxt
                
                # local_result_dict["E"] = np.sum(qoi_values, axis=0, dtype=np.float64) / self.numEvaluations
                self.result_dict[single_qoi_column][key]["E"] = np.mean(standard_qoi_values, 0)

                if self.dict_stat_to_compute.get("Var", False):
                    self.result_dict[single_qoi_column][key]["Var"] = np.var(standard_qoi_values, ddof=1)
                    # self.result_dict[single_qoi_column][key]["Var"] = np.sum(
                    #     (standard_qoi_values - self.result_dict[single_qoi_column][key]["E"]) ** 2, axis=0,
                    #     dtype=np.float64) / (self.numEvaluations - 1)
                if self.dict_stat_to_compute.get("StdDev", False):
                    # local_result_dict["StdDev"] = np.sqrt(local_result_dict["Var"], dtype=np.float64)
                    self.result_dict[single_qoi_column][key]["StdDev"] = np.std(standard_qoi_values, 0, ddof=1)
                if self.dict_stat_to_compute.get("Skew", False):
                    self.result_dict[single_qoi_column][key]["Skew"] = scipy.stats.skew(standard_qoi_values, axis=0,
                                                                                        bias=True)
                if self.dict_stat_to_compute.get("Kurt", False):
                    self.result_dict[single_qoi_column][key]["Kurt"] = scipy.stats.kurtosis(standard_qoi_values, axis=0,
                                                                                            bias=True)
                if self.dict_stat_to_compute.get("P10", False):
                    self.result_dict[single_qoi_column][key]["P10"] = np.percentile(standard_qoi_values, 10, axis=0)
                    if isinstance(self.result_dict[single_qoi_column][key]["P10"], list) and len(
                        self.result_dict[single_qoi_column][key]["P10"]) == 1:
                        self.result_dict[single_qoi_column][key]["P10"] = \
                        self.result_dict[single_qoi_column][key]["P10"][0]

                if self.dict_stat_to_compute.get("P90", False):
                    self.result_dict[single_qoi_column][key]["P90"] = np.percentile(standard_qoi_values, 90, axis=0)
                    if isinstance(self.result_dict[single_qoi_column][key]["P90"], list) and len(
                        self.result_dict[single_qoi_column][key]["P90"]) == 1:
                        self.result_dict[single_qoi_column][key]["P90"] = \
                        self.result_dict[single_qoi_column][key]["P90"][0]

                if self.compute_sobol_indices_with_samples and self.nodes is not None:
                    if self.compute_Sobol_m:
                        self.result_dict[single_qoi_column][key]["Sobol_m"] = \
                                sens_indices_sampling_based_utils.compute_sens_indices_based_on_samples_rank_based(
                                    samples=self.nodes.T[:self.numEvaluations],
                                    Y=qoi_values[:self.numEvaluations, np.newaxis], D=self.dim, N=self.numEvaluations)
                else:
                    if self.compute_Sobol_t or self.compute_Sobol_m:
                        s_i, s_t = sens_indices_sampling_based_utils.compute_first_and_total_order_sens_indices_based_on_samples_pick_freeze(
                            qoi_values_saltelli, self.dim, self.numEvaluations, compute_first=self.compute_Sobol_m, 
                            compute_total=self.compute_Sobol_t, code_first=3, code_total=4,
                            do_printing=False
                            )
                        if self.compute_Sobol_t:
                            self.result_dict[single_qoi_column][key]["Sobol_t"] = s_t
                        if self.compute_Sobol_m:
                            self.result_dict[single_qoi_column][key]["Sobol_m"] = s_i

                self._process_chunk_result_single_qoi_single_time_step(
                    single_qoi_column=single_qoi_column, timestamp=key, result_dict=self.result_dict[single_qoi_column][key])

            self.save_print_plot_and_clear_result_dict_single_qoi(single_qoi_column)

            solver_time_end = time.time()
            solver_time = solver_time_end - solver_time_start
            print(f"solver_time for qoi {single_qoi_column}: {solver_time}")


    # =================================================================================================

    def calcStatisticsForSc(self, rawSamples=None, timesteps=None,
                            simulationNodes=None, order=None, regression=None, solverTimes=None,
                            work_package_indexes=None, original_runtime_estimator=None, 
                            poly_normed=None, poly_rule=None, cross_truncation=1.0,
                            *args, **kwargs):
        """
        This function groups result by time column and then iterates over all the qois of interest columns
        and updates the self.result_dict by adding the following entries:
        for each [single_qoi_column][key/single time step]
        if self.compute_kl_expansion_of_qoi is False or self.compute_timewise_gpce_next_to_kl_expansion is True:
            qoi_values[optional], gPCE[optional]
            gpce_coeff[optional]
            E
            if compute_other_stat_besides_pce_surrogate is True and relevan entries in :
                Var, StdDev, Skew, Kurt, P10, P90, Sobol_t[optional], Sobol_m[optional], Sobol_m2[optional]
        if self.compute_kl_expansion_of_qoi is True:
            qoi_values[optional], E
        """
        self.result_dict = dict()
        self._groupby_df_simulation_results(columns_to_group_by=[self.time_column_name,])
        # keyIter = list(self.groups.keys())

        compute_other_stat_besides_pce_surrogate = kwargs.get("compute_other_stat_besides_pce_surrogate", self.compute_other_stat_besides_pce_surrogate)

        for single_qoi_column in self.list_qoi_column:
            self.result_dict[single_qoi_column] = defaultdict(dict)

            print(f"computation of statistics for qoi {single_qoi_column} started...")
            solver_time_start = time.time()
            print(f"computation for qoi {single_qoi_column} - waits for shutdown...")
            sys.stdout.flush()
            print(f"computation for qoi {single_qoi_column} - shut down...")
            sys.stdout.flush()

            for key, val_indices in self.groups.items():
                qoi_values = self.samples.df_simulation_result.loc[val_indices.values][single_qoi_column].values
                self.result_dict[single_qoi_column][key] = dict()
                if self.store_qoi_data_in_stat_dict:
                    self.result_dict[single_qoi_column][key]["qoi_values"] = qoi_values

                if self.compute_kl_expansion_of_qoi and not self.compute_timewise_gpce_next_to_kl_expansion:
                    if self.weights is None or self.regression:
                        self.result_dict[single_qoi_column][key]["E"] = np.mean(qoi_values, 0)
                    else:
                        self.result_dict[single_qoi_column][key]["E"] = np.dot(qoi_values, self.weights)
                else:
                    if self.regression:
                        qoi_gPCE, qoi_coeff = cp.fit_regression(
                            polynomials=self.polynomial_expansion, abscissas=self.nodes, evals=qoi_values, retall=True, 
                            model=None  # classical least-square; one can use as well sklearn.linear_model.LinearRegression(fit_intercept=False)
                        )
                    else:
                        #qoi_gPCE, qoi_coeff = cp.fit_quadrature(self.polynomial_expansion, self.nodes, self.weights, qoi_values, retall=True)
                        qoi_gPCE, qoi_coeff = cp.fit_quadrature(
                            orth=self.polynomial_expansion, nodes=self.nodes, weights=self.weights, solves=qoi_values, retall=True, norms=self.polynomial_norms)
                    self.result_dict[single_qoi_column][key]['gpce_coeff'] = qoi_coeff
                    
                    self._calc_stats_for_gPCE_single_qoi(
                        single_qoi_column, key, self.dist, qoi_gPCE, compute_other_stat_besides_pce_surrogate)

                self._process_chunk_result_single_qoi_single_time_step(
                    single_qoi_column=single_qoi_column, timestamp=key, result_dict=self.result_dict[single_qoi_column][key])
  
            self._postprocess_pce_results_single_qoi(single_qoi_column)
            self.save_print_plot_and_clear_result_dict_single_qoi(single_qoi_column)

            solver_time_end = time.time()
            solver_time = solver_time_end - solver_time_start
            print(f"solver_time for qoi {single_qoi_column}: {solver_time}")

    # =================================================================================================

    def _calc_stats_for_gPCE_single_qoi(self, single_qoi_column, key, dist, qoi_gPCE, compute_other_stat_besides_pce_surrogate=True):
        """
        Calculate statistics for a single quantity of interest (qoi) using gPCE surrogate.

        Args:
            single_qoi_column (str): The name of the single qoi column.
            key (str): The timestamp key.
            dist (chaospy.Distribution): The distribution of the qoi.
            qoi_gPCE (chaospy.Poly): The gPCE surrogate model for the qoi.
            compute_other_stat_besides_pce_surrogate (bool, optional): Flag to compute other statistics besides the gPCE surrogate. Defaults to True.

        Returns:
            None
        """
        if qoi_gPCE is None:
            qoi_gPCE = self.qoi_gPCE

        numPercSamples = 10 ** 5

        if self.store_gpce_surrogate_in_stat_dict:
            self.result_dict[single_qoi_column][key]["gPCE"] = qoi_gPCE
        if self.save_gpce_surrogate: # and "gPCE" in self.result_dict[single_qoi_column][key]:
            utility.save_gpce_surrogate_model(workingDir=self.workingDir, gpce=qoi_gPCE, qoi=single_qoi_column, timestamp=key)
            if "gpce_coeff" in self.result_dict[single_qoi_column][key]:
                utility.save_gpce_coeffs(
                    workingDir=self.workingDir,
                    coeff=self.result_dict[single_qoi_column][key]["gpce_coeff"], qoi=single_qoi_column, timestamp=key)

        self.result_dict[single_qoi_column][key]["E"] = float(cp.E(qoi_gPCE, dist))

        if compute_other_stat_besides_pce_surrogate:

            if self.dict_stat_to_compute.get("Var", False):
                self.result_dict[single_qoi_column][key]["Var"] = float(cp.Var(qoi_gPCE, dist))
            if self.dict_stat_to_compute.get("StdDev", False):
                self.result_dict[single_qoi_column][key]["StdDev"] = float(cp.Std(qoi_gPCE, dist))
            #self.result_dict[single_qoi_column][key]["qoi_dist"] = cp.QoI_Dist(qoi_gPCE, dist) # not working!

            if self.dict_stat_to_compute.get("Skew", False):
                self.result_dict[single_qoi_column][key]["Skew"] = cp.Skew(qoi_gPCE, dist).round(4)
            if self.dict_stat_to_compute.get("Kurt", False):
                self.result_dict[single_qoi_column][key]["Kurt"] = cp.Kurt(qoi_gPCE, dist)

            if self.dict_stat_to_compute.get("qoi_dist", False):
                self.result_dict[single_qoi_column][key]["qoi_dist"] = cp.QoI_Dist(qoi_gPCE, dist)
                # # An example from Chaospy - generate QoI dist
                # qoi_dist = cp.QoI_Dist(qoi_gPCE, dist)
                # # generate sampling values for the qoi dist (you should know the min/max values for doing this)
                # dist_sampling_values = np.linspace(min_value, max_value, 1e4, endpoint=True)
                # # sample the QoI dist on the generated sampling values
                # pdf_samples = qoi_dist.pdf(dist_sampling_values)

            if self.dict_stat_to_compute.get("P10", False):
                self.result_dict[single_qoi_column][key]["P10"] = float(cp.Perc(qoi_gPCE, 10, dist, numPercSamples))
                if isinstance(self.result_dict[single_qoi_column][key]["P10"], list) and len(self.result_dict[single_qoi_column][key]["P10"]) == 1:
                    self.result_dict[single_qoi_column][key]["P10"]= self.result_dict[single_qoi_column][key]["P10"][0]

            if self.dict_stat_to_compute.get("P90", False):
                self.result_dict[single_qoi_column][key]["P90"] = float(cp.Perc(qoi_gPCE, 90, dist, numPercSamples))
                if isinstance(self.result_dict[single_qoi_column][key]["P90"], list) and len(self.result_dict[single_qoi_column][key]["P90"]) == 1:
                    self.result_dict[single_qoi_column][key]["P90"]= self.result_dict[single_qoi_column][key]["P90"][0]

            if self.compute_Sobol_t:
                self.result_dict[single_qoi_column][key]["Sobol_t"] = cp.Sens_t(qoi_gPCE, dist)
            if self.compute_Sobol_m:
                self.result_dict[single_qoi_column][key]["Sobol_m"] = cp.Sens_m(qoi_gPCE, dist)
            if self.compute_Sobol_m2:
                self.result_dict[single_qoi_column][key]["Sobol_m2"] = cp.Sens_m2(qoi_gPCE, dist)

    # =================================================================================================

    def _check_if_Sobol_t_computed(self, timestamp=None, qoi_column=None):
        if qoi_column is None:
            qoi_column = self.list_qoi_column[0]
        if timestamp is None:
            timestamp = self.pdTimesteps[0] # self.timesteps[0]
        try:
            self._is_Sobol_t_computed = "Sobol_t" in self.result_dict[qoi_column][
                timestamp]  # hasattr(self.result_dict[keyIter[0], "Sobol_t")
        except KeyError as e:
            self._is_Sobol_t_computed = False

    def _check_if_Sobol_m_computed(self, timestamp=None, qoi_column=None):
        if qoi_column is None:
            qoi_column = self.list_qoi_column[0]
        if timestamp is None:
            timestamp = self.pdTimesteps[0] # self.timesteps[0]
        try:
            self._is_Sobol_m_computed = "Sobol_m" in self.result_dict[qoi_column][timestamp]
        except KeyError as e:
            self._is_Sobol_m_computed = False

    def _check_if_Sobol_m2_computed(self, timestamp=None, qoi_column=None):
        if qoi_column is None:
            qoi_column = self.list_qoi_column[0]
        if timestamp is None:
            timestamp = self.pdTimesteps[0] # self.timesteps[0]
        try:
            self._is_Sobol_m2_computed = "Sobol_m2" in self.result_dict[qoi_column][timestamp]
        except KeyError as e:
            self._is_Sobol_m2_computed = False

    # =================================================================================================

    def saveToFile(self, fileName="statistics_dict", fileNameIdent="", directory="./",
                   fileNameIdentIsFullName=False, **kwargs):

        #fileName = self.generateFileName(fileName, fileNameIdent, directory, fileNameIdentIsFullName)
        #statFileName = fileName + '.stat'

        if self.result_dict is not None and self.result_dict:

            # df_statistics = self.create_df_from_statistics_data()
            # df_statistics.to_pickle(
            #     os.path.abspath(os.path.join(str(self.workingDir), "df_statistics.pkl")), compression="gzip")

            for single_qoi_column in self.list_qoi_column:
                try:
                    fileName = "statistics_dictionary_qoi_" + single_qoi_column + ".pkl"
                    fullFileName = os.path.abspath(os.path.join(str(self.workingDir), fileName))
                    if self.result_dict[single_qoi_column]:
                        with open(fullFileName, 'wb') as handle:
                            pickle.dump(self.result_dict[single_qoi_column], handle, protocol=pickle.HIGHEST_PROTOCOL)
                    else:
                        print(f"TimeDependentStatistics.saveToFile() - Entry {single_qoi_column} does not exist anymore in "
                              f"TimeDependentStatistics.result_dict, therefore will not be saved")
                except KeyError as e:
                    print(f"TimeDependentStatistics.saveToFile() - Entry {single_qoi_column} does not exist anymore in "
                          f"TimeDependentStatistics.result_dict, therefore will not be saved")

        # else:
        #     self.result_dict = uqef_dynamic_utils.read_all_saved_statistics_dict(
        #         workingDir=self.workingDir, list_qoi_column=self.list_qoi_column, 
        #         single_timestamp_single_file=self.instantly_save_results_for_each_time_step, 
        #         throw_error=throw_error=True, convert_to_pd_timestamp=self.convert_to_pd_timestamp)
        #     df_statistics = self.create_df_from_statistics_data()
        #     df_statistics.to_pickle(
        #         os.path.abspath(os.path.join(str(self.workingDir), "df_statistics.pkl")), compression="gzip")


        if self.active_scores_dict is not None:
            fileName = "active_scores_dict.pkl"
            fullFileName = os.path.abspath(os.path.join(str(self.workingDir), fileName))
            with open(fullFileName, 'wb') as handle:
                pickle.dump(self.active_scores_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # self.save_param_grad_analysis(self.workingDir)
        if self.df_time_varying_grad_analysis is not None:
            fileName = "df_time_varying_grad_analysis.pkl"
            fullFileName = os.path.abspath(os.path.join(str(self.workingDir), fileName))
            self.df_time_varying_grad_analysis.to_pickle(fullFileName, compression="gzip")

        if self.df_time_aggregated_grad_analysis is not None:
            fileName = "df_time_aggregated_grad_analysis.pkl"
            fullFileName = os.path.abspath(os.path.join(str(self.workingDir), fileName))
            self.df_time_aggregated_grad_analysis.to_pickle(fullFileName, compression="gzip")
    
    # =================================================================================================

    def compute_centered_single_qoi_data(self, single_qoi_column):
        """
        This function computes the centered data of the QoI, i.e., the difference between the QoI and its mean value.
        This method can be called only after the self.result_dict is computed.
        It will populate a new column in self.samples.df_simulation_result

        Note: This method modifies self.samples.df_simulation_result!

        Note: this is similar function to utility.add_centered_qoi_column_to_df_simulation_result
        """
        if self.groups is None:
            self._groupby_df_simulation_results(columns_to_group_by=[self.time_column_name,])
        keyIter = list(self.groups.keys())  # list of all the dates
        # adding cantered data of the QoI
        single_qoi_column_centered = single_qoi_column + "_centered"
        for key in keyIter:  # for a single time stamp
            if self.result_dict and self.result_dict is not None:
                mean = self.result_dict[single_qoi_column][key]['E']
                self.samples.df_simulation_result.loc[self.groups[key].values, single_qoi_column_centered] = \
                self.samples.df_simulation_result.loc[self.groups[key].values, single_qoi_column] - mean
            else:
                raise Exception("[STAT ERROR] - Trying to compute centered data of the QoI, however, result_dict is missing")
        # re-save the df_simulation_result after the centered data is added
        # if self.save_all_simulations:
        #     self.samples.save_simulation_results_to_file(self.workingDir)

    def center_output_single_qoi(self, single_qoi_column):
        """
        Centers the output for a single quantity of interest (QoI).

        Args:
            single_qoi_column (str): The name of the column representing the single QoI.

        Returns:
            numpy.ndarray: An array containing the centered output for the single QoI.
        """
        centered_output = np.empty((self.numEvaluations, self.N_quad))

        single_qoi_column_centered = single_qoi_column + "_centered"
        if single_qoi_column_centered not in self.samples.df_simulation_result.columns:
            self.compute_centered_single_qoi_data(single_qoi_column)

        grouped_by_index = self.samples.df_simulation_result.groupby([self.index_column_name,])
        groups_by_index = grouped_by_index.groups
        index = 0
        for key, val_indices in groups_by_index.items():
            centered_output[index, :] = self.samples.df_simulation_result.loc[val_indices, single_qoi_column_centered].values
            index += 1
        # print(f"[DEBUGGIN - center_output_single_qoi centered_output.shape] - {centered_output.shape}")
        return centered_output

    def compute_centered_output(self):
        """
        Computes the centered output for each QOI column.

        This method iterates over each QOI column in the list_qoi_column attribute and
        computes the centered output for that column using the center_output_single_qoi method.

        Populats self.centered_output variable
        
        Returns:
            None
        """
        for single_qoi_column in self.list_qoi_column:
            self.centered_output[single_qoi_column] = self.center_output_single_qoi(single_qoi_column)

    def compute_covariance_matrix_in_time_based_on_centered_output(self, centered_output):
        # if weights is None:
        #     weights = self.weights
        covariance_matrix = np.empty((self.N_quad, self.N_quad))
        for c in range(self.N_quad):
            for s in range(self.N_quad):
                if self.uq_method == "samples" or self.uq_method == "sampling" or self.uq_method == "mc" or self.uq_method == "ensemble" or self.uq_method == "saltelli":
                    covariance_matrix[s, c] = 1/(self.numEvaluations-1) * \
                    np.dot(centered_output[:, c], centered_output[:, s])
                elif self.uq_method == "quadrature" or self.uq_method =="pce" or self.uq_method == "sc" or self.uq_method == "kl":
                    if self.weights is None:
                        raise ValueError("[STAT ERROR] - Weights must be provided for quadrature-based algorithms")
                    covariance_matrix[s, c] = np.dot(self.weights, centered_output[:, c]*centered_output[:,s])
                else:
                    raise ValueError(f"[STAT ERROR] - Unknown algorithm - {self.uq_method}")
        # print(f"[DEBUGGIN  \
        # def compute_covariance_matrix_in_time_based_on_centered_output(self, centered_output): - covariance_matrix.shape] - {covariance_matrix.shape}")
        return covariance_matrix
    
    def compute_covariance_matrix_in_time_single_qoi(self, single_qoi_column):
        # if weights is None:
        #     weights = self.weights
        centered_output = self.center_output_single_qoi(single_qoi_column)
        covariance_matrix = self.compute_covariance_matrix_in_time_based_on_centered_output(centered_output)
        # utility.save_covariance_matrix(covariance_matrix, self.workingDir, single_qoi_column)
        # utility.plot_covariance_matrix(covariance_matrix, self.workingDir, filname=f"covariance_matrix_{single_qoi_column}.png")
        return covariance_matrix

    def compute_covariance_matrix_in_time_for_all_qois(self):
        if self.result_dict is None or not self.result_dict:
            print(f"Sorry, you can not compute compute_covariance_matrix_in_time without computing the statistics first and having the result_dict")
        for single_qoi_column in self.list_qoi_column:
            self.covariance_matrix[single_qoi_column] = self.compute_covariance_matrix_in_time_single_qoi(single_qoi_column)

    def compute_kl_expansion_single_qoi(self, single_qoi_column):
        # 3.1 Create a matrix with centered_outputs and 3.2 Compute of the covariance matrix
        centered_output = self.center_output_single_qoi(single_qoi_column)
        covariance_matrix_loc = self.compute_covariance_matrix_in_time_based_on_centered_output(centered_output)
        # Save and plot the covariance matrix
        utility.save_covariance_matrix(covariance_matrix_loc, self.workingDir, single_qoi_column)
        utility.plot_covariance_matrix(covariance_matrix_loc, self.workingDir, filname=f"covariance_matrix_{single_qoi_column}.png")
        # 3.3 Solve Discretized (generalized) Eigenvalue Problem
        eigenvalues, eigenvectors = utility.solve_eigenvalue_problem(covariance_matrix_loc, self.weights_time)
        # print("Eigenvalues:\n", eigenvalues)
        # print("Eigenvectors:\n", eigenvectors)
        # print(f"[DEBUGGING compute_kl_expansion_single_qoi eigenvalues.shape] - {eigenvalues.shape}")
        # print(f"[DEBUGGING compute_kl_expansion_single_qoi eigenvectors.shape] - {eigenvectors.shape}")
        # Save Eigenvalues and eigenvactors
        fileName = f"eigenvalues_{single_qoi_column}.npy"
        fullFileName = os.path.abspath(os.path.join(str(self.workingDir), fileName))
        np.save(fullFileName, eigenvalues)
        fileName = f"eigenvectors_{single_qoi_column}.npy"
        fullFileName = os.path.abspath(os.path.join(str(self.workingDir), fileName))
        np.save(fullFileName, eigenvectors)
        # Plotting the eigenvalues
        utility.plot_eigenvalues(eigenvalues, self.workingDir)
        # 3.4 Approximating the KL Expansion
        Var_kl_approx = np.sum(eigenvalues)
        # self.kl_expansion_order =  60 # [2, 4, 6, 8, 10]
        f_kl_eval_at_params = utility.setup_kl_expansion_matrix(eigenvalues, self.kl_expansion_order, self.numEvaluations, self.N_quad, self.weights_time, centered_output, eigenvectors)
        print(f"DEBUGGING - f_kl_eval_at_params.shape {f_kl_eval_at_params.shape}")
        # 3.5 PCE of the KL Expansion
        f_kl_surrogate_dict, f_kl_surrogate_coefficients = utility.pce_of_kl_expansion(
            N_kl=self.kl_expansion_order, 
            polynomial_expansion=self.polynomial_expansion, 
            nodes=self.nodes, weights=self.weights, 
            f_kl_eval_at_params=f_kl_eval_at_params, 
            regression=self.regression,
            polynomial_norms=self.polynomial_norms
            )
        # # 3.6 Generalized Sobol Indices
        # if self.compute_generalized_sobol_indices:
        #     fileName = self.workingDir / f"generalized_sobol_indices_{single_qoi_column}.pkl"
        #     param_name_generalized_sobol_total_indices = utility.computing_generalized_sobol_total_indices_from_kl_expan(
        #         f_kl_surrogate_coefficients, self.polynomial_expansion, self.weights_time, self.nodeNames, fileName, total_variance=Var_kl_approx)
        # else:
        #     return param_name_generalized_sobol_total_indices
        # print(f"INFO: computation of generalized S.S.I based on KL+gPCE finished...")
        # Save KL Surrogate model
        fileName = f"f_kl_surrogate_coefficients_{single_qoi_column}.npy"
        fullFileName = os.path.abspath(os.path.join(str(self.workingDir), fileName))
        np.save(fullFileName, f_kl_surrogate_coefficients)
        f_kl_surrogate_df = pd.DataFrame.from_dict(f_kl_surrogate_dict, orient='index')
        f_kl_surrogate_df.to_pickle(
            os.path.abspath(os.path.join(str(self.workingDir), f"f_kl_surrogate_df_{single_qoi_column}.pkl")), compression="gzip")
        return eigenvalues, eigenvectors, f_kl_surrogate_dict, f_kl_surrogate_coefficients, Var_kl_approx

    # =================================================================================================

    def _get_measured_single_qoi(
        self, timestepRange=None, time_column_name="TimeStamp", qoi_column_measured="measured", **kwargs):
        return None
        # raise NotImplementedError

    def get_measured_data(self, timestepRange=None, time_column_name=None, qoi_column_name=None,
                              **kwargs):
        """

         :param timestepRange:
         :param time_column_name:
         :param qoi_column_name:
         :param kwargs: transform_measured_data_as_original_model; if read measured data shold be transformed in the same way
         as original data; default is True
         :return: set self.df_measured to be a pd.DataFrame with three columns "TimeStamp", "qoi", "measured"

         Note: this function rely on previously computed dict_qoi_column_and_measured_info 
         (computed in utility.read_simulation_settings_from_configuration_object function)
         """

        if timestepRange is None:
            timestepRange = (self.timesteps_min, self.timesteps_max)

        if time_column_name is None:
            time_column_name = self.time_column_name

        if qoi_column_name is None:
            qoi_column_name = self.list_original_model_output_columns 
        if not isinstance(qoi_column_name, list):
            qoi_column_name = [qoi_column_name, ]

        transform_measured_data_as_original_model = kwargs.get(
            "transform_measured_data_as_original_model", True)

        list_df_measured_single_qoi = []
        for single_qoi_column in qoi_column_name:

            # Trying to find the original model output column name corresponding to the single_qoi_column
            # e.g., single_qoi_column = "streamflow" and the original model output column name is "Q_cms"
            if single_qoi_column not in self.list_original_model_output_columns:
                # in this case, single_qoi_column is either one of the measured column names (e.g., streamflow),
                # or is one of the new qoi column names (e.g., delta_Q_cms)
                is_single_qoi_column_in_measured_column_names = False
                for temp in self.list_original_model_output_columns:
                    if single_qoi_column == self.dict_qoi_column_and_measured_info[temp][1] \
                            or temp == self.dict_corresponding_original_qoi_column[single_qoi_column]:
                        single_qoi_column = temp
                        is_single_qoi_column_in_measured_column_names = True
                        break
                if not is_single_qoi_column_in_measured_column_names:
                    continue

            # finally, single_qoi_column should be among the original model output columns
            single_qoi_column_info = self.dict_qoi_column_and_measured_info[single_qoi_column]
            single_qoi_read_measured_data = single_qoi_column_info[0]
            single_qoi_column_measured = single_qoi_column_info[1]
            single_qoi_transform_model_output = single_qoi_column_info[2]

            if not single_qoi_read_measured_data or single_qoi_read_measured_data is None:
                continue

            df_measured_single_qoi = self._get_measured_single_qoi(
                timestepRange=timestepRange, time_column_name=time_column_name,
                qoi_column_measured=single_qoi_column_measured, **kwargs)

            if df_measured_single_qoi is None:
                continue

            # This data will be used for plotting or comparing with approximated data
            # Perform the same transformation as on original model output
            if transform_measured_data_as_original_model:
                if single_qoi_transform_model_output is not None and single_qoi_transform_model_output != "None":
                    utility.transform_column_in_df(
                        df_measured_single_qoi,
                        transformation_function_str=single_qoi_transform_model_output,
                        column_name=single_qoi_column_measured,
                        new_column_name=single_qoi_column_measured)

            if df_measured_single_qoi.index.name == time_column_name:
                df_measured_single_qoi.reset_index(inplace=True)
                df_measured_single_qoi.rename(columns={df_measured_single_qoi.index.name: time_column_name},
                                              inplace=True)

            df_measured_single_qoi.rename(columns={single_qoi_column_measured: "measured"},
                                          inplace=True)

            df_measured_single_qoi["qoi"] = single_qoi_column
            df_measured_single_qoi = df_measured_single_qoi[[time_column_name, "measured", "qoi"]]
            list_df_measured_single_qoi.append(df_measured_single_qoi)

        if list_df_measured_single_qoi:
            self.df_measured = pd.concat(list_df_measured_single_qoi, ignore_index=True, sort=False, axis=0)
            self.measured_fetched = True
        else:
            self.df_measured = None
            self.measured_fetched = False

    def get_unaltered_run_data(
        self, timestepRange=None, time_column_name=None, qoi_column_name=None, **kwargs):
        """
        This function fetches the unaltered run data from the simulation results file.
        The unaltered run data is stored in the self.df_unaltered attribute.
        Reimplement this function in the child class if the unaltered run data is needed.
        """
        if timestepRange is None:
            timestepRange = (self.timesteps_min, self.timesteps_max)
        if time_column_name is None:
            time_column_name = self.time_column_name
        if qoi_column_name is None:
            qoi_column_name = self.list_original_model_output_columns.copy()
        if not isinstance(qoi_column_name, list):
            qoi_column_name = [qoi_column_name, ]
        self.df_unaltered = None
        self.unaltered_computed = False

    def get_forcing_data(self, timestepRange=None, time_column_name=None, **kwargs):
        """
        This function fetches the forcing data from the forcing file.
        The forcing data is stored in the self.forcing_df attribute.
        Reimplement this function in the child class if the forcing data is needed.
        """
        if timestepRange is None:
            timestepRange = (self.timesteps_min, self.timesteps_max)
        if time_column_name is None:
            time_column_name = self.time_column_name
        self.forcing_df = None
        self.forcing_data_fetched = False

    # =================================================================================================
    # =================================================================================================

    def extract_mean_time_series(self):
        """
        Extracts the mean time series for each quantity of interest (QoI).

        Raises:
            Exception: If self.result_dict is None, indicating that the statistics need to be calculated first.

        Returns:
            pd.DataFrame or None: A DataFrame containing the mean time series for each QoI, or None if no QoI is available.
        """
        if self.result_dict is None:
            raise Exception('[STAT INFO] extract_mean_time_series - self.result_dict is None. '
                            'Calculate the statistics first!')
        list_of_single_qoi_mean_df = []
        for single_qoi_column in self.list_qoi_column:
            keyIter = list(self.pdTimesteps)  #self.timesteps (?)
            try:
                mean_time_series = [self.result_dict[single_qoi_column][key]["E"] for key in keyIter]
            except KeyError as e:
                continue
            qoi_column = [single_qoi_column] * len(keyIter)
            mean_df_single_qoi = pd.DataFrame(list(zip(qoi_column, mean_time_series, self.pdTimesteps)),
                                              columns=['qoi', 'mean_qoi', self.time_column_name])  # self.timesteps (?)
            list_of_single_qoi_mean_df.append(mean_df_single_qoi) 

        if list_of_single_qoi_mean_df:
            self.qoi_mean_df = pd.concat(list_of_single_qoi_mean_df, ignore_index=True, sort=False, axis=0)
        else:
            self.qoi_mean_df = None

    def create_df_from_statistics_data(self, compute_measured_normalized_data=False, set_lower_predictions_to_zero=False):
        """
        Create a DataFrame from the statistics data.
        Iterates over each QoI column and creates a DataFrame containing the statistics data for each QoI.
        Take a look at the create_df_from_statistics_data_single_qoi function for more details.

        Args:
            compute_measured_normalized_data (bool, optional): Flag to compute measured normalized data. Defaults to False.
            set_lower_predictions_to_zero (bool, optional): Flag to set lower predictions to zero. Defaults to False.

        Returns:
            None

        """
        list_of_single_qoi_dfs = []
        for single_qoi_column in self.list_qoi_column:
            df_statistics_single_qoi = self.create_df_from_statistics_data_single_qoi(
                qoi_column=single_qoi_column, 
                compute_measured_normalized_data=compute_measured_normalized_data,
                set_lower_predictions_to_zero=set_lower_predictions_to_zero
                )
            if df_statistics_single_qoi is not None:
                if "qoi" not in df_statistics_single_qoi.columns:
                    df_statistics_single_qoi["qoi"] = single_qoi_column
                list_of_single_qoi_dfs.append(df_statistics_single_qoi)
        if list_of_single_qoi_dfs:
            self.df_statistics = pd.concat(list_of_single_qoi_dfs, axis=0)
            self.df_statistics.sort_values(by=self.time_column_name, ascending=True, inplace=True)
        else:
            self.df_statistics = None

    def _check_if_df_statistics_is_computed(self, recompute_if_not=False):
        if self.df_statistics is None or self.df_statistics.empty:
            if recompute_if_not:
                self.create_df_from_statistics_data()
            else:
                raise Exception(f"You are trying to call a plotting utiltiy function whereas "
                                f"self.df_statistics object is still not computed - make sure to first call"
                                f"self.create_df_from_statistics_data")
        else:
            return

    def merge_df_statistics_data_with_forcing_data(self, **kwargs):
        """
        Merges the statistics data with the forcing data based on the time column.

        Args:
            **kwargs: Additional keyword arguments to be passed to the `get_forcing_data` method.

        Returns:
            DataFrame: The merged DataFrame containing the statistics data and the measured forcing data.
        """
        if not self.forcing_data_fetched or self.forcing_df is None or self.forcing_df.empty:
            self.get_forcing_data(**kwargs)
        if self.df_statistics is None or self.df_statistics.empty:
            self.create_df_from_statistics_data()
        df_statistics_and_measured = pd.merge(
            self.df_statistics, self.forcing_df, left_on=self.time_column_name,
            right_index=True)
        return df_statistics_and_measured

    def merge_df_statistics_data_with_measured_and_forcing_data(self, add_measured_data=True, add_forcing_data=True, **kwargs):
        """
        Merges the statistics data with measured data and forcing data based on the time column.

        Args:
            **kwargs: Additional keyword arguments to be passed to the `get_measured_data `
            or  `get_forcing_data` method.

        Returns:
            DataFrame: The merged DataFrame containing the statistics data and the measured forcing data.
        """
        if add_measured_data and (not self.measured_fetched or self.df_measured is None or self.df_measured.empty):
            transform_measured_data_as_original_model = kwargs.pop(
                "transform_measured_data_as_original_model", True)
            self.get_measured_data(
                timestepRange=(self.timesteps_min, self.timesteps_max),
                transform_measured_data_as_original_model=transform_measured_data_as_original_model, **kwargs)
            self.create_df_from_statistics_data()  # make sure that measured data is added to the statistics data in a single DataFrame

        if add_forcing_data and (not self.forcing_data_fetched or self.forcing_df is None or self.forcing_df.empty):
            self.get_forcing_data(**kwargs)
        
        if self.df_statistics is None:
            self.create_df_from_statistics_data()

        if self.df_statistics is None or self.df_statistics.empty:
            raise Exception("The statistics data is not available.")
        if add_forcing_data and (self.forcing_df is None or self.forcing_df.empty):
            raise Exception("The forcing data is not available.")
        if add_measured_data and (self.df_measured is None or self.df_measured.empty):
            raise Exception("The measured data is not available.")

        if add_forcing_data:
            return pd.merge(
                self.df_statistics, self.forcing_df, left_on=self.time_column_name,
                right_index=True)
        else:
            return self.df_statistics

    def describe_df_statistics(self):
        """
        Prints descriptive statistics for each QOI in the dataframe.

        This method computes and prints descriptive statistics for each QOI (Quantity of Interest)
        in the dataframe `df_statistics`. It first checks if the statistics have been computed,
        and if not, it recomputes them. Then, it iterates over each QOI, subsets the dataframe
        for that QOI, and prints the descriptive statistics using the `describe` method.

        Note: This method assumes that the dataframe `df_statistics` has a column named 'qoi'
        which contains the QOI values.

        Returns:
            None
        """
        self._check_if_df_statistics_is_computed(recompute_if_not=True)
        for single_qoi in self.list_qoi_column:
            df_statistics_single_qoi_subset = self.df_statistics.loc[
                self.df_statistics['qoi'] == single_qoi]
            print(f"{single_qoi}\n\n")
            print(df_statistics_single_qoi_subset.describe(include=np.number))

    def create_df_from_sensitivity_indices(
        self, si_type="Sobol_t", compute_measured_normalized_data=False):
        """
        Creates one big Pandas DataFrame for all QoIs.

        :param si_type: The type of sensitivity indices to compute. 
        Should be one of "Sobol_t", "Sobol_m", or "Sobol_m2". Defaults to "Sobol_t". (default: "Sobol_t").
        :param compute_measured_normalized_data: Whether to compute measured normalized data (default: False).
        :return: The combined DataFrame containing sensitivity indices for all QoIs.
        """
        si_df = None
        list_of_single_qoi_dfs = []
        for single_qoi_column in self.list_qoi_column:
            single_si_df = self.create_df_from_sensitivity_indices_single_qoi(
                qoi_column=single_qoi_column, si_type=si_type,
                compute_measured_normalized_data=compute_measured_normalized_data
            )
            if single_si_df is not None:
                single_si_df["qoi"] = single_qoi_column
                single_si_df.reset_index(inplace=True)
                single_si_df.rename(columns={single_si_df.index.name: self.time_column_name}, inplace=True)
                list_of_single_qoi_dfs.append(single_si_df)
        if list_of_single_qoi_dfs:
            si_df = pd.concat(list_of_single_qoi_dfs, axis=0)
            si_df.sort_values(by=self.time_column_name, ascending=True, inplace=True)
        return si_df

    def create_df_from_statistics_data_single_qoi(
        self, qoi_column, compute_measured_normalized_data=False, set_lower_predictions_to_zero=False):
        """
        Creates a pandas DataFrame from the statistics data for a single quantity of interest (qoi).

        Args:
            qoi_column (str): The column name of the quantity of interest.
            compute_measured_normalized_data (bool, optional): Flag indicating whether to compute normalized measured data. Defaults to False.
            set_lower_predictions_to_zero (bool, optional): Flag indicating whether to set lower predictions to zero. Defaults to False.

        Returns:
            pandas.DataFrame: The DataFrame containing the statistics data for the specified qoi.

        Raises:

        Note:
            This method retrieves the statistics data from the result_dict attribute and constructs a DataFrame
            with columns representing different statistical measures such as mean, standard deviation, percentiles, etc.
            The DataFrame also includes the time column and the qoi column.

            If measured data is available (i.e., df_measured is not Noe) the method addes measured data to the final 
            data frame as well, and if compute_measured_normalized_data is True, the method computes
            the normalized measured data and adds it as a column in the DataFrame.

            If the unaltered_computed flag is True, the plan is to add it to the final df
        """
        # try:
        #     self.result_dict[qoi_column]
        # except KeyError as e:
        #     return None
        if not self.result_dict[qoi_column]:
            return None

        keyIter = list(self.pdTimesteps)  # self.timesteps (?)

        list_of_columns = [self.pdTimesteps, ]  # self.timesteps (?)
        list_of_columns_names = [self.time_column_name, ]
        # list_of_columns = [self.pdTimesteps, mean_time_series, std_time_series,
        #                    p10_time_series, p90_time_series]
        # list_of_columns_names = [self.time_column_name, "E", "StdDev", "P10", "P90"]

        if "E" in self.result_dict[qoi_column][keyIter[0]]:
            mean_time_series = [self.result_dict[qoi_column][key]["E"] for key in keyIter]
            list_of_columns.append(mean_time_series)
            list_of_columns_names.append("E")
        if "gpce_coeff" in self.result_dict[qoi_column][keyIter[0]]:
            list_of_columns.append([self.result_dict[qoi_column][key]["gpce_coeff"] for key in keyIter])
            list_of_columns_names.append("gpce_coeff")
        if "gPCE" in self.result_dict[qoi_column][keyIter[0]]:
            list_of_columns.append([self.result_dict[qoi_column][key]["gPCE"] for key in keyIter])
            list_of_columns_names.append("gPCE")
        if "Var" in self.result_dict[qoi_column][keyIter[0]]:
            std_time_series = [self.result_dict[qoi_column][key]["Var"] for key in keyIter]
            list_of_columns.append(std_time_series)
            list_of_columns_names.append("Var")
        if "StdDev" in self.result_dict[qoi_column][keyIter[0]]:
            std_time_series = [self.result_dict[qoi_column][key]["StdDev"] for key in keyIter]
            list_of_columns.append(std_time_series)
            list_of_columns_names.append("StdDev")
        if "P10" in self.result_dict[qoi_column][keyIter[0]]:
            p10_time_series = [self.result_dict[qoi_column][key]["P10"] for key in keyIter]
            list_of_columns.append(p10_time_series)
            list_of_columns_names.append("P10")
        if "P90" in self.result_dict[qoi_column][keyIter[0]]:
            p90_time_series = [self.result_dict[qoi_column][key]["P90"] for key in keyIter]
            list_of_columns.append(p90_time_series)
            list_of_columns_names.append("P90")
        if "Skew" in self.result_dict[qoi_column][keyIter[0]]:
            list_of_columns.append([self.result_dict[qoi_column][key]["Skew"] for key in keyIter])
            list_of_columns_names.append("Skew")
        if "Kurt" in self.result_dict[qoi_column][keyIter[0]]:
            list_of_columns.append([self.result_dict[qoi_column][key]["Kurt"] for key in keyIter])
            list_of_columns_names.append("Kurt")
        if "qoi_dist" in self.result_dict[qoi_column][keyIter[0]]:
            list_of_columns.append([self.result_dict[qoi_column][key]["qoi_dist"] for key in keyIter])
            list_of_columns_names.append("qoi_dist")

        # self._check_if_Sobol_t_computed(keyIter[0], qoi_column=qoi_column)
        # self._check_if_Sobol_m_computed(keyIter[0], qoi_column=qoi_column)
        is_Sobol_t_computed = "Sobol_t" in self.result_dict[qoi_column][keyIter[0]]
        is_Sobol_m_computed = "Sobol_m" in self.result_dict[qoi_column][keyIter[0]]
        is_Sobol_m2_computed = "Sobol_m2" in self.result_dict[qoi_column][keyIter[0]]

        if is_Sobol_m_computed:
            for i in range(len(self.labels)):
                sobol_m_time_series = [self.result_dict[qoi_column][key]["Sobol_m"][i] for key in keyIter]
                list_of_columns.append(sobol_m_time_series)
                temp = "Sobol_m_" + self.labels[i]
                list_of_columns_names.append(temp)
        if is_Sobol_m2_computed:
            for i in range(len(self.labels)):
                sobol_m2_time_series = [self.result_dict[qoi_column][key]["Sobol_m2"][i] for key in keyIter]
                list_of_columns.append(sobol_m2_time_series)
                temp = "Sobol_m2_" + self.labels[i]
                list_of_columns_names.append(temp)
        if is_Sobol_t_computed:
            for i in range(len(self.labels)):
                sobol_t_time_series = [self.result_dict[qoi_column][key]["Sobol_t"][i] for key in keyIter]
                list_of_columns.append(sobol_t_time_series)
                temp = "Sobol_t_" + self.labels[i]
                list_of_columns_names.append(temp)

        if f'generalized_sobol_total_index_{self.labels[0]}' in self.result_dict[qoi_column][keyIter[-1]]:
            for i in range(len(self.labels)):
                name = f"generalized_sobol_total_index_{self.labels[i]}"
                generalized_sobol_total_index_values_temp = []
                at_least_one_entry_found = False
                for key in keyIter:
                    if name in self.result_dict[qoi_column][key]:
                        at_least_one_entry_found = True
                        temp = self.result_dict[qoi_column][key][name]
                        generalized_sobol_total_index_values_temp.append(temp)
                if at_least_one_entry_found:
                    list_of_columns_names.append(name)
                    if len(generalized_sobol_total_index_values_temp)==1:
                        # print(f"[DEBUGGING] {type(generalized_sobol_total_index_values_temp)}; {len(generalized_sobol_total_index_values_temp)}")
                        generalized_sobol_total_index_values_temp = [generalized_sobol_total_index_values_temp[0],]*len(keyIter)
                        # print(f"[DEBUGGING] {type(generalized_sobol_total_index_values_temp)}; {len(generalized_sobol_total_index_values_temp)}")
                    list_of_columns.append(generalized_sobol_total_index_values_temp)

        if not list_of_columns:
            return None

        df_statistics_single_qoi = pd.DataFrame(list(zip(*list_of_columns)), columns=list_of_columns_names)
        df_statistics_single_qoi["qoi"] = qoi_column

        if 'E' in df_statistics_single_qoi.columns:
            if 'StdDev' in df_statistics_single_qoi.columns:
                df_statistics_single_qoi["E_minus_std"] = df_statistics_single_qoi['E'] - df_statistics_single_qoi['StdDev']
                df_statistics_single_qoi["E_plus_std"] = df_statistics_single_qoi['E'] + df_statistics_single_qoi['StdDev']
                df_statistics_single_qoi["E_minus_2std"] = df_statistics_single_qoi['E'] - 2*df_statistics_single_qoi['StdDev']
                df_statistics_single_qoi["E_plus_2std"] = df_statistics_single_qoi['E'] + 2*df_statistics_single_qoi['StdDev']
            elif 'Var' in df_statistics_single_qoi.columns:
                df_statistics_single_qoi["E_minus_std"] = df_statistics_single_qoi['E'] - np.sqrt(df_statistics_single_qoi['Var'])
                df_statistics_single_qoi["E_plus_std"] = df_statistics_single_qoi['E'] + np.sqrt(df_statistics_single_qoi['Var'])
                df_statistics_single_qoi["E_minus_2std"] = df_statistics_single_qoi['E'] - 2*np.sqrt(df_statistics_single_qoi['Var'])
                df_statistics_single_qoi["E_plus_2std"] = df_statistics_single_qoi['E'] + 2*np.sqrt(df_statistics_single_qoi['Var'])

        if set_lower_predictions_to_zero:
            if 'E_minus_std' in df_statistics_single_qoi:
                df_statistics_single_qoi.loc[df_statistics_single_qoi["E_minus_std"] < 0, "E_minus_std"] = 0
            if 'E_minus_2std' in df_statistics_single_qoi:
                df_statistics_single_qoi.loc[df_statistics_single_qoi["E_minus_2std"] < 0, "E_minus_2std"] = 0
            if 'P10' in df_statistics_single_qoi:
                df_statistics_single_qoi['P10'] = df_statistics_single_qoi['P10'].apply(lambda x: max(0, x))

        if self.measured_fetched and self.df_measured is not None:
            if qoi_column in list(self.df_measured["qoi"].unique()):
                # print(f"{qoi_column}")
                df_measured_subset = self.df_measured.loc[self.df_measured["qoi"] == qoi_column][[
                    self.time_column_name, "measured"]]
                # df_measured_subset.drop("qoi", inplace=True)
                df_statistics_single_qoi = pd.merge(df_statistics_single_qoi, df_measured_subset,
                                                    on=[self.time_column_name, ], how='left')
            elif self.dict_corresponding_original_qoi_column[qoi_column] in list(self.df_measured["qoi"].unique()):
                df_measured_subset = self.df_measured.loc[
                    self.df_measured["qoi"] == self.dict_corresponding_original_qoi_column[qoi_column]][[
                    self.time_column_name, "measured"]]
                # df_measured_subset.drop("qoi", inplace=True)
                df_statistics_single_qoi = pd.merge(df_statistics_single_qoi, df_measured_subset,
                                                    on=[self.time_column_name, ], how='left')
            else:
                df_statistics_single_qoi["measured"] = np.nan
        else:
            df_statistics_single_qoi["measured"] = np.nan

        if self.unaltered_computed:
            pass  # TODO

        return df_statistics_single_qoi

    def create_df_from_sensitivity_indices_single_qoi(
        self, qoi_column, si_type="Sobol_t", compute_measured_normalized_data=False):
        """
        Creates a DataFrame from sensitivity indices for a single quantity of interest (QoI).

        Args:
            qoi_column (str): The column name of the quantity of interest.
            si_type (str, optional): The type of sensitivity index to compute. 
                Should be one of "Sobol_t", "Sobol_m", or "Sobol_m2". Defaults to "Sobol_t".
            compute_measured_normalized_data (bool, optional): Whether to compute normalized measured data. 
                Defaults to False.

        Returns:
            pandas.DataFrame: The DataFrame containing the sensitivity indices.

        Raises:

        Note:
            - The sensitivity indices are computed based on the result_dict, which should contain the necessary data.
            - The result_dict should have the sensitivity indices computed for the specified qoi_column and si_type.
            - If the sensitivity indices are not computed for the specified qoi_column and si_type, 
              the method returns None.

        """
        # try:
        #     self.result_dict[qoi_column]
        # except KeyError as e:
        #     return None
        if not self.result_dict[qoi_column]:
            return None

        keyIter = list(self.pdTimesteps)  # self.timesteps (?)
        is_Sobol_t_computed = "Sobol_t" in self.result_dict[qoi_column][keyIter[0]]
        is_Sobol_m_computed = "Sobol_m" in self.result_dict[qoi_column][keyIter[0]]
        is_Sobol_m2_computed = "Sobol_m2" in self.result_dict[qoi_column][keyIter[0]]

        if si_type == "Sobol_t" and not is_Sobol_t_computed:
            #raise Exception("Sobol Total Order Indices are not computed")
            return None
        elif si_type == "Sobol_m" and not is_Sobol_m_computed:
            #raise Exception("Sobol Main Order Indices are not computed")
            return None
        elif si_type == "Sobol_m2" and not is_Sobol_m2_computed:
            #raise Exception("Sobol Second Order Indices are not computed")
            return None

        list_of_df_over_parameters = []
        for i in range(len(self.labels)):
            # if uq_method == "saltelli":
            #     si_single_param = [self.result_dict[key][si_type][i][0] for key in keyIter]
            # else:
            si_single_param = [self.result_dict[qoi_column][key][si_type][i] for key in keyIter]
            df_temp = pd.DataFrame(list(zip(si_single_param, self.pdTimesteps)),
                                   columns=[si_type + "_" + self.labels[i], self.time_column_name])  #self.timesteps (?)
            list_of_df_over_parameters.append(df_temp)
        si_df = reduce(lambda left, right: pd.merge(left, right, on=self.time_column_name, how='outer'),
                       list_of_df_over_parameters)

        if self.measured_fetched and self.df_measured is not None:
            if qoi_column in list(self.df_measured["qoi"].unique()):
                df_measured_subset = self.df_measured.loc[self.df_measured["qoi"] == qoi_column][[
                    self.time_column_name, "measured"]]
                si_df = pd.merge(si_df, df_measured_subset, on=[self.time_column_name,], how='left')
                if compute_measured_normalized_data:
                    # df_statistics_single_qoi["measured_norm"] = MinMaxScaler().fit_transform(
                    #     np.array(df_statistics_single_qoi["measured"]).reshape(-1, 1))
                    si_df["measured_norm"] = (si_df["measured"] - si_df["measured"].min()) / (
                            si_df["measured"].max() - si_df["measured"].min())
            elif self.dict_corresponding_original_qoi_column[qoi_column] in list(self.df_measured["qoi"].unique()):
                df_measured_subset = self.df_measured.loc[
                    self.df_measured["qoi"] == self.dict_corresponding_original_qoi_column[qoi_column]][[
                    self.time_column_name, "measured"]]
                # df_measured_subset.drop("qoi", inplace=True)
                si_df = pd.merge(si_df, df_measured_subset, on=[self.time_column_name, ], how='left')
                if compute_measured_normalized_data:
                    # df_statistics_single_qoi["measured_norm"] = MinMaxScaler().fit_transform(
                    #     np.array(df_statistics_single_qoi["measured"]).reshape(-1, 1))
                    si_df["measured_norm"] = (si_df["measured"] - si_df["measured"].min()) / (
                            si_df["measured"].max() - si_df["measured"].min())
            else:
                si_df["measured"] = np.nan
        else:
            si_df["measured"] = np.nan

        si_df.set_index(self.time_column_name, inplace=True)
        si_df.sort_index(ascending=True, inplace=True)
        return si_df

    # =================================================================================================
    # Set of functions for plotting
    # =================================================================================================

    def printResults(self, timestep=-1, **kwargs):
        pass    

    def printResults_single_qoi(self, single_qoi_column, dict_time_vs_qoi_stat=None, timestep=-1, **kwargs):
        pass  
    
    def prepare_for_plotting(self, timestep=-1, display=False,
                    fileName="", fileNameIdent="", directory="./",
                    fileNameIdentIsFullName=False, safe=True, **kwargs):
        pass

    def plotResults_single_qoi(self, single_qoi_column, dict_time_vs_qoi_stat=None, timestep=-1, display=False, fileName="",
                               fileNameIdent="", directory="./", fileNameIdentIsFullName=False, safe=True,
                               dict_what_to_plot=None, **kwargs):
        """
        This function plots the statistics of a single QoI.
        It is more relevant that the plotResults function, which plots all the QoIs.
        """
        # TODO - This might be a memory problem, why not just self.result_dict[single_qoi_column]!
        if dict_time_vs_qoi_stat is None:
            dict_time_vs_qoi_stat = self.result_dict[single_qoi_column]

        if fileName == "":
            fileName = single_qoi_column

        single_fileName = self.generateFileName(fileName=fileName, fileNameIdent=".html",
                                                directory=directory, fileNameIdentIsFullName=fileNameIdentIsFullName)

        fig = self._plotStatisticsDict_plotly_single_qoi(
            single_qoi_column=single_qoi_column, 
            dict_time_vs_qoi_stat=dict_time_vs_qoi_stat,
            filename=single_fileName, display=display,
            dict_what_to_plot=dict_what_to_plot, 
            **kwargs
        )
        if display:
            fig.show()
        print(f"[STAT INFO] plotResults for QoI-{single_qoi_column} function is done!")

    def _plotStatisticsDict_plotly_single_qoi(self, single_qoi_column, dict_time_vs_qoi_stat=None, unalatered=False,
                                              measured=False, forcing=False, recalculateTimesteps=False,
                                              window_title='Forward UQ & SA', filename="sim-plotly.html", display=False,
                                              dict_what_to_plot=None, **kwargs):
        """
        This function plots the statistics of a single QoI.
        It should implement the plotting based on plotly library, and present a background engine for the plotResults_single_qoi function.
        """
        pass

    def plotResults(self, timestep=-1, display=False, fileName="", fileNameIdent="", directory="./",
                    fileNameIdentIsFullName=False, safe=True, dict_what_to_plot=None, **kwargs):
        """
        This function plots the statistics of a single, or multiple, QoI.
        Thake a look at the plotResults_single_qoi function for more details.
        """
        pass

    def _plotStatisticsDict_plotly(self, unalatered=False, measured=False, forcing=False, recalculateTimesteps=False,
                                   window_title='Forward UQ & SA', filename="sim-plotly.html", display=False,
                                   dict_what_to_plot=None, **kwargs):
        pass

    def _compute_number_of_rows_for_plotting(self, dict_what_to_plot=None, forcing=False,
                                             list_qoi_column_to_plot=None, result_dict=None, **kwargs):
        keyIter = list(self.pdTimesteps)  # self.timesteps

        n_rows = 0

        if list_qoi_column_to_plot is None:
            list_qoi_column_to_plot = self.list_qoi_column

        starting_row = 1
        n_rows += len(list_qoi_column_to_plot)

        if forcing and self.forcing_data_fetched:
            n_rows += 3
            starting_row = 4

        # current_row = starting_row

        if dict_what_to_plot is None:
            if self.dict_what_to_plot is not None:
                dict_what_to_plot = self.dict_what_to_plot
            else:
                dict_what_to_plot = {
                    "E_minus_std": False, "E_plus_std": False, "P10": False, "P90": False,
                    "StdDev": False, "Skew": False, "Kurt": False, "Sobol_m": False, "Sobol_m2": False, "Sobol_t": False
                }
                self.dict_what_to_plot = dict_what_to_plot

        # dict_qoi_vs_plot_rows = defaultdict(dict, {single_qoi_column: {} for single_qoi_column in list_qoi_column_to_plot})

        for single_qoi_column in list_qoi_column_to_plot:
            # dict_qoi_vs_plot_rows[single_qoi_column]["qoi"] = current_row
            # current_row += 1
            if result_dict is None:
                if self.result_dict[single_qoi_column]:
                    result_dict = self.result_dict[single_qoi_column]
                else:
                    continue
            if result_dict:
                if "StdDev" in result_dict[keyIter[0]] and dict_what_to_plot.get("StdDev", False):
                    n_rows += 1
                    # dict_qoi_vs_plot_rows[single_qoi_column]["StdDev"] = current_row
                    # current_row += 1
                if "Skew" in result_dict[keyIter[0]] and dict_what_to_plot.get("Skew", False):
                    n_rows += 1
                    # dict_qoi_vs_plot_rows[single_qoi_column]["Skew"] = current_row
                    # current_row += 1
                if "Kurt" in result_dict[keyIter[0]] and dict_what_to_plot.get("Kurt", False):
                    n_rows += 1
                    # dict_qoi_vs_plot_rows[single_qoi_column]["Kurt"] = current_row
                    # current_row += 1
                if "Sobol_m" in result_dict[keyIter[0]] and dict_what_to_plot.get("Sobol_m", False):
                    n_rows += 1
                    # dict_qoi_vs_plot_rows[single_qoi_column]["Sobol_m"] = current_row
                    # current_row += 1
                if "Sobol_m2" in result_dict[keyIter[0]] and dict_what_to_plot.get("Sobol_m2", False):
                    n_rows += 1
                    # dict_qoi_vs_plot_rows[single_qoi_column]["Sobol_m2"] = current_row
                    # current_row += 1
                if "Sobol_t" in result_dict[keyIter[0]] and dict_what_to_plot.get("Sobol_t", False):
                    n_rows += 1
                    # dict_qoi_vs_plot_rows[single_qoi_column]["Sobol_t"] = current_row
                    # current_row += 1
                plot_generalized_sobol_indices = False
                for key in result_dict[keyIter[-1]].keys():
                    if key.startswith("generalized_sobol_total_index_"):
                        plot_generalized_sobol_indices = True
                        break
                if plot_generalized_sobol_indices:   
                    n_rows += 1
            else:
                continue

        return n_rows, starting_row

    # =================================================================================================

    def plot_measured_data_single_qoi(self, single_qoi: str, fig=None, add_to_subplot=False, **kwargs):
        """
        Plots the measured data for a single quantity of interest (QoI).

        Args:
            single_qoi (str): The name of the quantity of interest (QoI) to plot.
            fig (go.Figure, optional): The plotly figure to add the measured data to. Defaults to None.
            add_to_subplot (bool, optional): Whether to add the measured data to a subplot. Defaults to False.
            if true, then n_rows and n_col should be provided in kwargs; the function on that case just extend the 
            propagated figure and retuns None

        Returns:
            None or go.Figure: The plot of the measured data for the specified QoI.

        Raises:
            None
        """
        if fig is None:
            fig = go.Figure()
            add_to_subplot = False

        if add_to_subplot:
            n_rows=kwargs.get("n_rows", 1)
            n_col=kwargs.get("n_col", 1)

        if self.measured_fetched and self.df_measured is not None and not self.df_measured.empty:
            df_measured_subset = self.df_measured[self.df_measured["qoi"]==single_qoi]
            if not df_measured_subset.empty:
                if add_to_subplot:
                    fig.add_trace(
                        go.Scatter(
                            x=df_measured_subset[self.time_column_name], 
                            y=df_measured_subset["measured"],
                            name=f"measured {single_qoi}",
                            line_color='red', mode="lines", opacity=1.0, showlegend=True,
                        ), row=n_rows, col=n_col
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=df_measured_subset[self.time_column_name], 
                            y=df_measured_subset["measured"],
                            name=f"measured {single_qoi}",
                            line_color='red', mode="lines", opacity=1.0, showlegend=True,
                        )
                    )
                    return fig
            else:
                print(f"[STAT INFO] DF subset storing measured data for QoI-{single_qoi} is empty!")
        else:
            print(f"[STAT INFO] No plotting in plot_measured_data_single_qoi - df_measured is empty!")

    def plot_mean_data_single_qoi(self, single_qoi: str, df: pd.DataFrame=None, fig=None, add_to_subplot=False, **kwargs):
        """
        Plots mean computed data for a single quantity of interest (QoI) from pandas DataFrame!
        Args:
            single_qoi (str): The name of the quantity of interest (QoI) to plot.
            df (): Df storinf mean value. Defaults to None; if None self.df_statistics is used and assumed it is already computed.
                   If DF is provided it is assumed that it stores only data for the particular QoI and that it has 'E' column
            fig (go.Figure, optional): The plotly figure to add the measured data to. Defaults to None.
            add_to_subplot (bool, optional): Whether to add the measured data to a subplot. Defaults to False.
            if true, then n_rows and n_col should be provided in kwargs; the function on that case just extend the 
            propagated figure and retuns None

        Returns:
            None or go.Figure: The plot of the measured data for the specified QoI.

        Raises:
            None
        """
        if fig is None:
            fig = go.Figure()
            add_to_subplot = False

        if add_to_subplot:
            n_rows=kwargs.get("n_rows", 1)
            n_col=kwargs.get("n_col", 1)

        if df is None:
            if self.df_statistics is None or self.df_statistics.empty:
                print(f"[STAT INFO] DF subset storing mean computed data for QoI-{single_qoi} is not provided and df_statistics empty!")
                return
            else:
                df = self.df_statistics[self.df_statistics["qoi"]==single_qoi]

        if not df.empty:
            if add_to_subplot:
                fig.add_trace(
                    go.Scatter(
                        x=df[self.time_column_name], 
                        y=df["E"],
                        name=f"Mean {single_qoi}",
                        line_color='green', mode="lines", opacity=1.0, showlegend=True,
                    ), row=n_rows, col=n_col
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=df[self.time_column_name], 
                        y=df["E"],
                        name=f"Mean {single_qoi}",
                        line_color='green', mode="lines", opacity=1.0, showlegend=True,
                    )
                )
                return fig
        else:
            print(f"[STAT INFO] DF subset storing mean computed data for QoI-{single_qoi} is not provided and df_statistics empty!")

    def get_info_for_plotting_forcing_data(self, **kwargs):
        """
        Gets the information needed for plotting the forcing data.

        Args:
            kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the information needed for plotting the forcing data.
        Reimplement this function in the child class if the forcing data is needed.
        """
        return {
            "n_rows": 0,
            "subplot_titles": []
        }

    def plot_forcing_data(self, df: pd.DataFrame=None, fig=None, add_to_subplot=False, **kwargs):
        """
        This function should plot the forcing data.
        The forcing data should be read from the self.forcing_df attribute in case df is None.
        Args:
            df (pd.DataFrame, optional): The DataFrame containing the forcing data. Defaults to None. 
                if None self.forcing_df is used.
            fig (go.Figure, optional): The plotly figure to add the measured data to. Defaults to None.
            add_to_subplot (bool, optional): Whether to add the measured data to a subplot. Defaults to False.
            kwargs: Additional keyword arguments.

        Reimplement this function in the child class if the forcing data is needed.
        """
        pass

    def plot_conditioned_simulation_runs_single_qoi(
        self, single_qoi: str, df_simulation_result: pd.DataFrame=None, fig=None, add_to_subplot=False, **kwargs):
        """
        Plots the conditioned simulation runs for a single quantity of interest (QoI).

        Args:
            single_qoi (str): The name of the quantity of interest (QoI) to plot.
            df_simulation_result (pd.DataFrame, optional): The DataFrame containing the simulation results. 
                If not provided, it will be retrieved from `self.samples` if available.
            fig (go.Figure, optional): The Figure object to add the plot to. If not provided, a new Figure object will be created.
            add_to_subplot (bool, optional): Whether to add the plot to an existing subplot. Default is False.
            **kwargs: Additional keyword arguments for customization.
                - df_index_parameter_gof (pd.DataFrame, optional): The DataFrame containing the index-parameter goodness-of-fit values.
                - allow_conditioning_results_based_on_metric (bool, optional): Flag indicating whether to condition the results based on a metric. Default is False.
                - condition_results_based_on_metric (str, optional): The metric to condition the results on. Default is None.
                - condition_results_based_on_metric_value (float, optional): The value of the metric to condition the results on. Default is None.
                - condition_results_based_on_metric_sign (str, optional): The sign of the metric to condition the results on. Default is None.  
        Returns:
            go.Figure: The Figure object with the plot.

        """
        df_index_parameter_gof = kwargs.get("df_index_parameter_gof", None)
        allow_conditioning_results_based_on_metric = kwargs.get("allow_conditioning_results_based_on_metric", False)
        if fig is None:
            fig = go.Figure()
            add_to_subplot = False

        if add_to_subplot:
            n_rows=kwargs.get("n_rows", 1)
            n_col=kwargs.get("n_col", 1)

        if df_simulation_result is None and self.samples is not None:
            df_simulation_result = self.samples.get_df_simulation_result()

        if df_simulation_result is not None and not df_simulation_result.empty:
            if allow_conditioning_results_based_on_metric:
                condition_results_based_on_metric = kwargs.get("condition_results_based_on_metric", None)
                condition_results_based_on_metric_value = kwargs.get("condition_results_based_on_metric_value", None)
                condition_results_based_on_metric_sign = kwargs.get("condition_results_based_on_metric_sign", None)
                if df_index_parameter_gof is None and self.samples is not None:
                    df_index_parameter_gof = self.samples.get_df_index_parameter_gof_values()
                if df_index_parameter_gof is not None and not df_index_parameter_gof.empty:
                    df_simulation_result_for_plotting = uqef_dynamic_utils.filter_df_simulation_result_based_on_gof_condition(
                        df_simulation_result, df_index_parameter_gof,
                        condition_results_based_on_metric, condition_results_based_on_metric_value,
                        condition_results_based_on_metric_sign, index_column_name=self.index_column_name, 
                        time_column_name=self.time_column_name
                    )
                else:
                    df_simulation_result_for_plotting = df_simulation_result
            else:
                df_simulation_result_for_plotting = df_simulation_result
                
            grouped = df_simulation_result_for_plotting.groupby(self.index_column_name)
            groups = grouped.groups
            keyIter = list(groups.keys())
            for key in keyIter:
                temp = df_simulation_result_for_plotting.loc[groups[key].values]
                fig.add_trace(
                    go.Scatter(
                        x=temp[self.time_column_name], 
                        y=temp[single_qoi],
                        line_color='LightSkyBlue', mode="lines", opacity=0.3, showlegend=False,
                    ), row=n_rows, col=n_col
                )
            if not add_to_subplot:
                return fig
        else:
            print(f"[STAT INFO] DF simulation result for QoI-{single_qoi} is empty!")

    def plot_filtered_data_results_measured_and_forcing_single_qoi(
        self, single_qoi, plot_measured=True, plot_forcing=True, plot_df_simulation_result=True, plot_mean_data = True,
        df_simulation_result=None, df_index_parameter_gof=None, allow_conditioning_results_based_on_metric=True,
        directory_for_saving_plots=None, fileName: str=None,
        title=None, **kwargs):
        """
        Plots the filtered data results for a single quantity of interest (QoI).
        
        Args:
            single_qoi (str): The name of the quantity of interest (QoI) to plot.
            plot_measured (bool, optional): Whether to plot the measured data. Defaults to True.
            plot_forcing (bool, optional): Whether to plot the forcing data. Defaults to True.
            plot_df_simulation_result (bool, optional): Whether to plot the simulation results. Defaults to True.
            plot_mean_data (bool, optional): Whether to plot the mean data. Defaults to True.
            df_simulation_result (pd.DataFrame, optional): The simulation results dataframe. Defaults to None.
            df_index_parameter_gof (pd.DataFrame, optional): The index parameter goodness-of-fit dataframe. Defaults to None.
            allow_conditioning_results_based_on_metric (bool, optional): Whether to allow conditioning results based on a metric. Defaults to True.
            directory_for_saving_plots (str or pathlib.Path, optional): The directory to save the plots. Defaults to None.
            fileName (str, optional): The file name for the plot. Defaults to None.
            title (str, optional): The title of the plot. Defaults to None.
            **kwargs: Additional keyword arguments to pass to other functions (i.e., merge_df_statistics_data_with_measured_and_forcing_data
                get_info_for_plotting_forcing_data, and plot_forcing_data functions)
                - condition_results_based_on_metric (str, optional): The metric to condition the results on. Default is None. 
                    relevant only if allow_conditioning_results_based_on_metric is True.
                - condition_results_based_on_metric_value (float, optional): The value of the metric to condition the results on. Default is None.
                                relevant only if allow_conditioning_results_based_on_metric is True.
                - condition_results_based_on_metric_sign (str, optional): The sign of the metric to condition the results on. Default is None. 
                                relevant only if allow_conditioning_results_based_on_metric is True. 
        
        Returns:
            fig: The plotly figure object.
        """
        df_statistics_and_measured = self.merge_df_statistics_data_with_measured_and_forcing_data(
            add_measured_data=plot_measured, add_forcing_data=plot_forcing, **kwargs)
        # filter only relevant qoi
        df_statistics_and_measured_single_qoi = df_statistics_and_measured[df_statistics_and_measured["qoi"]==single_qoi]

        if plot_forcing:
            get_info_for_plotting_forcing_data = self.get_info_for_plotting_forcing_data(**kwargs)
            n_rows = get_info_for_plotting_forcing_data.get("n_rows", 0)
            subplot_titles = get_info_for_plotting_forcing_data.get("subplot_titles", [])
            # number_rows_forcing = n_rows
            # subplot_titles_forcing = subplot_titles
        else:
            n_rows = 0
            subplot_titles = []
        n_rows = n_rows + 1
        if plot_measured:
            subplot_titles = subplot_titles + [f"Measured & Simulated {single_qoi}"]
        else:
            subplot_titles = subplot_titles + [f"Simulated {single_qoi}"]
        
        fig = make_subplots(
            rows=n_rows, cols=1,
            subplot_titles=subplot_titles,
            shared_xaxes=False,
            vertical_spacing=0.04
        )

        if plot_forcing:
            self.plot_forcing_data(df=df_statistics_and_measured_single_qoi, fig=fig, add_to_subplot=True, n_rows=1, n_col=1, **kwargs)

        if plot_df_simulation_result:
            condition_results_based_on_metric = kwargs.get("condition_results_based_on_metric", None)
            condition_results_based_on_metric_value = kwargs.get("condition_results_based_on_metric_value", None)
            condition_results_based_on_metric_sign = kwargs.get("condition_results_based_on_metric_sign", None)
            self.plot_conditioned_simulation_runs_single_qoi(
                single_qoi, df_simulation_result, fig=fig, add_to_subplot=True, n_rows=n_rows, n_col=1, 
                df_index_parameter_gof=df_index_parameter_gof,
                allow_conditioning_results_based_on_metric=allow_conditioning_results_based_on_metric, 
                condition_results_based_on_metric=condition_results_based_on_metric,
                condition_results_based_on_metric_value=condition_results_based_on_metric_value,
                condition_results_based_on_metric_sign=condition_results_based_on_metric_sign
                )

        if plot_measured:
            self.plot_measured_data_single_qoi(single_qoi=single_qoi, fig=fig, add_to_subplot=True, n_rows=n_rows, n_col=1)

        if plot_mean_data:
            self.plot_mean_data_single_qoi(single_qoi=single_qoi, fig=fig, add_to_subplot=True, n_rows=n_rows, n_col=1)

        fig.update_layout(
            xaxis=dict(
                rangemode='normal',
                range=[self.timesteps_min, self.timesteps_max],
                type="date"
            ),
            yaxis=dict(
                rangemode='normal',  # Ensures the range is not padded for markers
                autorange=True       # Auto-range is enabled
            )
        )

        if title is None:
            title = f"Runs, measured and forcing data ({single_qoi})"
        fig.update_layout(height=1100, width=1100)
        fig.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
                title=title,
                showlegend=True,
                # template="plotly_white",
            )

        if directory_for_saving_plots is None:
            directory_for_saving_plots = self.workingDir
        if not str(directory_for_saving_plots).endswith("/"):
            directory_for_saving_plots = str(directory_for_saving_plots) + "/"
        # directory_for_saving_plots = pathlib.Path(directory_for_saving_plots)
        if os.path.isdir(str(directory_for_saving_plots)):
            if fileName is None:
                fileName = f"measured_forcing_and_model_runs_plot_{single_qoi}.html"
            fileName = str(directory_for_saving_plots) + fileName        
            pyo.plot(fig, filename=fileName)
            
        # fig.show()
        return fig

    def plot_heatmap_si_single_qoi(self, qoi_column, si_df=None, si_type="Sobol_t"):
        """
        Plots a heatmap of sensitivity indices for a single quantity of interest (QoI).

        Args:
            qoi_column (str): The column name of the quantity of interest.
            si_df (pandas.DataFrame, optional): The sensitivity indices DataFrame. If not provided, it will be created using
                the `create_df_from_sensitivity_indices_single_qoi` method.
            si_type (str, optional): The type of sensitivity indices to use. Defaults to "Sobol_t".

        Returns:
            plotly.graph_objects.Figure: The heatmap figure.

        Raises:
            None

        """
        if si_df is None:
            si_df = self.create_df_from_sensitivity_indices_single_qoi(qoi_column, si_type)

        if si_df is None:
            print(f"Error in plot_heatmap_si_single_qoi - {si_type} is probably not computed for {qoi_column}")
            return None

        reset_index_at_the_end = False
        if si_df.index.name != self.time_column_name:
            si_df.set_index(self.time_column_name, inplace=True)
            reset_index_at_the_end = True

        si_columns_to_plot = [x for x in si_df.columns.tolist() if x != 'measured' \
                              and x != 'measured_norm' and x != 'qoi']
        
        si_columns_to_label = [single_column.split("_", 2)[2] for single_column in si_columns_to_plot]

        if 'qoi' in si_df.columns.tolist():
            fig = px.imshow(si_df.loc[si_df['qoi'] == qoi_column][si_columns_to_plot].T,
                            y=si_columns_to_label,
                            labels=dict(y='Parameters', x='Dates'))
        else:
            fig = px.imshow(si_df[si_columns_to_plot].T,
                            y=si_columns_to_label,
                            labels=dict(y='Parameters', x='Dates'))

        if reset_index_at_the_end:
            si_df.reset_index(inplace=True)
            si_df.rename(columns={si_df.index.name: self.time_column_name}, inplace=True)

        fig.update_xaxes(
            tickformat='%b %y',            # Format dates as "Month Day" (e.g., "Jan 01")
            dtick="M1"                     # Set tick interval to 1 day for denser ticks
        )

        return fig

    def plot_si_indices_over_time_single_qoi(self, qoi_column, si_type="Sobol_t"):
        """
        Plots the sensitivity indices over time for a single quantity of interest (QoI).

        Args:
            qoi_column (str): The column name of the quantity of interest.
            si_type (str, optional): The type of sensitivity index to plot. Defaults to "Sobol_t".

        Returns:
            go.Figure: The plot figure object.
        """
        fig = go.Figure()
        keyIter = list(self.pdTimesteps)  # self.timesteps (?)
        for i in range(len(self.labels)):
            try:
                fig.add_trace(
                    go.Scatter(
                        x=self.pdTimesteps, y=[self.result_dict[qoi_column][key][si_type][i] for key in keyIter],
                        name=self.labels[i], legendgroup=self.labels[i], line_color=colors.COLORS[i]),
                        mode='lines'
                        ) #self.timesteps (?)
            except KeyError as e:
                print(f"Error in plot_si_indices_over_time_single_qoi - "
                      f"StatisticsObject.result_dict has not key {qoi_column}")
                raise
        fig.update_xaxes(
            tickformat='%b %y',            # Format dates as "Month Day" (e.g., "Jan 01")
            dtick="M1"                     # Set tick interval to 1 day for denser ticks
        )
        return fig

    # =================================================================================================
    # Set of functions which require some measured/observed data
    # =================================================================================================

    def compare_mean_time_series_and_measured(self):
        # TODO Finish this
        raise NotImplementedError

    def compute_gof_over_different_time_series_single_qoi(self, objective_function=None, qoi_column="Q",
                                                          measuredDF_column_names="measured"):

        self._check_if_df_statistics_is_computed(recompute_if_not=True)
        if objective_function is None:
            objective_function = self.objective_function
        # TODO move compute_gof_over_different_time_series to utility; when importing uqef_dynamic_utils these is a circular import
        # uqef_dynamic_utils.compute_gof_over_different_time_series(df_statistics=self.df_statistics,
        #                                                         objective_function=objective_function,
        #                                                         qoi_column=qoi_column,
        #                                                         measuredDF_column_names=measuredDF_column_names)

    def calculate_p_factor_single_qoi(self, qoi_column, df_statistics=None,
                           column_lower_uncertainty_bound="P10", column_upper_uncertainty_bound="P90",
                           observed_column="measured"):

        if df_statistics is None:
            df_statistics = self.create_df_from_statistics_data_single_qoi(qoi_column)

        if observed_column not in df_statistics.columns or df_statistics[observed_column].isnull().all():
            raise Exception(f"{observed_column} is missing from df_statistics!")

        df_statistics_subset = df_statistics.loc[df_statistics["qoi"] == qoi_column]

        condition = df_statistics_subset[
            (df_statistics_subset[observed_column] >= df_statistics_subset[column_lower_uncertainty_bound]) & (
                    df_statistics_subset[observed_column] <= df_statistics_subset[column_upper_uncertainty_bound])]

        p = len(condition.index) / len(df_statistics_subset.index)
        print(f"P factor for QoI-{qoi_column} is: {p * 100}%")
        return p

    def compute_stat_of_uncertainty_band(self, qoi_column, df_statistics=None,
                                         column_lower_uncertainty_bound="P10", column_upper_uncertainty_bound="P90",
                                         observed_column="measured"):
        if df_statistics is None:
            df_statistics = self.create_df_from_statistics_data_single_qoi(qoi_column)

        df_statistics_subset = df_statistics.loc[df_statistics["qoi"] == qoi_column]

        mean_uncertainty_band = np.mean(
            df_statistics_subset[column_upper_uncertainty_bound] - df_statistics_subset[
                column_lower_uncertainty_bound])
        std_uncertainty_band = np.std(
            df_statistics_subset[column_upper_uncertainty_bound] - df_statistics_subset[
                column_lower_uncertainty_bound])
        print(f"mean_uncertainty_band for QoI-{qoi_column} is: {mean_uncertainty_band} \n")
        print(f"std_uncertainty_band for QoI-{qoi_column} is: {std_uncertainty_band} \n")

        mean_observed = None
        std_observed = None

        if observed_column in df_statistics_subset.columns and not df_statistics_subset[observed_column].isnull().all():
            mean_observed = df_statistics_subset[observed_column].mean()
            std_observed = df_statistics_subset[observed_column].std(ddof=1)
            print(f"mean_observed for QoI-{qoi_column} is: {mean_observed} \n")
            print(f"std_observed for QoI-{qoi_column} is: {std_observed} \n")

        return mean_uncertainty_band, std_uncertainty_band, mean_observed, std_observed

    def plot_si_and_normalized_measured_time_signal_single_qoi(
            self, qoi_column, si_df=None, si_type="Sobol_t",
            observed_column_normalized="measured_norm", plot_forcing_data=False):

        if si_df is None:
            si_df = self.create_df_from_sensitivity_indices_single_qoi(
                qoi_column, si_type, compute_measured_normalized_data=True
            )

        if si_df is None:
            print(f"Error in plot_si_and_normalized_measured_time_signal_single_qoi - {si_type} is probably not computed for {qoi_column}")
            return None

        if 'qoi' in si_df.columns.tolist():
            si_df = si_df.loc[si_df['qoi'] == qoi_column]

        reset_index_at_the_end = False
        if si_df.index.name != self.time_column_name:
            si_df.set_index(self.time_column_name, inplace=True)
            reset_index_at_the_end = True

        si_columns_to_plot = [x for x in si_df.columns.tolist() if x != 'measured' \
                              and x != 'measured_norm' and x != 'qoi']

        si_columns_to_label = [single_column.split("_", 2)[2] for single_column in si_columns_to_plot]
        # fig = px.line(
        #     si_df, x=si_df.index, y=si_columns_to_plot) #mode='lines+markers' markers=True
        fig = go.Figure()
        for i, single_column in enumerate(si_columns_to_plot):
            fig.add_trace(
                go.Scatter(
                    x=si_df.index, y=si_df[single_column],
                    name=si_columns_to_label[i], line_color=colors.COLORS[i],
                    text=si_columns_to_label[i], mode='lines'
                )
            ) #self.timesteps (?)

        if observed_column_normalized in si_df.columns.tolist() and not si_df[observed_column_normalized].isnull().all():
            fig.add_trace(go.Scatter(
                x=si_df.index, y=si_df[observed_column_normalized], fill='tozeroy',
                name=f"Normalized {self.dict_corresponding_original_qoi_column[qoi_column]}")
            )

        if reset_index_at_the_end:
            si_df.reset_index(inplace=True)
            si_df.rename(columns={si_df.index.name: self.time_column_name}, inplace=True)

        # if plot_forcing_data:
        #     pass
        fig.update_xaxes(
            tickformat='%b %y',            # Format dates as "Month Day" (e.g., "Jan 01")
            dtick="M1"                     # Set tick interval to 1 day for denser ticks
        )
        return fig
    
    # =================================================================================================

    @staticmethod
    def _single_qoi_single_param_grad_analysis(df, qoi_column, time_column_name="TimeStamp"):

        if df is None:
            raise Exception()

        grouped = df.groupby([time_column_name, ])
        groups = grouped.groups
        keyIter = list(groups.keys())
        result_dict_time_aggregated = dict()
        result_dict_time_aggregated["qoi_column"] = qoi_column

        # all_list_of_values = [
        #             df.loc[groups[key].values][qoi_column].values
        #             for key in keyIter
        #         ]
        # all_list_of_values = df[qoi_column].values
        all_list_of_values_abs = np.abs(df[qoi_column].values)
        # Calculate relative frequencies


        count, bins_count = np.histogram(all_list_of_values_abs, bins=len(all_list_of_values_abs))
        # finding the PDF of the histogram using count values
        pdf = count / sum(count)
        # using numpy np.cumsum to calculate the CDF
        cdf = np.cumsum(pdf)
        mean_cdf = np.mean(cdf)

        # x, counts = np.unique(all_list_of_values_abs, return_counts=True)
        # cusum = np.cumsum(counts) # cusum / cusum[-1]

        result_dict_time_aggregated["total_time_relative_frequency_np"] = pdf
        result_dict_time_aggregated["total_time_cumulative_sum_np"] = cdf
        result_dict_time_aggregated["total_time_mean_cumulative_sum_np"] = mean_cdf

        pdf = stats.relfreq(all_list_of_values_abs, numbins=len(all_list_of_values_abs))
        cdf = stats.cumfreq(all_list_of_values_abs, numbins=len(all_list_of_values_abs))
        mean_cdf = np.mean(cdf)
        result_dict_time_aggregated["total_time_relative_frequency"] = pdf
        result_dict_time_aggregated["total_time_cumulative_sum"] = cdf
        result_dict_time_aggregated["total_time_mean_cumulative_sum"] = mean_cdf

        # result_dict_over_time = defaultdict(list)

        result_dict_over_time = defaultdict(list)
        for key in keyIter:
            # single_timestep_grad_values = df.loc[groups[key].values][qoi_column].values
            single_timestep_abs_grad_values = np.abs(
                df.loc[groups[key].values][qoi_column].values)
            mean_cdf = np.mean(
                stats.cumfreq(single_timestep_abs_grad_values, numbins=len(single_timestep_abs_grad_values)))
            result_dict_over_time[time_column_name].append(key)
            result_dict_over_time[qoi_column].append(mean_cdf)

        result_df_over_time = pd.DataFrame.from_dict(result_dict_over_time)
        result_df_over_time.set_index(time_column_name, inplace=True)

        return result_dict_time_aggregated, result_df_over_time

    @staticmethod
    def _compute_active_score(dict_of_matrix_c_eigen_decomposition):
        dict_of_active_scores = dict()
        if dict_of_matrix_c_eigen_decomposition is not None:
            for key in dict_of_matrix_c_eigen_decomposition.keys():
                # w, v = np.linalg.eigh(self.samples.dict_of_approx_matrix_c[key])
                w, v = dict_of_matrix_c_eigen_decomposition[key]
                scores_vect = []
                for i in range(len(w)):
                    # temp = np.dot(w, np.power(v[i,:], 2))
                    temp = 0
                    for j in range(v.shape[1]):
                        temp += w[j] * v[i, j] ** 2
                    scores_vect.append(temp)
                dict_of_active_scores[key] = scores_vect
        return dict_of_active_scores

