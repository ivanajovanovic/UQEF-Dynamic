import chaospy as cp
from collections import defaultdict
from distutils.util import strtobool
from functools import reduce
import json
import itertools
import matplotlib.pyplot as plotter
import more_itertools
from mpi4py import MPI
import mpi4py.futures as futures
import numpy as np
import os
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import pathlib
import pickle
from plotly.offline import iplot, plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy import stats
import sys
import time

from uqef.stat import Statistics

from common import saltelliSobolIndicesHelpingFunctions
from common import parallelStatistics
from common import colors

from common import utility

# TODO Differe btw two cases - time_column_name is or not an index column in returned data!?
class Samples(object):
    def __init__(self, rawSamples, qoi_columns="Value", time_column_name="TimeStamp", **kwargs):
        """

        :param rawSamples:
        :param qoi_columns: should be a list or a string containing column names from the result DF,
        :param time_column_name:
        :param kwargs:
        """
        self.time_column_name = time_column_name
        original_model_output_column = kwargs.get('original_model_output_column', "Value")
        qoi_is_a_single_number = kwargs.get('qoi_is_a_single_number', False)
        grad_columns = kwargs.get('grad_columns', [])

        if not isinstance(qoi_columns, list):
            qoi_columns = [qoi_columns, ]

        qoi_columns = qoi_columns + [time_column_name, "Index_run"]

        if grad_columns:
            qoi_columns = qoi_columns + grad_columns

        extract_only_qoi_columns = kwargs.get('extract_only_qoi_columns', False)

        list_of_single_df = []
        list_index_parameters_dict = []
        list_of_single_index_parameter_gof_df = []
        list_of_gradient_matrix_dict = []
        for index_run, value in enumerate(rawSamples, ):
            if value is None:
                # TODO write in some log file runs which have returned None, in case of sc break!
                continue
            if isinstance(value, tuple):
                df_result = value[0]
                list_index_parameters_dict.append(value[1])
            elif isinstance(value, dict):
                if "result_time_series" in value:
                    df_result = value["result_time_series"]
                else:
                    df_result = None
                if "parameters_dict" in value:
                    list_index_parameters_dict.append(value["parameters_dict"])
                if "gof_df" in value:
                    list_of_single_index_parameter_gof_df.append(value["gof_df"])
                if "grad_matrix" in value:
                    gradient_matrix_dict = value["grad_matrix"]
                    if gradient_matrix_dict is not None:
                        # TODO Extract only entry for station and one or multiple gofs
                        list_of_gradient_matrix_dict.append(gradient_matrix_dict)
            else:
                df_result = value

            # logic in Statistics is opposite the one in a Model, e.g.,
            # it is assumed that time_column is not an index in DFs
            if isinstance(df_result, pd.DataFrame) and df_result.index.name == time_column_name:
                df_result = df_result.reset_index()
                df_result.rename(columns={df_result.index.name: time_column_name}, inplace=True)

            if time_column_name not in list(df_result):
                raise Exception(f"Error in Samples class - {time_column_name} is not in the "
                                f"columns of the result DataFrame")

            if df_result is not None:
                if extract_only_qoi_columns:
                    list_of_single_df.append(df_result[qoi_columns])
                else:
                    list_of_single_df.append(df_result)

        # TODO Add 'qoi' column
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

        # TODO Move this outside Sample class!?
        # In case compute_gradients mode was turned on and compute_active_subspaces set to True
        if list_of_gradient_matrix_dict:
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
        else:
            self.dict_of_approx_matrix_c = None
            self.dict_of_matrix_c_eigen_decomposition = None

    def save_samples_to_file(self, file_path='./'):
        file_path = str(file_path)
        if self.df_simulation_result is not None:
            self.df_simulation_result.to_pickle(
                os.path.abspath(os.path.join(file_path, "df_all_simulations.pkl")), compression="gzip")

    def save_index_parameter_values(self, file_path='./'):
        file_path = str(file_path)
        if self.df_index_parameter_values is not None:
            self.df_index_parameter_values.to_pickle(
                os.path.abspath(os.path.join(file_path, "df_all_index_parameter_values.pkl")), compression="gzip")

    def save_index_parameter_gof_values(self, file_path='./'):
        file_path = str(file_path)
        if self.df_index_parameter_gof_values is not None:
            self.df_index_parameter_gof_values.to_pickle(
                os.path.abspath(os.path.join(file_path, "df_all_index_parameter_gof_values.pkl")), compression="gzip")

    def save_dict_of_approx_matrix_c(self, file_path='./'):
        file_path = str(file_path)
        if self.dict_of_matrix_c_eigen_decomposition is not None:
            fileName = os.path.abspath(os.path.join(file_path, "dict_of_approx_matrix_c.pkl"))
            with open(fileName, 'wb') as handle:
                pickle.dump(self.dict_of_approx_matrix_c, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_dict_of_matrix_c_eigen_decomposition(self, file_path='./'):
        file_path = str(file_path)
        if self.dict_of_matrix_c_eigen_decomposition is not None:
            fileName = os.path.abspath(os.path.join(file_path, "dict_of_matrix_c_eigen_decomposition.pkl"))
            with open(fileName, 'wb') as handle:
                pickle.dump(self.dict_of_matrix_c_eigen_decomposition, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_number_of_runs(self, index_run_column_name="Index_run"):
        return self.df_simulation_result[index_run_column_name].nunique()

    def get_simulation_timesteps(self, time_stamp_column="TimeStamp"):
        if self.df_simulation_result is not None:
            return list(self.df_simulation_result[time_stamp_column].unique())
        else:
            return None

    def get_timesteps_min(self, time_stamp_column="TimeStamp"):
        if self.df_simulation_result is not None:
            return self.df_simulation_result[time_stamp_column].min()
        else:
            return None

    def get_timesteps_max(self, time_stamp_column="TimeStamp"):
        if self.df_simulation_result is not None:
            return self.df_simulation_result[time_stamp_column].max()
        else:
            return None


class HydroStatistics(Statistics):
    def __init__(self, configurationObject, workingDir=None, *args, **kwargs):
        Statistics.__init__(self)

        self.configurationObject = utility.check_if_configurationObject_is_in_right_format_and_return(
            configurationObject, raise_error=True)

        # print(f"[DEBUGGING STAT INFO] - {self.configurationObject}")

        # in the statistics class specification of the workingDir is necessary
        self.workingDir = pathlib.Path(workingDir)

        #####################################
        # Set of configuration variables propagated via UQsim.args and/or **kwargs
        # These are mainly UQ simulation - related configurations
        #####################################
        self.uq_method = kwargs.get('uq_method', None)

        self.sampleFromStandardDist = kwargs.get('sampleFromStandardDist', False)

        # whether to store all original model outputs in the stat dict; note - this might take a lot of space
        self.store_qoi_data_in_stat_dict = kwargs.get('store_qoi_data_in_stat_dict', False)
        self.store_gpce_surrogate = kwargs.get('store_gpce_surrogate', False)
        self.save_gpce_surrogate = kwargs.get('save_gpce_surrogate', False)

        # TODO: eventually make a non-MPI version
        self.parallel_statistics = kwargs.get('parallel_statistics', False)
        if self.parallel_statistics:
            self.size = MPI.COMM_WORLD.Get_size()
            self.rank = MPI.COMM_WORLD.Get_rank()
            self.name = MPI.Get_processor_name()
            self.version = MPI.Get_library_version()
            self.mpi_chunksize = kwargs.get('mpi_chunksize', 1)
            self.unordered = kwargs.get('unordered', False)

        self.save_samples = kwargs.get('save_samples', True)

        self._compute_Sobol_t = kwargs.get('compute_Sobol_t', True)
        self._compute_Sobol_m = kwargs.get('compute_Sobol_m', True)
        self._compute_Sobol_m2 = kwargs.get('compute_Sobol_m2', False)

        #####################################
        # Set of configuration variables propagated via **kwargs or read from configurationObject
        # These are mainly model related configurations
        #####################################
        self.time_column_name = kwargs.get("time_column_name", "TimeStamp")
        self.forcing_data_column_names = kwargs.get("forcing_data_column_names", "precipitation")

        if "corrupt_forcing_data" in kwargs:
            self.corrupt_forcing_data = kwargs['corrupt_forcing_data']
        else:
            self.corrupt_forcing_data = strtobool(self.configurationObject["model_settings"].get(
                "corrupt_forcing_data", False))
        #####################################
        # Parameters related set-up part
        #####################################
        self.nodeNames = []
        try:
            list_of_parameters = self.configurationObject["parameters"]
        except KeyError as e:
            print(f"Statistics: parameters key does "
                  f"not exists in the configurationObject{e}")
            raise
        for i in list_of_parameters:
            if self.uq_method == "ensemble" or i["distribution"] != "None":
                self.nodeNames.append(i["name"])
        self.dim = len(self.nodeNames)
        self.labels = [nodeName.strip() for nodeName in self.nodeNames]

        #####################################
        # Initialize different variables of the Statistics class
        #####################################
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
        self.numbTimesteps = None
        self.pdTimesteps = None
        self.timestep_qoi = None
        self.number_of_unique_index_runs = None
        self.numEvaluations = None
        self.samples = None
        # self.result_dict = dict()
        self.result_dict = None

        self.qoi_mean_df = None
        self.gof_mean_measured = None

        self.active_scores_dict = None
        self.df_time_varying_grad_analysis = None
        self.df_time_aggregated_grad_analysis = None
        self.solverTimes = None
        self.work_package_indexes = None

        ##################################
        self.dict_processed_simulation_settings_from_config_file = \
            utility.read_simulation_settings_from_configuration_object(self.configurationObject, **kwargs)

        self.qoi = self.dict_processed_simulation_settings_from_config_file["qoi"]
        self.qoi_column = self.dict_processed_simulation_settings_from_config_file["qoi_column"]
        self.transform_model_output = self.dict_processed_simulation_settings_from_config_file["transform_model_output"]
        self.multiple_qoi = self.dict_processed_simulation_settings_from_config_file["multiple_qoi"]
        self.number_of_qois = self.dict_processed_simulation_settings_from_config_file["number_of_qois"]
        self.qoi_column_measured = self.dict_processed_simulation_settings_from_config_file["qoi_column_measured"]
        self.read_measured_data = self.dict_processed_simulation_settings_from_config_file["read_measured_data"]

        # self.objective_function_qoi = self.dict_processed_simulation_settings_from_config_file["objective_function_qoi"]
        # self.objective_function_names_qoi = self.dict_processed_simulation_settings_from_config_file["objective_function_names_qoi"]

        # list versions of the above variables
        self.list_qoi_column = self.dict_processed_simulation_settings_from_config_file["list_qoi_column"]
        self.list_qoi_column_measured = self.dict_processed_simulation_settings_from_config_file["list_qoi_column_measured"]
        self.list_read_measured_data = self.dict_processed_simulation_settings_from_config_file["list_read_measured_data"]
        self.list_transform_model_output = self.dict_processed_simulation_settings_from_config_file["list_transform_model_output"]

        self.dict_qoi_column_and_measured_info = self.dict_processed_simulation_settings_from_config_file[
            "dict_qoi_column_and_measured_info"]

        self.list_objective_function_qoi = self.dict_processed_simulation_settings_from_config_file["list_objective_function_qoi"]
        self.list_objective_function_names_qoi = self.dict_processed_simulation_settings_from_config_file[
            "list_objective_function_names_qoi"]

        self.mode = self.dict_processed_simulation_settings_from_config_file["mode"]
        self.method = self.dict_processed_simulation_settings_from_config_file["method"]

        self.compute_gradients = self.dict_processed_simulation_settings_from_config_file["compute_gradients"]
        self.compute_active_subspaces = self.dict_processed_simulation_settings_from_config_file["compute_active_subspaces"]
        self.save_gradient_related_runs = self.dict_processed_simulation_settings_from_config_file["save_gradient_related_runs"]
        self.gradient_analysis = self.dict_processed_simulation_settings_from_config_file["gradient_analysis"]

        self.list_original_model_output_columns = self.list_qoi_column.copy()
        self.additional_qoi_columns_besides_original_model_output = False
        self.qoi_is_a_single_number = False
        self.list_grad_columns = []

        self._infer_qoi_column_names(**kwargs)

    def _set_dict_processed_simulation_settings_from_config_file(self, dict_processed_simulation_settings_from_config_file=None, **kwargs):
        if dict_processed_simulation_settings_from_config_file is None:
            self.dict_processed_simulation_settings_from_config_file = utility.read_simulation_settings_from_configuration_object(
            self.configurationObject, **kwargs)

    def _set_attributes_based_on_dict_processed_simulation_settings_from_config_file(self, **kwargs):
        self.qoi = self.dict_processed_simulation_settings_from_config_file["qoi"]
        self.qoi_column = self.dict_processed_simulation_settings_from_config_file["qoi_column"]
        self.transform_model_output = self.dict_processed_simulation_settings_from_config_file["transform_model_output"]
        self.multiple_qoi = self.dict_processed_simulation_settings_from_config_file["multiple_qoi"]
        self.number_of_qois = self.dict_processed_simulation_settings_from_config_file["number_of_qois"]
        self.qoi_column_measured = self.dict_processed_simulation_settings_from_config_file["qoi_column_measured"]
        self.read_measured_data = self.dict_processed_simulation_settings_from_config_file["read_measured_data"]

        # self.objective_function_qoi = self.dict_processed_simulation_settings_from_config_file["objective_function_qoi"]
        # self.objective_function_names_qoi = self.dict_processed_simulation_settings_from_config_file["objective_function_names_qoi"]

        # list versions of the above variables
        self.list_qoi_column = self.dict_processed_simulation_settings_from_config_file["list_qoi_column"]
        self.list_qoi_column_measured = self.dict_processed_simulation_settings_from_config_file["list_qoi_column_measured"]
        self.list_read_measured_data = self.dict_processed_simulation_settings_from_config_file["list_read_measured_data"]
        self.list_transform_model_output = self.dict_processed_simulation_settings_from_config_file["list_transform_model_output"]

        self.dict_qoi_column_and_measured_info = self.dict_processed_simulation_settings_from_config_file[
            "dict_qoi_column_and_measured_info"]

        self.list_objective_function_qoi = self.dict_processed_simulation_settings_from_config_file["list_objective_function_qoi"]
        self.list_objective_function_names_qoi = self.dict_processed_simulation_settings_from_config_file[
            "list_objective_function_names_qoi"]

        self.mode = self.dict_processed_simulation_settings_from_config_file["mode"]
        self.method = self.dict_processed_simulation_settings_from_config_file["method"]

        self.compute_gradients = self.dict_processed_simulation_settings_from_config_file["compute_gradients"]
        self.compute_active_subspaces = self.dict_processed_simulation_settings_from_config_file["compute_active_subspaces"]
        self.save_gradient_related_runs = self.dict_processed_simulation_settings_from_config_file["save_gradient_related_runs"]
        self.gradient_analysis = self.dict_processed_simulation_settings_from_config_file["gradient_analysis"]

        self.list_original_model_output_columns = self.list_qoi_column.copy()
        self.additional_qoi_columns_besides_original_model_output = False
        self.qoi_is_a_single_number = False
        self.list_grad_columns = []

        self._infer_qoi_column_names(**kwargs)

    def _infer_qoi_column_names(self, **kwargs):
        # TODO Is this redundant with self.store_qoi_data_in_stat_dict
        always_process_original_model_output = kwargs.get("always_process_original_model_output", False)
        list_qoi_column_processed = []

        if self.mode == "continuous":
            if self.qoi == "GoF":
                for idx, single_qoi_column in enumerate(self.list_qoi_column):
                    if self.list_read_measured_data[idx]:
                        for single_objective_function_name_qoi in self.list_objective_function_names_qoi:
                            new_column_name = single_objective_function_name_qoi + "_" + single_qoi_column
                            list_qoi_column_processed.append(new_column_name)
                            self.additional_qoi_columns_besides_original_model_output = True
                            self.qoi_is_a_single_number = True
                            # TODO in this case QoI is a single number, not a time-series!!!
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
                            self.additional_qoi_columns_besides_original_model_output = True
            else:
                for idx, single_qoi_column in enumerate(self.list_qoi_column):
                    new_column_name = single_qoi_column + "_" + self.method + "_sliding_window"
                    list_qoi_column_processed.append(new_column_name)
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
            else:
                self.list_qoi_column = list_qoi_column_processed

        # print(f"[STAT INFO] Statistics class will process the following QoIs:\n {self.list_qoi_column}\n"
        #       f"whereas the columns representing the model output itself are:{self.list_original_model_output_columns}"
        #       f"and additional QoI columns are { list_qoi_column_processed}. "
        #       f"Plus the following gradient columns exist in the result DF {self.list_grad_columns}")

    ###################################################################################################################

    def prepare(self, rawSamples, **kwargs):
        self.timesteps = kwargs.get('timesteps', None)
        self.solverTimes = kwargs.get('solverTimes', None)
        self.work_package_indexes = kwargs.get('work_package_indexes', None)
        self.extract_only_qoi_columns = kwargs.get('extract_only_qoi_columns', False)

        # TODO a couple of similar/redundant variables
        # self.store_qoi_data_in_stat_dict, self.extract_only_qoi_columns always_process_original_model_output

        list_of_columns_to_filter_from_results = self.list_qoi_column.copy()
        if self.corrupt_forcing_data:
            list_of_columns_to_filter_from_results = self.list_qoi_column + [self.forcing_data_column_names]

        self.samples = Samples(rawSamples, qoi_column=list_of_columns_to_filter_from_results,
                               time_column_name=self.time_column_name,
                               extract_only_qoi_columns=self.extract_only_qoi_columns,
                               original_model_output_column=self.list_original_model_output_columns,
                               qoi_is_a_single_number=self.qoi_is_a_single_number,
                               grad_columns=self.list_grad_columns
                               )

        if self.samples.df_simulation_result is not None:
            self.samples.df_simulation_result.sort_values(
                by=["Index_run", self.time_column_name], ascending=[True, True], inplace=True, kind='quicksort',
                na_position='last'
            )

        if self.compute_gradients and self.compute_active_subspaces:
            self.active_scores_dict = HydroStatistics._compute_active_score(self.samples.dict_of_matrix_c_eigen_decomposition)

        if self.compute_gradients and self.gradient_analysis:
            self.param_grad_analysis()

        if self.save_samples:
            self.samples.save_samples_to_file(self.workingDir)
            self.samples.save_index_parameter_values(self.workingDir)
            self.samples.save_index_parameter_gof_values(self.workingDir)
            if self.compute_gradients and self.compute_active_subspaces:
                self.samples.save_dict_of_approx_matrix_c(self.workingDir)
                self.samples.save_dict_of_matrix_c_eigen_decomposition(self.workingDir)

        # Read info about the time from propagated model runs, i.e., samples
        self.timesteps = self.samples.get_simulation_timesteps()
        if self.qoi_is_a_single_number:
            if self.timesteps is not None and isinstance(self.timesteps, list):
                # TODO take a middle element
                self.timestep_qoi = self.timesteps[-1]
        else:
            self.timestep_qoi = self.timesteps
        self.timesteps_min = self.samples.get_timesteps_min()
        self.timesteps_max = self.samples.get_timesteps_max()
        self.pdTimesteps = [pd.Timestamp(timestep) for timestep in self.timesteps]
        self.number_of_unique_index_runs = self.samples.get_number_of_runs()
        self.numEvaluations = self.number_of_unique_index_runs

        self.numbTimesteps = len(self.timesteps)

    ###################################################################################################################
    # TODO Write getters and setters!

    def set_timesteps(self, timesteps=None):
        if timesteps is not None:
            self.timesteps = timesteps
        elif self.samples is not None:
            self.timesteps = self.samples.get_simulation_timesteps()
        self.pdTimesteps = [pd.Timestamp(timestep) for timestep in self.timesteps]

    def set_pdTimesteps(self, pdTimesteps):
        self.pdTimesteps = pdTimesteps

    def set_timesteps_min(self, timesteps_min):
        if timesteps_min is not None:
            self.timesteps_min = timesteps_min
        elif self.samples is not None:
            self.timesteps_min = self.samples.get_timesteps_min()

    def set_timesteps_max(self, timesteps_max):
        if timesteps_max is not None:
            self.timesteps_max = timesteps_max
        elif self.samples is not None:
            self.timesteps_max = self.samples.get_timesteps_max()

    def set_result_dict(self, result_dict):
        self.result_dict = result_dict

    def set_number_of_unique_index_runs(self, number_of_unique_index_runs):
        self.number_of_unique_index_runs = number_of_unique_index_runs
        self.numEvaluations = self.number_of_unique_index_runs

    def set_numbTimesteps(self, numbTimesteps):
        self.numbTimesteps = numbTimesteps
    ###################################################################################################################

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
                    HydroStatistics._single_qoi_single_param_grad_analysis(
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
    ###################################################################################################################

    def preparePolyExpanForMc(self, simulationNodes, numEvaluations, regression=None, order=None,
                              poly_normed=None, poly_rule=None, *args, **kwargs):
        self.numEvaluations = numEvaluations
        # TODO Think about this, tricky for saltelli, makes sense for mc
        # self.numEvaluations = self.number_of_unique_index_runs
        if regression:
            self.nodes = simulationNodes.distNodes
            self.weights = None
            if self.sampleFromStandardDist:
                self.dist = simulationNodes.joinedStandardDists
            else:
                self.dist = simulationNodes.joinedDists
            self.polynomial_expansion = cp.generate_expansion(order, self.dist, rule=poly_rule, normed=poly_normed)

    def preparePolyExpanForSc(self, simulationNodes, order, poly_normed, poly_rule, *args, **kwargs):
        self.nodes = simulationNodes.distNodes
        if self.sampleFromStandardDist:
            self.dist = simulationNodes.joinedStandardDists
        else:
            self.dist = simulationNodes.joinedDists
        self.weights = simulationNodes.weights
        self.polynomial_expansion = cp.generate_expansion(order, self.dist, rule=poly_rule, normed=poly_normed)

    def preparePolyExpanForSaltelli(self, simulationNodes, numEvaluations=None, regression=None, order=None,
                                    poly_normed=None, poly_rule=None, *args, **kwargs):
        self.preparePolyExpanForMc(simulationNodes, numEvaluations, regression, order, poly_normed, poly_rule,
                                   *args, **kwargs)

    ###################################################################################################################

    def calcStatisticsForMcParallel(self, chunksize=1, regression=False, *args, **kwargs):
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

                regressionChunks = [regression] * len(keyIter_chunk)
                compute_Sobol_t_Chunks = [self._compute_Sobol_t] * len(keyIter_chunk)
                compute_Sobol_m_Chunks = [self._compute_Sobol_m] * len(keyIter_chunk)
                store_qoi_data_in_stat_dict_Chunks = [self.store_qoi_data_in_stat_dict] * len(keyIter_chunk)

                if regression:
                    nodesChunks = [self.nodes] * len(keyIter_chunk)
                    weightsChunks = [self.weights] * len(keyIter_chunk)
                    distChunks = [self.dist] * len(keyIter_chunk)
                    dimChunks = [self.dim] * len(keyIter_chunk)
                    polynomial_expansionChunks = [self.polynomial_expansion] * len(keyIter_chunk)

            with futures.MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
                if executor is not None:  # master process
                    print(f"{self.rank}: computation of statistics for qoi {single_qoi_column} started...")
                    solver_time_start = time.time()
                    if regression:
                        chunk_results_it = executor.map(parallelStatistics._my_parallel_calc_stats_for_gPCE,
                                                        keyIter_chunk,
                                                        list_of_simulations_df_chunk,
                                                        distChunks,
                                                        polynomial_expansionChunks,
                                                        nodesChunks,
                                                        weightsChunks,
                                                        regressionChunks,
                                                        compute_Sobol_t_Chunks,
                                                        compute_Sobol_m_Chunks,
                                                        store_qoi_data_in_stat_dict_Chunks,
                                                        chunksize=self.mpi_chunksize,
                                                        unordered=self.unordered)
                    else:
                        chunk_results_it = executor.map(parallelStatistics._my_parallel_calc_stats_for_MC,
                                                        keyIter_chunk,
                                                        list_of_simulations_df_chunk,
                                                        numEvaluations_chunk,
                                                        store_qoi_data_in_stat_dict_Chunks,
                                                        chunksize=self.mpi_chunksize,
                                                        unordered=self.unordered)
                    print(f"{self.rank}: computation for qoi {single_qoi_column} - waits for shutdown...")
                    sys.stdout.flush()
                    executor.shutdown(wait=True)
                    print(f"{self.rank}: computation for qoi {single_qoi_column} - shut down...")
                    sys.stdout.flush()

                    solver_time_end = time.time()
                    solver_time = solver_time_end - solver_time_start
                    print(f"solver_time for qoi {single_qoi_column}: {solver_time}")

                    chunk_results = list(chunk_results_it)
                    for chunk_result in chunk_results:
                        for result in chunk_result:
                            self.result_dict[single_qoi_column][result[0]] = result[1]

                    # Saving Stat Dict as soon as it is computed
                    # TODO - current problem of saving at this stage -
                    #  single_qoi_column info is missed once saved dict is read
                    fileName = "statistics_dictionary_qoi_" + single_qoi_column + ".pkl"
                    fullFileName = os.path.abspath(os.path.join(str(self.workingDir), fileName))
                    with open(fullFileName, 'wb') as handle:
                        pickle.dump(self.result_dict[single_qoi_column], handle, protocol=pickle.HIGHEST_PROTOCOL)

                    # TODO Plotting Stat Dict as soon as it is computed


    def calcStatisticsForEnsembleParallel(self, chunksize=1, regression=False, *args, **kwargs):
        self.calcStatisticsForMcParallel(chunksize=chunksize, regression=False, *args, **kwargs)

    def calcStatisticsForSaltelliParallel(self, chunksize=1, regression=False, *args, **kwargs):
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

            with futures.MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
                if executor is not None:  # master process
                    print(f"{self.rank}: computation of statistics for qoi {single_qoi_column} started...")
                    solver_time_start = time.time()
                    chunk_results_it = executor.map(parallelStatistics._my_parallel_calc_stats_for_mc_saltelli,
                                                    keyIter_chunk,
                                                    list_of_simulations_df_chunk,
                                                    numEvaluations_chunk,
                                                    dimChunks,
                                                    compute_Sobol_t_Chunks,
                                                    compute_Sobol_m_Chunks,
                                                    store_qoi_data_in_stat_dict_Chunks,
                                                    chunksize=self.mpi_chunksize,
                                                    unordered=self.unordered)
                    print(f"{self.rank}: computation for qoi {single_qoi_column} - waits for shutdown...")
                    sys.stdout.flush()
                    executor.shutdown(wait=True)
                    print(f"{self.rank}: computation for qoi {single_qoi_column} - shut down...")
                    sys.stdout.flush()

                    solver_time_end = time.time()
                    solver_time = solver_time_end - solver_time_start
                    print(f"solver_time for qoi {single_qoi_column}: {solver_time}")

                    chunk_results = list(chunk_results_it)
                    for chunk_result in chunk_results:
                        for result in chunk_result:
                            self.result_dict[single_qoi_column][result[0]] = result[1]

                    # Saving Stat Dict as soon as it is computed
                    # TODO - current problem, single_qoi_column info is missed once saved dict is read
                    fileName = "statistics_dictionary_qoi_" + single_qoi_column + ".pkl"
                    fullFileName = os.path.abspath(os.path.join(str(self.workingDir), fileName))
                    with open(fullFileName, 'wb') as handle:
                        pickle.dump(self.result_dict[single_qoi_column], handle, protocol=pickle.HIGHEST_PROTOCOL)

                    # TODO Plotting Stat Dict as soon as it is computed

    def calcStatisticsForScParallel(self, chunksize=1, regression=False, *args, **kwargs):
        self.result_dict = defaultdict(dict)

        if self.rank == 0:
            grouped = self.samples.df_simulation_result.groupby([self.time_column_name, ])
            groups = grouped.groups

            keyIter = list(groups.keys())

        for single_qoi_column in self.list_qoi_column:
            if self.rank == 0:
                # TODO Potential Memory Problem
                list_of_simulations_df = [
                    self.samples.df_simulation_result.loc[groups[key].values][single_qoi_column].values
                    for key in keyIter
                ]

                keyIter_chunk = list(more_itertools.chunked(keyIter, chunksize))
                # TODO Potential Memory Problem
                list_of_simulations_df_chunk = list(more_itertools.chunked(list_of_simulations_df, chunksize))

                nodesChunks = [self.nodes] * len(keyIter_chunk)
                distChunks = [self.dist] * len(keyIter_chunk)
                weightsChunks = [self.weights] * len(keyIter_chunk)
                polynomial_expansionChunks = [self.polynomial_expansion] * len(keyIter_chunk)

                regressionChunks = [regression] * len(keyIter_chunk)
                compute_Sobol_t_Chunks = [self._compute_Sobol_t] * len(keyIter_chunk)
                compute_Sobol_m_Chunks = [self._compute_Sobol_m] * len(keyIter_chunk)
                store_qoi_data_in_stat_dict_Chunks = [self.store_qoi_data_in_stat_dict] * len(keyIter_chunk)
                store_gpce_surrogate_Chunks = [self.store_gpce_surrogate] * len(keyIter_chunk)
                save_gpce_surrogate_Chunks = [self.save_gpce_surrogate] * len(keyIter_chunk)

            with futures.MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
                if executor is not None:  # master process
                    print(f"{self.rank}: computation of statistics for qoi {single_qoi_column} started...")
                    solver_time_start = time.time()
                    # TODO Potential Memory Problem
                    chunk_results_it = executor.map(parallelStatistics._my_parallel_calc_stats_for_gPCE,
                                                    keyIter_chunk,
                                                    list_of_simulations_df_chunk,
                                                    distChunks,
                                                    polynomial_expansionChunks,
                                                    nodesChunks,
                                                    weightsChunks,
                                                    regressionChunks,
                                                    compute_Sobol_t_Chunks,
                                                    compute_Sobol_m_Chunks,
                                                    store_qoi_data_in_stat_dict_Chunks,
                                                    store_gpce_surrogate_Chunks,
                                                    save_gpce_surrogate_Chunks,
                                                    chunksize=self.mpi_chunksize,
                                                    unordered=self.unordered)
                    print(f"{self.rank}: computation for qoi {single_qoi_column} - waits for shutdown...")
                    sys.stdout.flush()
                    executor.shutdown(wait=True)
                    print(f"{self.rank}: computation for qoi {single_qoi_column} - shut down...")
                    sys.stdout.flush()

                    solver_time_end = time.time()
                    solver_time = solver_time_end - solver_time_start
                    print(f"solver_time for qoi {single_qoi_column}: {solver_time}")

                    chunk_results = list(chunk_results_it)
                    for chunk_result in chunk_results:
                        for result in chunk_result:
                            # TODO - extend result[1] with {"qoi":single_qoi_column}
                            result[1].update({'qoi': single_qoi_column})
                            self.result_dict[single_qoi_column][result[0]] = result[1]
                            # final_result_dict = {'qoi': single_qoi_column, **result[1]}
                            # self.result_dict[single_qoi_column][result[0]] = final_result_dict
                            # TODO should I doo result[1].clean(); would that delete the entries in self.result_dict?

                    # Saving Stat Dict as soon as it is computed
                    # TODO - current problem, single_qoi_column info is missed once saved dict is read
                    fileName = "statistics_dictionary_qoi_" + single_qoi_column + ".pkl"
                    fullFileName = os.path.abspath(os.path.join(str(self.workingDir), fileName))
                    with open(fullFileName, 'wb') as handle:
                        pickle.dump(self.result_dict[single_qoi_column], handle, protocol=pickle.HIGHEST_PROTOCOL)

                    # TODO Plotting Stat Dict as soon as it is computed

                    # TODO - maybe freeing up the memory - self.result_dict[single_qoi_column].clean(),
                    #  it will letter be used only in saveToFile(), _plotStatisticsDict_plotly()
                    #  and functions which do postprocessing

    ###################################################################################################################

    def _check_if_Sobol_t_computed(self, timestamp=None, qoi_column=None):
        if qoi_column is None:
            qoi_column = self.list_qoi_column[0]
        if timestamp is None:
            timestamp = self.pdTimesteps[0]
        self._is_Sobol_t_computed = "Sobol_t" in self.result_dict[qoi_column][timestamp] #hasattr(self.result_dict[keyIter[0], "Sobol_t")

    def _check_if_Sobol_m_computed(self, timestamp=None, qoi_column=None):
        if qoi_column is None:
            qoi_column = self.list_qoi_column[0]
        if timestamp is None:
            timestamp = self.pdTimesteps[0]
        self._is_Sobol_m_computed = "Sobol_m" in self.result_dict[qoi_column][timestamp]

    def _check_if_Sobol_m2_computed(self, timestamp=None, qoi_column=None):
        if qoi_column is None:
            qoi_column = self.list_qoi_column[0]
        if timestamp is None:
            timestamp = self.pdTimesteps[0]
        self._is_Sobol_m2_computed = "Sobol_m2" in self.result_dict[qoi_column][timestamp]

    ###################################################################################################################

    def saveToFile(self, fileName="statistics_dict", fileNameIdent="", directory="./",
                   fileNameIdentIsFullName=False, **kwargs):

        for single_qoi_column in self.list_qoi_column:
            fileName = "statistics_dictionary_qoi_" + single_qoi_column + ".pkl"
            fullFileName = os.path.abspath(os.path.join(str(self.workingDir), fileName))
            with open(fullFileName, 'wb') as handle:
                pickle.dump(self.result_dict[single_qoi_column], handle, protocol=pickle.HIGHEST_PROTOCOL)

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

    ###################################################################################################################

    def _get_measured_single_qoi(self, timestepRange=None, time_column_name="TimeStamp",
                qoi_column_measured="measured", **kwargs):
        raise NotImplementedError

    def get_measured_data(self, timestepRange=None, time_column_name="TimeStamp", qoi_column_name="measured",
                              **kwargs):
        """

         :param timestepRange:
         :param time_column_name:
         :param qoi_column_name:
         :param kwargs:
         :return: set self.df_measured to be a pd.DataFrame with three columns "TimeStamp", "qoi", "measured"
         """
        # In this particular set-up, we only have access to the measured streamflow

        if not isinstance(qoi_column_name, list):
            qoi_column_name = [qoi_column_name, ]

        transform_measured_data_as_original_model = kwargs.get(
            "transform_measured_data_as_original_model", True)

        list_df_measured_single_qoi = []
        for single_qoi_column in qoi_column_name:

            if single_qoi_column not in self.list_original_model_output_columns:
                is_single_qoi_column_in_measured_column_names = False
                for temp in self.list_original_model_output_columns:
                    if single_qoi_column == self.dict_qoi_column_and_measured_info[temp][1]:
                        is_single_qoi_column_in_measured_column_names = True
                        single_qoi_column = temp
                        break
                if not is_single_qoi_column_in_measured_column_names:
                    continue

            single_qoi_column_info = self.dict_qoi_column_and_measured_info[single_qoi_column]

            single_qoi_read_measured_data = single_qoi_column_info[0]
            single_qoi_column_measured = single_qoi_column_info[1]
            single_qoi_transform_model_output = single_qoi_column_info[2]

            if not single_qoi_read_measured_data:
                continue

            df_measured_single_qoi = self._get_measured_single_qoi(
                timestepRange=timestepRange, time_column_name=time_column_name,
                qoi_column_measured=single_qoi_column_measured, **kwargs)

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

    def get_unaltered_run_data(self, timestepRange=None, time_column_name="TimeStamp", qoi_column_name="streamflow",
                              **kwargs):
        raise NotImplementedError

    def get_forcing_data(self, timestepRange=None, time_column_name="TimeStamp", forcing_column_names="precipitation",
                         **kwargs):
        raise NotImplementedError

    def plotResults(self, timestep=-1, display=False,
                    fileName="", fileNameIdent="", directory="./",
                    fileNameIdentIsFullName=False, safe=True, **kwargs):
        raise NotImplementedError

    def _plotStatisticsDict_plotly(self, unalatered=False, measured=False, forcing=False, recalculateTimesteps=False,
                                   window_title='Forward UQ & SA', filename="sim-plotly.html",
                                   display=False, **kwargs):
        raise NotImplementedError

    def _compute_number_of_rows_for_plotting(self, starting_row=1):
        sobol_t_row = sobol_m_row = None
        if self.forcing_data_fetched:
            starting_row += 2
        if self._is_Sobol_t_computed and self._is_Sobol_m_computed:
            n_rows = 4
            sobol_m_row = starting_row+2
            sobol_t_row = starting_row+3
        elif self._is_Sobol_t_computed:
            n_rows = 3
            sobol_t_row = starting_row+2
        elif self._is_Sobol_m_computed:
            n_rows = 3
            sobol_m_row = starting_row+2
        else:
            n_rows = 2
        if self.forcing_data_fetched:
            n_rows += 2
        return n_rows, starting_row, sobol_t_row, sobol_m_row

    ###################################################################################################################
    def prepare_for_plotting(self, timestep=-1, display=False,
                    fileName="", fileNameIdent="", directory="./",
                    fileNameIdentIsFullName=False, safe=True, **kwargs):
        raise NotImplementedError

    def plotResults_single_qoi(self, single_qoi_column, dict_time_vs_qoi_stat, timestep=-1, display=False,
                    fileName="", fileNameIdent="", directory="./",
                    fileNameIdentIsFullName=False, safe=True, **kwargs):
        raise NotImplementedError

    def _plotStatisticsDict_plotly_single_qoi(
            self, single_qoi_column, dict_time_vs_qoi_stat=None,
            unalatered=False, measured=False, forcing=False,
            recalculateTimesteps=False,
            window_title='Forward UQ & SA', filename="sim-plotly.html", display=False, **kwargs):
        raise NotImplementedError

    ###################################################################################################################

    def extract_mean_time_series(self):
        if self.result_dict is None:
            raise Exception('[STAT INFO] extract_mean_time_series - self.result_dict is None. '
                            'Calculate the statistics first!')
        list_of_single_qoi_mean_df = []
        for single_qoi_column in self.list_qoi_column:
            keyIter = list(self.pdTimesteps)
            mean_time_series = [self.result_dict[single_qoi_column][key]["E"] for key in keyIter]
            qoi_column = [single_qoi_column] * len(keyIter)
            mean_df_single_qoi = pd.DataFrame(list(zip(qoi_column, mean_time_series, self.pdTimesteps)),
                                              columns=['qoi', 'mean_qoi', self.time_column_name])
            list_of_single_qoi_mean_df.append(mean_df_single_qoi)
        self.qoi_mean_df = pd.concat(list_of_single_qoi_mean_df, ignore_index=True, sort=False, axis=0)

    def create_df_from_statistics_data(self, uq_method="sc"):
        list_of_single_qoi_dfs = []
        for single_qoi_column in self.list_qoi_column:
            df_statistics_single_qoi = self.create_df_from_statistics_data_single_qoi(
                qoi_column=single_qoi_column, uq_method=uq_method)
            df_statistics_single_qoi["qoi"] = single_qoi_column
            list_of_single_qoi_dfs.append(df_statistics_single_qoi)
        if list_of_single_qoi_dfs:
            self.df_statistics = pd.concat(list_of_single_qoi_dfs, axis=0)

    def create_df_from_sensitivity_indices(self, si_type="Sobol_t", uq_method="sc"):
        """
        Creates one big Padans.DataFrame for all QoIs
        :param si_type:
        :param uq_method:
        :return:
        """
        si_df = None
        list_of_single_qoi_dfs = []
        for single_qoi_column in self.list_qoi_column:
            single_si_df = self.create_df_from_sensitivity_indices_single_qoi(
                qoi_column=single_qoi_column, si_type=si_type, uq_method=uq_method)
            single_si_df["qoi"] = single_qoi_column
            single_si_df.reset_index(inplace=True)
            single_si_df.rename(columns={single_si_df.index.name: self.time_column_name}, inplace=True)
            list_of_single_qoi_dfs.append(single_si_df)
        if list_of_single_qoi_dfs:
            si_df = pd.concat(list_of_single_qoi_dfs, axis=0)
        return si_df

    def create_df_from_statistics_data_single_qoi(self, qoi_column, uq_method="sc"):
        keyIter = list(self.pdTimesteps)
        mean_time_series = [self.result_dict[qoi_column][key]["E"] for key in keyIter]
        std_time_series = [self.result_dict[qoi_column][key]["StdDev"] for key in keyIter]
        p10_time_series = [self.result_dict[qoi_column][key]["P10"] for key in keyIter]
        p90_time_series = [self.result_dict[qoi_column][key]["P90"] for key in keyIter]

        list_of_columns = [self.pdTimesteps, mean_time_series, std_time_series,
                           p10_time_series, p90_time_series]
        list_of_columns_names = [self.time_column_name, "E", "Std", "P10", "P90"]

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
        # is_Sobol_m2_computed = "Sobol_m2" in self.result_dict[qoi_column][keyIter[0]]

        if is_Sobol_m_computed:
            for i in range(len(self.labels)):
                sobol_m_time_series = [self.result_dict[qoi_column][key]["Sobol_m"][i] for key in keyIter]
                list_of_columns.append(sobol_m_time_series)
                temp = "sobol_m_" + self.labels[i]
                list_of_columns_names.append(temp)
        if is_Sobol_t_computed:
            for i in range(len(self.labels)):
                sobol_t_time_series = [self.result_dict[qoi_column][key]["Sobol_t"][i] for key in keyIter]
                list_of_columns.append(sobol_t_time_series)
                temp = "sobol_t_" + self.labels[i]
                list_of_columns_names.append(temp)

        df_statistics_single_qoi = pd.DataFrame(list(zip(*list_of_columns)), columns=list_of_columns_names)

        df_statistics_single_qoi["E_minus_std"] = df_statistics_single_qoi['E'] - df_statistics_single_qoi['Std']
        df_statistics_single_qoi["E_plus_std"] = df_statistics_single_qoi['E'] + df_statistics_single_qoi['Std']

        if self.measured_fetched and self.df_measured is not None:
            if qoi_column in list(self.df_measured["qoi"].unique()):
                # print(f"{qoi_column}")
                df_measured_subset = self.df_measured.loc[self.df_measured["qoi"] == qoi_column][[
                    self.time_column_name, "measured"]]
                # df_measured_subset.drop("qoi", inplace=True)
                df_statistics_single_qoi = pd.merge(df_statistics_single_qoi, df_measured_subset, on=[self.time_column_name, ], how='left')
            else:
                df_statistics_single_qoi["measured"] = np.nan
        else:
            df_statistics_single_qoi["measured"] = np.nan

        if self.unaltered_computed:
            pass  # TODO

        return df_statistics_single_qoi

    def create_df_from_sensitivity_indices_single_qoi(self, qoi_column, si_type="Sobol_t", uq_method="sc"):
        """
        si_type should be: Sobol_t, Sobol_m or Sobol_m2
        """
        keyIter = list(self.pdTimesteps)
        is_Sobol_t_computed = "Sobol_t" in self.result_dict[qoi_column][keyIter[0]]
        is_Sobol_m_computed = "Sobol_m" in self.result_dict[qoi_column][keyIter[0]]
        is_Sobol_m2_computed = "Sobol_m2" in self.result_dict[qoi_column][keyIter[0]]

        if si_type == "Sobol_t" and not is_Sobol_t_computed:
            raise Exception("Sobol Total Order Indices are not computed")
        elif si_type == "Sobol_m" and not is_Sobol_m_computed:
            raise Exception("Sobol Main Order Indices are not computed")
        elif si_type == "Sobol_m2" and not is_Sobol_m2_computed:
            raise Exception("Sobol Second Order Indices are not computed")

        list_of_df_over_parameters = []
        for i in range(len(self.labels)):
            # if uq_method == "saltelli":
            #     si_single_param = [self.result_dict[key][si_type][i][0] for key in keyIter]
            # else:
            si_single_param = [self.result_dict[qoi_column][key][si_type][i] for key in keyIter]
            df_temp = pd.DataFrame(list(zip(si_single_param, self.pdTimesteps)),
                                   columns=[si_type + "_" + self.labels[i], self.time_column_name])
            list_of_df_over_parameters.append(df_temp)
        si_df = reduce(lambda left, right: pd.merge(left, right, on=self.time_column_name, how='outer'),
                       list_of_df_over_parameters)

        if self.measured_fetched and self.df_measured is not None:
            if qoi_column in list(self.df_measured["qoi"].unique()):
                df_measured_subset = self.df_measured.loc[self.df_measured["qoi"] == qoi_column][[
                    self.time_column_name, "measured"]]
                si_df = pd.merge(si_df, df_measured_subset, on=[self.time_column_name,], how='left')
                # TODO Add normalized measured for plotting later on "measured_norm"
            else:
                si_df["measured"] = np.nan
        else:
            si_df["measured"] = np.nan

        si_df.set_index(self.time_column_name, inplace=True)
        return si_df

    def plot_heatmap_si_single_qoi(self, qoi_column, si_df=None, si_type="Sobol_t", uq_method="sc"):
        if si_df is None:
            si_df = self.create_df_from_sensitivity_indices_single_qoi(qoi_column, si_type, uq_method)
        si_columns = [x for x in si_df.columns.tolist() if x != 'measured']
        fig = px.imshow(si_df[si_columns].T, labels=dict(y='Parameter'))
        return fig

    def plot_si_indices_over_time_single_qoi(self, qoi_column, si_type="Sobol_t", uq_method="sc"):
        fig = go.Figure()
        keyIter = list(self.pdTimesteps)
        for i in range(len(self.labels)):
            # if uq_method == "saltelli":
            #     fig.add_trace(
            #         go.Scatter(x=self.pdTimesteps, y=[self.result_dict[key][si_type][i][0] for key in keyIter],
            #                    name=self.labels[i], legendgroup=self.labels[i], line_color=colors.COLORS[i]))
            # else:
            fig.add_trace(
                go.Scatter(x=self.pdTimesteps, y=[self.result_dict[qoi_column][key][si_type][i] for key in keyIter],
                           name=self.labels[i], legendgroup=self.labels[i], line_color=colors.COLORS[i]))

        return fig
    ###################################################################################################################
    # Set of functions which require some measured/observed data
    ###################################################################################################################

    def compare_mean_time_series_and_measured(self):
        # TODO Finish this
        raise NotImplementedError

    def compute_gof_over_different_time_series_single_qoi(self, df_statistics,
                                               objective_function="MAE", qoi="Q", measuredDF_column_names=["measured"]):
        # TODO Finish this
        raise NotImplementedError

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
            observed_column="measured_norm", uq_method="sc", plot_precipitation=False):

        if si_df is None:
            si_df = self.create_df_from_sensitivity_indices_single_qoi(qoi_column, si_type, uq_method)

        si_columns = [x for x in si_df.columns.tolist() if x != 'measured']

        fig = px.line(si_df, x=si_df.index, y=si_columns)

        if observed_column in si_df.columns and not si_df[observed_column].isnull().all():
            fig.add_trace(go.Scatter(x=si_df.index,
                                     y=si_df[observed_column],
                                     fill='tozeroy', name=f"Normalized {qoi_column}"))
        # if plot_precipitation:
        #     self._add_precipitation_to_graph(fig)
        return fig

    ###################################################################################################################
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

