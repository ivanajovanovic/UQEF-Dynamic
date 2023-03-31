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
from hbv_sask import hbvsask_utility as hbv


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

    # def save_grad_analysis_to_file(self, file_path='./'):
    #     file_path = str(file_path)
    #     if self.df_grad_analysis is not None:
    #         self.df_grad_analysis.to_pickle(
    #             os.path.abspath(os.path.join(file_path, "df_grad_analysis.pkl")), compression="gzip")

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


class HBVSASKStatistics(Statistics):
    def __init__(self, configurationObject, workingDir=None, *args, **kwargs):
        Statistics.__init__(self)

        self.configurationObject = utility.check_if_configurationObject_is_in_right_format_and_return(
            configurationObject, raise_error=True)

        # print(f"[DEBUGGING STAT INFO] - {self.configurationObject}")

        # in the statistics class specification of the workingDir is necessary
        self.workingDir = pathlib.Path(workingDir)

        if "basis" in kwargs:
            self.basis = kwargs['basis']
        else:
            self.basis = self.configurationObject["model_settings"].get("basis", 'Oldman_Basin')

        inputModelDir = kwargs.get('inputModelDir', self.workingDir)
        self.inputModelDir = pathlib.Path(inputModelDir)
        self.inputModelDir_basis = self.inputModelDir / self.basis

        #####################################
        # Set of configuration variables propagated via UQsim.args and/or **kwargs
        # These are mainly UQ simulation - related configurations
        #####################################
        self.uq_method = kwargs.get('uq_method', None)

        self.sampleFromStandardDist = kwargs.get('sampleFromStandardDist', False)

        # whether to store all original model outputs in the stat dict; note - this might take a lot of space
        self.store_qoi_data_in_stat_dict = kwargs.get('store_qoi_data_in_stat_dict', False)

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
        # This is actually index name in the propageted results DataFrame
        self.time_column_name = kwargs.get("time_column_name", "TimeStamp")
        self.precipitation_column_name = kwargs.get("precipitation_column_name", "precipitation")
        self.temperature_column_name = kwargs.get("temperature_column_name", "temperature")

        if "run_full_timespan" in kwargs:
            self.run_full_timespan = kwargs['run_full_timespan']
        else:
            self.run_full_timespan = strtobool(
                self.configurationObject["time_settings"].get("run_full_timespan", 'False'))

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

        self.solverTimes = None
        self.work_package_indexes = None

        ##################################
        dict_processed_config_simulation_settings = utility.read_simulation_settings_from_configuration_object(
            self.configurationObject, **kwargs)

        self.qoi = dict_processed_config_simulation_settings["qoi"]
        self.qoi_column = dict_processed_config_simulation_settings["qoi_column"]
        self.transform_model_output = dict_processed_config_simulation_settings["transform_model_output"]
        self.multiple_qoi = dict_processed_config_simulation_settings["multiple_qoi"]
        self.number_of_qois = dict_processed_config_simulation_settings["number_of_qois"]
        self.qoi_column_measured = dict_processed_config_simulation_settings["qoi_column_measured"]
        self.read_measured_data = dict_processed_config_simulation_settings["read_measured_data"]

        # self.objective_function_qoi = dict_processed_config_simulation_settings["objective_function_qoi"]
        # self.objective_function_names_qoi = dict_processed_config_simulation_settings["objective_function_names_qoi"]

        # list versions of the above variables
        self.list_qoi_column = dict_processed_config_simulation_settings["list_qoi_column"]
        self.list_qoi_column_measured = dict_processed_config_simulation_settings["list_qoi_column_measured"]
        self.list_read_measured_data = dict_processed_config_simulation_settings["list_read_measured_data"]
        self.list_transform_model_output = dict_processed_config_simulation_settings["list_transform_model_output"]
        self.list_objective_function_qoi = dict_processed_config_simulation_settings["list_objective_function_qoi"]
        self.list_objective_function_names_qoi = dict_processed_config_simulation_settings[
            "list_objective_function_names_qoi"]

        self.mode = dict_processed_config_simulation_settings["mode"]
        self.method = dict_processed_config_simulation_settings["method"]

        self.compute_gradients = dict_processed_config_simulation_settings["compute_gradients"]
        self.compute_active_subspaces = dict_processed_config_simulation_settings["compute_active_subspaces"]
        self.save_gradient_related_runs = dict_processed_config_simulation_settings["save_gradient_related_runs"]
        self.gradient_analysis = dict_processed_config_simulation_settings["gradient_analysis"]

        # streamflow is of special importance here, since we have saved/measured/ground truth that for it and it is inside input data
        # self.streamflow_column_name = kwargs.get("streamflow_column_name", "streamflow")
        self.read_measured_streamflow = False
        if self.multiple_qoi:
            for idx, single_qoi_column in enumerate(self.qoi_column):
                if single_qoi_column == "Q_cms" or single_qoi_column == "Q" or single_qoi_column == "streamflow":
                    self.read_measured_streamflow = self.read_measured_data[idx]
                    self.streamflow_column_name = self.qoi_column_measured[idx]
        else:
            if self.qoi_column == "Q_cms" or self.qoi_column == "Q" or self.qoi_column == "streamflow":
                self.read_measured_streamflow = self.read_measured_data
                self.streamflow_column_name = self.qoi_column_measured

        self.list_original_model_output_columns = self.list_qoi_column.copy()
        self.additional_qoi_columns_besides_original_model_output = False
        self.qoi_is_a_single_number = False
        self.list_grad_columns = []
        self.df_grad_analysis = None

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
                                              "_" + self.method + "_sliding_window"
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
                                                          single_qoi_column + "_" + self.method + "_sliding_window" \
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
            list_of_columns_to_filter_from_results = self.list_qoi_column + [self.precipitation_column_name]

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
            self.active_scores_dict = self._compute_active_score(self.samples.dict_of_matrix_c_eigen_decomposition)

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
                    print(f"{self.rank}: computation for qoi {single_qoi_column} - shutted down...")
                    sys.stdout.flush()

                    solver_time_end = time.time()
                    solver_time = solver_time_end - solver_time_start
                    print(f"solver_time for qoi {single_qoi_column}: {solver_time}")

                    chunk_results = list(chunk_results_it)
                    for chunk_result in chunk_results:
                        for result in chunk_result:
                            self.result_dict[single_qoi_column][result[0]] = result[1]

    def calcStatisticsForEnsembleParallel(self, chunksize=1, regression=False, *args, **kwargs):
        self.calcStatisticsForMcParallel(chunksize=chunksize, regression=False, *args, **kwargs)

    def calcStatisticsForScParallel(self, chunksize=1, regression=False, *args, **kwargs):
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

                nodesChunks = [self.nodes] * len(keyIter_chunk)
                distChunks = [self.dist] * len(keyIter_chunk)
                weightsChunks = [self.weights] * len(keyIter_chunk)
                polynomial_expansionChunks = [self.polynomial_expansion] * len(keyIter_chunk)

                regressionChunks = [regression] * len(keyIter_chunk)
                compute_Sobol_t_Chunks = [self._compute_Sobol_t] * len(keyIter_chunk)
                compute_Sobol_m_Chunks = [self._compute_Sobol_m] * len(keyIter_chunk)
                store_qoi_data_in_stat_dict_Chunks = [self.store_qoi_data_in_stat_dict] * len(keyIter_chunk)

            with futures.MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
                if executor is not None:  # master process
                    print(f"{self.rank}: computation of statistics for qoi {single_qoi_column} started...")
                    solver_time_start = time.time()
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
                    print(f"{self.rank}: computation for qoi {single_qoi_column} - waits for shutdown...")
                    sys.stdout.flush()
                    executor.shutdown(wait=True)
                    print(f"{self.rank}: computation for qoi {single_qoi_column} - shutted down...")
                    sys.stdout.flush()

                    solver_time_end = time.time()
                    solver_time = solver_time_end - solver_time_start
                    print(f"solver_time for qoi {single_qoi_column}: {solver_time}")

                    chunk_results = list(chunk_results_it)
                    for chunk_result in chunk_results:
                        for result in chunk_result:
                            self.result_dict[single_qoi_column][result[0]] = result[1]

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
                    print(f"{self.rank}: computation for qoi {single_qoi_column} - shutted down...")
                    sys.stdout.flush()

                    solver_time_end = time.time()
                    solver_time = solver_time_end - solver_time_start
                    print(f"solver_time for qoi {single_qoi_column}: {solver_time}")

                    chunk_results = list(chunk_results_it)
                    for chunk_result in chunk_results:
                        for result in chunk_result:
                            self.result_dict[single_qoi_column][result[0]] = result[1]

    ###################################################################################################################
    def param_grad_analysis(self):
        # In case compute_gradients mode was turned on and gradient_analysis set to True
        if self.samples.df_simulation_result is not None and self.compute_gradients \
                and self.gradient_analysis and self.list_grad_columns:
            list_of_single_df_grad_analysis = []
            if not isinstance(self.list_grad_columns, list):
                self.list_grad_columns = [self.list_grad_columns, ]
            for single_grad_columns in self.list_grad_columns:
                df_single_grad_analysis = HBVSASKStatistics._single_qoi_single_param_grad_analysis(
                    self.samples.df_simulation_result, single_grad_columns, self.time_column_name)
                list_of_single_df_grad_analysis.append(df_single_grad_analysis)
            self.df_grad_analysis = pd.concat(list_of_single_df_grad_analysis, ignore_index=True, sort=False, axis=0)
        else:
            raise Exception("[STAT ERROR] - Calling method param_grad_analysis "
                            "when not all the requirements for the analysis are meet")

    @staticmethod
    def _single_qoi_single_param_grad_analysis(df, qoi_column, time_column_name="TimeStamp"):

        if df is None:
            raise Exception()

        grouped = df.groupby([time_column_name, ])
        groups = grouped.groups
        keyIter = list(groups.keys())
        result_dict = dict()
        result_dict["qoi_column"] = qoi_column

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
        result_dict["total_time_relative_frequency_np"] = pdf
        result_dict["total_time_cumulative_sum_np"] = cdf
        result_dict["total_time_mean_cumulative_sum_np"] = mean_cdf

        pdf = stats.relfreq(all_list_of_values_abs, numbins=len(all_list_of_values_abs))
        cdf = stats.cumfreq(all_list_of_values_abs, numbins=len(all_list_of_values_abs))
        mean_cdf = np.mean(cdf)
        result_dict["total_time_relative_frequency"] = pdf
        result_dict["total_time_cumulative_sum"] = cdf
        result_dict["total_time_mean_cumulative_sum"] = mean_cdf

        # result_dict_over_time = defaultdict(list)
        result_dict["TimeStamp"] = []
        result_dict["mean_cumulative_sum"] = []
        for key in keyIter:
            # single_timestep_grad_values = df.loc[groups[key].values][qoi_column].values
            single_timestep_abs_grad_values = np.abs(
                df.loc[groups[key].values][qoi_column].values)
            mean_cdf = np.mean(
                stats.cumfreq(single_timestep_abs_grad_values, numbins=len(single_timestep_abs_grad_values)))
            result_dict["TimeStamp"].append(key)
            result_dict["mean_cumulative_sum"].append(mean_cdf)

        result_df = pd.DataFrame.from_dict(result_dict)
        return result_df
    ###################################################################################################################

    def _check_if_Sobol_t_computed(self, keyIter, qoi_column=None):
        if qoi_column is None:
            qoi_column = self.list_qoi_column[0]
        self._is_Sobol_t_computed = "Sobol_t" in self.result_dict[qoi_column][keyIter[0]] #hasattr(self.result_dict[keyIter[0], "Sobol_t")

    def _check_if_Sobol_m_computed(self, keyIter, qoi_column=None):
        if qoi_column is None:
            qoi_column = self.list_qoi_column[0]
        self._is_Sobol_m_computed = "Sobol_m" in self.result_dict[qoi_column][keyIter[0]]

    def _check_if_Sobol_m2_computed(self, keyIter, qoi_column=None):
        if qoi_column is None:
            qoi_column = self.list_qoi_column[0]
        self._is_Sobol_m2_computed = "Sobol_m2" in self.result_dict[qoi_column][keyIter[0]]

    ###################################################################################################################

    def saveToFile(self, fileName="statistics_dict", fileNameIdent="", directory="./",
                   fileNameIdentIsFullName=False, **kwargs):

        for single_qoi_column in self.list_qoi_column:
            fileName = "statistics_dictionary_qoi_" + single_qoi_column + ".pkl"
            statFileName = os.path.abspath(os.path.join(str(self.workingDir), fileName))
            with open(statFileName, 'wb') as handle:
                pickle.dump(self.result_dict[single_qoi_column], handle, protocol=pickle.HIGHEST_PROTOCOL)

        if self.active_scores_dict is not None:
            fileName = "active_scores_dict.pkl"
            statFileName = os.path.abspath(os.path.join(str(self.workingDir), fileName))
            with open(statFileName, 'wb') as handle:
                pickle.dump(self.active_scores_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ###################################################################################################################
    # TODO What about AET or other QoI/Model output, make this more general
    # TODO Make below functions more general
    def get_measured_data(self, timestepRange=None, time_column_name="TimeStamp", qoi_column_name="streamflow",
                          **kwargs):
        # In this particular set-up, we only have access to the measured streamflow
        self.df_measured = self._get_measured_streamflow(
            time_column_name=time_column_name, qoi_column_name=qoi_column_name, **kwargs)

        # This data will be used for plotting or comparing with approximated data
        # Perform the same transformation as on original model output
        for idx, single_qoi_column in enumerate(self.list_original_model_output_columns):
            single_transformation = self.list_transform_model_output[idx]
            if single_transformation is not None and single_transformation!="None":
                if self.list_read_measured_data[idx]:
                    utility.transform_column_in_df(
                        self.df_measured,
                        transformation_function_str=single_transformation,
                        column_name=self.list_qoi_column_measured[idx],
                        new_column_name=self.list_qoi_column_measured[idx])
        self.measured_fetched = True

    def _get_measured_streamflow(self, time_column_name="TimeStamp", qoi_column_name="streamflow",
                                 **kwargs):
        streamflow_inp = kwargs.get("streamflow_inp", "streamflow.inp")
        streamflow_inp = self.inputModelDir_basis / streamflow_inp

        if qoi_column_name is None:
            qoi_column_name = self.streamflow_column_name
        streamflow_df = hbv.read_streamflow(streamflow_inp,
                                            time_column_name=time_column_name,
                                            streamflow_column_name=qoi_column_name)
        # Parse input based on some timeframe
        if time_column_name in streamflow_df.columns:
            streamflow_df = streamflow_df.loc[
                (streamflow_df[time_column_name] >= self.timesteps_min) & (
                        streamflow_df[time_column_name] <= self.timesteps_max)]
        else:
            streamflow_df = streamflow_df[self.timesteps_min:self.timesteps_max]
        return streamflow_df

    def get_unaltered_run_data(self):
        self.df_unaltered = None
        self.unaltered_computed = False

    def get_precipitation_temperature_input_data(self, time_column_name="TimeStamp",
                          precipitation_column_name="precipitation", temperature_column_name="temperature", **kwargs):
        precipitation_temperature_inp = kwargs.get("precipitation_temperature_inp", "Precipitation_Temperature.inp")
        precipitation_temperature_inp = self.inputModelDir_basis / precipitation_temperature_inp

        self.precipitation_temperature_df = hbv.read_precipitation_temperature(
            precipitation_temperature_inp, time_column_name=time_column_name,
            precipitation_column_name=precipitation_column_name, temperature_column_name=temperature_column_name
        )

        # Parse input based on some timeframe
        if time_column_name in self.precipitation_temperature_df.columns:
            self.precipitation_temperature_df = self.precipitation_temperature_df.loc[
                (self.precipitation_temperature_df[time_column_name] >= self.timesteps_min) & (self.precipitation_temperature_df[time_column_name] <= self.timesteps_max)]
        else:
            self.precipitation_temperature_df = self.precipitation_temperature_df[self.timesteps_min:self.timesteps_max]

        self.forcing_data_fetched = True

    def _input_and_measured_data_setup(self, time_column_name="TimeStamp", precipitation_column_name="precipitation",
                                       temperature_column_name="temperature",
                                       read_measured_streamflow=None, streamflow_column_name="streamflow",
                                       **kwargs):
        # % ********  Forcing (Precipitation and Temperature)  *********
        self.get_precipitation_temperature_input_data(time_column_name=time_column_name,
                                                      precipitation_column_name=precipitation_column_name,
                                                      temperature_column_name=temperature_column_name, **kwargs)
        # % ********  Observed Streamflow  *********

        if read_measured_streamflow is None:
            read_measured_streamflow = self.read_measured_streamflow
        if read_measured_streamflow:
            self.get_measured_data(time_column_name=time_column_name, qoi_column_name=self.streamflow_column_name,
                                   **kwargs)

            self.time_series_measured_data_df = pd.merge(
                self.df_measured, self.precipitation_temperature_df,  left_index=True, right_index=True
            )
        else:
            self.time_series_measured_data_df = self.precipitation_temperature_df

    ###################################################################################################################

    def plotResults(self, timestep=-1, display=False,
                    fileName="", fileNameIdent="", directory="./",
                    fileNameIdentIsFullName=False, safe=True, **kwargs):

        plot_measured_timeseries = kwargs.get('plot_measured_timeseries', False)
        plot_unaltered_timeseries = kwargs.get('plot_unalteres_timeseries', False)
        plot_forcing_timeseries = kwargs.get('plot_forcing_timeseries', False)
        time_column_name = kwargs.get('time_column_name', self.time_column_name)

        if plot_measured_timeseries:
            self.get_measured_data(time_column_name=time_column_name,
                                   qoi_column_name=kwargs.get('measured_df_column_to_draw', "streamflow"))
        if plot_unaltered_timeseries:
            self.get_unaltered_run_data()
        if plot_forcing_timeseries:
            self.get_precipitation_temperature_input_data(
                time_column_name=time_column_name,
                precipitation_column_name=kwargs.get('precipitation_df_column_to_draw', "precipitation"),
                temperature_column_name=kwargs.get('temperature_df_column_to_draw', "temperature")
            )

        single_fileName = self.generateFileName(fileName=fileName, fileNameIdent=".html",
                                                directory=directory, fileNameIdentIsFullName=fileNameIdentIsFullName)
        fig = self._plotStatisticsDict_plotly(
            unalatered=self.unaltered_computed, measured=self.measured_fetched, forcing=self.forcing_data_fetched,
            recalculateTimesteps=False, filename=single_fileName, display=display, **kwargs)
        if display:
            fig.show()
        print(f"[STAT INFO] plotResults function is done!")

    def _plotStatisticsDict_plotly(self, unalatered=False, measured=False, forcing=False, recalculateTimesteps=False,
                                   window_title='Forward UQ & SA', filename="sim-plotly.html",
                                   display=False, **kwargs):
        pdTimesteps = self.pdTimesteps
        keyIter = list(pdTimesteps)
        self._check_if_Sobol_t_computed(keyIter)
        self._check_if_Sobol_m_computed(keyIter)

        n_rows = 0
        starting_row = 1
        if forcing and self.forcing_data_fetched:
            n_rows += 2
            starting_row = 3
        n_rows += len(self.list_qoi_column)
        if self._is_Sobol_m_computed:
            n_rows += len(self.list_qoi_column)
        if self._is_Sobol_t_computed:
            n_rows += len(self.list_qoi_column)

        fig = make_subplots(rows=n_rows, cols=1,
                            print_grid=True, shared_xaxes=False,
                            vertical_spacing=0.1)

        if forcing and self.forcing_data_fetched:
            # Precipitation
            column_to_draw = kwargs.get('precipitation_df_column_to_draw', 'precipitation')
            timestamp_column = kwargs.get('precipitation_df_timestamp_column', 'TimeStamp')
            N_max = self.precipitation_temperature_df[column_to_draw].max()
            if timestamp_column == "index":
                fig.add_trace(go.Bar(x=self.precipitation_temperature_df.index,
                                     y=self.precipitation_temperature_df[column_to_draw],
                                     name="Precipitation", marker_color='magenta'),
                              row=1, col=1)
            else:
                fig.add_trace(go.Bar(x=self.precipitation_temperature_df[timestamp_column],
                                     y=self.precipitation_temperature_df[column_to_draw],
                                     name="Precipitation", marker_color='magenta'),
                              row=1, col=1)

            # Temperature
            column_to_draw = kwargs.get('temperature_df_column_to_draw', 'temperature')
            timestamp_column = kwargs.get('temperature_df_timestamp_column', 'TimeStamp')
            if timestamp_column == "index":
                fig.add_trace(go.Scatter(x=self.precipitation_temperature_df.index,
                                         y=self.precipitation_temperature_df[column_to_draw],
                                         name="Temperature", line_color='blue', mode = 'lines+markers'),
                              row=2, col=1)
            else:
                fig.add_trace(go.Scatter(x=self.precipitation_temperature_df[timestamp_column],
                                         y=self.precipitation_temperature_df[column_to_draw],
                                         name="Temperature", line_color='blue', mode = 'lines+markers'),
                              row=2, col=1)

        dict_qoi_vs_plot_rows = defaultdict(dict, {single_qoi_column: {} for single_qoi_column in self.list_qoi_column})
        # One big Figure for each QoI; Note: self.list_qoi_column contain first original model output
        for idx, single_qoi_column in enumerate(self.list_qoi_column):
            if single_qoi_column in self.list_original_model_output_columns:
                if measured and self.measured_fetched and self.list_read_measured_data[idx]:
                    column_to_draw = self.list_qoi_column_measured[idx]
                    timestamp_column = kwargs.get('measured_df_timestamp_column', 'TimeStamp')
                    if timestamp_column == "index":
                        fig.add_trace(go.Scatter(x=self.df_measured.index, y=self.df_measured[column_to_draw],
                                                 name=f"{single_qoi_column} (measured)", line_color='red', mode='lines'),
                                      row=starting_row, col=1)
                    else:
                        fig.add_trace(go.Scatter(x=self.df_measured[timestamp_column], y=self.df_measured[column_to_draw],
                                                 name=f"{single_qoi_column} (measured)", line_color='red', mode='lines'),
                                      row=starting_row, col=1)

            fig.add_trace(go.Scatter(x=pdTimesteps,
                                     y=[self.result_dict[single_qoi_column][key]["E"] for key in keyIter],
                                     name=f'E[{single_qoi_column}]',
                                     line_color='green', mode='lines'),
                          row=starting_row, col=1)
            fig.add_trace(go.Scatter(x=pdTimesteps,
                                     y=[(self.result_dict[single_qoi_column][key]["E"] \
                                         - self.result_dict[single_qoi_column][key]["StdDev"]) for key in keyIter],
                                     name='mean - std. dev', line_color='darkviolet', mode='lines'),
                          row=starting_row, col=1)
            fig.add_trace(go.Scatter(x=pdTimesteps,
                                     y=[(self.result_dict[single_qoi_column][key]["E"] +\
                                         self.result_dict[single_qoi_column][key]["StdDev"]) for key in keyIter],
                                     name='mean + std. dev', line_color='darkviolet', mode='lines', fill='tonexty'),
                          row=starting_row, col=1)
            fig.add_trace(go.Scatter(x=pdTimesteps,
                                     y=[self.result_dict[single_qoi_column][key]["P10"] for key in keyIter],
                                     name='10th percentile', line_color='yellow', mode='lines'),
                          row=starting_row, col=1)
            fig.add_trace(go.Scatter(x=pdTimesteps,
                                     y=[self.result_dict[single_qoi_column][key]["P90"] for key in keyIter],
                                     name='90th percentile', line_color='yellow', mode='lines', fill='tonexty'),
                          row=starting_row, col=1)
            dict_qoi_vs_plot_rows[single_qoi_column]["qoi"] = starting_row
            starting_row += 1

            # fig.add_trace(go.Scatter(x=pdTimesteps,
            #                          y=[self.result_dict[single_qoi_column][key]["StdDev"] for key in keyIter],
            #                          name='std. dev', line_color='darkviolet', mode='lines'),
            #               row=starting_row+1, col=1)

        if self._is_Sobol_m_computed:
            for single_qoi_column in self.list_qoi_column:
                for i in range(len(self.labels)):
                    name = self.labels[i] + "_" + single_qoi_column + "_S_m"
                    fig.add_trace(go.Scatter(
                        x=pdTimesteps,
                        y=[self.result_dict[single_qoi_column][key]["Sobol_m"][i] for key in keyIter],
                        name=name, legendgroup=self.labels[i], line_color=colors.COLORS[i], mode='lines'),
                        row=starting_row, col=1)
                dict_qoi_vs_plot_rows[single_qoi_column]["sobol_m"] = starting_row
                starting_row += 1

        if self._is_Sobol_t_computed:
            for single_qoi_column in self.list_qoi_column:
                for i in range(len(self.labels)):
                    name = self.labels[i] + "_" + single_qoi_column + "_S_t"
                    fig.add_trace(go.Scatter(
                        x=pdTimesteps,
                        y=[self.result_dict[single_qoi_column][key]["Sobol_t"][i] for key in keyIter],
                        name=name, legendgroup=self.labels[i], line_color=colors.COLORS[i], mode='lines'),
                        row=starting_row, col=1)
                dict_qoi_vs_plot_rows[single_qoi_column]["sobol_t"] = starting_row
                starting_row += 1

        # fig.update_traces(mode='lines')
        #fig.update_xaxes(title_text="Time")
        if forcing and self.forcing_data_fetched:
            fig.update_yaxes(title_text="N [mm/h]", side='left', showgrid=True, row=1, col=1)
            fig.update_yaxes(autorange="reversed", row=1, col=1)
            fig.update_yaxes(title_text="T [c]", side='left', showgrid=True, row=2, col=1)

        for single_qoi_column in self.list_qoi_column:
            fig.update_yaxes(title_text=single_qoi_column, side='left', showgrid=True,
                             row=dict_qoi_vs_plot_rows[single_qoi_column]["qoi"], col=1)
            # fig.update_yaxes(title_text=f"Std. Dev. [{single_qoi_column}]", side='left', showgrid=True,
            #                  row=starting_row+1, col=1)
            if self._is_Sobol_m_computed:
                fig.update_yaxes(title_text=f"{single_qoi_column}_m", side='left', showgrid=True, range=[0, 1],
                                 row=dict_qoi_vs_plot_rows[single_qoi_column]["sobol_m"], col=1)
            if self._is_Sobol_t_computed:
                fig.update_yaxes(title_text=f"{single_qoi_column}_t", side='left', showgrid=True, range=[0, 1],
                                 row=dict_qoi_vs_plot_rows[single_qoi_column]["sobol_t"], col=1)

        fig.update_layout(width=1000)
        fig.update_layout(title_text=window_title)
        fig.update_layout(xaxis=dict(type="date"))

        print(f"[HVB STAT INFO] _plotStatisticsDict_plotly function is almost over, just to save the plot!")

        # filename = pathlib.Path(filename)
        plot(fig, filename=filename, auto_open=display)
        return fig

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
                                              columns=['QoI', 'Mean_QoI', 'TimeStamp'])
            list_of_single_qoi_mean_df.append(mean_df_single_qoi)
        self.qoi_mean_df = pd.concat(list_of_single_qoi_mean_df, ignore_index=True, sort=False, axis=0)

    def create_df_from_statistics_data_single_qoi(self, qoi_column, uq_method="sc"):
        keyIter = list(self.pdTimesteps)
        mean_time_series = [self.result_dict[qoi_column][key]["E"] for key in keyIter]
        std_time_series = [self.result_dict[qoi_column][key]["StdDev"] for key in keyIter]
        p10_time_series = [self.result_dict[qoi_column][key]["P10"] for key in keyIter]
        p90_time_series = [self.result_dict[qoi_column][key]["P90"] for key in keyIter]
        list_of_columns = [self.pdTimesteps, mean_time_series, std_time_series,
                           p10_time_series, p90_time_series]
        list_of_columns_names = ['TimeStamp', "E", "Std", "P10", "P90"]

        self._check_if_Sobol_t_computed(keyIter, qoi_column=qoi_column)
        self._check_if_Sobol_m_computed(keyIter, qoi_column=qoi_column)
        if self._is_Sobol_m_computed:
            for i in range(len(self.labels)):
                sobol_m_time_series = [self.result_dict[qoi_column][key]["Sobol_m"][i] for key in keyIter]
                list_of_columns.append(sobol_m_time_series)
                temp = "sobol_m_" + self.labels[i]
                list_of_columns_names.append(temp)
        if self._is_Sobol_t_computed:
            for i in range(len(self.labels)):
                sobol_t_time_series = [self.result_dict[qoi_column][key]["Sobol_t"][i] for key in keyIter]
                list_of_columns.append(sobol_t_time_series)
                temp = "sobol_t_" + self.labels[i]
                list_of_columns_names.append(temp)

        df_statistics = pd.DataFrame(list(zip(*list_of_columns)), columns=list_of_columns_names)

        if self.measured_fetched:
            pass  # TODO

        if self.unaltered_computed:
            pass  # TODO

        df_statistics["E_minus_std"] = df_statistics['E'] - df_statistics['Std']
        df_statistics["E_plus_std"] = df_statistics['E'] + df_statistics['Std']
        return df_statistics

    ###################################################################################################################

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

    ###################################################################################################################

    # def _timespan_setup(self, **kwargs):
    #     if self.run_full_timespan:
    #         self.start_date, self.end_date = hbv._get_full_time_span(self.basis)
    #     else:
    #         try:
    #             self.start_date = pd.Timestamp(
    #                 year=self.configurationObject["time_settings"]["start_year"],
    #                 month=self.configurationObject["time_settings"]["start_month"],
    #                 day=self.configurationObject["time_settings"]["start_day"],
    #                 hour=self.configurationObject["time_settings"]["start_hour"]
    #             )
    #             self.end_date = pd.Timestamp(
    #                 year=self.configurationObject["time_settings"]["end_year"],
    #                 month=self.configurationObject["time_settings"]["end_month"],
    #                 day=self.configurationObject["time_settings"]["end_day"],
    #                 hour=self.configurationObject["time_settings"]["end_hour"]
    #             )
    #         except KeyError:
    #             self.start_date, self.end_date = hbv._get_full_time_span(self.basis)
    #
    #     if "spin_up_length" in kwargs:
    #         self.spin_up_length = kwargs["spin_up_length"]
    #     else:
    #         try:
    #             self.spin_up_length = self.configurationObject["time_settings"]["spin_up_length"]
    #         except KeyError:
    #             self.spin_up_length = 0  # 365*3
    #
    #     if "simulation_length" in kwargs:
    #         self.simulation_length = kwargs["simulation_length"]
    #     else:
    #         try:
    #             self.simulation_length = self.configurationObject["time_settings"]["simulation_length"]
    #         except KeyError:
    #             self.simulation_length = (self.end_date - self.start_date).days - self.spin_up_length
    #             if self.simulation_length <= 0:
    #                 self.simulation_length = 365
    #
    #     self.start_date_predictions = pd.to_datetime(self.start_date) + pd.DateOffset(days=self.spin_up_length)
    #     self.end_date = pd.to_datetime(self.start_date_predictions) + pd.DateOffset(days=self.simulation_length)
    #     self.full_data_range = pd.date_range(start=self.start_date, end=self.end_date, freq="1D")
    #     self.simulation_range = pd.date_range(start=self.start_date_predictions, end=self.end_date, freq="1D")
    #
    #     self.start_date = pd.Timestamp(self.start_date)
    #     self.end_date = pd.Timestamp(self.end_date)
    #     self.start_date_predictions = pd.Timestamp(self.start_date_predictions)
    #
    #     # print(f"start_date-{self.start_date}; spin_up_length-{self.spin_up_length};
    #     # start_date_predictions-{self.start_date_predictions}")
    #     # print(
    #     #     f"start_date_predictions-{self.start_date_predictions}; simulation_length-{self.simulation_length}; end_date-{self.end_date}")
    #     # print(len(self.simulation_range), (self.end_date - self.start_date_predictions).days)
    #     # assert len(self.time_series_data_df[self.start_date:self.end_date]) == len(self.full_data_range)
