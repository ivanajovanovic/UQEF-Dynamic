import chaospy as cp
from collections import defaultdict
from distutils.util import strtobool
from functools import reduce
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
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import time
import sys

from uqef.stat import Statistics

from common import saltelliSobolIndicesHelpingFunctions
from common import parallelStatistics
from common import colors

# import hbvsask_utility as hbv
from hbv_sask import hbvsask_utility as hbv


class HBVSASKSamples(object):
    def __init__(self, rawSamples, configurationObject, qoi_column="Value", **kwargs):
        if isinstance(configurationObject, dict):
            self.configurationObject = configurationObject
        else:
            with open(configurationObject) as f:
                self.configurationObject = json.load(f)

        qoi_columns = [qoi_column, ]  # ["Q_cms", ]
        qoi_columns = qoi_columns + ["TimeStamp", "Index_run"]  # "streamflow"
        self.extract_only_qoi_columns = kwargs.get('extract_only_qoi_columns', False)

        try:
            calculate_GoF = strtobool(self.configurationObject["output_settings"]["calculate_GoF"])
            compute_gradients = strtobool(self.configurationObject["model_run_settings"]["compute_gradients"])
        except KeyError:
            calculate_GoF = False
            compute_gradients = False

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
                if "gof_df" in value and calculate_GoF:
                    list_of_single_index_parameter_gof_df.append(value["gof_df"])
                if "gradient_matrix_dict" in value and compute_gradients:
                    gradient_matrix_dict = value["gradient_matrix_dict"]
                    if gradient_matrix_dict is not None:
                        # TODO Extract only entry for station and one or multiple gofs
                        list_of_gradient_matrix_dict.append(gradient_matrix_dict)
            else:
                df_result = value

            if df_result is not None:
                if self.extract_only_qoi_columns:
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


class HBVSASKStatistics(Statistics):
    def __init__(self, configurationObject, workingDir=None, *args, **kwargs):
        Statistics.__init__(self)

        if isinstance(configurationObject, dict):
            self.configurationObject = configurationObject
        else:
            with open(configurationObject) as f:
                self.configurationObject = json.load(f)

        self.workingDir = pathlib.Path(workingDir)

        #####################################
        # Set of configuration variables propagated via UQsim.args and **kwargs
        #####################################
        self.sampleFromStandardDist = kwargs.get('sampleFromStandardDist', False)

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

        self.uq_method = kwargs.get('uq_method', None)

        self.save_samples = kwargs.get('save_samples', True)

        self._compute_Sobol_t = kwargs.get('compute_Sobol_t', True)
        self._compute_Sobol_m = kwargs.get('compute_Sobol_m', True)
        self._compute_Sobol_m2 = kwargs.get('compute_Sobol_m2', False)

        self.qoi_column = kwargs.get('qoi_column', "Value")

        #####################################
        if "run_full_timespan" in kwargs:
            self.run_full_timespan = kwargs['run_full_timespan']
        else:
            self.run_full_timespan = strtobool(self.configurationObject["time_settings"].get("run_full_timespan", 'False'))

        if "basis" in kwargs:
            self.basis = kwargs['basis']
        else:
            self.basis = self.configurationObject["model_settings"].get("basis", 'Oldman_Basin')

        inputModelDir = kwargs.get('inputModelDir', self.workingDir)
        self.inputModelDir = pathlib.Path(inputModelDir)
        self.inputModelDir_basis = self.inputModelDir / self.basis
        #####################################
        self.nodeNames = []
        try:
            list_of_parameters = configurationObject["parameters"]
        except KeyError as e:
            print(f"HBVSASK Statistics: parameters key does "
                  f"not exists in the configurationObject{e}")
            raise
        for i in list_of_parameters:
            if self.uq_method == "ensemble" or i["distribution"] != "None":
                self.nodeNames.append(i["name"])
        self.dim = len(self.nodeNames)
        self.labels = [nodeName.strip() for nodeName in self.nodeNames]

        self.df_unaltered = None
        self.df_measured = None
        self.unaltered_computed = False
        self.groundTruth_computed = False

        self._is_Sobol_t_computed = False
        self._is_Sobol_m_computed = False
        self._is_Sobol_m2_computed = False

        self.timesteps = None
        self.timesteps_min = None
        self.timesteps_max = None
        self.numbTimesteps = None
        self.pdTimesteps = None
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
        self.timesteps = kwargs.get('timesteps') if 'timesteps' in kwargs else None
        self.solverTimes = kwargs.get('solverTimes') if 'solverTimes' in kwargs else None
        self.work_package_indexes = kwargs.get('work_package_indexes') if 'work_package_indexes' in kwargs else None

        try:
            compute_gradients = strtobool(self.configurationObject["model_run_settings"]["compute_gradients"])
        except KeyError:
            compute_gradients = False

        self.samples = HBVSASKSamples(
            rawSamples,
            configurationObject=self.configurationObject,
            qoi_column=self.qoi_column,
            extract_only_qoi_columns=True
        )

        if self.samples.df_simulation_result is not None:
            self.samples.df_simulation_result.sort_values(
                by=["Index_run", "TimeStamp"], axis=0, ascending=True, inplace=True, kind='quicksort', na_position='last'
            )

        if self.save_samples:
            self.samples.save_samples_to_file(self.workingDir)
            self.samples.save_index_parameter_values(self.workingDir)
            self.samples.save_index_parameter_gof_values(self.workingDir)
            if compute_gradients:
                self.active_scores_dict = self._compute_active_score(self.samples.dict_of_matrix_c_eigen_decomposition)
                self.samples.save_dict_of_approx_matrix_c(self.workingDir)
                self.samples.save_dict_of_matrix_c_eigen_decomposition(self.workingDir)

        self.timesteps = self.samples.get_simulation_timesteps()
        self.timesteps_min = self.samples.get_timesteps_min()
        self.timesteps_max = self.samples.get_timesteps_max()
        self.pdTimesteps = [pd.Timestamp(timestep) for timestep in self.timesteps]
        self.number_of_unique_index_runs = self.samples.get_number_of_runs()
        self.numEvaluations = self.number_of_unique_index_runs

        self.numbTimesteps = len(self.timesteps)

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

    def calcStatisticsForMcParallel(self, chunksize=1, regression=False, *args, **kwargs):
        if self.rank == 0:
            grouped = self.samples.df_simulation_result.groupby(['TimeStamp',])
            groups = grouped.groups

            keyIter = list(groups.keys())
            list_of_simulations_df = [
                self.samples.df_simulation_result.loc[groups[key].values][self.qoi_column].values
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
                print(f"{self.rank}: waits for shutdown...")
                sys.stdout.flush()
                executor.shutdown(wait=True)
                print(f"{self.rank}: shutted down...")
                sys.stdout.flush()

                solver_time_end = time.time()
                solver_time = solver_time_end - solver_time_start
                print(f"solver_time: {solver_time}")

                chunk_results = list(chunk_results_it)
                self.result_dict = dict()
                for chunk_result in chunk_results:
                    for result in chunk_result:
                        self.result_dict[result[0]] = result[1]

    def calcStatisticsForEnsembleParallel(self, chunksize=1, regression=False, *args, **kwargs):
        self.calcStatisticsForMcParallel(chunksize=chunksize, regression=False, *args, **kwargs)

    def calcStatisticsForScParallel(self, chunksize=1, regression=False, *args, **kwargs):
        if self.rank == 0:
            grouped = self.samples.df_simulation_result.groupby(['TimeStamp', ])
            groups = grouped.groups

            keyIter = list(groups.keys())
            list_of_simulations_df = [
                self.samples.df_simulation_result.loc[groups[key].values][self.qoi_column].values
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
                print(f"{self.rank}: waits for shutdown...")
                sys.stdout.flush()
                executor.shutdown(wait=True)
                print(f"{self.rank}: shutted down...")
                sys.stdout.flush()

                solver_time_end = time.time()
                solver_time = solver_time_end - solver_time_start
                print(f"solver_time: {solver_time}")

                chunk_results = list(chunk_results_it)
                self.result_dict = dict()
                for chunk_result in chunk_results:
                    for result in chunk_result:
                        self.result_dict[result[0]] = result[1]

    def calcStatisticsForSaltelliParallel(self, chunksize=1, regression=False, *args, **kwargs):
        if self.rank == 0:
            grouped = self.samples.df_simulation_result.groupby(['TimeStamp', ])
            groups = grouped.groups

            keyIter = list(groups.keys())
            list_of_simulations_df = [
                self.samples.df_simulation_result.loc[groups[key].values][self.qoi_column].values
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
                print(f"{self.rank}: waits for shutdown...")
                sys.stdout.flush()
                executor.shutdown(wait=True)
                print(f"{self.rank}: shutted down...")
                sys.stdout.flush()

                solver_time_end = time.time()
                solver_time = solver_time_end - solver_time_start
                print(f"solver_time: {solver_time}")

                chunk_results = list(chunk_results_it)
                self.result_dict = dict()
                for chunk_result in chunk_results:
                    for result in chunk_result:
                        self.result_dict[result[0]] = result[1]

    ###################################################################################################################

    def _check_if_Sobol_t_computed(self, keyIter):
        self._is_Sobol_t_computed = "Sobol_t" in self.result_dict[keyIter[0]] #hasattr(self.result_dict[keyIter[0], "Sobol_t")

    def _check_if_Sobol_m_computed(self, keyIter):
        self._is_Sobol_m_computed = "Sobol_m" in self.result_dict[keyIter[0]]

    def _check_if_Sobol_m2_computed(self, keyIter):
        self._is_Sobol_m2_computed = "Sobol_m2" in self.result_dict[keyIter[0]]

    ###################################################################################################################

    def saveToFile(self, fileName="statistics_dict", fileNameIdent="", directory="./",
                   fileNameIdentIsFullName=False, **kwargs):

        fileName = "statistics_dictionary_qoi_" + self.qoi_column + ".pkl"
        statFileName = os.path.abspath(os.path.join(str(self.workingDir), fileName))
        with open(statFileName, 'wb') as handle:
            pickle.dump(self.result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if self.active_scores_dict is not None:
            fileName = "active_scores_dict.pkl"
            statFileName = os.path.abspath(os.path.join(str(self.workingDir), fileName))
            with open(statFileName, 'wb') as handle:
                pickle.dump(self.active_scores_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    ###################################################################################################################
    def get_measured_data(self, timestepRange=None, time_column_name="TimeStamp",
                          streamflow_column_name="streamflow", **kwargs):
        streamflow_inp = kwargs.get("streamflow_inp", "streamflow.inp")
        streamflow_inp = self.inputModelDir_basis / streamflow_inp
        self.df_measured = hbv.read_streamflow(streamflow_inp,
                                               time_column_name=time_column_name,
                                               streamflow_column_name=streamflow_column_name)
        # Parse input based on some timeframe
        if time_column_name in self.df_measured.columns:
            self.df_measured = self.df_measured.loc[
                (self.df_measured[time_column_name] >= self.timesteps_min) & (self.df_measured[time_column_name] <= self.timesteps_max)]
        else:
            self.df_measured = self.df_measured[self.timesteps_min:self.timesteps_max]
        self.groundTruth_computed = True

    def get_unaltered_run_data(self):
        self.df_unaltered = None
        self.unaltered_computed = False

    ###################################################################################################################

    def plotResults(self, timestep=-1, display=False,
                    fileName="", fileNameIdent="", directory="./",
                    fileNameIdentIsFullName=False, safe=True, **kwargs):

        plot_measured_timeseries = kwargs.get('plot_measured_timeseries', False)
        plot_unalteres_timeseries = kwargs.get('plot_unalteres_timeseries', False)
        if plot_measured_timeseries:
            self.get_measured_data()
        if plot_unalteres_timeseries:
            self.get_unaltered_run_data()

        single_fileName = self.generateFileName(fileName=fileName, fileNameIdent=".html",
                                                directory=directory, fileNameIdentIsFullName=fileNameIdentIsFullName)
        fig = self._plotStatisticsDict_plotly(unalatered=self.unaltered_computed, measured=self.groundTruth_computed,
                                              recalculateTimesteps=False, filename=single_fileName, display=display, **kwargs)
        if display:
            fig.show()
        print(f"[HBV STAT INFO] plotResults function is done!")

    def _plotStatisticsDict_plotly(self, unalatered=False, measured=False, recalculateTimesteps=False,
                                   window_title='HBVSASK Forward UQ & SA', filename="sim-plotly.html",
                                   display=False, **kwargs):
        pdTimesteps = self.pdTimesteps
        keyIter = list(pdTimesteps)
        self._check_if_Sobol_t_computed(keyIter)
        self._check_if_Sobol_m_computed(keyIter)

        starting_row = 1
        n_rows, sobol_t_row, sobol_m_row = self._compute_number_of_rows_for_plotting(starting_row)
        fig = make_subplots(rows=n_rows, cols=1, print_grid=True, shared_xaxes=False)

        if unalatered and self.unaltered_computed:
            column_to_draw = self.qoi_column if self.qoi_column in self.df_unaltered.columns \
                else kwargs.get('unaltered_df_column_to_draw', 'Value')
            timestamp_column = kwargs.get('unaltered_df_timestamp_column', 'TimeStamp')
            # TODO change this logic
            if timestamp_column == "index":
                fig.add_trace(go.Scatter(x=self.df_unaltered.index, y=self.df_unaltered[column_to_draw],
                                         name="Q (unaltered simulation)", line_color='deepskyblue'),
                              row=1, col=1)
            else:
                fig.add_trace(go.Scatter(x=self.df_unaltered[timestamp_column], y=self.df_unaltered[column_to_draw],
                                         name="Q (unaltered simulation)", line_color='deepskyblue'),
                              row=1, col=1)
        if measured and self.groundTruth_computed:
            column_to_draw = self.qoi_column if self.qoi_column in self.df_measured.columns \
                else kwargs.get('measured_df_column_to_draw', 'Value')
            timestamp_column = kwargs.get('measured_df_timestamp_column', 'TimeStamp')
            # TODO change this logic
            if timestamp_column == "index":
                fig.add_trace(go.Scatter(x=self.df_measured.index, y=self.df_measured[column_to_draw],
                                         name="Q (measured)", line_color='red'),
                              row=1, col=1)
            else:
                fig.add_trace(go.Scatter(x=self.df_measured[timestamp_column], y=self.df_measured[column_to_draw],
                                         name="Q (measured)",line_color='red'),
                              row=1, col=1)

        fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.result_dict[key]["E"] for key in keyIter], name='E[QoI]',
                                 line_color='green', mode='lines'),
                      row=starting_row, col=1)
        fig.add_trace(go.Scatter(x=pdTimesteps,
                                 y=[(self.result_dict[key]["E"] - self.result_dict[key]["StdDev"]) for key in keyIter],
                                 name='mean - std. dev', line_color='darkviolet', mode='lines'),
                      row=starting_row, col=1)
        fig.add_trace(go.Scatter(x=pdTimesteps,
                                 y=[(self.result_dict[key]["E"] + self.result_dict[key]["StdDev"]) for key in keyIter],
                                 name='mean + std. dev', line_color='darkviolet', mode='lines', fill='tonexty'),
                      row=starting_row, col=1)
        fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.result_dict[key]["P10"] for key in keyIter],
                                 name='10th percentile', line_color='yellow', mode='lines'),
                      row=starting_row, col=1)
        fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.result_dict[key]["P90"] for key in keyIter],
                                 name='90th percentile', line_color='yellow', mode='lines', fill='tonexty'),
                      row=starting_row, col=1)

        fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.result_dict[key]["StdDev"] for key in keyIter],
                                 name='std. dev', line_color='darkviolet'),
                      row=starting_row+1, col=1)

        if self._is_Sobol_m_computed:
            for i in range(len(self.labels)):
                name = self.labels[i] + "_S_m"
                fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.result_dict[key]["Sobol_m"][i] for key in keyIter],
                                         name=name, legendgroup=self.labels[i], line_color=colors.COLORS[i]),
                              row=sobol_m_row, col=1)
        if self._is_Sobol_t_computed:
            for i in range(len(self.labels)):
                name = self.labels[i] + "_S_t"
                fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.result_dict[key]["Sobol_t"][i] for key in keyIter],
                                         name=name, legendgroup=self.labels[i], line_color=colors.COLORS[i]),
                              row=sobol_t_row, col=1)

        fig.update_traces(mode='lines')
        #fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text=self.qoi_column, side='left', showgrid=True, row=starting_row, col=1)
        fig.update_yaxes(title_text="Std. Dev. [QoI]", side='left', showgrid=True, row=starting_row+1, col=1)

        if self._is_Sobol_m_computed:
            fig.update_yaxes(title_text="Sobol_m", side='left', showgrid=True, range=[0, 1], row=sobol_m_row, col=1)
        if self._is_Sobol_t_computed:
            fig.update_yaxes(title_text="Sobol_t", side='left', showgrid=True, range=[0, 1], row=sobol_t_row, col=1)

        fig.update_layout(height=600, width=1000, title_text=window_title)

        print(f"[HVB STAT INFO] _plotStatisticsDict_plotly function is almost over!")

        plot(fig, filename=filename, auto_open=display)
        return fig

    def _compute_number_of_rows_for_plotting(self, starting_row):
        sobol_t_row = sobol_m_row = None
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
        return n_rows, sobol_t_row, sobol_m_row
    ###################################################################################################################

    def extract_mean_time_series(self):
        if self.result_dict is None:
            raise Exception('[HBV STAT INFO] extract_mean_time_series - self.result_dict is None. '
                            'Calculate the statistics first!')
        keyIter = list(self.pdTimesteps)
        mean_time_series = [self.result_dict[key]["E"] for key in keyIter]
        self.qoi_mean_df = pd.DataFrame(list(zip(mean_time_series, self.pdTimesteps)), columns=['Mean_QoI', 'TimeStamp'])

    def create_df_from_statistics_data_single_station(self, station=None, uq_method="sc"):
        keyIter = list(self.pdTimesteps)
        mean_time_series = [self.result_dict[key]["E"] for key in keyIter]
        std_time_series = [self.result_dict[key]["StdDev"] for key in keyIter]
        p10_time_series = [self.result_dict[key]["P10"] for key in keyIter]
        p90_time_series = [self.result_dict[key]["P90"] for key in keyIter]
        list_of_columns = [self.pdTimesteps, mean_time_series, std_time_series,
                           p10_time_series, p90_time_series]
        list_of_columns_names = ['TimeStamp', "E", "Std", "P10", "P90"]

        self._check_if_Sobol_t_computed(keyIter)
        self._check_if_Sobol_m_computed(keyIter)
        if self._is_Sobol_m_computed:
            for i in range(len(self.labels)):
                sobol_m_time_series = [self.result_dict[key]["Sobol_m"][i] for key in keyIter]
                list_of_columns.append(sobol_m_time_series)
                temp = "sobol_m_" + self.labels[i]
                list_of_columns_names.append(temp)
        if self._is_Sobol_t_computed:
            for i in range(len(self.labels)):
                sobol_t_time_series = [self.result_dict[key]["Sobol_t"][i] for key in keyIter]
                list_of_columns.append(sobol_t_time_series)
                temp = "sobol_t_" + self.labels[i]
                list_of_columns_names.append(temp)

        df_statistics_station = pd.DataFrame(list(zip(*list_of_columns)), columns=list_of_columns_names)

        if self.groundTruth_computed:
            pass  # TODO

        if self.unaltered_computed:
            pass  # TODO

        df_statistics_station["E_minus_std"] = df_statistics_station['E'] - df_statistics_station['Std']
        df_statistics_station["E_plus_std"] = df_statistics_station['E'] + df_statistics_station['Std']
        return df_statistics_station

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
