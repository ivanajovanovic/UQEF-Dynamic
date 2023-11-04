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

from LarsimUtilityFunctions import larsimDataPostProcessing
from LarsimUtilityFunctions import larsimIO
from LarsimUtilityFunctions import larsimConfig
import LarsimUtilityFunctions.larsimPaths as paths
from LarsimUtilityFunctions.larsimModel import LarsimConfigurations

#from UQEFPP.common import saltelliSobolIndicesHelpingFunctions
from common import saltelliSobolIndicesHelpingFunctions
from common import parallelStatistics
from common import colors

# from numba import jit, prange

class LarsimSamples(object):
    """
     Samples is a collection of the (filtered) sampled results of a whole UQ simulation
     Prepares results from Model to Statistics
    """
    def __init__(self, rawSamples, configurationObject, QoI="Value", **kwargs):

        if isinstance(configurationObject, LarsimConfigurations):
            self.larsimConfObject = configurationObject
        else:
            self.larsimConfObject = LarsimConfigurations(configurationObject, False, **kwargs)

        self.list_objective_function_qoi = self.larsimConfObject.list_objective_function_qoi
        self.qoi_columns = self.larsimConfObject.qoi_columns

        list_of_single_df = []
        list_index_parameters_dict = []
        list_of_single_index_parameter_gof_df = []
        list_of_gradient_matrix_dict = []
        # Important that the results inside rawSamples (resulted paths)
        # are sorted in order which corresponds to the parameters order
        for index_run, value in enumerate(rawSamples,):
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
                if "gof_df" in value and self.larsimConfObject.calculate_GoF:
                    list_of_single_index_parameter_gof_df.append(value["gof_df"])
                if "gradient_matrix_dict" in value and self.larsimConfObject.compute_gradients:
                    gradient_matrix_dict = value["gradient_matrix_dict"]
                    if gradient_matrix_dict is not None:
                        # TODO Extract only entry for station and one or multiple gofs
                        list_of_gradient_matrix_dict.append(gradient_matrix_dict)
            else:
                df_result = value

            df_single_result = larsimDataPostProcessing.read_process_write_discharge(df=df_result,
                                                                                     index_run=index_run,
                                                                                     type_of_output=self.larsimConfObject.type_of_output_of_Interest,
                                                                                     station=self.larsimConfObject.station_of_Interest)

            #larsimIO._postProcessing_DataFrame_after_reading(df_single_result)
            #simulation_start_timestamp = pd.Timestamp(df_single_result.TimeStamp.min()) + datetime.timedelta(hours=self.warm_up_duration)
            #df_single_result = larsimDataPostProcessing.parse_df_based_on_time(df_single_result, (simulation_start_timestamp, None))

            if df_single_result is not None:
                list_of_single_df.append(df_single_result)

        if list_of_single_df:
            self.df_simulation_result = pd.concat(list_of_single_df, ignore_index=True, sort=False, axis=0)
            larsimIO._postProcessing_DataFrame_after_reading(self.df_simulation_result)
            print(f"[LARSIM STAT INFO] Number of Unique TimeStamps (Hourly): "
                  f"{len(self.df_simulation_result.TimeStamp.unique())}")
            # TODO remove/refactor resample_time_series_df
            if self.larsimConfObject.dailyOutput:
                print(f"[LarsimSamples INFO] Transformation to daily output")
                self.transform_dict = dict()
                for one_qoi_column in self.qoi_columns:
                    self.transform_dict[one_qoi_column] = 'mean'
                self.df_simulation_result = larsimDataPostProcessing.resample_time_series_df(self.df_simulation_result,
                                                                                             transform_dict=self.transform_dict,
                                                                                             groupby_some_columns=True,
                                                                                             columns_to_groupby=[
                                                                                                 "Stationskennung",
                                                                                                 "Type", "Index_run"],
                                                                                             time_column="TimeStamp",
                                                                                             resample_freq="D"
                                                                                             )
                ##### Debugging - remove afterwards
                # print(self.df_simulation_result)
                # temp = os.path.abspath( os.path.join("/gpfs/scratch/pr63so/ga45met2", "Larsim_runs", 'larsim_run_siam_cse_v4','model_runs'))
                # self.save_samples_to_file(temp)
                print(
                    f"[LARSIM STAT INFO] Number of Unique TimeStamps (Daily): {self.df_simulation_result.TimeStamp.nunique()}")

            # save only the real values, independent on QoI
            # TODO Does not work when only one time step!!!
            # self.df_time_discharges = self.df_simulation_result.groupby(["Stationskennung","TimeStamp"])["Value"].apply(
            # lambda df: df.reset_index(drop=True)).unstack()
            self.df_time_discharges = None
        else:
            self.df_simulation_result = None
            self.df_time_discharges = None

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

    def save_time_samples_to_file(self, file_path='./'):
        file_path = str(file_path)
        if self.df_time_discharges is not None:
            self.df_time_discharges.to_pickle(
                os.path.abspath(os.path.join(file_path, "df_all_time_simulations.pkl")), compression="gzip")

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

    def get_simulation_stations(self, stations_column_name="Stationskennung"):
        if self.df_simulation_result is not None:
            return list(self.df_simulation_result[stations_column_name].unique())
        else:
            return None

    def get_list_of_qoi_column_names(self):
        return self.qoi_columns

########################################################################################################################


class LarsimStatistics(Statistics):
    """
       LarsimStatistics calculates the statistics for the LarsimModel
       One LarsimStatistics Object should compute statistics for a multiple station and single QoI!
       TODO The problem is that LarsimSamples currently supports vice versa logic - single stations and multiple Qoi/GoFs
    """

    def __init__(self, configurationObject, workingDir=None, *args, **kwargs):
        Statistics.__init__(self)

        # self.configurationObject = configurationObject
        if isinstance(configurationObject, LarsimConfigurations):
            self.larsimConfObject = configurationObject
        else:
            self.larsimConfObject = LarsimConfigurations(configurationObject, False, **kwargs)

        self.workingDir = pathlib.Path(workingDir)
        if self.workingDir is None:
            self.workingDir = pathlib.Path(paths.workingDir)

        #####################################
        # Set of configuration variables propagated via UQsim.args and **kwargs
        #####################################
        self.sampleFromStandardDist = kwargs.get('sampleFromStandardDist', False)

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

        self.uq_method = kwargs.get('uq_method', None)

        self.save_samples = kwargs.get('save_samples', True)

        self._compute_Sobol_t = kwargs.get('compute_Sobol_t', True)
        self._compute_Sobol_m = kwargs.get('compute_Sobol_m', True)
        self._compute_Sobol_m2 = kwargs.get('compute_Sobol_m2', False)

        #####################################
        # Set of configuration variables propagated via either kwargs or config json file, i.e., larsimConfObject
        # important for Statistics computation, i.e., parsing the results DF
        #####################################
        # TODO for now only a single self.qoi_column is supported, but multiple stations!
        #  Change this so that only a single station is assumed

        self.qoi_column = kwargs.get('qoi_column', self.larsimConfObject.qoi_column)

        self.station_of_Interest = kwargs.get('station_of_Interest', self.larsimConfObject.station_of_Interest)
        if not isinstance(self.station_of_Interest, list):
            self.station_of_Interest = [self.station_of_Interest,]

        #####################################
        # Only the names of the stochastic parameters
        self.nodeNames = []
        try:
            list_of_parameters = self.larsimConfObject.configurationObject["parameters"]
        except KeyError as e:
            print(f"Larsim Statistics: parameters key does "
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
        self.samples_station_names = None

        # df_statistics_station = None
        # si_t_df = None
        # si_m_df = None

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

    def reset_station_of_interest(self):
        # Update self.station_of_Interest, though there is probably no need for this!
        self.station_of_Interest = list(set(self.samples_station_names).intersection(self.station_of_Interest))
        if not self.station_of_Interest:
            self.station_of_Interest = self.samples_station_names
    ###################################################################################################################

    def prepare(self, rawSamples, **kwargs):
        self.timesteps = kwargs.get('timesteps') if 'timesteps' in kwargs else None
        self.solverTimes = kwargs.get('solverTimes') if 'solverTimes' in kwargs else None
        self.work_package_indexes = kwargs.get('work_package_indexes') if 'work_package_indexes' in kwargs else None

        self.samples = LarsimSamples(rawSamples, configurationObject=self.larsimConfObject)

        # TODO - not sure if this is ineed needed
        if self.samples.df_simulation_result is not None:
            self.samples.df_simulation_result.sort_values(
                by=["Index_run", "TimeStamp"], axis=0, ascending=True, inplace=True, kind='quicksort', na_position='last'
            )

        if self.save_samples:
            self.samples.save_samples_to_file(self.workingDir)
            self.samples.save_index_parameter_values(self.workingDir)
            self.samples.save_index_parameter_gof_values(self.workingDir)
            if self.larsimConfObject.compute_gradients:
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

        # Update self.station_of_Interest, though there is probably no need for this!
        self.samples_station_names = self.samples.get_simulation_stations()
        self.reset_station_of_interest()

    def prepareForMcStatistics(self, simulationNodes, numEvaluations, regression=None, order=None,
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

    def prepareForScStatistics(self, simulationNodes, order, poly_normed, poly_rule, *args, **kwargs):
        self.nodes = simulationNodes.distNodes
        if self.sampleFromStandardDist:
            self.dist = simulationNodes.joinedStandardDists
        else:
            self.dist = simulationNodes.joinedDists
        self.weights = simulationNodes.weights
        self.polynomial_expansion = cp.generate_expansion(order, self.dist, rule=poly_rule, normed=poly_normed)

    def prepareForMcSaltelliStatistics(self, simulationNodes, numEvaluations=None, regression=None, order=None,
                                    poly_normed=None, poly_rule=None, *args, **kwargs):
        self.prepareForMcStatistics(simulationNodes, numEvaluations, regression, order, poly_normed, poly_rule,
                                   *args, **kwargs)

    def calcStatisticsForMcParallel(self, chunksize=1, regression=False, *args, **kwargs):
        if self.rank == 0:
            grouped = self.samples.df_simulation_result.groupby(['Stationskennung', 'TimeStamp'])
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
                    chunk_results_it = executor.map(parallelStatistics._parallel_calc_stats_for_gPCE,
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
                    chunk_results_it = executor.map(parallelStatistics._parallel_calc_stats_for_MC,
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
            grouped = self.samples.df_simulation_result.groupby(['Stationskennung', 'TimeStamp'])
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
            store_gpce_surrogate_Chunks = [self.store_gpce_surrogate] * len(keyIter_chunk)
            save_gpce_surrogate_Chunks = [self.save_gpce_surrogate] * len(keyIter_chunk)

        with futures.MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
            if executor is not None:  # master process
                solver_time_start = time.time()
                chunk_results_it = executor.map(parallelStatistics._parallel_calc_stats_for_gPCE,
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

    def calcStatisticsForMcSaltelliParallel(self, chunksize=1, regression=False, *args, **kwargs):
        if self.rank == 0:
            grouped = self.samples.df_simulation_result.groupby(['Stationskennung', 'TimeStamp'])
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
                chunk_results_it = executor.map(parallelStatistics._parallel_calc_stats_for_mc_saltelli,
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

    def calcStatisticsForMc(self, rawSamples, timesteps, simulationNodes,
                            numEvaluations, order, regression, poly_normed, poly_rule, solverTimes,
                            work_package_indexes, original_runtime_estimator=None, *args, **kwargs):

        self.samples = LarsimSamples(rawSamples, configurationObject=self.larsimConfObject)
        if self.save_samples:
            self.samples.save_samples_to_file(self.workingDir)
            self.samples.save_index_parameter_values(self.workingDir)
            self.samples.save_index_parameter_gof_values(self.workingDir)
            if self.larsimConfObject.compute_gradients:
                self.active_scores_dict = self._compute_active_score(self.samples.dict_of_matrix_c_eigen_decomposition)
                self.samples.save_dict_of_approx_matrix_c(self.workingDir)
                self.samples.save_dict_of_matrix_c_eigen_decomposition(self.workingDir)

        self.timesteps = self.samples.get_simulation_timesteps()
        self.timesteps_min = self.samples.get_timesteps_min()
        self.timesteps_max = self.samples.get_timesteps_max()
        self.pdTimesteps = [pd.Timestamp(timestep) for timestep in self.timesteps]
        self.number_of_unique_index_runs = self.samples.get_number_of_runs()

        self.numEvaluations = numEvaluations
        # TODO Think about this, tricky for saltelli, makes sense for mc
        # self.numEvaluations = self.number_of_unique_index_runs

        self.numbTimesteps = len(self.timesteps)
        print(f"[LARSIM STAT INFO] numbTimesteps is: {self.numbTimesteps}")

        self.samples_station_names = self.samples.get_simulation_stations()
        # self.samples_qoi_columns = self.samples.get_list_of_qoi_column_names()
        self.station_of_Interest = list(set(self.samples_station_names).intersection(self.station_of_Interest))
        if not self.station_of_Interest:
            self.station_of_Interest = self.samples_station_names

        grouped = self.samples.df_simulation_result.groupby(['Stationskennung','TimeStamp'])
        groups = grouped.groups

        if regression:
            nodes = simulationNodes.distNodes
            if self.sampleFromStandardDist:
                dist = simulationNodes.joinedStandardDists
            else:
                dist = simulationNodes.joinedDists
            polynomial_expansion = cp.generate_expansion(order, dist, rule=poly_rule, normed=poly_normed)

        for key, val_indices in groups.items():
            qoi_values = self.samples.df_simulation_result.loc[val_indices.values][self.qoi_column].values #numpy array nx1
            #qoi_values = self.samples.df_simulation_result.Value.loc[val_indices].values
            self.result_dict[key] = dict()
            if self.store_qoi_data_in_stat_dict:
                self.result_dict[key]["qoi_values"] = qoi_values
            if regression:
                qoi_gPCE = cp.fit_regression(polynomial_expansion, nodes, qoi_values)
                self._calc_stats_for_gPCE(dist, key, qoi_gPCE)
            else:
                # self.result_dict[key]["E"] = np.sum(qoi_values, axis=0, dtype=np.float64)/ numEvaluations
                self.result_dict[key]["E"] = np.mean(qoi_values, 0)
                #self.result_dict[key]["Var"] = float(np.sum(power(qoi_values)) / numEvaluations - self.result_dict[key]["E"]**2)
                self.result_dict[key]["Var"] = np.sum((qoi_values - self.result_dict[key]["E"]) ** 2,
                                                      axis=0, dtype=np.float64) / (numEvaluations - 1)
                # self.result_dict[key]["StdDev"] = np.sqrt(self.result_dict[key]["Var"], dtype=np.float64)
                self.result_dict[key]["StdDev"] = np.std(qoi_values, 0, ddof=1)
                self.result_dict[key]["P10"] = np.percentile(qoi_values, 10, axis=0)
                self.result_dict[key]["P90"] = np.percentile(qoi_values, 90, axis=0)
                if isinstance(self.result_dict[key]["P10"], list) and len(self.result_dict[key]["P10"]) == 1:
                    self.result_dict[key]["P10"] = self.result_dict[key]["P10"][0]
                    self.result_dict[key]["P90"] = self.result_dict[key]["P90"][0]

        print(f"[LARSIM STAT INFO] calcStatisticsForMc function is done!")

    def calcStatisticsForEnsemble(self, rawSamples, timesteps, simulationNodes, numEvaluations, solverTimes,
                                  work_package_indexes, original_runtime_estimator=None, *args, **kwargs):
        self.calcStatisticsForMc(
            rawSamples=rawSamples, timesteps=timesteps, simulationNodes=simulationNodes, numEvaluations=numEvaluations,
            order=None, regression=False, poly_normed=None, poly_rule=None, solverTimes=solverTimes,
            work_package_indexes=work_package_indexes, original_runtime_estimator=original_runtime_estimator,
            *args, **kwargs
        )

    def calcStatisticsForSc(self, rawSamples, timesteps,
                           simulationNodes, order, regression, poly_normed, poly_rule, solverTimes,
                           work_package_indexes, original_runtime_estimator=None,  *args, **kwargs):

        self.samples = LarsimSamples(rawSamples, configurationObject=self.larsimConfObject)
        if self.save_samples:
            self.samples.save_samples_to_file(self.workingDir)
            self.samples.save_index_parameter_values(self.workingDir)
            self.samples.save_index_parameter_gof_values(self.workingDir)
            if self.larsimConfObject.compute_gradients:
                self.active_scores_dict = self._compute_active_score(self.samples.dict_of_matrix_c_eigen_decomposition)
                self.samples.save_dict_of_approx_matrix_c(self.workingDir)
                self.samples.save_dict_of_matrix_c_eigen_decomposition(self.workingDir)

        self.timesteps = self.samples.get_simulation_timesteps()
        self.timesteps_min = self.samples.get_timesteps_min()
        self.timesteps_max = self.samples.get_timesteps_max()
        self.pdTimesteps = [pd.Timestamp(timestep) for timestep in self.timesteps]
        self.number_of_unique_index_runs = self.samples.get_number_of_runs()

        self.numbTimesteps = len(self.timesteps)
        print(f"[LARSIM STAT INFO] numbTimesteps is: {self.numbTimesteps}")

        self.samples_station_names = self.samples.get_simulation_stations()
        # self.samples_qoi_columns = self.samples.get_list_of_qoi_column_names()
        #self.nodeNames = simulationNodes.nodeNames
        self.station_of_Interest = list(set(self.samples_station_names).intersection(self.station_of_Interest))
        if not self.station_of_Interest:
            self.station_of_Interest = self.samples_station_names

        # components independent on model evaluations, i.e., defined a priori, based solely on the underlying distribution
        nodes = simulationNodes.distNodes
        if self.sampleFromStandardDist:
            dist = simulationNodes.joinedStandardDists
        else:
            dist = simulationNodes.joinedDists

        weights = simulationNodes.weights
        polynomial_expansion = cp.generate_expansion(order, dist, rule=poly_rule, normed=poly_normed)

        grouped = self.samples.df_simulation_result.groupby(['Stationskennung','TimeStamp'])
        groups = grouped.groups

        for key, val_indices in groups.items():
            qoi_values = self.samples.df_simulation_result.loc[val_indices.values][self.qoi_column].values
            self.result_dict[key] = dict()
            if self.store_qoi_data_in_stat_dict:
                self.result_dict[key]["qoi_values"] = qoi_values
            if regression:
                qoi_gPCE = cp.fit_regression(polynomial_expansion, nodes, qoi_values)
            else:
                qoi_gPCE = cp.fit_quadrature(polynomial_expansion, nodes, weights, qoi_values)
            self._calc_stats_for_gPCE(dist, key, qoi_gPCE)

        print(f"[LARSIM STAT INFO] calcStatisticsForSc function is done!")

    def _calc_stats_for_gPCE(self, dist, key, qoi_gPCE):
        if qoi_gPCE is None:
            qoi_gPCE = self.qoi_gPCE

        numPercSamples = 10 ** 5

        self.result_dict[key]["gPCE"] = qoi_gPCE
        self.result_dict[key]["E"] = float(cp.E(qoi_gPCE, dist))
        self.result_dict[key]["Var"] = float(cp.Var(qoi_gPCE, dist))
        self.result_dict[key]["StdDev"] = float(cp.Std(qoi_gPCE, dist))
        #self.result_dict[key]["qoi_dist"] = cp.QoI_Dist(qoi_gPCE, dist) # not working!

        # # generate QoI dist
        # qoi_dist = cp.QoI_Dist(self.qoi_gPCE, dist)
        # # generate sampling values for the qoi dist (you should know the min/max values for doing this)
        # dist_sampling_values = np.linspace(min_value, max_value, 1e4, endpoint=True)
        # # sample the QoI dist on the generated sampling values
        # pdf_samples = qoi_dist.pdf(dist_sampling_values)
        # # plot it (for example with matplotlib) ...

        self.result_dict[key]["P10"] = float(cp.Perc(qoi_gPCE, 10, dist, numPercSamples))
        self.result_dict[key]["P90"] = float(cp.Perc(qoi_gPCE, 90, dist, numPercSamples))
        if isinstance(self.result_dict[key]["P10"], list) and len(self.result_dict[key]["P10"]) == 1:
            self.result_dict[key]["P10"]= self.result_dict[key]["P10"][0]
            self.result_dict[key]["P90"] = self.result_dict[key]["P90"][0]

        if self._compute_Sobol_t:
            self.result_dict[key]["Sobol_t"] = cp.Sens_t(qoi_gPCE, dist)
        if self._compute_Sobol_m:
            self.result_dict[key]["Sobol_m"] = cp.Sens_m(qoi_gPCE, dist)
        if self._compute_Sobol_m2:
            self.result_dict[key]["Sobol_m2"] = cp.Sens_m2(qoi_gPCE, dist)  # second order sensitivity indices

    def calcStatisticsForMcSaltelli(self, rawSamples, timesteps,
                            simulationNodes, numEvaluations, order, regression, poly_normed, poly_rule, solverTimes,
                            work_package_indexes, original_runtime_estimator=None,  *args, **kwargs):

        self.samples = LarsimSamples(rawSamples, configurationObject=self.larsimConfObject)
        if self.save_samples:
            self.samples.save_samples_to_file(self.workingDir)
            self.samples.save_index_parameter_values(self.workingDir)
            self.samples.save_index_parameter_gof_values(self.workingDir)
            if self.larsimConfObject.compute_gradients:
                self.active_scores_dict = self._compute_active_score(self.samples.dict_of_matrix_c_eigen_decomposition)
                self.samples.save_dict_of_approx_matrix_c(self.workingDir)
                self.samples.save_dict_of_matrix_c_eigen_decomposition(self.workingDir)

        self.timesteps = self.samples.get_simulation_timesteps()
        self.timesteps_min = self.samples.get_timesteps_min()
        self.timesteps_max = self.samples.get_timesteps_max()
        self.pdTimesteps = [pd.Timestamp(timestep) for timestep in self.timesteps]
        self.number_of_unique_index_runs = self.samples.get_number_of_runs()

        self.numEvaluations = numEvaluations
        # TODO Think about this, tricky for saltelli, makes sense for mc
        # self.numEvaluations = self.number_of_unique_index_runs

        self.numbTimesteps = len(self.timesteps)
        print(f"[LARSIM STAT INFO] numbTimesteps is: {self.numbTimesteps}")

        self.samples_station_names = self.samples.get_simulation_stations()
        # self.samples_qoi_columns = self.samples.get_list_of_qoi_column_names()
        self.station_of_Interest = list(set(self.samples_station_names).intersection(self.station_of_Interest))
        if not self.station_of_Interest:
            self.station_of_Interest = self.samples_station_names

        grouped = self.samples.df_simulation_result.groupby(['Stationskennung','TimeStamp'])
        groups = grouped.groups

        self.dim = len(simulationNodes.distNodes[0])

        for key, val_indices in groups.items():
            self.result_dict[key] = dict()

            # numpy array - for sartelli it should be n(2+d)x1
            qoi_values = self.samples.df_simulation_result.loc[val_indices.values][self.qoi_column].values
            # extended_standard_qoi_values = qoi_values[:(2*numEvaluations)]
            qoi_values_saltelli = qoi_values[:, np.newaxis]
            # values based on which we calculate standard statistics
            standard_qoi_values = qoi_values_saltelli[:numEvaluations, :]
            extended_standard_qoi_values = qoi_values_saltelli[:(2*numEvaluations), :]
            if self.store_qoi_data_in_stat_dict:
                self.result_dict[key]["qoi_values"] = standard_qoi_values

            #self.result_dict[key]["min_q"] = np.amin(qoi_values) #standard_qoi_values.min()
            #self.result_dict[key]["max_q"] = np.amax(qoi_values) #standard_qoi_values.max()

            #self.result_dict[key]["E"] = np.sum(extended_standard_qoi_values, axis=0, dtype=np.float64) / (2*numEvaluations)
            self.result_dict[key]["E"] = np.mean(qoi_values[:(2 * numEvaluations)], 0)
            #self.result_dict[key]["Var"] = float(np.sum(power(standard_qoi_values)) / numEvaluations - self.result_dict[key]["E"] ** 2)
            self.result_dict[key]["Var"] = np.sum((extended_standard_qoi_values - self.result_dict[key]["E"]) ** 2,
                                                  axis=0, dtype=np.float64) / (2*numEvaluations - 1)
            #self.result_dict[key]["StdDev"] = np.sqrt(self.result_dict[key]["Var"], dtype=np.float64)
            self.result_dict[key]["StdDev"] = np.std(qoi_values[:(2 * numEvaluations)], 0, ddof=1)

            #self.result_dict[key]["P10"] = np.percentile(qoi_values[:numEvaluations], 10, axis=0)
            #self.result_dict[key]["P90"] = np.percentile(qoi_values[:numEvaluations], 90, axis=0)
            self.result_dict[key]["P10"] = np.percentile(qoi_values[:(2 * numEvaluations)], 10, axis=0)
            self.result_dict[key]["P90"] = np.percentile(qoi_values[:(2 * numEvaluations)], 90, axis=0)
            if isinstance(self.result_dict[key]["P10"], list) and len(self.result_dict[key]["P10"]) == 1:
                self.result_dict[key]["P10"] = self.result_dict[key]["P10"][0]
                self.result_dict[key]["P90"] = self.result_dict[key]["P90"][0]

            if self._compute_Sobol_t:
                self.result_dict[key]["Sobol_t"] = saltelliSobolIndicesHelpingFunctions._Sens_t_sample_4(
                    qoi_values_saltelli, self.dim, numEvaluations)
            if self._compute_Sobol_m:
                self.result_dict[key]["Sobol_m"] = saltelliSobolIndicesHelpingFunctions._Sens_m_sample_4(
                    qoi_values_saltelli, self.dim, numEvaluations)
                # self.result_dict[key]["Sobol_m"] = saltelliSobolIndicesHelpingFunctions._Sens_m_sample_3
                # (qoi_values_saltelli, self.dim, numEvaluations)

        print(f"[LARSIM STAT INFO] calcStatisticsForMcSaltelli function is done!")

    ###################################################################################################################

    def _check_if_Sobol_t_computed(self, keyIter):
        self._is_Sobol_t_computed = "Sobol_t" in self.result_dict[keyIter[0]] #hasattr(self.result_dict[keyIter[0], "Sobol_t")

    def _check_if_Sobol_m_computed(self, keyIter):
        self._is_Sobol_m_computed = "Sobol_m" in self.result_dict[keyIter[0]]

    def _check_if_Sobol_m2_computed(self, keyIter):
        self._is_Sobol_m2_computed = "Sobol_m2" in self.result_dict[keyIter[0]]

    # TODO timestepRange calculated based on whole Time-series
    def get_measured_data(self, timestepRange=None):
        transform_measured_to_daily = False
        if self.larsimConfObject.dailyOutput and self.qoi_column == "Value":
            transform_measured_to_daily = True

        local_measurment_file = os.path.abspath(os.path.join(str(self.workingDir), "df_measured.pkl"))
        if os.path.exists(local_measurment_file):
            self.df_measured = larsimDataPostProcessing.read_process_write_discharge(df=local_measurment_file,
                                                                                     timeframe=timestepRange,
                                                                                     station=self.station_of_Interest,
                                                                                     dailyOutput=transform_measured_to_daily,
                                                                                     compression="gzip")
        else:
            # TODO remove this, the path to mesured data is hardcoded
            self.df_measured = larsimConfig.extract_measured_discharge(
                timestepRange[0], timestepRange[1], index_run=0
            )
            self.df_measured = larsimDataPostProcessing.filterResultForStationAndTypeOfOutpu(self.df_measured,
                                                                                             station=self.station_of_Interest,
                                                                                             type_of_output=self.larsimConfObject.type_of_output_of_Interest_measured)
            if transform_measured_to_daily:
                # TODO resample_time_series_df was refactored
                if "Value" in self.df_measured.columns:
                    transform_dict = {"Value": 'mean', }
                else:
                    transform_dict = dict()
                    for single_station in self.station_of_Interest:
                        transform_dict[single_station] = 'mean'
                self.df_measured = larsimDataPostProcessing.resample_time_series_df(
                    self.df_measured,
                    transform_dict=transform_dict,
                    groupby_some_columns=True,
                    columns_to_groupby=["Stationskennung", ],
                    time_column="TimeStamp",
                    resample_freq="D"
                )
        self.groundTruth_computed = True
        #self.result_dict["Ground_Truth_Measurements"] = self.measured

    # TODO timestepRange calculated based on whole Time-series
    def get_unaltered_run_data(self, timestepRange=None):
        transform_unaltered_to_daily = False
        if self.larsimConfObject.dailyOutput and self.qoi_column == "Value":
            transform_unaltered_to_daily = True
        df_unaltered_file = os.path.abspath(os.path.join(str(self.workingDir), "df_unaltered.pkl"))
        if paths.check_if_file_exists(df_unaltered_file,
                                      message="[LARSIM STAT INFO] df_unaltered_file does not exist"):
            self.df_unaltered = larsimDataPostProcessing.read_process_write_discharge(
                df=df_unaltered_file,
                timeframe=timestepRange,
                type_of_output=self.larsimConfObject.type_of_output_of_Interest,
                station=self.station_of_Interest,
                dailyOutput=transform_unaltered_to_daily,
                compression="gzip")
            self.unaltered_computed = True
        else:
            self.unaltered_computed = False
        #self.result_dict["Unaltered"] = self.unalatered

    ###################################################################################################################

    def plotResults(self, timestep=-1, display=False,
                    fileName="", fileNameIdent="", directory="./",
                    fileNameIdentIsFullName=False, safe=True, **kwargs):

        plot_measured_timeseries = kwargs.get('plot_measured_timeseries') \
            if "plot_measured_timeseries" in kwargs else False
        plot_unaltered_timeseries = kwargs.get('plot_unaltered_timeseries') \
            if "plot_unaltered_timeseries" in kwargs else False
        if self.timesteps_min is None or self.timesteps_max is None:
            if plot_measured_timeseries:
                self.get_measured_data()
            if plot_unaltered_timeseries:
                self.get_unaltered_run_data()
        else:
            # timestepRange = (pd.Timestamp(min(self.timesteps)), pd.Timestamp(max(self.timesteps)))
            timestepRange = (pd.Timestamp(self.timesteps_min), pd.Timestamp(self.timesteps_max))
            if plot_measured_timeseries:
                self.get_measured_data(timestepRange=timestepRange)
            if plot_unaltered_timeseries:
                self.get_unaltered_run_data(timestepRange=timestepRange)

        print(f"[LARSIM STAT INFO] plotResults function is called!")

        for single_station in self.station_of_Interest:
            fileName = single_station + "_" + self.qoi_column
            single_fileName = self.generateFileName(fileName=fileName, fileNameIdent=".html",
                                             directory=directory, fileNameIdentIsFullName=fileNameIdentIsFullName)
            fig = self._plotStatisticsDict_plotly(unalatered=self.unaltered_computed, measured=self.groundTruth_computed,
                                                  station=single_station, recalculateTimesteps=False,
                                                  filename=single_fileName, display=display)
            # self._plotStatisticsDict_plotter(unalatered=None, measured=None, station=single_station,
            #                                  recalculateTimesteps=False, filename=single_fileName, display=display)
            if display:
                fig.show()
        print(f"[LARSIM STAT INFO] plotResults function is done!")

    def _plotStatisticsDict_plotly(self, unalatered=False, measured=False, station="MARI",
                                   recalculateTimesteps=False, window_title='Larsim Forward UQ & SA',
                                   filename="sim-plotly.html", display=False, uq_method=None):

        print(f"[LARSIM STAT INFO] _plotStatisticsDict_plotly function is called!")

        # if uq_method is None:
        #     if self.uq_method is None:
        #         raise Exception("_plotStatisticsDict_plotly - Please specify uq_method argument!")
        #     else:
        #         uq_method = self.uq_method

        #TODO Access to timesteps in a two different ways, e.g. one for QoI one for Q
        if recalculateTimesteps:
            result_dict_keys_list = list(self.result_dict.keys())#[:]
            pdTimesteps = []
            for i in range(0, len(result_dict_keys_list)):
                pdTimesteps.append(pd.Timestamp(result_dict_keys_list[i][1]))
        else:
            pdTimesteps = self.pdTimesteps

        keyIter = list(itertools.product([station,], pdTimesteps))

        self._check_if_Sobol_t_computed(keyIter)
        self._check_if_Sobol_m_computed(keyIter)

        if self.qoi_column == "Value":
            starting_row = 1
        else:
            starting_row = 2

        n_rows, sobol_t_row, sobol_m_row = self._compute_number_of_rows_for_plotting(starting_row)

        if self.qoi_column != "Value":
            n_rows = n_rows+1

        fig = make_subplots(rows=n_rows, cols=1, print_grid=True, shared_xaxes=False)

        if unalatered:
            column_to_draw = 'Value' if 'Value' in self.df_unaltered.columns else station
            fig.add_trace(go.Scatter(x=self.df_unaltered['TimeStamp'], y=self.df_unaltered[column_to_draw],
                                     name="Q (unaltered simulation)", line_color='deepskyblue'),
                          row=1, col=1)
        if measured:
            column_to_draw = 'Value' if 'Value' in self.df_measured.columns else station
            fig.add_trace(go.Scatter(x=self.df_measured['TimeStamp'], y=self.df_measured[column_to_draw],
                                     name="Q (measured)",line_color='red'),
                          row=1, col=1)

        fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.result_dict[key]["E"] for key in keyIter], name='E[QoI]',
                                 line_color='green', mode='lines'),
                      row=starting_row, col=1)
        #fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.result_dict[key]["min_q"] for key in keyIter],
        # name='min_q',line_color='indianred', mode='lines'), row=starting_row, col=1)
        #fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.result_dict[key]["max_q"] for key in keyIter],
        # name='max_q',line_color='yellow', mode='lines'), row=starting_row, col=1)
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
                # if uq_method == "saltelli":
                #     fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.result_dict[key]["Sobol_m"][i][0] for key in keyIter],
                #                              name=name, legendgroup=self.labels[i], line_color=colors.COLORS[i]),
                #                   row=sobol_m_row, col=1)
                # else:
                fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.result_dict[key]["Sobol_m"][i] for key in keyIter],
                                         name=name, legendgroup=self.labels[i], line_color=colors.COLORS[i]),
                              row=sobol_m_row, col=1)
        if self._is_Sobol_t_computed:
            for i in range(len(self.labels)):
                name = self.labels[i] + "_S_t"
                # if uq_method == "saltelli":
                #     fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.result_dict[key]["Sobol_t"][i][0] for key in keyIter],
                #                              name=name, legendgroup=self.labels[i], line_color=colors.COLORS[i]),
                #                   row=sobol_t_row, col=1)
                # else:
                fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.result_dict[key]["Sobol_t"][i] for key in keyIter],
                                         name=name, legendgroup=self.labels[i], line_color=colors.COLORS[i]),
                              row=sobol_t_row, col=1)

        # TODO - Make additional plots - each parameter independently with color + normalized Q measured

        fig.update_traces(mode='lines')
        #fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="Q [m^3/s]", side='left', showgrid=True, row=1, col=1)

        if self.qoi_column == "Value":
            fig.update_yaxes(title_text="Std. Dev. [m^3/s]", side='left', showgrid=True, row=2, col=1)
        else:
            fig.update_yaxes(title_text=self.qoi_column, side='left', showgrid=True, row=starting_row, col=1)
            fig.update_yaxes(title_text="Std. Dev. [QoI]", side='left', showgrid=True, row=starting_row+1, col=1)

        if self._is_Sobol_m_computed:
            fig.update_yaxes(title_text="Sobol_m", side='left', showgrid=True, range=[0, 1], row=sobol_m_row, col=1)
        if self._is_Sobol_t_computed:
            fig.update_yaxes(title_text="Sobol_t", side='left', showgrid=True, range=[0, 1], row=sobol_t_row, col=1)

        window_title = window_title + ": " + station
        fig.update_layout(height=600, width=1000, title_text=window_title)
        #xaxis4_rangeslider_visible=True, xaxis4_rangeslider_thickness=0.05)

        print(f"[LARSIM STAT INFO] _plotStatisticsDict_plotly function is almost over!")

        plot(fig, filename=filename, auto_open=display)
        #fig.write_image(filename)
        #fig.show()
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

    def _plotStatisticsDict_plotter(self, unalatered=False, measured=False, station="MARI",
                                    recalculateTimesteps=False, window_title='Larsim Forward UQ & SA - MARI',
                                    filename="sim-plotter", display=True):
        raise NotImplementedError("Should have implemented this")

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
    # Most of the methods below make sense only when QoI is the time-series
    ###################################################################################################################
    def extract_mean_time_series(self):
        if self.result_dict is None:
            raise Exception('[LARSIM STAT INFO] extract_mean_time_series - self.result_dict is None. '
                            'Calculate the statistics first!')
        list_of_df_over_stations = []
        for single_station in self.station_of_Interest:
            keyIter = list(itertools.product([single_station, ], self.pdTimesteps))
            mean_time_series = [self.result_dict[key]["E"] for key in keyIter]
            # station_list = [single_station]*len(mean_time_series)
            # df_temp = pd.DataFrame(list(zip(mean_time_series, pdTimesteps, station_list)),
            #                        columns=['E','TimeStamp','Stationskennung'])
            df_temp = pd.DataFrame(list(zip(mean_time_series, self.pdTimesteps)), columns=[single_station, 'TimeStamp'])
            list_of_df_over_stations.append(df_temp)
        # self.qoi_mean = pd.concat(list_of_df_over_stations)
        self.qoi_mean_df = reduce(lambda left, right: pd.merge(left, right, on="TimeStamp", how='outer'),
                                  list_of_df_over_stations)

    def compare_mean_time_series_and_measured(self):
        if self.result_dict is None:
            raise Exception('[LARSIM STAT INFO] compare_mean_time_series_and_measured - self.result_dict is None. '
                            'Calculate the statistics first!')
        if not self.groundTruth_computed:
            raise Exception('[LARSIM STAT INFO] compare_mean_time_series_and_measured - '
                            'compute the measured data frame first!')
        if self.qoi_mean_df is None:
            raise Exception('[LARSIM STAT INFO] compare_mean_time_series_and_measured - '
                            'compute the mean time series first!')
        self.gof_mean_measured = larsimDataPostProcessing.calculateGoodnessofFit(
            measuredDF=self.df_measured, predictedDF=self.qoi_mean_df, station="all", gof_list="all",
            measuredDF_column_name='station', simulatedDF_column_name='station', filter_station=True,
            filter_type_of_output=False, return_dict=True)

    ###################################################################################################################
    # It makes sense that the function below are executed for one input station
    ###################################################################################################################

    def create_df_from_statistics_data_single_station(self, station=None, uq_method="sc"):
        station = self._check_station_argument(station)
        keyIter = list(itertools.product([station, ], self.pdTimesteps))
        mean_time_series = [self.result_dict[key]["E"] for key in keyIter]
        std_time_series = [self.result_dict[key]["StdDev"] for key in keyIter]
        p10_time_series = [self.result_dict[key]["P10"] for key in keyIter]
        p90_time_series = [self.result_dict[key]["P90"] for key in keyIter]
        station_list = [station] * len(mean_time_series)

        list_of_columns = [self.pdTimesteps, mean_time_series, std_time_series,
                           p10_time_series, p90_time_series, station_list]
        list_of_columns_names = ['TimeStamp', "E", "Std", "P10", "P90", "Stationskennung"]

        if self._is_Sobol_m_computed:
            for i in range(len(self.labels)):
                # if uq_method == "saltelli":
                #     sobol_m_time_series = [self.result_dict[key]["Sobol_m"][i][0] for key in keyIter]
                # else:
                sobol_m_time_series = [self.result_dict[key]["Sobol_m"][i] for key in keyIter]
                list_of_columns.append(sobol_m_time_series)
                temp = "sobol_m_" + self.labels[i]
                list_of_columns_names.append(temp)
        if self._is_Sobol_t_computed:
            for i in range(len(self.labels)):
                # if uq_method == "saltelli":
                #     sobol_t_time_series = [self.result_dict[key]["Sobol_t"][i][0] for key in keyIter]
                # else:
                sobol_t_time_series = [self.result_dict[key]["Sobol_t"][i] for key in keyIter]
                list_of_columns.append(sobol_t_time_series)
                temp = "sobol_t_" + self.labels[i]
                list_of_columns_names.append(temp)

        df_statistics_station = pd.DataFrame(list(zip(*list_of_columns)), columns=list_of_columns_names)

        if self.groundTruth_computed:
            temp = larsimDataPostProcessing.filterResultForStation(self.df_measured, station=station)
            column_to_extract = 'Value' if 'Value' in self.df_measured.columns else station
            df_statistics_station = pd.merge_ordered(df_statistics_station,
                                                          temp[[column_to_extract, "TimeStamp"]],
                                                          on="TimeStamp", how='outer', fill_method="ffill")
            df_statistics_station.rename(columns={column_to_extract: "measured", }, inplace=True)

            #  df_statistics_station['measured_standardized'] =
            #  (df_statistics_station.measured-df_statistics_station.measured.mean())/df_statistics_station.measured.std()
            df_statistics_station['measured_norm'] = \
                (df_statistics_station.measured - df_statistics_station.measured.min()) / \
                (df_statistics_station.measured.max() - df_statistics_station.measured.min())

        if self.unaltered_computed:
            temp = larsimDataPostProcessing.filterResultForStation(self.df_unaltered, station=station)
            column_to_extract = 'Value' if 'Value' in self.df_unaltered.columns else station
            df_statistics_station = pd.merge_ordered(df_statistics_station,
                                                          temp[[column_to_extract, "TimeStamp"]],
                                                          on="TimeStamp", how='outer', fill_method="ffill")
            df_statistics_station.rename(columns={column_to_extract: "unaltered", }, inplace=True)

        df_statistics_station["E_minus_std"] = df_statistics_station['E'] - df_statistics_station['Std']
        df_statistics_station["E_plus_std"] = df_statistics_station['E'] + df_statistics_station['Std']
        return df_statistics_station

    def compute_gof_over_different_time_series(self, objective_function, df_statistics_station=None, station=None):
        station = self._check_station_argument(station)

        if df_statistics_station is None:
            df_statistics_station = self.create_df_from_statistics_data_single_station(station)

        if df_statistics_station is None:
            return

        if not callable(
                objective_function) and objective_function in larsimDataPostProcessing.mapping_gof_names_to_functions:
            objective_function = larsimDataPostProcessing.mapping_gof_names_to_functions[objective_function]
        elif not callable(
                objective_function) and objective_function not in larsimDataPostProcessing.mapping_gof_names_to_functions \
                or callable(objective_function) and objective_function not in larsimDataPostProcessing._all_functions:
            raise ValueError("Not proper specification of Goodness of Fit function name")

        gof_meas_unalt = None
        gof_meas_mean = None
        gof_meas_mean_m_std = None
        gof_meas_mean_p_std = None
        gof_meas_p10 = None
        gof_meas_p90 = None

        if 'unaltered' in df_statistics_station.columns:
            gof_meas_unalt = objective_function(df_statistics_station, df_statistics_station,
                                                measuredDF_column_name='measured', simulatedDF_column_name='unaltered')
        if 'E' in df_statistics_station.columns:
            gof_meas_mean = objective_function(df_statistics_station, df_statistics_station,
                                               measuredDF_column_name='measured', simulatedDF_column_name='E')
        if 'E_minus_std' in df_statistics_station.columns:
            gof_meas_mean_m_std = objective_function(df_statistics_station, df_statistics_station,
                                                     measuredDF_column_name='measured',
                                                     simulatedDF_column_name='E_minus_std')
        if 'E_plus_std' in df_statistics_station.columns:
            gof_meas_mean_p_std = objective_function(df_statistics_station, df_statistics_station,
                                                     measuredDF_column_name='measured',
                                                     simulatedDF_column_name='E_plus_std')
        if 'P10' in df_statistics_station.columns:
            gof_meas_p10 = objective_function(df_statistics_station, df_statistics_station,
                                              measuredDF_column_name='measured', simulatedDF_column_name='P10')
        if 'P90' in df_statistics_station.columns:
            gof_meas_p90 = objective_function(df_statistics_station, df_statistics_station,
                                              measuredDF_column_name='measured', simulatedDF_column_name='P90')

        print(f"gof_meas_unalt:{gof_meas_unalt} \ngof_meas_mean:{gof_meas_mean} \n"
              f"gof_meas_mean_m_std:{gof_meas_mean_m_std} \ngof_meas_mean_p_std:{gof_meas_mean_p_std} \n"
              f"gof_meas_p10:{gof_meas_p10} \ngof_meas_p90:{gof_meas_p90} \n")

    ###################################################################################################################

    def create_df_from_sensitivity_indices_for_singe_station(self, station=None, si_type="Sobol_t", uq_method="sc"):
        """
        si_type should be: Sobol_t, Sobol_m or Sobol_m2
        """
        if si_type == "Sobol_t" and not self._is_Sobol_t_computed:
            raise Exception("Sobol Total Order Indices are not computed")
        elif si_type == "Sobol_m" and not self._is_Sobol_m_computed:
            raise Exception("Sobol Main Order Indices are not computed")
        elif si_type == "Sobol_m2" and not self._is_Sobol_m2_computed:
            raise Exception("Sobol Second Order Indices are not computed")

        station = self._check_station_argument(station)

        keyIter = list(itertools.product([station, ], self.pdTimesteps))

        list_of_df_over_parameters = []
        for i in range(len(self.labels)):
            # if uq_method == "saltelli":
            #     si_single_param = [self.result_dict[key][si_type][i][0] for key in keyIter]
            # else:
            si_single_param = [self.result_dict[key][si_type][i] for key in keyIter]
            df_temp = pd.DataFrame(list(zip(si_single_param, self.pdTimesteps)),
                                   columns=[si_type + "_" + self.labels[i], 'TimeStamp'])
            list_of_df_over_parameters.append(df_temp)
        si_df = reduce(lambda left, right: pd.merge(left, right, on="TimeStamp", how='outer'),
                              list_of_df_over_parameters)

        if self.groundTruth_computed:
            temp = larsimDataPostProcessing.filterResultForStation(self.df_measured, station=station)
            column_to_extract = 'Value' if 'Value' in self.df_measured.columns else station
            si_df = pd.merge_ordered(si_df, temp[[column_to_extract, "TimeStamp"]], on="TimeStamp",
                                     how='outer', fill_method="ffill")
            si_df.rename(columns={column_to_extract: "measured", }, inplace=True)

        si_df.set_index("TimeStamp", inplace=True)
        return si_df

    def plot_heatmap_si_for_single_station(self, si_df=None, station=None, si_type="Sobol_t", uq_method="sc"):
        if si_df is None:
            si_df = self.create_df_from_sensitivity_indices_for_singe_station(station, si_type, uq_method)
        si_columns = [x for x in si_df.columns.tolist() if x != 'measured']
        fig = px.imshow(si_df[si_columns].T, labels=dict(y='Parameter'))
        return fig

    def plot_si_indices_over_time(self, station=None, si_type="Sobol_t", uq_method="sc"):
        fig = go.Figure()
        station = self._check_station_argument(station)
        keyIter = list(itertools.product([station, ], self.pdTimesteps))
        for i in range(len(self.labels)):
            # if uq_method == "saltelli":
            #     fig.add_trace(
            #         go.Scatter(x=self.pdTimesteps, y=[self.result_dict[key][si_type][i][0] for key in keyIter],
            #                    name=self.labels[i], legendgroup=self.labels[i], line_color=colors.COLORS[i]))
            # else:
            fig.add_trace(
                go.Scatter(x=self.pdTimesteps, y=[self.result_dict[key][si_type][i] for key in keyIter],
                           name=self.labels[i], legendgroup=self.labels[i], line_color=colors.COLORS[i]))

        return fig

    def plot_si_and_normalized_measured_time_signal(self, si_df=None,
                                                    df_statistics_station=None, station=None, si_type="Sobol_t",
                                                    measured_norm_columns_name="measured_norm", uq_method="sc",
                                                    plot_precipitation=False
                                                    ):
        station = self._check_station_argument(station)
        si_columns = [x for x in si_df.columns.tolist() if x != 'measured']

        if si_df is None:
            si_df = self.create_df_from_sensitivity_indices_for_singe_station(station, si_type, uq_method)
        if df_statistics_station is None:
            df_statistics_station = self.create_df_from_statistics_data_single_station(station, uq_method)
        fig = px.line(si_df, x=si_df.index, y=si_columns)
        fig.add_trace(go.Scatter(x=df_statistics_station['TimeStamp'],
                                 y=df_statistics_station[measured_norm_columns_name],
                                 fill='tozeroy', name="Normalized Q[m^3/s]"))
        if plot_precipitation:
            self._add_precipitation_to_graph(fig)
        return fig

    ###################################################################################################################

    def _check_station_argument(self, station=None):
        if station is None:
            if not isinstance(self.station_of_Interest, list):
                station = self.station_of_Interest
            else:
                station = self.station_of_Interest[0]
        return station

    def calculate_p_factor(self, df_statistics_station=None, station=None,
                                  column_lower_uncertainty_bound="P10", column_upper_uncertainty_bound="P90",
                                  observed_column="measured"):
        station = self._check_station_argument(station)

        if df_statistics_station is None:
            df_statistics_station = self.create_df_from_statistics_data_single_station(station)

        condition = df_statistics_station[
            (df_statistics_station[observed_column] >= df_statistics_station[column_lower_uncertainty_bound]) & (
                        df_statistics_station[observed_column] <= df_statistics_station[column_upper_uncertainty_bound])]

        p = len(condition.index) / len(df_statistics_station.index)
        print(f"P factor is: {p * 100}%")
        return p

    def compute_stat_of_uncertainty_band(self, df_statistics_station=None, station=None,
                                  column_lower_uncertainty_bound="P10", column_upper_uncertainty_bound="P90",
                                  observed_column="measured"):
        station = self._check_station_argument(station)
        if df_statistics_station is None:
            df_statistics_station = self.create_df_from_statistics_data_single_station(station)
        mean_uncertainty_band = np.mean(
            df_statistics_station[column_upper_uncertainty_bound] - df_statistics_station[
                column_lower_uncertainty_bound])
        std_uncertainty_band = np.std(
            df_statistics_station[column_upper_uncertainty_bound] - df_statistics_station[
                column_lower_uncertainty_bound])
        mean_observed = df_statistics_station[observed_column].mean()
        std_observed = df_statistics_station[observed_column].std(ddof=1)
        return mean_uncertainty_band, std_uncertainty_band, mean_observed, std_observed

    # TODO calculate_gof_over_leap_time (!?)
    def calculate_gof_over_lead_time(self):
        pass

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

    def _add_precipitation_to_graph(self, fig):
        n_lila_local_file = self.workingDir / 'master_configuration' / "station-n.lila"
        df_n = larsimIO.any_lila_parser_toPandas(n_lila_local_file)
        df_n = larsimDataPostProcessing.get_time_vs_station_values_df(df_n)
        df_n = larsimDataPostProcessing.parse_df_based_on_time(df_n, (self.timesteps_min,
                                                                      self.timesteps_max
                                                                      ))
        df_n.set_index("TimeStamp", inplace=True)
        mean_n = df_n.mean(axis=1)
        max_n = df_n.max(axis=1)
        max_N = max(list(df_n.max()))

        fig.add_trace(go.Scatter(x=df_n.index, \
                                 y=max_n, \
                                 text=max_n, \
                                 name="N_Max",
                                 yaxis="y2", ))
        fig.update_layout(
            xaxis=dict(
                autorange=True,
                range=[self.timesteps_min, self.timesteps_max],
                type="date"
            ),
            yaxis=dict(
                side="left",
                domain=[0, 0.7],
                mirror=True,
                tickfont={"color": "#d62728"},
                tickmode="auto",
                ticks="inside",
                titlefont={"color": "#d62728"},
            ),
            yaxis2=dict(
                anchor="x",
                domain=[0.7, 1],
                mirror=True,
                range=[max_N, 0],
                side="right",
                tickfont={"color": '#1f77b4'},
                nticks=3,
                tickmode="auto",
                ticks="inside",
                titlefont={"color": '#1f77b4'},
                title="Precipitation [mm/h]",
                type="linear",
            )
        )

        return fig