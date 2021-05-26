import chaospy as cp
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
import pickle
from plotly.offline import iplot, plot
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
import time
import sys

from uqef.stat import Statistics

from LarsimUtilityFunctions import larsimDataPostProcessing
from LarsimUtilityFunctions import larsimInputOutputUtilities
from LarsimUtilityFunctions import larsimConfigurationSettings
import LarsimUtilityFunctions.larsimPaths as paths

#from Larsim-UQ.common import saltelliSobolIndicesHelpingFunctions
from common import saltelliSobolIndicesHelpingFunctions

from numba import jit, prange

COLORS = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]

s = '''
        aliceblue, antiquewhite, aqua, aquamarine, azure,
        beige, bisque, black, blanchedalmond, blue,
        blueviolet, brown, burlywood, cadetblue,
        chartreuse, chocolate, coral, cornflowerblue,
        cornsilk, crimson, cyan, darkblue, darkcyan,
        darkgoldenrod, darkgray, darkgrey, darkgreen,
        darkkhaki, darkmagenta, darkolivegreen, darkorange,
        darkorchid, darkred, darksalmon, darkseagreen,
        darkslateblue, darkslategray, darkslategrey,
        darkturquoise, darkviolet, deeppink, deepskyblue,
        dimgray, dimgrey, dodgerblue, firebrick,
        floralwhite, forestgreen, fuchsia, gainsboro,
        ghostwhite, gold, goldenrod, gray, grey, green,
        greenyellow, honeydew, hotpink, indianred, indigo,
        ivory, khaki, lavender, lavenderblush, lawngreen,
        lemonchiffon, lightblue, lightcoral, lightcyan,
        lightgoldenrodyellow, lightgray, lightgrey,
        lightgreen, lightpink, lightsalmon, lightseagreen,
        lightskyblue, lightslategray, lightslategrey,
        lightsteelblue, lightyellow, lime, limegreen,
        linen, magenta, maroon, mediumaquamarine,
        mediumblue, mediumorchid, mediumpurple,
        mediumseagreen, mediumslateblue, mediumspringgreen,
        mediumturquoise, mediumvioletred, midnightblue,
        mintcream, mistyrose, moccasin, navajowhite, navy,
        oldlace, olive, olivedrab, orange, orangered,
        orchid, palegoldenrod, palegreen, paleturquoise,
        palevioletred, papayawhip, peachpuff, peru, pink,
        plum, powderblue, purple, red, rosybrown,
        royalblue, saddlebrown, salmon, sandybrown,
        seagreen, seashell, sienna, silver, skyblue,
        slateblue, slategray, slategrey, snow, springgreen,
        steelblue, tan, teal, thistle, tomato, turquoise,
        violet, wheat, white, whitesmoke, yellow,
        yellowgreen
        '''
COLORS_ALL = s.split(',')
COLORS_ALL = [l.replace('\n', '') for l in COLORS_ALL]
COLORS_ALL = [l.replace(' ', '') for l in COLORS_ALL]


class LarsimSamples(object):
    """
     Samples is a collection of the (filtered) sampled results of a whole UQ simulation
     Prepares results from Model to Statistics
    """
    def __init__(self, rawSamples, configurationObject, QoI="Value"):

        station = configurationObject["Output"]["station_calibration_postproc"] \
            if "station_calibration_postproc" in configurationObject["Output"] else "MARI"
        type_of_output = configurationObject["Output"]["type_of_output"] \
            if "type_of_output" in configurationObject["Output"] else "result_dict Messung + Vorhersage"
        dailyOutput = configurationObject["Output"]["dailyOutput"] \
            if "dailyOutput" in configurationObject["Output"] else "False"

        qoi = configurationObject["Output"]["QOI"] if "QOI" in configurationObject["Output"] else "Q"
        mode = configurationObject["Output"]["mode"] \
            if "mode" in configurationObject["Output"] else "continuous"
        calculate_GoF = configurationObject["Output"]["calculate_GoF"]
        compute_gradients = configurationObject["Output"]["compute_gradients"]

        # TODO SHOULD VALUE Always be among QoI columns!?
        self.qoi_columns = ["Value",]
        if qoi == "GoF":
            objective_function_qoi = configurationObject["Output"]["objective_function_qoi"]
            if isinstance(objective_function_qoi, list):
                self.qoi_columns = self.qoi_columns + [single_gof.__name__ for single_gof in objective_function_qoi]
            else:
                self.qoi_columns = self.qoi_columns + [objective_function_qoi.__name__, ]
        #objective_function, interval, min_periods, method

        list_of_single_df = []
        list_index_parameters_dict = []
        list_of_single_index_parameter_gof_df = []
        # Important that the results inside rawSamples (resulted paths)
        # are sorted in order which corresponds to the parameters order
        for index_run, value in enumerate(rawSamples,):
            if isinstance(value, tuple):
                df_result = value[0]
                list_index_parameters_dict.append(value[1])
            elif isinstance(value, dict):
                df_result = value["result_time_series"]
                list_index_parameters_dict.append(value["parameters_dict"])
                if strtobool(calculate_GoF):
                    list_of_single_index_parameter_gof_df.append(value["gof_df"])
                if strtobool(compute_gradients):
                    df_gradient = value["gradient"]
            else:
                df_result = value

            df_single_ergebnis = larsimDataPostProcessing.read_process_write_discharge(df=df_result,\
                                 index_run=index_run,\
                                 type_of_output=type_of_output,\
                                 station=station)

            #larsimInputOutputUtilities._postProcessing_DataFrame_after_reading(df_single_ergebnis)
            #simulation_start_timestamp = pd.Timestamp(df_single_ergebnis.TimeStamp.min()) + datetime.timedelta(hours=self.warm_up_duration)
            #df_single_ergebnis = larsimDataPostProcessing.parse_df_based_on_time(df_single_ergebnis, (simulation_start_timestamp, None))

            list_of_single_df.append(df_single_ergebnis)

        self.df_simulation_result = pd.concat(list_of_single_df, ignore_index=True, sort=False, axis=0)

        larsimInputOutputUtilities._postProcessing_DataFrame_after_reading(self.df_simulation_result)

        if list_index_parameters_dict:
            self.df_index_parameter_values = pd.DataFrame(list_index_parameters_dict)
        else:
            self.df_index_parameter_values = None

        if list_of_single_index_parameter_gof_df:
            self.df_index_parameter_gof_values = pd.concat(list_of_single_index_parameter_gof_df,
                                                           ignore_index=True, sort=False, axis=0)
        else:
            self.df_index_parameter_gof_values = None

        print(f"[LARSIM STAT INFO] Number of Unique TimeStamps (Hourly): "
              f"{len(self.df_simulation_result.TimeStamp.unique())}")

        # TODO remove/refactor resample_time_series_df
        if strtobool(dailyOutput):
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
            print(f"[LARSIM STAT INFO] Number of Unique TimeStamps (Daily): {self.df_simulation_result.TimeStamp.nunique()}")

        # save only the real values, independent on QoI
        # TODO Does not work when only one time step!!!
        # self.df_time_discharges = self.df_simulation_result.groupby(["Stationskennung","TimeStamp"])["Value"].apply(
        # lambda df: df.reset_index(drop=True)).unstack()

    def save_samples_to_file(self, file_path='./'):
        self.df_simulation_result.to_pickle(
            os.path.abspath(os.path.join(file_path, "df_all_simulations.pkl")), compression="gzip")

    def save_index_parameter_values(self, file_path='./'):
        self.df_index_parameter_values.to_pickle(
            os.path.abspath(os.path.join(file_path, "df_all_index_parameter_values.pkl")), compression="gzip")

    def save_index_parameter_gof_values(self, file_path='./'):
        self.df_index_parameter_gof_values.to_pickle(
            os.path.abspath(os.path.join(file_path, "df_all_index_parameter_gof_values.pkl")), compression="gzip")

    def save_time_samples_to_file(self, file_path='./'):
        self.df_time_discharges.to_pickle(
            os.path.abspath(os.path.join(file_path, "df_all_time_simulations.pkl")), compression="gzip")

    def get_simulation_timesteps(self):
        return list(self.df_simulation_result.TimeStamp.unique())

    def get_timesteps_min(self):
        return self.df_simulation_result.TimeStamp.min()

    def get_timesteps_max(self):
        return self.df_simulation_result.TimeStamp.max()

    def get_simulation_stations(self):
        return list(self.df_simulation_result.Stationskennung.unique())

    def get_list_of_qoi_column_names(self):
        return self.qoi_columns

########################################################################################################################


def _my_parallel_calc_stats_for_MC(keyIter_chunk, discharge_values_chunk, numEvaluations,
                                   store_qoi_data_in_stat_dict=False):
    results = []
    for ip in range(0, len(keyIter_chunk)):  # for each peace of work
        key = keyIter_chunk[ip]
        discharge_values = discharge_values_chunk[ip]
        local_result_dict = dict()
        if store_qoi_data_in_stat_dict:
            local_result_dict["Q"] = discharge_values

        local_result_dict["E"] = np.sum(discharge_values, axis=0, dtype=np.float64) / numEvaluations
        local_result_dict["E_numpy"] = np.mean(discharge_values, 0)
        local_result_dict["Var"] = np.sum((discharge_values - local_result_dict["E"]) ** 2, axis=0,
                                          dtype=np.float64) / (numEvaluations - 1)
        local_result_dict["StdDev"] = np.sqrt(local_result_dict[key]["Var"], dtype=np.float64)
        local_result_dict["StdDev_numpy"] = np.std(discharge_values, 0, ddof=1)

        local_result_dict["P10"] = np.percentile(discharge_values, 10, axis=0)
        local_result_dict["P90"] = np.percentile(discharge_values, 90, axis=0)
        if isinstance(local_result_dict["P10"], list) and len(local_result_dict["P10"]) == 1:
            local_result_dict["P10"] = local_result_dict["P10"][0]
            local_result_dict["P90"] = local_result_dict["P90"][0]

        results.append((key, local_result_dict))
    return results


def _my_parallel_calc_stats_for_SC(keyIter_chunk, discharge_values_chunk,
                                   dist, polynomial_expansion, nodes,
                                   compute_Sobol_t=False, compute_Sobol_m=False,
                                   store_qoi_data_in_stat_dict=False):
    pass


def _my_parallel_calc_stats_for_gPCE(keyIter_chunk, discharge_values_chunk,
                                     dist, polynomial_expansion, nodes, weights=None,
                                     regression=False, compute_Sobol_t=False, compute_Sobol_m=False,
                                     store_qoi_data_in_stat_dict=False):
    results = []
    for ip in range(0, len(keyIter_chunk)):  # for each peace of work
        key = keyIter_chunk[ip]
        discharge_values = discharge_values_chunk[ip]
        local_result_dict = dict()
        if store_qoi_data_in_stat_dict:
            local_result_dict["Q"] = discharge_values
        if regression:
            qoi_gPCE = cp.fit_regression(polynomial_expansion, nodes, discharge_values)
        else:
            qoi_gPCE = cp.fit_quadrature(polynomial_expansion, nodes, weights, discharge_values)

        numPercSamples = 10 ** 5
        local_result_dict["gPCE"] = qoi_gPCE
        local_result_dict["E"] = float(cp.E(qoi_gPCE, dist))
        local_result_dict["Var"] = float(cp.Var(qoi_gPCE, dist))
        local_result_dict["StdDev"] = float(cp.Std(qoi_gPCE, dist))
        #local_result_dict["qoi_dist"] = cp.QoI_Dist(qoi_gPCE, dist)

        local_result_dict["P10"] = float(cp.Perc(qoi_gPCE, 10, dist, numPercSamples))
        local_result_dict["P90"] = float(cp.Perc(qoi_gPCE, 90, dist, numPercSamples))
        if isinstance(local_result_dict["P10"], list) and len(local_result_dict["P10"]) == 1:
            local_result_dict["P10"] = local_result_dict["P10"][0]
            local_result_dict["P90"] = local_result_dict["P90"][0]

        if compute_Sobol_t:
            local_result_dict["Sobol_t"] = cp.Sens_t(qoi_gPCE, dist)
        if compute_Sobol_m:
            local_result_dict["Sobol_m"] = cp.Sens_m(qoi_gPCE, dist)
            #local_result_dict["Sobol_m2"] = cp.Sens_m2(qoi_gPCE, dist) # second order sensitivity indices

        results.append((key, local_result_dict))
    return results


def _my_parallel_calc_stats_for_mc_saltelli(keyIter_chunk, discharge_values_chunk, numEvaluations, dim,
                                             compute_Sobol_t=False, compute_Sobol_m=False,
                                             store_qoi_data_in_stat_dict=False):
    results = []
    for ip in range(0, len(keyIter_chunk)):  # for each peace of work
        key = keyIter_chunk[ip]
        discharge_values = discharge_values_chunk[ip]
        local_result_dict = dict()

        discharge_values_saltelli = discharge_values[:, np.newaxis]
        standard_discharge_values = discharge_values_saltelli[:numEvaluations, :]
        extended_standard_discharge_values = discharge_values_saltelli[:(2 * numEvaluations), :]

        if store_qoi_data_in_stat_dict:
            local_result_dict["Q"] = standard_discharge_values

        local_result_dict["E"] = np.mean(discharge_values[:(2 * numEvaluations)], 0)
        local_result_dict["Var"] = np.sum((extended_standard_discharge_values - local_result_dict["E"]) ** 2,
                                          axis=0, dtype=np.float64) / (2 * numEvaluations - 1)
        local_result_dict["StdDev"] = np.std(discharge_values[:(2 * numEvaluations)], 0, ddof=1)
        local_result_dict["P10"] = np.percentile(discharge_values[:(2 * numEvaluations)], 10, axis=0)
        local_result_dict["P90"] = np.percentile(discharge_values[:(2 * numEvaluations)], 90, axis=0)
        if isinstance(local_result_dict["P10"], list) and len(local_result_dict["P10"]) == 1:
            local_result_dict["P10"] = local_result_dict["P10"][0]
            local_result_dict["P90"] = local_result_dict["P90"][0]

        if compute_Sobol_t:
            local_result_dict["Sobol_t"] = saltelliSobolIndicesHelpingFunctions._Sens_t_sample_4(
                discharge_values_saltelli, dim, numEvaluations)
        if compute_Sobol_m:
            local_result_dict["Sobol_m"] = saltelliSobolIndicesHelpingFunctions._Sens_m_sample_4(
                discharge_values_saltelli, dim, numEvaluations)

        results.append((key, local_result_dict))
    return results
########################################################################################################################


class LarsimStatistics(Statistics):
    """
       LarsimStatistics calculates the statistics for the LarsimModel
       One LarsimStatistics Object should compute statistics for a multiple station and single QoI
    """

    def __init__(self, configurationObject, *args, **kwargs):
        Statistics.__init__(self)

        self.configurationObject = configurationObject

        if "workingDir" in kwargs:
            self.workingDir = kwargs.get('workingDir')
        else:
            try:
                self.workingDir = self.configurationObject["Directories"]["workingDir"]
            except KeyError:
                self.workingDir = paths.workingDir

        # TODO for now this is hardcoded such that only a single self.qoi_column is supported
        self.qoi_column = kwargs.get('qoi_column') if "qoi_column" in kwargs else "Value"
        self.qoi = configurationObject["Output"]["QOI"] if "QOI" in configurationObject["Output"] else "Q"
        if self.qoi == "GoF":
            objective_function_qoi = configurationObject["Output"]["objective_function_qoi"]
            if isinstance(objective_function_qoi, list):
                self.qoi_column = objective_function_qoi[0].__name__
            else:
                self.qoi_column = objective_function_qoi.__name__

        self.result_dict = dict()

        self.df_unaltered = None
        self.df_measured = None
        self.unaltered_computed = False
        self.groundTruth_computed = False

        # check if simulation results were already saved in LarsimModel - currently not used
        self.run_and_save_simulations = strtobool(self.configurationObject["Output"]["run_and_save_simulations"])\
                                        if "run_and_save_simulations" in self.configurationObject["Output"] else False

        self.station_of_Interest = self.configurationObject["Output"]["station_calibration_postproc"]
        if not isinstance(self.station_of_Interest, list):
            self.station_of_Interest = [self.station_of_Interest,]

        # Only the names of the stochastic parameters
        self.nodeNames = []
        for i in self.configurationObject["parameters"]:
            if i["distribution"] != "None":
                self.nodeNames.append(i["name"])
        self.dim = len(self.nodeNames)
        self.labels = [nodeName.strip() for nodeName in self.nodeNames]

        self.uq_method = kwargs.get('uq_method') if 'uq_method' in kwargs else None

        self._compute_Sobol_t = kwargs.get('compute_Sobol_t') if 'compute_Sobol_t' in kwargs else True
        self._compute_Sobol_m = kwargs.get('compute_Sobol_m') if 'compute_Sobol_m' in kwargs else True
        self._is_Sobol_t_computed = False
        self._is_Sobol_m_computed = False

        self.timesteps_min = None
        self.timesteps_max = None
        self.numbTimesteps = None
        self.timesteps = None

        self.samples = None
        self.result_dict = None

        self.qoi_mean_df = None
        self.gof_mean_measured = None

        # df_statistics_station = None
        # si_t_df = None
        # si_m_df = None

        self.store_qoi_data_in_stat_dict = kwargs.get('store_qoi_data_in_stat_dict') if 'store_qoi_data_in_stat_dict' \
                                                                                        in kwargs else False

        # TODO: eventually make a non-MPI version
        self.parallel_statistics = kwargs.get('parallel_statistics') if 'parallel_statistics' in kwargs else False
        if self.parallel_statistics:
            self.size = MPI.COMM_WORLD.Get_size()
            self.rank = MPI.COMM_WORLD.Get_rank()
            self.name = MPI.Get_processor_name()
            self.version = MPI.Get_library_version()
            self.mpi_chunksize = kwargs.get('mpi_chunksize') if 'mpi_chunksize' in kwargs else 1
            self.unordered = kwargs.get('unordered') if 'unordered' in kwargs else False

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

        self.samples = LarsimSamples(rawSamples, configurationObject=self.configurationObject)
        self.samples.save_samples_to_file(self.workingDir)
        self.samples.save_index_parameter_values(self.workingDir)
        self.samples.save_index_parameter_gof_values(self.workingDir)

        self.timesteps = self.samples.get_simulation_timesteps()
        self.timesteps_min = self.samples.get_timesteps_min()
        self.timesteps_max = self.samples.get_timesteps_max()
        self.pdTimesteps = [pd.Timestamp(timestep) for timestep in self.timesteps]

        self.numbTimesteps = len(self.timesteps)

        # Update self.station_of_Interest, though there is probably no need for this!
        self.samples_station_names = self.samples.get_simulation_stations()
        self.station_of_Interest = list(set(self.samples_station_names).intersection(self.station_of_Interest))
        if not self.station_of_Interest:
            self.station_of_Interest = self.samples_station_names

    def preparePolyExpanForMc(self, simulationNodes, numEvaluations, regression=None, order=None,
                              poly_normed=None, poly_rule=None, *args, **kwargs):
        self.numEvaluations = numEvaluations
        if regression:
            self.nodes = simulationNodes.distNodes
            self.weights = None
            self.dist = simulationNodes.joinedDists
            self.polynomial_expansion = cp.generate_expansion(order, self.dist, rule=poly_rule, normed=poly_normed)

    def preparePolyExpanForSc(self, simulationNodes, order, poly_normed, poly_rule, *args, **kwargs):
        self.nodes = simulationNodes.distNodes
        self.dist = simulationNodes.joinedDists
        self.weights = simulationNodes.weights
        self.polynomial_expansion = cp.generate_expansion(order, self.dist, rule=poly_rule, normed=poly_normed)

    def preparePolyExpanForSaltelli(self, simulationNodes, numEvaluations=None, regression=None, order=None,
                                    poly_normed=None, poly_rule=None, *args, **kwargs):
        pass

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
                    chunk_results_it = executor.map(_my_parallel_calc_stats_for_gPCE,
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
                    chunk_results_it = executor.map(_my_parallel_calc_stats_for_MC,
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

        with futures.MPICommExecutor(MPI.COMM_WORLD, root=0) as executor:
            if executor is not None:  # master process
                solver_time_start = time.time()
                chunk_results_it = executor.map(_my_parallel_calc_stats_for_gPCE,
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
                chunk_results_it = executor.map(_my_parallel_calc_stats_for_mc_saltelli,
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
                            work_package_indexes, original_runtime_estimator, **kwargs):

        self.samples = LarsimSamples(rawSamples, configurationObject=self.configurationObject)
        self.samples.save_samples_to_file(self.workingDir)
        self.samples.save_index_parameter_values(self.workingDir)
        self.samples.save_index_parameter_gof_values(self.workingDir)

        self.timesteps = self.samples.get_simulation_timesteps()
        self.timesteps_min = self.samples.get_timesteps_min()
        self.timesteps_max = self.samples.get_timesteps_max()
        self.pdTimesteps = [pd.Timestamp(timestep) for timestep in self.timesteps]

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
            dist = simulationNodes.joinedDists
            polynomial_expansion = cp.generate_expansion(order, dist, rule=poly_rule, normed=poly_normed)

        for key, val_indices in groups.items():
            discharge_values = self.samples.df_simulation_result.loc[val_indices.values][self.qoi_column].values #numpy array nx1
            #discharge_values = self.samples.df_simulation_result.Value.loc[val_indices].values
            self.result_dict[key] = dict()
            if self.store_qoi_data_in_stat_dict:
                self.result_dict[key]["Q"] = discharge_values
            if regression:
                self.qoi_gPCE = cp.fit_regression(polynomial_expansion, nodes, discharge_values)
                self._calc_stats_for_gPCE(dist, key)
            else:
                # self.result_dict[key]["E"] = np.sum(discharge_values, axis=0, dtype=np.float64)/ numEvaluations
                self.result_dict[key]["E"] = np.mean(discharge_values, 0)
                #self.result_dict[key]["Var"] = float(np.sum(power(discharge_values)) / numEvaluations - self.result_dict[key]["E"]**2)
                self.result_dict[key]["Var"] = np.sum((discharge_values - self.result_dict[key]["E"]) ** 2,
                                                      axis=0, dtype=np.float64) / (numEvaluations - 1)
                # self.result_dict[key]["StdDev"] = np.sqrt(self.result_dict[key]["Var"], dtype=np.float64)
                self.result_dict[key]["StdDev"] = np.std(discharge_values, 0, ddof=1)
                self.result_dict[key]["P10"] = np.percentile(discharge_values, 10, axis=0)
                self.result_dict[key]["P90"] = np.percentile(discharge_values, 90, axis=0)
                if isinstance(self.result_dict[key]["P10"], list) and len(self.result_dict[key]["P10"]) == 1:
                    self.result_dict[key]["P10"] = self.result_dict[key]["P10"][0]
                    self.result_dict[key]["P90"] = self.result_dict[key]["P90"][0]

        print(f"[LARSIM STAT INFO] calcStatisticsForMc function is done!")

    def calcStatisticsForSc(self, rawSamples, timesteps,
                           simulationNodes, order, regression, poly_normed, poly_rule, solverTimes,
                           work_package_indexes, original_runtime_estimator, **kwargs):

        self.samples = LarsimSamples(rawSamples, configurationObject=self.configurationObject)
        self.samples.save_samples_to_file(self.workingDir)
        self.samples.save_index_parameter_values(self.workingDir)
        self.samples.save_index_parameter_gof_values(self.workingDir)

        self.timesteps = self.samples.get_simulation_timesteps()
        self.timesteps_min = self.samples.get_timesteps_min()
        self.timesteps_max = self.samples.get_timesteps_max()
        self.pdTimesteps = [pd.Timestamp(timestep) for timestep in self.timesteps]

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
        dist = simulationNodes.joinedDists
        weights = simulationNodes.weights
        polynomial_expansion = cp.generate_expansion(order, dist, rule=poly_rule, normed=poly_normed)

        grouped = self.samples.df_simulation_result.groupby(['Stationskennung','TimeStamp'])
        groups = grouped.groups

        for key, val_indices in groups.items():
            discharge_values = self.samples.df_simulation_result.loc[val_indices.values][self.qoi_column].values
            self.result_dict[key] = dict()
            if self.store_qoi_data_in_stat_dict:
                self.result_dict[key]["Q"] = discharge_values
            if regression:
                self.qoi_gPCE = cp.fit_regression(polynomial_expansion, nodes, discharge_values)
            else:
                self.qoi_gPCE = cp.fit_quadrature(polynomial_expansion, nodes, weights, discharge_values)
            self._calc_stats_for_gPCE(dist, key)

        print(f"[LARSIM STAT INFO] calcStatisticsForSc function is done!")

    def _calc_stats_for_gPCE(self, dist, key, qoi_gPCE=None):
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
            #self.result_dict[key]["Sobol_m2"] = cp.Sens_m2(qoi_gPCE, dist) # second order sensitivity indices

    def calcStatisticsForSaltelli(self, rawSamples, timesteps,
                            simulationNodes, numEvaluations, order, regression, poly_normed, poly_rule, solverTimes,
                            work_package_indexes, original_runtime_estimator=None, **kwargs):

        self.samples = LarsimSamples(rawSamples, configurationObject=self.configurationObject)
        self.samples.save_samples_to_file(self.workingDir)
        self.samples.save_index_parameter_values(self.workingDir)
        self.samples.save_index_parameter_gof_values(self.workingDir)

        self.timesteps = self.samples.get_simulation_timesteps()
        self.timesteps_min = self.samples.get_timesteps_min()
        self.timesteps_max = self.samples.get_timesteps_max()
        self.pdTimesteps = [pd.Timestamp(timestep) for timestep in self.timesteps]

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
            discharge_values = self.samples.df_simulation_result.loc[val_indices.values][self.qoi_column].values
            # extended_standard_discharge_values = discharge_values[:(2*numEvaluations)]
            discharge_values_saltelli = discharge_values[:, np.newaxis]
            # values based on which we calculate standard statistics
            standard_discharge_values = discharge_values_saltelli[:numEvaluations, :]
            extended_standard_discharge_values = discharge_values_saltelli[:(2*numEvaluations), :]
            if self.store_qoi_data_in_stat_dict:
                self.result_dict[key]["Q"] = standard_discharge_values

            #self.result_dict[key]["min_q"] = np.amin(discharge_values) #standard_discharge_values.min()
            #self.result_dict[key]["max_q"] = np.amax(discharge_values) #standard_discharge_values.max()

            #self.result_dict[key]["E"] = np.sum(extended_standard_discharge_values, axis=0, dtype=np.float64) / (2*numEvaluations)
            self.result_dict[key]["E"] = np.mean(discharge_values[:(2 * numEvaluations)], 0)
            #self.result_dict[key]["Var"] = float(np.sum(power(standard_discharge_values)) / numEvaluations - self.result_dict[key]["E"] ** 2)
            self.result_dict[key]["Var"] = np.sum((extended_standard_discharge_values - self.result_dict[key]["E"]) ** 2,
                                                  axis=0, dtype=np.float64) / (2*numEvaluations - 1)
            #self.result_dict[key]["StdDev"] = np.sqrt(self.result_dict[key]["Var"], dtype=np.float64)
            self.result_dict[key]["StdDev"] = np.std(discharge_values[:(2 * numEvaluations)], 0, ddof=1)

            #self.result_dict[key]["P10"] = np.percentile(discharge_values[:numEvaluations], 10, axis=0)
            #self.result_dict[key]["P90"] = np.percentile(discharge_values[:numEvaluations], 90, axis=0)
            self.result_dict[key]["P10"] = np.percentile(discharge_values[:(2 * numEvaluations)], 10, axis=0)
            self.result_dict[key]["P90"] = np.percentile(discharge_values[:(2 * numEvaluations)], 90, axis=0)
            if isinstance(self.result_dict[key]["P10"], list) and len(self.result_dict[key]["P10"]) == 1:
                self.result_dict[key]["P10"] = self.result_dict[key]["P10"][0]
                self.result_dict[key]["P90"] = self.result_dict[key]["P90"][0]

            if self._compute_Sobol_t:
                self.result_dict[key]["Sobol_t"] = saltelliSobolIndicesHelpingFunctions._Sens_t_sample_4(
                    discharge_values_saltelli, self.dim, numEvaluations)
            if self._compute_Sobol_m:
                self.result_dict[key]["Sobol_m"] = saltelliSobolIndicesHelpingFunctions._Sens_m_sample_4(
                    discharge_values_saltelli, self.dim, numEvaluations)
                # self.result_dict[key]["Sobol_m"] = saltelliSobolIndicesHelpingFunctions._Sens_m_sample_3
                # (discharge_values_saltelli, self.dim, numEvaluations)

        print(f"[LARSIM STAT INFO] calcStatisticsForSaltelli function is done!")

    ###################################################################################################################

    def _check_if_Sobol_t_computed(self, keyIter):
        self._is_Sobol_t_computed = "Sobol_t" in self.result_dict[keyIter[0]] #hasattr(self.result_dict[keyIter[0], "Sobol_t")

    def _check_if_Sobol_m_computed(self, keyIter):
        self._is_Sobol_m_computed = "Sobol_m" in self.result_dict[keyIter[0]] \
                                    or "Sobol_m2" in self.result_dict[keyIter[0]] #hasattr(self.result_dict[keyIter[0], "Sobol_m")

    # TODO timestepRange calculated based on whole Time-series
    def get_measured_discharge(self, timestepRange=None):
        transform_measured_to_daily = False
        if strtobool(self.configurationObject["Output"]["dailyOutput"]) and self.qoi_column == "Value":
            transform_measured_to_daily = True

        local_measurment_file = os.path.abspath(os.path.join(self.workingDir, "df_measured.pkl"))
        if os.path.exists(local_measurment_file):
            self.df_measured = larsimDataPostProcessing.read_process_write_discharge(df=local_measurment_file,\
                                     timeframe=timestepRange,\
                                     station=self.station_of_Interest,\
                                     dailyOutput=transform_measured_to_daily,\
                                     compression="gzip")
        else:
            self.df_measured = larsimConfigurationSettings.extract_measured_discharge(timestepRange[0], timestepRange[1], index_run=0)
            self.df_measured = larsimDataPostProcessing.filterResultForStationAndTypeOfOutpu(self.df_measured,\
                                                       station=self.station_of_Interest,\
                                                       type_of_output=self.configurationObject["Output"]["type_of_output_measured"])
            if transform_measured_to_daily:
                # TODO resample_time_series_df was refactored
                if "Value" in self.df_measured.columns:
                    transform_dict = {"Value": 'mean', }
                else:
                    transform_dict = dict()
                    for single_station in self.station_of_Interest:
                        transform_dict[single_station] = 'mean'
                self.df_measured = larsimDataPostProcessing.resample_time_series_df(self.df_measured,
                                                                                    transform_dict=transform_dict,
                                                                                    groupby_some_columns=True,
                                                                                    columns_to_groupby=["Stationskennung",],
                                                                                    time_column="TimeStamp",
                                                                                    resample_freq="D"
                                                                                    )
        self.groundTruth_computed = True
        #self.result_dict["Ground_Truth_Measurements"] = self.measured

    # TODO timestepRange calculated based on whole Time-series
    def get_unaltered_discharge(self, timestepRange=None):
        transform_unaltered_to_daily = False
        if strtobool(self.configurationObject["Output"]["dailyOutput"]) and self.qoi_column == "Value":
            transform_unaltered_to_daily = True
        df_unaltered_file = os.path.abspath(os.path.join(self.workingDir, "df_unaltered.pkl"))
        if paths.check_if_file_exists(df_unaltered_file,
                                      message="[LARSIM STAT INFO] df_unaltered_file does not exist"):
            self.df_unaltered = larsimDataPostProcessing.read_process_write_discharge(
                df=df_unaltered_file,
                timeframe=timestepRange,
                type_of_output=self.configurationObject["Output"]["type_of_output"],
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
            if "plot_measured_timeseries" in kwargs else True
        plot_unalteres_timeseries = kwargs.get('plot_unalteres_timeseries') \
            if "plot_unalteres_timeseries" in kwargs else True
        if self.timesteps_min is None or self.timesteps_max is None:
            if plot_measured_timeseries:
                self.get_measured_discharge()
            if plot_unalteres_timeseries:
                self.get_unaltered_discharge()
        else:
            # timestepRange = (pd.Timestamp(min(self.timesteps)), pd.Timestamp(max(self.timesteps)))
            timestepRange = (pd.Timestamp(self.timesteps_min), pd.Timestamp(self.timesteps_max))
            if plot_measured_timeseries:
                self.get_measured_discharge(timestepRange=timestepRange)
            if plot_unalteres_timeseries:
                self.get_unaltered_discharge(timestepRange=timestepRange)

        print(f"[LARSIM STAT INFO] plotResults function is called!")

        for single_station in self.station_of_Interest:
            fileName = single_station
            single_fileName = self.generateFileName(fileName=fileName, fileNameIdent=".html",
                                             directory=directory, fileNameIdentIsFullName=fileNameIdentIsFullName)
            self._plotStatisticsDict_plotly(unalatered=self.unaltered_computed, measured=self.groundTruth_computed,
                                            station=single_station,
                                            recalculateTimesteps=False, filename=single_fileName, display=display)
            # self._plotStatisticsDict_plotter(unalatered=None, measured=None, station=single_station,
            #                                  recalculateTimesteps=False, filename=single_fileName, display=display)
        print(f"[LARSIM STAT INFO] plotResults function is done!")

    def _plotStatisticsDict_plotly(self, unalatered=False, measured=False, station="MARI",
                                   recalculateTimesteps=False, window_title='Larsim Forward UQ & SA',
                                   filename="sim-plotly.html", display=False):

        print(f"[LARSIM STAT INFO] _plotStatisticsDict_plotly function is called!")

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
                fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.result_dict[key]["Sobol_m"][i] for key in keyIter],
                                         name=name, legendgroup=self.labels[i], line_color=COLORS[i]),
                              row=sobol_m_row, col=1)
        if self._is_Sobol_t_computed:
            for i in range(len(self.labels)):
                name = self.labels[i] + "_S_t"
                fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.result_dict[key]["Sobol_t"][i] for key in keyIter],
                                         name=name, legendgroup=self.labels[i], line_color=COLORS[i]),
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
        fig.show()

    def _plotStatisticsDict_plotter(self, unalatered=False, measured=False, station="MARI",
                                    recalculateTimesteps=False, window_title='Larsim Forward UQ & SA - MARI',
                                    filename="sim-plotter", display=True):
        raise NotImplementedError("Should have implemented this")


    def saveToFile(self, fileName="statistics_dict", fileNameIdent="", directory="./",
                   fileNameIdentIsFullName=False):

        statFileName = os.path.abspath(os.path.join(self.workingDir,"statistics_dictionary.pkl"))
        with open(statFileName, 'wb') as handle:
            pickle.dump(self.result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

    def create_df_from_statistics_data_singe_station(self, station=None):
        if station is None:
            if not isinstance(self.station_of_Interest, list):
                station = self.station_of_Interest
            else:
                station = self.station_of_Interest[0]
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
            temp = larsimDataPostProcessing.filterResultForStation(self.df_measured,
                                                                   station=station)
            column_to_extract = 'Value' if 'Value' in self.df_measured.columns else station
            df_statistics_station = pd.merge_ordered(df_statistics_station,
                                                          temp[[column_to_extract, "TimeStamp"]],
                                                          on="TimeStamp", how='outer', fill_method="ffill")
            df_statistics_station.rename(columns={column_to_extract: "measured", }, inplace=True)

        if self.unaltered_computed:
            temp = larsimDataPostProcessing.filterResultForStation(self.df_unaltered,
                                                                   station=station)
            column_to_extract = 'Value' if 'Value' in self.df_unaltered.columns else station
            df_statistics_station = pd.merge_ordered(df_statistics_station,
                                                          temp[[column_to_extract, "TimeStamp"]],
                                                          on="TimeStamp", how='outer', fill_method="ffill")
            df_statistics_station.rename(columns={column_to_extract: "unaltered", }, inplace=True)

        #  df_statistics_station['measured_standardized'] =
        #  (df_statistics_station.measured-df_statistics_station.measured.mean())/df_statistics_station.measured.std()
        df_statistics_station['measured_norm'] = \
            (df_statistics_station.measured - df_statistics_station.measured.min()) / \
            (df_statistics_station.measured.max() - df_statistics_station.measured.min())

        df_statistics_station["E_minus_std"] = df_statistics_station['E'] - df_statistics_station['Std']
        df_statistics_station["E_plus_std"] = df_statistics_station['E'] + df_statistics_station['Std']
        return df_statistics_station

    def compute_gof_over_different_time_series(self, objective_function, station=None):
        if station is None:
            if not isinstance(self.station_of_Interest, list):
                station = self.station_of_Interest
            else:
                station = self.station_of_Interest[0]
        df_statistics_station = self.create_df_from_statistics_data_singe_station(station)
        if df_statistics_station is None:
            return
        if not callable(
                objective_function) and objective_function in larsimDataPostProcessing.mapping_gof_names_to_functions:
            objective_function = larsimDataPostProcessing.mapping_gof_names_to_functions[objective_function]
        elif not callable(
                objective_function) and objective_function not in larsimDataPostProcessing.mapping_gof_names_to_functions \
                or callable(objective_function) and objective_function not in larsimDataPostProcessing._all_functions:
            raise ValueError("Not proper specification of Goodness of Fit function name")

        gof_meas_unalt = objective_function(df_statistics_station, df_statistics_station,
                                            measuredDF_column_name='measured', simulatedDF_column_name='unaltered')
        gof_meas_mean = objective_function(df_statistics_station, df_statistics_station,
                                           measuredDF_column_name='measured', simulatedDF_column_name='E')
        gof_meas_mean_m_std = objective_function(df_statistics_station, df_statistics_station,
                                                 measuredDF_column_name='measured',
                                                 simulatedDF_column_name='E_minus_std')
        gof_meas_mean_p_std = objective_function(df_statistics_station, df_statistics_station,
                                                 measuredDF_column_name='measured',
                                                 simulatedDF_column_name='E_plus_std')
        gof_meas_p10 = objective_function(df_statistics_station, df_statistics_station,
                                          measuredDF_column_name='measured', simulatedDF_column_name='P10')
        gof_meas_p90 = objective_function(df_statistics_station, df_statistics_station,
                                          measuredDF_column_name='measured', simulatedDF_column_name='P90')

        print(f"gof_meas_unalt:{gof_meas_unalt} \ngof_meas_mean:{gof_meas_mean} \n"
              f"gof_meas_mean_m_std:{gof_meas_mean_m_std} \ngof_meas_mean_p_std:{gof_meas_mean_p_std} \n"
              f"gof_meas_p10:{gof_meas_p10} \ngof_meas_p90:{gof_meas_p90} \n")

    def create_df_from_sensitivity_total_indices_for_singe_station(self, station=None):
        if not self._is_Sobol_t_computed:
            raise Exception("Sobol Total Order Indices are not computed")

        if station is None:
            if not isinstance(self.station_of_Interest, list):
                station = self.station_of_Interest
            else:
                station = self.station_of_Interest[0]

        keyIter = list(itertools.product([station, ], self.pdTimesteps))

        list_of_df_over_parameters = []
        for i in range(len(self.labels)):
            si_t_single_param = [self.result_dict[key]["Sobol_t"][i] for key in keyIter]
            df_temp = pd.DataFrame(list(zip(si_t_single_param, self.pdTimesteps)),
                                   columns=[self.labels[i] + "_si_t", 'TimeStamp'])
            list_of_df_over_parameters.append(df_temp)
        si_t_df = reduce(lambda left, right: pd.merge(left, right, on="TimeStamp", how='outer'),
                              list_of_df_over_parameters)

        if self.groundTruth_computed:
            temp = larsimDataPostProcessing.filterResultForStation(self.df_measured, station=station)
            column_to_extract = 'Value' if 'Value' in self.df_measured.columns else station
            si_t_df = pd.merge_ordered(si_t_df, temp[[column_to_extract, "TimeStamp"]], on="TimeStamp",
                                            how='outer', fill_method="ffill")
            si_t_df.rename(columns={column_to_extract: "measured", }, inplace=True)

        si_t_df.set_index("TimeStamp", inplace=True)
        return si_t_df

    def create_df_from_sensitivity_first_indices_for_singe_station(self, station=None):
        if not self._is_Sobol_m_computed:
            raise Exception("Sobol First Order Indices are not computed")

        if station is None:
            if not isinstance(self.station_of_Interest, list):
                station = self.station_of_Interest
            else:
                station = self.station_of_Interest[0]

        keyIter = list(itertools.product([station, ], self.pdTimesteps))

        list_of_df_over_parameters = []
        for i in range(len(self.labels)):
            si_t_single_param = [self.result_dict[key]["Sobol_m"][i] for key in keyIter]
            df_temp = pd.DataFrame(list(zip(si_t_single_param, self.pdTimesteps)),
                                   columns=[self.labels[i] + "_si_t", 'TimeStamp'])
            list_of_df_over_parameters.append(df_temp)
        si_m_df = reduce(lambda left, right: pd.merge(left, right, on="TimeStamp", how='outer'),
                              list_of_df_over_parameters)

        if self.groundTruth_computed:
            temp = larsimDataPostProcessing.filterResultForStation(self.df_measured, station=station)
            column_to_extract = 'Value' if 'Value' in self.df_measured.columns else station
            si_m_df = pd.merge_ordered(si_m_df, temp[[column_to_extract, "TimeStamp"]], on="TimeStamp",
                                            how='outer', fill_method="ffill")
            si_m_df.rename(columns={column_to_extract: "measured", }, inplace=True)

        si_m_df.set_index("TimeStamp", inplace=True)

    # def plot_heatmap_si_t(self):
    #     si_t_columns = [x for x in si_t_df.columns.tolist() if x != 'measured']
    #     fig = px.imshow(si_t_df[si_t_columns].T, labels=dict(y='Parameter'))
    #     return fig
    #
    # def plot_heatmap_si_m(self):
    #     si_m_columns = [x for x in si_m_df.columns.tolist() if x != 'measured']
    #     fig = px.imshow(si_m_df[si_m_columns].T, labels=dict(y='Parameter'))
    #     return fig
    #
    # def plot_si_t_and_normalized_measured_time_signal(self, station=None):
    #     if station is None:
    #         if not isinstance(self.station_of_Interest, list):
    #             station = self.station_of_Interest
    #         else:
    #             station = self.station_of_Interest[0]
    #
    #     si_t_columns = [x for x in si_t_df.columns.tolist() if x != 'measured']
    #     fig = go.Figure()
    #     if si_t_df is not None:
    #         fig = px.line(si_t_df, x=si_t_df.index, y=si_t_columns)
    #     else:
    #         keyIter = list(itertools.product([station, ], self.pdTimesteps))
    #         for i in range(len(self.labels)):
    #             fig.add_trace(go.Scatter(x=self.pdTimesteps,
    #                                      y=[self.result_dict[key]["Sobol_t"][i] for key in keyIter],
    #                                      name=self.labels[i], legendgroup=self.labels[i],
    #                                      line_color=COLORS[i])
    #                           )
    #     if df_statistics_station is not None:
    #         fig.add_trace(go.Scatter(x=df_statistics_station['TimeStamp'],
    #                                  y=df_statistics_station['measured_norm'],
    #                                  fill='tozeroy'))
    #     return fig

    def calculate_p_and_r_factors(self):
        pass
