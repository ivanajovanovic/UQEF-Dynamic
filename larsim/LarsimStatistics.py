import chaospy as cp
import numpy as np
import pickle
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plotter
from plotly.offline import iplot, plot
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import itertools
import os
from distutils.util import strtobool

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
            if "type_of_output" in configurationObject["Output"] else "Abfluss Messung + Vorhersage"
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
            self.df_index_parameter_gof_values = pd.concat(list_of_single_index_parameter_gof_df, ignore_index=True, sort=False, axis=0)
        else:
            self.df_index_parameter_gof_values = None

        print(f"[LARSIM STAT INFO] Number of Unique TimeStamps (Hourly): "
              f"{len(self.df_simulation_result.TimeStamp.unique())}")

        # TODO remove/refactor transformToDailyResolution
        if strtobool(dailyOutput):
            print(f"[LarsimSamples INFO] Transformation to daily output")
            self.transform_dict = dict()
            for one_qoi_column in self.qoi_columns:
                self.transform_dict[one_qoi_column] = 'mean'
            self.df_simulation_result = larsimDataPostProcessing.transformToDailyResolution(self.df_simulation_result,
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
        # self.df_time_discharges = self.df_simulation_result.groupby(["Stationskennung","TimeStamp"])["Value"].apply(lambda df: df.reset_index(drop=True)).unstack()

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

        self.Abfluss = dict()

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

        self.uq_method = kwargs.get('uq_method') if 'uq_method' in kwargs else None

        self._compute_Sobol_t = kwargs.get('compute_Sobol_t') if 'compute_Sobol_t' in kwargs else True
        self._compute_Sobol_m = kwargs.get('compute_Sobol_m') if 'compute_Sobol_m' in kwargs else True
        self._is_Sobol_t_computed = False
        self._is_Sobol_m_computed = False

    def calcStatisticsForMc(self, rawSamples, timesteps, simulationNodes,
                            numEvaluations, order, regression, solverTimes,
                            work_package_indexes, original_runtime_estimator):

        samples = LarsimSamples(rawSamples, configurationObject=self.configurationObject)

        samples.save_samples_to_file(self.workingDir)
        samples.save_index_parameter_values(self.workingDir)
        samples.save_index_parameter_gof_values(self.workingDir)

        self.timesteps = samples.get_simulation_timesteps()
        self.timesteps_min = samples.get_timesteps_min()
        self.timesteps_max = samples.get_timesteps_max()

        self.numbTimesteps = len(self.timesteps)
        print(f"[LARSIM STAT INFO] numbTimesteps is: {self.numbTimesteps}")

        self.samples_station_names = samples.get_simulation_stations()
        # self.samples_qoi_columns = samples.get_list_of_qoi_column_names()
        self.station_of_Interest = list(set(self.samples_station_names).intersection(self.station_of_Interest))
        if not self.station_of_Interest:
            self.station_of_Interest = self.samples_station_names

        grouped = samples.df_simulation_result.groupby(['Stationskennung','TimeStamp'])
        groups = grouped.groups

        if regression:
            nodes = simulationNodes.distNodes
            dist = simulationNodes.joinedDists
            # TODO make for calcStatisticsForMc normed and rule as parameters
            polynomial_expansion = cp.generate_expansion(order, dist, rule="three_terms_recurrence", normed=True)

        for key, val_indices in groups.items():
            discharge_values = samples.df_simulation_result.loc[val_indices.sort_values().values][self.qoi_column].values #numpy array nx1
            #discharge_values = samples.df_simulation_result.Value.loc[val_indices].values
            self.Abfluss[key] = dict()
            self.Abfluss[key]["Q"] = discharge_values
            if regression:
                self.qoi_gPCE = cp.fit_regression(polynomial_expansion, nodes, discharge_values)
                self._calc_stats_for_gPCE(dist, key)
            else:
                self.Abfluss[key]["E"] = np.sum(discharge_values, axis=0, dtype=np.float64)/ numEvaluations
                self.Abfluss[key]["E_numpy"] = np.mean(discharge_values, 0)  #TODO!!!
                #self.Abfluss[key]["Var"] = float(np.sum(power(discharge_values)) / numEvaluations - self.Abfluss[key]["E"]**2)
                self.Abfluss[key]["Var"] = np.sum((discharge_values - self.Abfluss[key]["E"]) ** 2, axis=0, dtype=np.float64) / (numEvaluations - 1)
                self.Abfluss[key]["StdDev"] = np.sqrt(self.Abfluss[key]["Var"], dtype=np.float64)
                self.Abfluss[key]["StdDev_numpy"] = np.std(discharge_values, 0, ddof=1)  #TODO!!!
                self.Abfluss[key]["P10"] = np.percentile(discharge_values, 10, axis=0)
                self.Abfluss[key]["P90"] = np.percentile(discharge_values, 90, axis=0)
                if isinstance(self.Abfluss[key]["P10"], list) and len(self.Abfluss[key]["P10"]) == 1:
                    self.Abfluss[key]["P10"]=self.Abfluss[key]["P10"][0]
                    self.Abfluss[key]["P90"]=self.Abfluss[key]["P90"][0]

        print(f"[LARSIM STAT INFO] calcStatisticsForMc function is done!")

    def calcStatisticsForSc(self, rawSamples, timesteps,
                           simulationNodes, order, regression, poly_normed, poly_rule, solverTimes,
                           work_package_indexes, original_runtime_estimator):

        samples = LarsimSamples(rawSamples, configurationObject=self.configurationObject)

        samples.save_samples_to_file(self.workingDir)
        samples.save_index_parameter_values(self.workingDir)
        samples.save_index_parameter_gof_values(self.workingDir)

        self.timesteps = samples.get_simulation_timesteps()
        self.timesteps_min = samples.get_timesteps_min()
        self.timesteps_max = samples.get_timesteps_max()

        self.numbTimesteps = len(self.timesteps)
        print(f"[LARSIM STAT INFO] numbTimesteps is: {self.numbTimesteps}")

        self.samples_station_names = samples.get_simulation_stations()
        # self.samples_qoi_columns = samples.get_list_of_qoi_column_names()
        #self.nodeNames = simulationNodes.nodeNames
        self.station_of_Interest = list(set(self.samples_station_names).intersection(self.station_of_Interest))
        if not self.station_of_Interest:
            self.station_of_Interest = self.samples_station_names

        # components independent on model evaluations, i.e., defined a priori, based solely on the underlying distribution
        nodes = simulationNodes.distNodes
        dist = simulationNodes.joinedDists
        weights = simulationNodes.weights
        polynomial_expansion = cp.generate_expansion(order, dist, rule=poly_rule, normed=poly_normed)

        grouped = samples.df_simulation_result.groupby(['Stationskennung','TimeStamp'])
        groups = grouped.groups

        list_of_unique_timesteps = [pd.Timestamp(timestep) for timestep in self.timesteps]
        keyIter = list(itertools.product(self.station_of_Interest, list_of_unique_timesteps))

        @jit(nopython=True, parallel=True)
        def my_parallel_statistic_func(qoi_column, compute_Sobol_t, compute_Sobol_m):
            Abfluss = dict()
            for i in prange(len(keyIter)):
                key = keyIter[i]
                Abfluss[key] = dict()
                discharge_values = samples.df_simulation_result.loc[groups[key].sort_values().values][qoi_column].values
                Abfluss[key]["Q"] = discharge_values
                if regression:
                    qoi_gPCE = cp.fit_regression(polynomial_expansion, nodes, discharge_values)
                else:
                    qoi_gPCE = cp.fit_quadrature(polynomial_expansion, nodes, weights, discharge_values)
                #self._calc_stats_for_gPCE(dist, key)
                numPercSamples = 10 ** 5
                Abfluss[key]["gPCE"] = qoi_gPCE
                Abfluss[key]["E"] = float(cp.E(qoi_gPCE, dist))
                Abfluss[key]["Var"] = float(cp.Var(qoi_gPCE, dist))
                Abfluss[key]["StdDev"] = float(cp.Std(qoi_gPCE, dist))
                Abfluss[key]["qoi_dist"] = cp.QoI_Dist(qoi_gPCE, dist)
                if compute_Sobol_t:
                    Abfluss[key]["Sobol_t"] = cp.Sens_t(qoi_gPCE, dist)
                if compute_Sobol_m:
                    Abfluss[key]["Sobol_m"] = cp.Sens_m(qoi_gPCE, dist)
                    # Abfluss[key]["Sobol_m2"] = cp.Sens_m2(qoi_gPCE, dist) # second order sensitivity indices

                Abfluss[key]["P10"] = float(cp.Perc(qoi_gPCE, 10, dist, numPercSamples))
                Abfluss[key]["P90"] = float(cp.Perc(qoi_gPCE, 90, dist, numPercSamples))
                if isinstance(Abfluss[key]["P10"], list) and len(Abfluss[key]["P10"]) == 1:
                    Abfluss[key]["P10"] = Abfluss[key]["P10"][0]
                    Abfluss[key]["P90"] = Abfluss[key]["P90"][0]
            return Abfluss
        self.Abfluss = my_parallel_statistic_func(self.qoi_column, self._compute_Sobol_t, self._compute_Sobol_m)

        # for key, val_indices in groups.items():
        #     # make come coupling of these values and nodes through the index
        #     discharge_values = samples.df_simulation_result.loc[val_indices.sort_values().values][self.qoi_column].values
        #     self.Abfluss[key] = dict()
        #     self.Abfluss[key]["Q"] = discharge_values
        #     if regression:
        #         self.qoi_gPCE = cp.fit_regression(polynomial_expansion, nodes, discharge_values)
        #     else:
        #         self.qoi_gPCE = cp.fit_quadrature(polynomial_expansion, nodes, weights, discharge_values)
        #     self._calc_stats_for_gPCE(dist, key)

        print(f"[LARSIM STAT INFO] calcStatisticsForSc function is done!")

    def _calc_stats_for_gPCE(self, dist, key):
        numPercSamples = 10 ** 5
        self.Abfluss[key]["gPCE"] = self.qoi_gPCE
        self.Abfluss[key]["E"] = float(cp.E(self.qoi_gPCE, dist))
        self.Abfluss[key]["Var"] = float(cp.Var(self.qoi_gPCE, dist))
        self.Abfluss[key]["StdDev"] = float(cp.Std(self.qoi_gPCE, dist))

        self.Abfluss[key]["qoi_dist"] = cp.QoI_Dist(self.qoi_gPCE, dist)

        # # generate QoI dist
        # qoi_dist = cp.QoI_Dist(self.qoi_gPCE, dist)
        # # generate sampling values for the qoi dist (you should know the min/max values for doing this)
        # dist_sampling_values = np.linspace(min_value, max_value, 1e4, endpoint=True)
        # # sample the QoI dist on the generated sampling values
        # pdf_samples = qoi_dist.pdf(dist_sampling_values)
        # # plot it (for example with matplotlib) ...

        if self._compute_Sobol_t:
            self.Abfluss[key]["Sobol_t"] = cp.Sens_t(self.qoi_gPCE, dist)
        if self._compute_Sobol_m:
            self.Abfluss[key]["Sobol_m"] = cp.Sens_m(self.qoi_gPCE, dist)
            #self.Abfluss[key]["Sobol_m2"] = cp.Sens_m2(self.qoi_gPCE, dist) # second order sensitivity indices

        self.Abfluss[key]["P10"] = float(cp.Perc(self.qoi_gPCE, 10, dist, numPercSamples))
        self.Abfluss[key]["P90"] = float(cp.Perc(self.qoi_gPCE, 90, dist, numPercSamples))
        if isinstance(self.Abfluss[key]["P10"], list) and len(self.Abfluss[key]["P10"]) == 1:
            self.Abfluss[key]["P10"]= self.Abfluss[key]["P10"][0]
            self.Abfluss[key]["P90"] = self.Abfluss[key]["P90"][0]

    def calcStatisticsForSaltelli(self, rawSamples, timesteps,
                            simulationNodes, numEvaluations, order, regression, solverTimes,
                            work_package_indexes, original_runtime_estimator=None):

        samples = LarsimSamples(rawSamples, configurationObject=self.configurationObject)

        samples.save_samples_to_file(self.workingDir)
        samples.save_index_parameter_values(self.workingDir)
        samples.save_index_parameter_gof_values(self.workingDir)

        self.timesteps = samples.get_simulation_timesteps()
        self.timesteps_min = samples.get_timesteps_min()
        self.timesteps_max = samples.get_timesteps_max()

        self.numbTimesteps = len(self.timesteps)
        print(f"[LARSIM STAT INFO] numbTimesteps is: {self.numbTimesteps}")

        self.samples_station_names = samples.get_simulation_stations()
        # self.samples_qoi_columns = samples.get_list_of_qoi_column_names()
        #self.nodeNames = simulationNodes.nodeNames
        self.station_of_Interest = list(set(self.samples_station_names).intersection(self.station_of_Interest))
        if not self.station_of_Interest:
            self.station_of_Interest = self.samples_station_names

        grouped = samples.df_simulation_result.groupby(['Stationskennung','TimeStamp'])
        groups = grouped.groups

        self.dim = len(simulationNodes.distNodes[0])

        list_of_unique_timesteps = [pd.Timestamp(timestep) for timestep in self.timesteps]
        keyIter = list(itertools.product(self.station_of_Interest, list_of_unique_timesteps))

        @jit(nopython=True, parallel=True)
        def my_parallel_statistic_func(numEvaluations, qoi_column, dim, compute_Sobol_t, compute_Sobol_m):
            Abfluss = dict()
            for i in prange(len(keyIter)):
                key = keyIter[i]
                Abfluss[key] = dict()
                # numpy array - for sartelli it should be n(2+d)x1
                discharge_values = samples.df_simulation_result.\
                    loc[groups[key].sort_values().values][qoi_column].values
                # extended_standard_discharge_values = discharge_values[:(2*numEvaluations)]
                discharge_values_saltelli = discharge_values[:, np.newaxis]
                # values based on which we calculate standard statistics
                standard_discharge_values = discharge_values_saltelli[:numEvaluations, :]
                extended_standard_discharge_values = discharge_values_saltelli[:(2 * numEvaluations), :]

                Abfluss[key]["Q"] = standard_discharge_values

                # Abfluss[key]["min_q"] = np.amin(discharge_values) #standard_discharge_values.min()
                # Abfluss[key]["max_q"] = np.amax(discharge_values) #standard_discharge_values.max()

                # Abfluss[key]["E"] = np.sum(extended_standard_discharge_values, axis=0, dtype=np.float64) / (2*numEvaluations)
                Abfluss[key]["E"] = np.mean(discharge_values[:(2 * numEvaluations)], 0)

                # Abfluss[key]["Var"] = float(np.sum(power(standard_discharge_values)) / numEvaluations - Abfluss[key]["E"] ** 2)
                # Abfluss[key]["Var"] = np.sum((extended_standard_discharge_values - Abfluss[key]["E"]) ** 2, axis=0, dtype=np.float64) / (2*numEvaluations - 1)
                # Abfluss[key]["StdDev"] = np.sqrt(Abfluss[key]["Var"], dtype=np.float64)
                Abfluss[key]["StdDev"] = np.std(discharge_values[:(2 * numEvaluations)], 0, ddof=1)

                # Abfluss[key]["P10"] = np.percentile(discharge_values[:numEvaluations], 10, axis=0)
                # Abfluss[key]["P90"] = np.percentile(discharge_values[:numEvaluations], 90, axis=0)
                Abfluss[key]["P10"] = np.percentile(discharge_values[:(2 * numEvaluations)], 10, axis=0)
                Abfluss[key]["P90"] = np.percentile(discharge_values[:(2 * numEvaluations)], 90, axis=0)

                if compute_Sobol_t:
                    Abfluss[key]["Sobol_t"] = saltelliSobolIndicesHelpingFunctions._Sens_t_sample_4(
                        discharge_values_saltelli, dim, numEvaluations)
                if compute_Sobol_m:
                    Abfluss[key]["Sobol_m"] = saltelliSobolIndicesHelpingFunctions._Sens_m_sample_4(
                        discharge_values_saltelli, dim, numEvaluations)
                    # Abfluss[key]["Sobol_m"] = saltelliSobolIndicesHelpingFunctions._Sens_m_sample_3
                    # (discharge_values_saltelli, self.dim, numEvaluations)

                if isinstance(Abfluss[key]["P10"], list) and len(Abfluss[key]["P10"]) == 1:
                    Abfluss[key]["P10"] = Abfluss[key]["P10"][0]
                    Abfluss[key]["P90"] = Abfluss[key]["P90"][0]
            return Abfluss

        self.Abfluss = my_parallel_statistic_func(numEvaluations, self.qoi_column, self.dim,
                                                  self._compute_Sobol_t, self._compute_Sobol_m)

        print(f"[LARSIM STAT INFO] calcStatisticsForSaltelli function is done!")

    def check_if_Sobol_t_computed(self, keyIter):
        self._is_Sobol_t_computed = "Sobol_t" in self.Abfluss[keyIter[0]] #hasattr(self.Abfluss[keyIter[0], "Sobol_t")

    def check_if_Sobol_m_computed(self, keyIter):
        self._is_Sobol_m_computed = "Sobol_m" in self.Abfluss[keyIter[0]] \
                                    or "Sobol_m2" in self.Abfluss[keyIter[0]] #hasattr(self.Abfluss[keyIter[0], "Sobol_m")

    # TODO timestepRange calculated based on whole Time-series
    def get_measured_discharge(self, timestepRange=None):
        transform_measured_to_daily = False
        if strtobool(self.configurationObject["Output"]["dailyOutput"]) and self.qoi_column == "Value":
            transform_measured_to_daily = False

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
                # TODO transformToDailyResolution was refactored
                if "Value" in self.df_measured.columns:
                    transform_dict = {"Value": 'mean', }
                else:
                    transform_dict = dict()
                    for single_station in self.station_of_Interest:
                        transform_dict[single_station] = 'mean'
                self.df_measured = larsimDataPostProcessing.transformToDailyResolution(self.df_measured,
                                                                                       transform_dict=transform_dict,
                                                                                       groupby_some_columns=True,
                                                                                       columns_to_groupby=["Stationskennung",],
                                                                                       time_column="TimeStamp",
                                                                                       resample_freq="D"
                                                                                       )
        self.groundTruth_computed = True
        #self.Abfluss["Ground_Truth_Measurements"] = self.measured

    # TODO timestepRange calculated based on whole Time-series
    def get_unaltered_discharge(self, timestepRange=None):
        transform_unaltered_to_daily = False
        if strtobool(self.configurationObject["Output"]["dailyOutput"]) and self.qoi_column == "Value":
            transform_unaltered_to_daily = False
        self.df_unaltered = larsimDataPostProcessing.read_process_write_discharge(
            df=os.path.abspath(os.path.join(self.workingDir, "df_unaltered.pkl")),
            timeframe=timestepRange,
            type_of_output=self.configurationObject["Output"]["type_of_output"],
            station=self.station_of_Interest,
            dailyOutput=transform_unaltered_to_daily,
            compression="gzip")
        self.unaltered_computed = True
        #self.Abfluss["Unaltered"] = self.unalatered

    def plotResults(self, timestep=-1, display=False,
                    fileName="", fileNameIdent="", directory="./",
                    fileNameIdentIsFullName=False, safe=True):

        #timestepRange = (pd.Timestamp(min(self.timesteps)), pd.Timestamp(max(self.timesteps)))
        timestepRange = (self.timesteps_min, self.timesteps_max)

        # TODO timestepRange calculated based on whole Time-series
        self.get_measured_discharge(timestepRange=timestepRange)
        self.get_unaltered_discharge(timestepRange=timestepRange)

        print(f"[LARSIM STAT INFO] plotResults function is called!")

        for single_station in self.station_of_Interest:
            fileName = single_station
            single_fileName = self.generateFileName(fileName=fileName, fileNameIdent=".html",
                                             directory=directory, fileNameIdentIsFullName=fileNameIdentIsFullName)
            self._plotStatisticsDict_plotly(unalatered=self.unaltered_computed, measured=self.groundTruth_computed,
                                            station=single_station,
                                            recalculateTimesteps=False, filename=single_fileName, display=display)
            #self._plotStatisticsDict_plotter(unalatered=None, measured=None,
            # station=single_station,
            # recalculateTimesteps=False, filename=single_fileName, display=display)
        print(f"[LARSIM STAT INFO] plotResults function is done!")

    def _plotStatisticsDict_plotly(self, unalatered=False, measured=False, station="MARI",
                                   recalculateTimesteps=False, window_title='Larsim Forward UQ & SA',
                                   filename="sim-plotly.html", display=False):

        print(f"[LARSIM STAT INFO] _plotStatisticsDict_plotly function is called!")

        #TODO Access to timesteps in a two different ways, e.g. one for QoI one for Q
        #timesteps = df_measured_aligned.TimeStamp.unique()
        #pdTimesteps = [pd.Timestamp(timestep) for timestep in timesteps]
        if recalculateTimesteps:
            Abfluss_keys_list = list(self.Abfluss.keys())[1:]
            pdTimesteps = []
            for i in range(0, len(Abfluss_keys_list)):
                pdTimesteps.append(pd.Timestamp(Abfluss_keys_list[i][1]))
        else:
            pdTimesteps = [pd.Timestamp(timestep) for timestep in self.timesteps]

        keyIter = list(itertools.product([station,], pdTimesteps))

        labels = [nodeName.strip() for nodeName in self.nodeNames]

        self.check_if_Sobol_t_computed(keyIter)
        self.check_if_Sobol_m_computed(keyIter)

        if self._is_Sobol_t_computed and self._is_Sobol_m_computed:
            n_rows = 4
        elif self._is_Sobol_t_computed or self._is_Sobol_m_computed:
            n_rows = 3
        else:
            n_rows = 2

        if self.qoi_column == "Value":
            starting_row = 1
        else:
            n_rows = n_rows+1
            starting_row = 2

        fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=False)

        if unalatered:
            # if 'Value' in self.df_unaltered.columns:
            #     df_unaltered = self.df_unaltered.loc[self.df_unaltered['Stationskennung'] == station]['Value']
            # elif station in self.df_unaltered.columns:
            #     df_unaltered = self.df_unaltered[station]
            # else:
            #     raise Exception(f"[LARSIM STAT ERROR:] in _plotStatisticsDict_plotly when extracting unalatered")

            column_to_draw = 'Value' if 'Value' in self.df_unaltered.columns else station
            #fig.add_trace(go.Scatter(x=pdTimesteps, y=self.unalatered['Value'], name="Q (unaltered simulation)",line_color='deepskyblue'), row=1, col=1)
            fig.add_trace(go.Scatter(x=self.df_unaltered['TimeStamp'], y=self.df_unaltered[column_to_draw],
                                     name="Q (unaltered simulation)", line_color='deepskyblue'), row=1, col=1)
        if measured:
            column_to_draw = 'Value' if 'Value' in self.df_measured.columns else station
            #fig.add_trace(go.Scatter(x=pdTimesteps, y=self.measured['Value'], name="Q (measured)",line_color='red'), row=1, col=1)
            fig.add_trace(go.Scatter(x=self.df_measured['TimeStamp'], y=self.df_measured[column_to_draw],
                                     name="Q (measured)",line_color='red'), row=1, col=1)

        fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.Abfluss[key]["E"] for key in keyIter], name='E[Q]',line_color='green', mode='lines'), row=starting_row, col=1)
        #fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.Abfluss[key]["min_q"] for key in keyIter], name='min_q',line_color='indianred', mode='lines'), row=starting_row, col=1)
        #fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.Abfluss[key]["max_q"] for key in keyIter], name='max_q',line_color='yellow', mode='lines'), row=starting_row, col=1)
        fig.add_trace(go.Scatter(x=pdTimesteps, y=[(self.Abfluss[key]["E"] - self.Abfluss[key]["StdDev"]) for key in keyIter], name='mean - std. dev', line_color='darkviolet', mode='lines'), row=starting_row, col=1)
        fig.add_trace(go.Scatter(x=pdTimesteps, y=[(self.Abfluss[key]["E"] + self.Abfluss[key]["StdDev"]) for key in keyIter], name='mean + std. dev', line_color='darkviolet', mode='lines', fill='tonexty'), row=starting_row, col=1)

        if self.uq_method=="saltelli":
            fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.Abfluss[key]["P10"] for key in keyIter], name='10th percentile',line_color='yellow', mode='lines'), row=starting_row, col=1)
            fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.Abfluss[key]["P90"] for key in keyIter], name='90th percentile',line_color='yellow', mode='lines',fill='tonexty'), row=starting_row, col=1)
        else:
            fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.Abfluss[key]["P10"] for key in keyIter], name='10th percentile',line_color='yellow', mode='lines'), row=starting_row, col=1)
            fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.Abfluss[key]["P90"] for key in keyIter], name='90th percentile',line_color='yellow', mode='lines',fill='tonexty'), row=starting_row, col=1)

        fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.Abfluss[key]["StdDev"] for key in keyIter], name='std. dev', line_color='darkviolet'), row=starting_row+1, col=1)

        # TODO - Make additional plots - each parameter independently with color + normalized Q measured
        if self._is_Sobol_m_computed:
            for i in range(len(labels)):
                if self.uq_method == "saltelli":
                    fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.Abfluss[key]["Sobol_m"][i] for key in keyIter],
                                             name=labels[i], legendgroup=labels[i], line_color=COLORS[i]), row=starting_row+2, col=1)
                else:
                    fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.Abfluss[key]["Sobol_m"][i] for key in keyIter],
                                             name=labels[i], legendgroup=labels[i], line_color=COLORS[i]), row=starting_row+2, col=1)
        if self._is_Sobol_t_computed:
            for i in range(len(labels)):
                if self.uq_method == "saltelli":
                    fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.Abfluss[key]["Sobol_t"][i] for key in keyIter],
                                             legendgroup=labels[i], showlegend = False, line_color=COLORS[i]), row=starting_row+3, col=1)
                else:
                    fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.Abfluss[key]["Sobol_t"][i] for key in keyIter],
                                             legendgroup=labels[i], showlegend = False, line_color=COLORS[i]), row=starting_row+3, col=1)

        fig.update_traces(mode='lines')
        #fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="Q [m^3/s]", side='left', showgrid=True, row=1, col=1)

        if self.qoi_column == "Value":
            fig.update_yaxes(title_text="Std. Dev. [m^3/s]", side='left', showgrid=True, row=2, col=1)
        else:
            fig.update_yaxes(title_text=self.qoi_column, side='left', showgrid=True, row=starting_row, col=1)
            fig.update_yaxes(title_text="Std. Dev. [QoI]", side='left', showgrid=True, row=starting_row+1, col=1)

        if self._is_Sobol_m_computed:
            fig.update_yaxes(title_text="Sobol_m", side='left', showgrid=True, range=[0, 1], row=starting_row+2, col=1)
        if self._is_Sobol_t_computed:
            fig.update_yaxes(title_text="Sobol_t", side='left', showgrid=True, range=[0, 1], row=starting_row+3, col=1)

        window_title = window_title + station
        #fig.update_layout(height=1200, width=1200,
        # title_text='Larsim Forward UQ & SA - MARI', xaxis4_rangeslider_visible=True, xaxis4_rangeslider_thickness=0.05)
        fig.update_layout(height=800, width=1200, title_text=window_title)
        #xaxis4_rangeslider_visible=True, xaxis4_rangeslider_thickness=0.05)

        print(f"[LARSIM STAT INFO] _plotStatisticsDict_plotly function is almost over!")

        plot(fig, filename=filename, auto_open=display)
        #fig.write_image("sim-09-plotly.png")
        fig.show()

    def _plotStatisticsDict_plotter(self, unalatered=False, measured=False, station="MARI",
                                    recalculateTimesteps=False, window_title='Larsim Forward UQ & SA - MARI',
                                    filename="sim-plotter", display=True):
        figure = plotter.figure(1, figsize=(13, 10))
        figure.canvas.set_window_title(window_title)

        if recalculateTimesteps:
            Abfluss_keys_list = list(self.Abfluss.keys())[1:]
            pdTimesteps = []
            for i in range(0, len(Abfluss_keys_list)):
                pdTimesteps.append(pd.Timestamp(Abfluss_keys_list[i][1]))
        else:
            pdTimesteps = [pd.Timestamp(timestep) for timestep in self.timesteps]

        keyIter = list(itertools.product([station,], pdTimesteps))

        #sobol_labels = ["BSF", "A2", "EQD", "EQD2"]
        labels = [nodeName.strip() for nodeName in self.nodeNames]
        #labels = list(map(str.strip, self.nodeNames))

        self.check_if_Sobol_t_computed(keyIter)
        self.check_if_Sobol_m_computed(keyIter)
        if self._is_Sobol_t_computed or self._is_Sobol_m_computed:
            n_rows = 4
        else:
            n_rows = 3

        plotter.subplot(411)

        if unalatered:
            column_to_draw = 'Value' if 'Value' in self.df_unaltered.columns else station
            #plotter.plot(pdTimesteps, self.unalatered['Value'], label="Q (unaltered simulation)")
            plotter.plot(self.df_unaltered['TimeStamp'], self.df_unaltered[column_to_draw], label="Q (unaltered simulation)")
        if measured:
            column_to_draw = 'Value' if 'Value' in self.df_measured.columns else station
            #plotter.plot(pdTimesteps, self.measured['Value'], label="Q (measured)")
            plotter.plot(self.df_measured['TimeStamp'], self.df_measured[column_to_draw], label="Q (measured)")

        #plotter.plot(pdTimesteps, [Abfluss[key]["E"] for key in keyIter], '-r', label='E[Q_sim]')
        plotter.fill_between(pdTimesteps, [self.Abfluss[key]["P10"] for key in keyIter], [self.Abfluss[key]["P90"] for key in keyIter], facecolor='#5dcec6')
        plotter.plot(pdTimesteps, [self.Abfluss[key]["P10"] for key in keyIter], label='10th percentile')
        plotter.plot(pdTimesteps,[self.Abfluss[key]["P90"] for key in keyIter], label='90th percentile')
        plotter.xlabel('time', fontsize=13)
        plotter.ylabel('Q [m^3/s]', fontsize=13)
        #plotter.xticks(rotation=45)plotter.legend()
        plotter.grid(True)

        plotter.subplot(412)
        plotter.plot(pdTimesteps, [self.Abfluss[key]["StdDev"] for key in keyIter], label='std. dev. of the simulations')
        plotter.xlabel('time', fontsize=13)
        plotter.ylabel('Std. Dev. [m^3/s]', fontsize=13)
        #plotter.xlim(0, 200)
        #plotter.ylim(0, 20)
        #plotter.xticks(rotation=45)
        plotter.legend()
        plotter.grid(True)

        if self._is_Sobol_m_computed:
            plotter.subplot(413)
            for i in range(len(labels)):
                plotter.plot(pdTimesteps, [self.Abfluss[key]["Sobol_m"][i][0] for key in keyIter],\
                label=labels[i])
            plotter.xlabel('time', fontsize=13)
            plotter.ylabel('First O. Sobol Indices', fontsize=13)
            #plotter.xticks(rotation=45)
            plotter.legend()
            plotter.grid(True)

        if self._is_Sobol_t_computed:
            plotter.subplot(414)
            for i in range(len(labels)):
                plotter.plot(pdTimesteps, [self.Abfluss[key]["Sobol_t"][i][0] for key in keyIter],\
                label=labels[i])
            plotter.xlabel('time', fontsize=13)
            plotter.ylabel('Total Sobol Indices', fontsize=13)
            #plotter.xticks(rotation=45)
            plotter.legend()
            plotter.grid(True)

        #plotter.savefig(pdfFileName, format='pdf')
        plotter.savefig(filename, format='png')

        if display:
            plotter.show()

        plotter.close()

    def saveToFile(self, fileName="statistics_dict", fileNameIdent="", directory="./",
                   fileNameIdentIsFullName=False):

        statFileName = os.path.abspath(os.path.join(self.workingDir,"statistics_dictionary.pkl"))
        with open(statFileName, 'wb') as handle:
            pickle.dump(self.Abfluss, handle, protocol=pickle.HIGHEST_PROTOCOL)