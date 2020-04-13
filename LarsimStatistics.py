import chaospy as cp
import numpy as np
import pickle
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from tabulate import tabulate
import matplotlib.pyplot as plotter
from plotly.offline import iplot, plot
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import itertools
import os
import string as str
from distutils.util import strtobool

from uqef.stat import Statistics

import larsimDataPostProcessing
import larsimInputOutputUtilities

class LarsimSamples(object):
    """
     Samples is a collection of the (filtered) sampled results of a whole UQ simulation
    """

    #TODO write get/set methods for the attributes of the class

    def __init__(self, rawSamples, station=None, type_of_output='Abfluss Messung', dailyOutput="False"):
        """
        :param rawSamples: results returned by solver. Either list of paths to diff. ergebnis files or pandas.DataFrame object containing output of all the runs
        :param station: Name of the station in which output we are interested in, if None - filter out all the stations
        :param type_of_output: can be Abfluss Simulation or Abfluss Messung, corresponds to the entries in ergebnis.lils
        """

        list_of_single_df = []
        for index_run, value in enumerate(rawSamples,): #Important that the results inside rawSamples (resulted paths) are in sorted order and correspond to the parameters order
        #TODO-Ivana Add code which will disregard non-valid simulations with value=None and the corresponding nodes!
            df_single_ergebnis = larsimDataPostProcessing.read_process_write_discharge(df=value,\
                                 index_run=index_run,\
                                 type_of_output=type_of_output,\
                                 station=station)
            list_of_single_df.append(df_single_ergebnis)

        self.df_simulation_result = pd.concat(list_of_single_df, ignore_index=True, sort=False, axis=0)

        self.df_simulation_result['Value'] = self.df_simulation_result['Value'].astype(float)

        print("LARSIM STAT INFO: Number of Unique TimeStamps (Hourly): {}".format(len(self.df_simulation_result.TimeStamp.unique())))

        if strtobool(dailyOutput):
            # Average over time. i.e. change column TimeStamp and Value
            self.df_simulation_result = larsimDataPostProcessing.transformToDailyResolution(self.df_simulation_result)
            print("LARSIM STAT INFO: Number of Unique TimeStamps (Daily): {}".format(len(self.df_simulation_result.TimeStamp.unique())))

        self.df_time_discharges = self.df_simulation_result.groupby(["Stationskennung","TimeStamp"])["Value"].apply(lambda df: df.reset_index(drop=True)).unstack()

    def save_samples_to_file(self, file_path='./'):
        self.df_simulation_result.to_pickle(
            os.path.abspath(os.path.join(file_path, "df_all_simulations.pkl")), compression="gzip")

    def save_time_samples_to_file(self, file_path='./'):
        self.df_time_discharges.to_pickle(
            os.path.abspath(os.path.join(file_path, "df_all_time_simulations.pkl")), compression="gzip")

    def get_simulation_timesteps(self):
        return self.df_simulation_result.TimeStamp.unique()

    def get_simulation_stations(self):
        return self.df_simulation_result.Stationskennung.unique()


class LarsimStatistics(Statistics):
    """
       LarsimStatistics calculates the statistics for the LarsimModel
    """

    def __init__(self, configurationObject):
        Statistics.__init__(self)

        self.configurationObject = configurationObject

        #try:
        self.working_dir = self.configurationObject["Directories"]["working_dir"]
        #except KeyError:
        #    self.working_dir = paths.working_dir

        self.Abfluss = {}

        self.df_unalatered = None
        self.df_measured = None
        self.unalatered_computed = False
        self.groundTruth_computed = False

    def calcStatisticsForMc(self, rawSamples, timesteps,
                            simulationNodes, numEvaluations, order, regression, solverTimes,
                            work_package_indexes, original_runtime_estimator):

        samples = LarsimSamples(rawSamples, station=self.configurationObject["Output"]["station"],
                          type_of_output=self.configurationObject["Output"]["type_of_output"],
                          dailyOutput=self.configurationObject["Output"]["dailyOutput"])

        # Save the DataFrame containing all the simulation results - This is really important
        samples.save_samples_to_file(self.working_dir)

        self.timesteps = samples.get_simulation_timesteps()
        self.numbTimesteps = len(self.timesteps)
        print("LARSIM STAT INFO: numbTimesteps is: {}".format(self.numbTimesteps))

        self.station_names = samples.get_simulation_stations()
        self.nodeNames = simulationNodes.nodeNames

        grouped = samples.df_simulation_result.groupby(['Stationskennung','TimeStamp'])
        groups = grouped.groups

        if regression:
            nodes = simulationNodes.distNodes
            dist = simulationNodes.joinedDists
            P = cp.orth_ttr(order, dist)

        for key,val in groups.items():
            discharge_values = samples.df_simulation_result.iloc[val.values].Value.values #numpy array nx1
            self.Abfluss[key] = {}
            self.Abfluss[key]["Q"] = discharge_values
            if regression:
                self.qoi_gPCE = cp.fit_regression(P, nodes, discharge_values)
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
                if isinstance(self.Abfluss[key]["P10"], (list)) and len(self.Abfluss[key]["P10"]) == 1:
                    self.Abfluss[key]["P10"]=self.Abfluss[key]["P10"][0]
                    self.Abfluss[key]["P90"]=self.Abfluss[key]["P90"][0]


    def calcStatisticsForSc(self, rawSamples, timesteps,
                           simulationNodes, order, regression, solverTimes,
                           work_package_indexes, original_runtime_estimator):

        samples = LarsimSamples(rawSamples, station=self.configurationObject["Output"]["station"],
                                 type_of_output=self.configurationObject["Output"]["type_of_output"],
                                 dailyOutput=self.configurationObject["Output"]["dailyOutput"])

        samples.save_samples_to_file(self.working_dir)

        self.timesteps = samples.get_simulation_timesteps()
        self.numbTimesteps = len(self.timesteps)
        print("LARSIM STAT INFO: numbTimesteps is: {}".format(self.numbTimesteps))

        self.station_names = samples.get_simulation_stations()
        self.nodeNames = simulationNodes.nodeNames

        nodes = simulationNodes.distNodes
        dist = simulationNodes.joinedDists
        weights = simulationNodes.weights
        P = cp.orth_ttr(order, dist)

        grouped = samples.df_simulation_result.groupby(['Stationskennung','TimeStamp'])
        groups = grouped.groups
        for key,val in groups.items():
            discharge_values = samples.df_simulation_result.iloc[val.values].Value.values
            self.Abfluss[key] = {}
            self.Abfluss[key]["Q"] = discharge_values
            if regression:
                self.qoi_gPCE = cp.fit_regression(P, nodes, discharge_values)
            else:
                self.qoi_gPCE = cp.fit_quadrature(P, nodes, weights, discharge_values)
            self._calc_stats_for_gPCE(dist, key)


    def _calc_stats_for_gPCE(self, dist, key):
        #percentiles
        numPercSamples = 10 ** 5
        self.Abfluss[key]["gPCE"] = self.qoi_gPCE
        self.Abfluss[key]["E"] = float((cp.E(self.qoi_gPCE, dist)))
        self.Abfluss[key]["Var"] = float((cp.Var(self.qoi_gPCE, dist)))
        self.Abfluss[key]["StdDev"] = float((cp.Std(self.qoi_gPCE, dist)))
        self.Abfluss[key]["Sobol_m"] = cp.Sens_m(self.qoi_gPCE, dist)
        self.Abfluss[key]["Sobol_m2"] = cp.Sens_m2(self.qoi_gPCE, dist)
        self.Abfluss[key]["Sobol_t"] = cp.Sens_t(self.qoi_gPCE, dist)
        self.Abfluss[key]["P10"] = float(cp.Perc(self.qoi_gPCE, 10, dist, numPercSamples))
        self.Abfluss[key]["P90"] = float(cp.Perc(self.qoi_gPCE, 90, dist, numPercSamples))
        if isinstance(self.Abfluss[key]["P10"], (list)) and len(self.Abfluss[key]["P10"]) == 1:
            self.Abfluss[key]["P10"]= self.Abfluss[key]["P10"][0]
            self.Abfluss[key]["P90"] = self.Abfluss[key]["P90"][0]


    def calcStatisticsForSaltelli(self, rawSamples, timesteps,
                            simulationNodes, numEvaluations, order, regression, solverTimes,
                            work_package_indexes, original_runtime_estimator=None):

        samples = LarsimSamples(rawSamples, station=self.configurationObject["Output"]["station"],
                          type_of_output=self.configurationObject["Output"]["type_of_output"],
                          dailyOutput=self.configurationObject["Output"]["dailyOutput"])

        # Save the DataFrame containing all the simulation results - This is really important
        samples.save_samples_to_file(self.working_dir)

        self.timesteps = samples.get_simulation_timesteps()
        self.numbTimesteps = len(self.timesteps)
        print("LARSIM STAT INFO: numbTimesteps is: {}".format(self.numbTimesteps))

        self.station_names = samples.get_simulation_stations()
        self.nodeNames = simulationNodes.nodeNames

        grouped = samples.df_simulation_result.groupby(['Stationskennung','TimeStamp'])
        groups = grouped.groups

        self.dim = len(simulationNodes.nodeNames)

        for key,val in groups.items():
            self.Abfluss[key] = {}

            discharge_values = samples.df_simulation_result.iloc[val.values].Value.values #numpy array - for sartelli it should be n(2+d)x1
            #extended_standard_discharge_values = discharge_values[:(2*numEvaluations)]
            discharge_values_saltelli = discharge_values[:, np.newaxis]
            standard_discharge_values = discharge_values_saltelli[:numEvaluations,:] #values based on which we calculate standard statistics
            extended_standard_discharge_values = discharge_values_saltelli[:(2*numEvaluations),:]

            self.Abfluss[key]["Q"] = standard_discharge_values

            self.Abfluss[key]["min_q"] = np.amin(discharge_values) #standard_discharge_values.min()
            self.Abfluss[key]["max_q"] = np.amax(discharge_values) #standard_discharge_values.max()

            self.Abfluss[key]["E"] = np.sum(extended_standard_discharge_values, axis=0, dtype=np.float64) / (2*numEvaluations)
            self.Abfluss[key]["E_numpy"] = np.mean(discharge_values, 0) #TODO!!!
            #self.Abfluss[key]["Var"] = float(np.sum(power(standard_discharge_values)) / numEvaluations - self.Abfluss[key]["E"] ** 2)
            self.Abfluss[key]["Var"] = np.sum((extended_standard_discharge_values - self.Abfluss[key]["E"]) ** 2, axis=0, dtype=np.float64) / (2*numEvaluations - 1)
            self.Abfluss[key]["StdDev"] = np.sqrt(self.Abfluss[key]["Var"], dtype=np.float64)
            self.Abfluss[key]["StdDev_numpy"] = np.std(discharge_values, 0, ddof=1)  #TODO!!!

            #self.Abfluss[key]["P10"] = np.percentile(discharge_values[:numEvaluations], 10, axis=0)
            #self.Abfluss[key]["P90"] = np.percentile(discharge_values[:numEvaluations], 90, axis=0)
            self.Abfluss[key]["P10"] = np.percentile(extended_standard_discharge_values, 10, axis=0)
            self.Abfluss[key]["P90"] = np.percentile(extended_standard_discharge_values, 90, axis=0)

            #self.Abfluss[key]["Sobol_m"] = _Sens_m_sample_4(discharge_values_saltelli, self.dim, numEvaluations)
            self.Abfluss[key]["Sobol_m"] = _Sens_m_sample_3(discharge_values_saltelli, self.dim, numEvaluations)
            self.Abfluss[key]["Sobol_t"] = _Sens_t_sample_4(discharge_values_saltelli, self.dim, numEvaluations)

            if isinstance(self.Abfluss[key]["P10"], (list)) and len(self.Abfluss[key]["P10"]) == 1:
                self.Abfluss[key]["P10"]=self.Abfluss[key]["P10"][0]
                self.Abfluss[key]["P90"]=self.Abfluss[key]["P90"][0]


    def  _compute_Sobol_t(self):
        is_Sobol_t_computed = "Sobol_t" in self.Abfluss[self.keyIter[0]] #hasattr(self.Abfluss[self.keyIter[0], "Sobol_t")
        return is_Sobol_t_computed


    def  _compute_Sobol_m(self):
        is_Sobol_m_computed = "Sobol_m" in self.Abfluss[self.keyIter[0]] #hasattr(self.Abfluss[self.keyIter[0], "Sobol_m")
        return is_Sobol_m_computed


    def get_measured_discharge(self, timestepRange=None):
        self.df_measured = larsimDataPostProcessing.read_process_write_discharge(df=os.path.abspath(os.path.join(self.working_dir, "df_measured.csv")),\
                             timeframe=timestepRange,\
                             station=self.configurationObject["Output"]["station"],\
                             dailyOutput=strtobool(self.configurationObject["Output"]["dailyOutput"]))
        self.groundTruth_computed = True
        #self.Abfluss["Ground_Truth_Measurements"] = self.measured


    def get_unaltered_discharge(self, timestepRange=None):
        self.df_unalatered = larsimDataPostProcessing.read_process_write_discharge(df=os.path.abspath(os.path.join(self.working_dir, "df_unaltered_ergebnis.csv")),\
                             timeframe=timestepRange,\
                             type_of_output=self.configurationObject["Output"]["type_of_output"],\
                             station=self.configurationObject["Output"]["station"],\
                             dailyOutput=strtobool(self.configurationObject["Output"]["dailyOutput"]))
        self.unalatered_computed = True
        #self.Abfluss["Unaltered"] = self.unalatered


    def plotResults(self, timestep=-1, display=False,
                    fileName="", fileNameIdent="", directory="./",
                    fileNameIdentIsFullName=False, safe=True):

        fileName = self.generateFileName(fileName=fileName, fileNameIdent=".html", directory=directory, fileNameIdentIsFullName=fileNameIdentIsFullName)

        timestepRange = (pd.Timestamp(self.timesteps.min()), pd.Timestamp(self.timesteps.max()))
        self.get_measured_discharge(timestepRange=timestepRange)
        self.get_unaltered_discharge(timestepRange=timestepRange)

        self._plotStatisticsDict_plotly(unalatered=self.unalatered_computed, measured=self.groundTruth_computed, station=self.configurationObject["Output"]["station"], recalculateTimesteps=False, filename=fileName, display=display)
        #self._plotStatisticsDict_plotter(unalatered=None, measured=None, station=self.configurationObject["Output"]["station"], recalculateTimesteps=False, filename=fileName, display=display)


    def _plotStatisticsDict_plotly(self, unalatered=False, measured=False, station="MARI", recalculateTimesteps=False, window_title='Larsim Forward UQ & SA - MARI', filename="sim-plotly.html", display=False):

        #TODO Access to timesteps in a different way
        #timesteps = df_measured_aligned.TimeStamp.unique()
        #pdTimesteps = [pd.Timestamp(timestep) for timestep in timesteps]
        if recalculateTimesteps:
            Abfluss_keys_list = list(self.Abfluss.keys())[1:]
            pdTimesteps = []
            for i in range(0, len(Abfluss_keys_list)):
                pdTimesteps.append(pd.Timestamp(Abfluss_keys_list[i][1]))
        else:
            pdTimesteps = [pd.Timestamp(timestep) for timestep in self.timesteps]

        self.keyIter = list(itertools.product([station,],pdTimesteps))

        colors = ['darkred', 'midnightblue', 'mediumseagreen', 'darkorange']
        #sobol_labels = ["BSF", "A2", "EQD", "EQD2"]
        labels = [nodeName.strip() for nodeName in self.nodeNames]
        #labels = list(map(str.strip, self.nodeNames))

        is_Sobol_t_computed = self._compute_Sobol_t()
        is_Sobol_m_computed = self._compute_Sobol_m()

        if is_Sobol_t_computed and is_Sobol_m_computed:
            n_rows = 4
        elif is_Sobol_t_computed or is_Sobol_m_computed:
            n_rows = 3
        else:
            n_rows = 2

        fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=False)

        if unalatered:
            #fig.add_trace(go.Scatter(x=pdTimesteps, y=self.unalatered['Value'], name="Q (unaltered simulation)",line_color='deepskyblue'), row=1, col=1)
            fig.add_trace(go.Scatter(x=self.df_unalatered['TimeStamp'], y=self.df_unalatered['Value'], name="Q (unaltered simulation)",line_color='deepskyblue'), row=1, col=1)
        if measured:
            #fig.add_trace(go.Scatter(x=pdTimesteps, y=self.measured['Value'], name="Q (measured)",line_color='red'), row=1, col=1)
            fig.add_trace(go.Scatter(x=self.df_measured['TimeStamp'], y=self.df_measured['Value'], name="Q (measured)",line_color='red'), row=1, col=1)

        #fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.Abfluss[key]["E"][0] for key in self.keyIter], name='E[Q]',line_color='green', mode='lines'), row=1, col=1)
        fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.Abfluss[key]["P10"] for key in self.keyIter], name='10th percentile',line_color='indianred', mode='lines'), row=1, col=1)
        fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.Abfluss[key]["P90"] for key in self.keyIter], name='90th percentile',line_color='yellow', mode='lines'), row=1, col=1)
        #fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.Abfluss[key]["StdDev"][0] for key in self.keyIter], name='std. dev', line_color='darkviolet'), row=2, col=1)

        fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.Abfluss[key]["E_numpy"] for key in self.keyIter], name='E[Q]',line_color='green', mode='lines'), row=1, col=1)
        #fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.Abfluss[key]["min_q"] for key in self.keyIter], name='min_q',line_color='indianred', mode='lines'), row=1, col=1)
        #fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.Abfluss[key]["max_q"] for key in self.keyIter], name='max_q',line_color='yellow', mode='lines'), row=1, col=1)
        fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.Abfluss[key]["StdDev_numpy"] for key in self.keyIter], name='std. dev', line_color='darkviolet'), row=2, col=1)

        if is_Sobol_m_computed:
            for i in range(len(labels)):
                fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.Abfluss[key]["Sobol_m"][i] for key in self.keyIter], name=labels[i], legendgroup=labels[i], line_color=colors[i]), row=3, col=1)
        if is_Sobol_t_computed:
            for i in range(len(labels)):
                fig.add_trace(go.Scatter(x=pdTimesteps, y=[self.Abfluss[key]["Sobol_t"][i] for key in self.keyIter], legendgroup=labels[i], showlegend = False, line_color=colors[i]), row=4, col=1)

        fig.update_traces(mode='lines')
        #fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="Q [m^3/s]", side='left', showgrid=True, row=1, col=1)
        fig.update_yaxes(title_text="Std. Dev. [m^3/s]", side='left', showgrid=True, row=2, col=1)
        if is_Sobol_m_computed:
            fig.update_yaxes(title_text="Sobol_m", side='left', showgrid=True, range=[0, 1], row=3, col=1)
        if is_Sobol_t_computed:
            fig.update_yaxes(title_text="Sobol_t", side='left', showgrid=True, range=[0, 1], row=4, col=1)
        #fig.update_layout(height=1200, width=1200, title_text='Larsim Forward UQ & SA - MARI',xaxis4_rangeslider_visible=True, xaxis4_rangeslider_thickness=0.05)
        fig.update_layout(height=800, width=1200, title_text=window_title)

        plot(fig, filename=filename, auto_open=display)
        #fig.write_image("sim-09-plotly.png")
        fig.show()


    def _plotStatisticsDict_plotter(self, unalatered=False, measured=False, station="MARI", recalculateTimesteps=False, window_title='Larsim Forward UQ & SA - MARI', filename="sim-plotter", display=True):
        figure = plotter.figure(1, figsize=(13, 10))
        figure.canvas.set_window_title(window_title)

        if recalculateTimesteps:
            Abfluss_keys_list = list(self.Abfluss.keys())[1:]
            pdTimesteps = []
            for i in range(0, len(Abfluss_keys_list)):
                pdTimesteps.append(pd.Timestamp(Abfluss_keys_list[i][1]))
        else:
            pdTimesteps = [pd.Timestamp(timestep) for timestep in self.timesteps]

        #sobol_labels = ["BSF", "A2", "EQD", "EQD2"]
        labels = [nodeName.strip() for nodeName in self.nodeNames]
        #labels = list(map(str.strip, self.nodeNames))

        is_Sobol_t_computed = self._compute_Sobol_t()
        is_Sobol_m_computed = self._compute_Sobol_m()

        if is_Sobol_t_computed or is_Sobol_m_computed:
            n_rows = 4
        else:
            n_rows = 3

        plotter.subplot(411)

        if unalatered:
            #plotter.plot(pdTimesteps, self.unalatered['Value'], label="Q (unaltered simulation)")
            plotter.plot(self.df_unalatered['TimeStamp'], self.df_unalatered['Value'], label="Q (unaltered simulation)")
        if measured:
            #plotter.plot(pdTimesteps, self.measured['Value'], label="Q (measured)")
            plotter.plot(self.df_measured['TimeStamp'], self.df_measured['Value'], label="Q (measured)")

        self.keyIter = list(itertools.product([station,],pdTimesteps))

        #plotter.plot(pdTimesteps, [Abfluss[key]["E"] for key in self.keyIter], '-r', label='E[Q_sim]')
        plotter.fill_between(pdTimesteps, [self.Abfluss[key]["P10"] for key in self.keyIter], [self.Abfluss[key]["P90"] for key in self.keyIter], facecolor='#5dcec6')
        plotter.plot(pdTimesteps, [self.Abfluss[key]["P10"] for key in self.keyIter], label='10th percentile')
        plotter.plot(pdTimesteps,[self.Abfluss[key]["P90"] for key in self.keyIter], label='90th percentile')
        plotter.xlabel('time', fontsize=13)
        plotter.ylabel('Q [m^3/s]', fontsize=13)
        #plotter.xticks(rotation=45)plotter.legend()
        plotter.grid(True)

        plotter.subplot(412)
        plotter.plot(pdTimesteps, [self.Abfluss[key]["StdDev"] for key in self.keyIter], label='std. dev. of the simulations')
        plotter.xlabel('time', fontsize=13)
        plotter.ylabel('Std. Dev. [m^3/s]', fontsize=13)
        #plotter.xlim(0, 200)
        #plotter.ylim(0, 20)
        #plotter.xticks(rotation=45)
        plotter.legend()
        plotter.grid(True)

        if is_Sobol_m_computed:
            plotter.subplot(413)
            for i in range(len(labels)):
                plotter.plot(pdTimesteps, [self.Abfluss[key]["Sobol_m"][i] for key in self.keyIter],\
                label=labels[i])
            plotter.xlabel('time', fontsize=13)
            plotter.ylabel('First O. Sobol Indices', fontsize=13)
            #plotter.xticks(rotation=45)
            plotter.legend()
            plotter.grid(True)

        if is_Sobol_t_computed:
            plotter.subplot(414)
            for i in range(len(labels)):
                plotter.plot(pdTimesteps, [self.Abfluss[key]["Sobol_t"][i] for key in self.keyIter],\
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

        #save state to a file

        fileName = self.generateFileName(fileName=fileName, fileNameIdent=fileNameIdent,\
         directory=self.working_dir, fileNameIdentIsFullName=fileNameIdentIsFullName)

        statFileName = fileName + '.pkl'

        with open(statFileName, 'wb') as handle:
            pickle.dump(self.Abfluss, handle, protocol=pickle.HIGHEST_PROTOCOL)
            #dill.dump(self, f)
        #pickle_out = open(statFileName,"wb")
        #pickle.dump(self.Abfluss, pickle_out)
        #pickle_out.close()




#helper function
def _power(my_list):
    return [ x**2 for x in my_list ]


# Functions needed for calculating Sobol's indices using MC samples and Saltelli's method

def _separate_output_values_2(Y, D, N):
    """
    Input:
    dim(Y) = (N*(2+D) x t)
    D - Stochastic dimension
    N - Numer of samples
    Return:
    A - function evaluations based on m1 Nxt
    B - function evaluations based on m2 Nxt
    A_B - array of function evaluations based on m1 with some rows from m2,
    len(A_B) = D;
    """
    A = Y[0:N,:]
    B = Y[N:2*N,:]

    A_B = []
    for j in range(D):
        start = (j + 2)*N
        end = start + N
        temp = np.array(Y[start:end,:])
        A_B.append(temp)

    #return A.T, B.T, A_B
    return A, B, A_B

def _separate_output_values_j(Y, D, N, j):
    """
    Input:
    dim(Y) = (N*(2+D) x t)
    D - Stochastic dimension
    N - Numer of samples
    j - in range(0,D)
    Return:
    A - function evaluations based on m1 Nxt
    B - function evaluations based on m2 Nxt
    A_B - function evaluations based on m1 with jth row from m2
    """

    A = Y[0:N,:]
    B = Y[N:2*N, :]
    start = (j + 2)*N
    end = start + N
    A_B = Y[start:end,:]

    return A, B, A_B


def _Sens_m_sample_1(Y, D, N):
    """
    First order sensitivity indices - Chaospy
    """
    A, B, A_B = _separate_output_values_2(Y, D, N)

    #variance = np.var(A, -1)
    variance = np.var(A, axis=0)

    mean = .5*(np.mean(A) + np.mean(B))
    #mean = .5*(np.mean(A, axis=0) + np.mean(B, axis=0))
    A -= mean
    B -= mean

    out = [
        #np.mean(B*((A_B[j].T-mean)-A), -1) /
        np.mean(B*((A_B[j]-mean)-A),axis=0) /
        np.where(variance, variance, 1)
        for j in range(D)
        ]

    return np.array(out)

def _Sens_m_sample_2(Y, D, N):
    """
    First order sensitivity indices - Homma(1996) & Sobolo (2007)
    """
    A, B, A_B = _separate_output_values_2(Y, D, N)

    #variance = np.var(A, -1)
    variance = np.var(A, axis=0)

    #mean = .5*(np.mean(A) + np.mean(B))
    mean = .5*(np.mean(A, axis=0) + np.mean(B, axis=0))

    out = [
        #(np.mean(B*A_B[j].T, -1) - mean**2) /
        (np.mean(B*A_B[j], axis=0) - mean**2) /
        np.where(variance, variance, 1)
        for j in range(D)
        ]

    return np.array(out)

def _Sens_m_sample_3(Y, D, N):
    """
    First order sensitivity indices - Saltelli 2010.
    """
    A, B, A_B = _separate_output_values_2(Y, D, N)

    #variance = np.var(A, -1)
    variance = np.var(A, axis=0, ddof=1)

    #out = [
    #    #np.mean(B*(A_B[j].T-A), -1) /
    #    np.mean(B*(A_B[j]-A), axis=0) /
    #    np.where(variance, variance, 1)
    #    for j in range(D)
    #    ]
    s_i = []
    for j in range(D):
        #np.dot(B, (A_B[j]-A))
        s_i_j = np.mean(B*(A_B[j]-A), axis=0) / np.where(variance, variance, 1)
        s_i.append(s_i_j)

    return np.array(s_i)

def _Sens_m_sample_4(Y, D, N):
    """
    First order sensitivity indices - Jensen.
    """
    A, B, A_B = _separate_output_values_2(Y, D, N)

    #variance = np.var(A, -1)
    variance = np.var(A, axis=0, ddof=1)

    out = [
        #1 - np.mean((A_B[j].T-B)**2, -1) /
        1 - np.mean((A_B[j]-B)**2, axis=0) /
        (2*np.where(variance, variance, 1))
        for j in range(D)
        ]

    return np.array(out)




def _Sens_t_sample_1(Y, D, N):
    """
    Total order sensitivity indices - Chaospy
    """
    A, B, A_B = _separate_output_values_2(Y, D, N)

    #variance = np.var(A, -1)
    variance = np.var(A, axis=0)

    out = [
        #1-np.mean((A_B[j].T-B)**2, -1) /
        1-np.mean((A_B[j]-B)**2, axis=0) /
        (2*np.where(variance, variance, 1))
        for j in range(D)
        ]

    return np.array(out)

def _Sens_t_sample_2(Y, D, N):
    """
    Total order sensitivity indices - Homma 96
    """
    A, B, A_B = _separate_output_values_2(Y, D, N)

    #mean = .5*(np.mean(A) + np.mean(B))
    mean = .5*(np.mean(A, axis=0) + np.mean(B, axis=0))

    #variance = np.var(A, -1)
    variance = np.var(A, axis=0)

    out = [
        #1-(np.mean(A*A_B[j].T, -1) - mean**2)/
        1-(np.mean(A*A_B[j], axis=0) - mean**2)/
        np.where(variance, variance, 1)
        for j in range(D)
    ]

    return np.array(out)

def _Sens_t_sample_3(Y, D, N):
    """
    Total order sensitivity indices - Sobel 2007
    """
    A, B, A_B = _separate_output_values_2(Y, D, N)

    #variance = np.var(A, -1)
    variance = np.var(A, axis=0, ddof=1)

    out = [
        #np.mean(A*(A-A_B[j].T), -1) /
        np.mean(A*(A-A_B[j]),  axis=0) /
        np.where(variance, variance, 1)
        for j in range(D)
        ]

    return np.array(out)


def _Sens_t_sample_4(Y, D, N):
    """
    Total order sensitivity indices - Saltelli 2010 & Jensen
    """
    A, B, A_B = _separate_output_values_2(Y, D, N)

    #variance = np.var(A, -1)
    variance = np.var(A, axis=0, ddof=1)

    out = [
        #np.mean((A-A_B[j].T)**2, -1) /
        np.mean((A-A_B[j])**2, axis=0) /
        (2*np.where(variance, variance, 1))
        for j in range(D)
        ]

    return np.array(out)






# Old code

def _separate_output_values(Y, D, N):
    """
    Return:
    A - function evaluations based on m1 Nxt
    B - function evaluations based on m2 Nxt
    A_B - function evaluations based on m1 with some rows from m2 Nxt
    """

    A_B = np.zeros((N, D))

    A = np.tile(Y[0:N], (D,1))
    A = A.T

    B = np.tile(Y[N:2*N], (D,1))
    B = B.T

    for j in range(D):
        start = (j + 2)*N
        end = start + N
        A_B[:, j] = (Y[start:end]).T

    return A, B, A_B

def _first_order(A, AB, B):
    # First order estimator following Saltelli et al. 2010 CPC, normalized by
    # sample variance
    #return np.mean(B * (AB - A), axis=0) / np.var(np.r_[A, B], axis=0)
    #return np.mean(B * (AB - A), axis=0, dtype=np.float64) / np.var(np.concatenate((A, B), axis=0), axis=0, ddof=1, dtype=np.float64)
    return np.mean(B * (AB - A), axis=0, dtype=np.float64) / np.var(np.r_[A, B], axis=0, ddof=1, dtype=np.float64)

def _total_order(A, AB, B):
    # Total order estimator following Saltelli et al. 2010 CPC, normalized by
    # sample variance
    #return 0.5 * np.mean((A - AB) ** 2, axis=0) / np.var(np.r_[A, B], axis=0)
    #return np.mean(B * (AB - A), axis=0, dtype=np.float64) / np.var(np.concatenate((A, B), axis=0), axis=0, ddof=1, dtype=np.float64)
    return np.mean(B * (AB - A), axis=0, dtype=np.float64) / np.var(np.r_[A, B], axis=0, ddof=1, dtype=np.float64)
