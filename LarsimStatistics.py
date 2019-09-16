import chaospy as cp
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plotter
import itertools
import os
from distutils.util import strtobool


from uqef.stat import Statistics

import paths
import LARSIM_configs as config

class Samples(object):
    """
     Samples is a collection of the sampled results of a whole UQ simulation

     """

    #collects values from every simulation for each time steps as array, saves in dictionary
    def __init__(self, rawSamples, station=None, type_of_output='Abfluss Messung', pathsDataFormat="False", dailyOutput="False"):
        """

        :param rawSamples: results returned by solver. Either list of paths to diff. ergebnis files or pandas.DataFrame object containing output of all the runs
        :param pathsDataFormat: wheather or not results are in form of the path to result file or concrete data arrays
        :param station: Name of the station in which output we are interested in, if None - filter out all the stations
        :param type_of_output: can be Abfluss Simulation or ....
        """

        #self.df_simulation_result = pd.DataFrame(columns=['Index_run', 'Stationskennung', 'Type', 'TimeStamp', 'Value'])

        list_of_single_df = []
        for index_run, value in enumerate(rawSamples,): #Important that the results inside rawSamples (resulted paths) are in sorted order and correspond to parameters order
        #TODO What happens if value is None - ergebnis file was not read inside of the model properly
            if strtobool(pathsDataFormat):
                df_single_ergebnis = config.result_parser_toPandas(value, index_run)
            else:
                df_single_ergebnis = value

            if station is not None:
                df_single_ergebnis = df_single_ergebnis.loc[
                    (df_single_ergebnis['Stationskennung'] == station) &
                    (df_single_ergebnis['Type'] == type_of_output)]
            else:
                df_single_ergebnis = df_single_ergebnis.loc[
                    df_single_ergebnis['Type'] == type_of_output]
            list_of_single_df.append(df_single_ergebnis)


        #TODO Calculate predictive power / correctness of the model


        df_simulation_result = pd.concat(list_of_single_df, ignore_index=True, sort=False, axis=0)

        df_simulation_result['Value'] = df_simulation_result['Value'].astype(float)
        #print("Data Frame with All the simulation results : {}".format(df_simulation_result.dtypes))

        print("Number of Unique TimeStamps (Hourly): {}".format(len(df_simulation_result.TimeStamp.unique())))

        if strtobool(dailyOutput):
            # Average over time - change colume TimeStamp and Value!!!
            #df_simulation_result = config.transformToDailyResolution(df_simulation_result)
            #or
            df_simulation_result['TimeStamp_Date'] = [entry.date() for entry in df_simulation_result['TimeStamp']]
            df_simulation_result['TimeStamp_Time'] = [entry.time() for entry in df_simulation_result['TimeStamp']]
            df_simulation_result = df_simulation_result.groupby(['Stationskennung', 'Type', 'TimeStamp_Date', 'Index_run'])['Value'].mean().reset_index()
            df_simulation_result = df_simulation_result.rename({'TimeStamp_Date' : 'TimeStamp'}, axis = 'columns')
            df_simulation_result['TimeStamp'] = df_simulation_result['TimeStamp'].apply(lambda x: pd.Timestamp(x))
            #print(df_simulation_result.dtypes)
            print("Number of Unique TimeStamps (Daily): {}".format(len(df_simulation_result.TimeStamp.unique())))

        # TODO write get/set methods for this attributes
        self.df_simulation_result = df_simulation_result

        # TODO write get/set methods for this attributes
        self.df_time_discharges = df_simulation_result.groupby(["Stationskennung","TimeStamp"])["Value"].apply(lambda df: df.reset_index(drop=True)).unstack()



class LarsimStatistics(Statistics):


    def __init__(self, configurationObject):
        Statistics.__init__(self)

        self.configurationObject = configurationObject

        try:
            self.working_dir = self.configurationObject["Directories"]["working_dir"]
        except KeyError:
            self.working_dir = paths.working_dir  # directoy for all the larsim runs


    def calcStatisticsForMc(self, rawSamples, timesteps,
                            simulationNodes, numEvaluations, solverTimes,
                            work_package_indexes, original_runtime_estimator, regression, saltelli, order):
        """

        :param rawSamples: simulation.solver.results
        :param timesteps: simulation.solver.timesteps (which are LarsimModel.timesteps() - duration of each simulation in discrete timesteps)
        :param simulationNodes: simulationNodes
        :param numEvaluations: simulation.numEvaluations
        :param solverTimes: simulation.solver.solverTimes
        :param work_package_indexes: simulation.solver.work_package_indexes
        :param original_runtime_estimator: simulation.original_runtime_estimator
        :return:
        """

        #samples = Samples(rawSamples, station='MARI', type_of_output='Abfluss Simulation', pathsDataFormat=True)
        samples = Samples(rawSamples, station=self.configurationObject["Output"]["station"],
                          type_of_output=self.configurationObject["Output"]["type_of_output"],
                          pathsDataFormat=self.configurationObject["Output"]["pathsDataFormat"],
                          dailyOutput=self.configurationObject["Output"]["dailyOutput"])

        # Save the DataFrame containing all the simulation results
        samples.df_time_discharges.to_csv(path_or_buf=os.path.abspath(os.path.join(self.working_dir, "df_all_simulations.csv")),
                                index=True)

        #self.df_simulation_result = samples.df_simulation_result


        if regression:
            nodes = simulationNodes.distNodes
            dist = simulationNodes.joinedDists
            P = cp.orth_ttr(order, dist)

        #self.timesteps = timesteps #atm this is just a scalar representing total number of timesteps
        self.timesteps = samples.df_simulation_result.TimeStamp.unique()
        self.numbTimesteps = len(self.timesteps)

        #print("timesteps Info")
        #print(type(self.timesteps))
        print("numbTimesteps is: {}".format(self.numbTimesteps))

        # percentiles
        numPercSamples = 10 ** 5

        self.station_names = samples.df_simulation_result.Stationskennung.unique()

        dim = len(simulationNodes.nodeNames)

        self.Abfluss = {}

        #Gate ground truth measurements in case you want to plot them together with other statistics
        gt_dataFrame = pd.read_csv(os.path.abspath(os.path.join(self.working_dir, "df_measured.csv")))
        gt_dataFrame['TimeStamp'] = gt_dataFrame['TimeStamp'].astype('datetime64[ns]')
        gt_dataFrame = config.filterResultForStation(gt_dataFrame, station=self.configurationObject["Output"]["station"])
        gt_dataFrame_aligned = config.align_dataFrames_timewise(gt_dataFrame,
                                                                      samples.df_simulation_result)  # TODO get rid of this eventually - make gt as long as predictions!
        if strtobool(self.configurationObject["Output"]["dailyOutput"]):
            gt_dataFrame_aligned_daily = config.transformToDailyResolution(gt_dataFrame_aligned)
            gt_measurements_array = gt_dataFrame_aligned_daily.Value.values

        else:
            gt_measurements_array = gt_dataFrame_aligned.Value.values
        self.Abfluss["Ground_Truth_Measurements"] = gt_measurements_array
        

        grouped = samples.df_simulation_result.groupby(['Stationskennung','TimeStamp'])
        groups = grouped.groups
        # Do statistics calulcations for each station for each time step
        for key,val in groups.items():
            discharge_values = samples.df_simulation_result.iloc[val.values].Value.values #numpy array nx1, for sartelli it should be n(2+d)x1
            #print("Size of a single discharge array (single station  - single timestep) is: ")
            #print(discharge_values.shape)

            if regression:
                qoi_gPCE = cp.fit_regression(P, nodes, discharge_values)
                self.Abfluss[key] = {}
                self.Abfluss[key]["Q"] = discharge_values
                self.Abfluss[key]["E"] = float((cp.E(qoi_gPCE, dist)))
                self.Abfluss[key]["Var"] = float((cp.Var(qoi_gPCE, dist)))
                self.Abfluss[key]["StdDev"] = float((cp.Std(qoi_gPCE, dist)))
                self.Abfluss[key]["Sobol_m"] = cp.Sens_m(qoi_gPCE, dist)
                self.Abfluss[key]["Sobol_m2"] = cp.Sens_m2(qoi_gPCE, dist)
                self.Abfluss[key]["Sobol_t"] = cp.Sens_t(qoi_gPCE, dist)
                self.Abfluss[key]["P10"] = float(cp.Perc(qoi_gPCE, 10, dist, numPercSamples))
                self.Abfluss[key]["P90"] = float(cp.Perc(qoi_gPCE, 90, dist, numPercSamples))
            elif saltelli:
                self.Abfluss[key] = {}
                discharge_values_saltelli = discharge_values[:, np.newaxis]
                standard_discharge_values = discharge_values_saltelli[:numEvaluations,:] #values based on which we calculate standard statistics

                self.Abfluss[key]["Q"] = standard_discharge_values
                self.Abfluss[key]["E"] = np.sum(standard_discharge_values, axis=0, dtype=np.float64) / numEvaluations
                #self.Abfluss[key]["Var"] = float(np.sum(power(standard_discharge_values)) / numEvaluations - self.Abfluss[key]["E"] ** 2)
                self.Abfluss[key]["Var"] = np.sum((standard_discharge_values - self.Abfluss[key]["E"]) ** 2, axis=0, dtype=np.float64) / (numEvaluations - 1)
                self.Abfluss[key]["StdDev"] = np.sqrt(self.Abfluss[key]["Var"], dtype=np.float64)

                self.Abfluss[key]["P10"] = np.percentile(discharge_values[:numEvaluations], 10, axis=0)
                self.Abfluss[key]["P90"] = np.percentile(discharge_values[:numEvaluations], 90, axis=0)

                self.Abfluss[key]["Sobol_m"] = _Sens_m_sample_4(discharge_values_saltelli, dim, numEvaluations)
                self.Abfluss[key]["Sobol_t"] = _Sens_t_sample_4(discharge_values_saltelli, dim, numEvaluations)
                #print("self.Abfluss[key]["Sobol_m"].shape")
                #print(self.Abfluss[key]["Sobol_m"].shape)
                #print("self.Abfluss[key]["Sobol_t"].shape")
                #print(self.Abfluss[key]["Sobol_t"].shape)

            else:
                self.Abfluss[key] = {}
                self.Abfluss[key]["Q"] = discharge_values
                self.Abfluss[key]["E"] = np.sum(discharge_values, axis=0, dtype=np.float64)/ numEvaluations
                #self.Abfluss[key]["Var"] = float(np.sum(power(discharge_values)) / numEvaluations - self.Abfluss[key]["E"]**2)
                self.Abfluss[key]["Var"] = np.sum((discharge_values - self.Abfluss[key]["E"]) ** 2, axis=0, dtype=np.float64) / (numEvaluations - 1)
                self.Abfluss[key]["StdDev"] = np.sqrt(self.Abfluss[key]["Var"], dtype=np.float64)
                self.Abfluss[key]["P10"] = np.percentile(discharge_values, 10, axis=0)
                self.Abfluss[key]["P90"] = np.percentile(discharge_values, 90, axis=0)


            if isinstance(self.Abfluss[key]["P10"], (list)) and len(self.Abfluss[key]["P10"]) == 1:
                self.Abfluss[key]["P10"]=self.Abfluss[key]["P10"][0]
                self.Abfluss[key]["P90"]=self.Abfluss[key]["P90"][0]

        statistics_dict_path_np=os.path.abspath(os.path.join(self.working_dir, "statistics_dict.npy"))
        np.save(statistics_dict_path_np, self.Abfluss)
        #pickle_out = open(paths.statistics_dict_path_pkl,"wb")
        #pickle.dump(self.Abfluss, pickle_out)
        #pickle_out.close()


    def calcStatisticsForSc(self, rawSamples, timesteps,
                            simulationNodes, order, solverTimes,
                            work_package_indexes, original_runtime_estimator, regression):
        """
        in ScSimulation.calculateStatistics
        statistics.calcStatisticsForSc(self.solver.results, self.solver.timesteps, simulationNodes, self.p_order, self.solver.solverTimes,
                                       self.solver.work_package_indexes, self.original_runtime_estimator=None)
        :param rawSamples:
        :param timesteps:
        :param simulationNodes:
        :param order:
        :param solverTimes:
        :param work_package_indexes:
        :param original_runtime_estimator:
        :return:
        """

        nodes = simulationNodes.distNodes
        weights = simulationNodes.weights
        dist = simulationNodes.joinedDists

        #samples = Samples(rawSamples, station='MARI', type_of_output='Abfluss Simulation', pathsDataFormat=True)
        samples = Samples(rawSamples, station=self.configurationObject["Output"]["station"],
                          type_of_output=self.configurationObject["Output"]["type_of_output"],
                          pathsDataFormat=self.configurationObject["Output"]["pathsDataFormat"],
                          dailyOutput=self.configurationObject["Output"]["dailyOutput"])

        # Save the DataFrame containing all the simulation results
        samples.df_time_discharges.to_csv(
            path_or_buf=os.path.abspath(os.path.join(self.working_dir, "df_all_simulations.csv")),
            index=True)

        #self.timesteps = timesteps #this is just a scalar representing total number of timesteps
        self.timesteps = samples.df_simulation_result.TimeStamp.unique()
        self.numbTimesteps = len(self.timesteps)

        #print("timesteps Info")
        #print(type(self.timesteps))
        print("numbTimesteps is: {}".format(self.numbTimesteps))


        P = cp.orth_ttr(order, dist)

        # percentiles
        numPercSamples = 10 ** 5

        self.station_names = samples.df_simulation_result.Stationskennung.unique()

        self.Abfluss = {}

        # Gate ground truth measurements in case you want to plot them together with other statistics
        gt_dataFrame = pd.read_csv(os.path.abspath(os.path.join(self.working_dir, "df_measured.csv")))
        gt_dataFrame['TimeStamp'] = gt_dataFrame['TimeStamp'].astype('datetime64[ns]')
        gt_dataFrame = config.filterResultForStation(gt_dataFrame,
                                                     station=self.configurationObject["Output"]["station"])
        gt_dataFrame_aligned = config.align_dataFrames_timewise(gt_dataFrame,
                                                                samples.df_simulation_result)  # TODO get rid of this eventually - make gt as long as predictions!
        if strtobool(self.configurationObject["Output"]["dailyOutput"]):
            gt_dataFrame_aligned_daily = config.transformToDailyResolution(gt_dataFrame_aligned)
            gt_measurements_array = gt_dataFrame_aligned_daily.Value.values

        else:
            gt_measurements_array = gt_dataFrame_aligned.Value.values
        self.Abfluss["Ground_Truth_Measurements"] = gt_measurements_array

        grouped = samples.df_simulation_result.groupby(['Stationskennung','TimeStamp'])
        groups = grouped.groups
        for key,val in groups.items():
            discharge_values = samples.df_simulation_result.iloc[val.values].Value.values


            if regression:
                qoi_gPCE = cp.fit_regression(P, nodes, discharge_values)
            else:
                qoi_gPCE = cp.fit_quadrature(P, nodes, weights, discharge_values) #fit_quadrature for each time step for this station over multiple runs

            print("Shape of the qoi gPCE cofficients:")
            #print(type(qoi_gPCE.shape))
            print(qoi_gPCE.shape)


            self.Abfluss[key] = {}
            self.Abfluss[key]["Q"] = discharge_values
            self.Abfluss[key]["E"] = (cp.E(qoi_gPCE, dist))
            self.Abfluss[key]["Var"] = (cp.Var(qoi_gPCE, dist))
            self.Abfluss[key]["StdDev"] = (cp.Std(qoi_gPCE, dist))
            self.Abfluss[key]["Sobol_m"] = cp.Sens_m(qoi_gPCE, dist)
            self.Abfluss[key]["Sobol_m2"] = cp.Sens_m2(qoi_gPCE, dist)
            self.Abfluss[key]["Sobol_t"] = cp.Sens_t(qoi_gPCE, dist)
            self.Abfluss[key]["P10"] = cp.Perc(qoi_gPCE, 10, dist, numPercSamples)
            self.Abfluss[key]["P90"] = cp.Perc(qoi_gPCE, 90, dist, numPercSamples)

            if isinstance(self.Abfluss[key]["P10"], (list)) and len(self.Abfluss[key]["P10"]) == 1:
                self.Abfluss[key]["P10"]= self.Abfluss[key]["P10"][0]
                self.Abfluss[key]["P90"] = self.Abfluss[key]["P90"][0]

        statistics_dict_path_np=os.path.abspath(os.path.join(self.working_dir, "statistics_dict.npy"))
        np.save(statistics_dict_path_np, self.Abfluss)
        #pickle_out = open(paths.statistics_dict_path_pkl,"wb")
        #pickle.dump(self.Abfluss, pickle_out)
        #pickle_out.close()

    def plotResults(self, simulationNodes, display=False, station='MARI',
                    fileName="", fileNameIdent="", directory="./",
                    fileNameIdentIsFullName=False, safe=True):

        #####################################
        ### plot: mean + percentiles
        #####################################

        figure = plotter.figure(1, figsize=(13, 10))
        window_title = 'LarsimModel statistics - ' + station
        figure.canvas.set_window_title(window_title)

        pdTimesteps = [pd.Timestamp(timestep) for timestep in self.timesteps]

        keyIter = list(itertools.product([station,],pdTimesteps))
        #self.Abfluss[((station,oneTimetep) for oneTimetep in pdTimesteps)]
        #listE = [self.Abfluss[key]["E"] for key in itertools.product([station,],pdTimesteps)]

        
        plotter.subplot(411)
        # plotter.title('mean')
        plotter.plot(pdTimesteps, [self.Abfluss[key]["E"] for key in keyIter], '-r', label='mean')
        plotter.plot(pdTimesteps, self.Abfluss["Ground_Truth_Measurements"], '-g', label='gt')
        #plotter.fill_between(pdTimesteps, [self.Abfluss[key]["P10"] for key in keyIter], [self.Abfluss[key]["P90"] for key in keyIter], facecolor='#5dcec6')
        #plotter.plot(pdTimesteps, [self.Abfluss[key]["P10"] for key in keyIter], label='10th percentile')
        #plotter.plot(pdTimesteps,[self.Abfluss[key]["P90"] for key in keyIter], label='90th percentile')
        plotter.xlabel('time', fontsize=13)
        plotter.ylabel('Larsim Stat. Values [cm/s]', fontsize=13)
        #plotter.xlim(0, 200)
        #ymin, ymax = plotter.ylim()
        #plotter.ylim(0, 20)
        plotter.xticks(rotation=45)
        plotter.legend()  # enable the legend
        plotter.grid(True)

        plotter.subplot(412)
        # plotter.title('standard deviation')
        plotter.plot(pdTimesteps, [self.Abfluss[key]["StdDev"] for key in keyIter], label='std. dev.')
        plotter.xlabel('time', fontsize=13)
        plotter.ylabel('Standard Deviation [cm/s]', fontsize=13)
        #plotter.xlim(0, 200)
        #plotter.ylim(0, 20)
        plotter.xticks(rotation=45)
        plotter.legend()  # enable the legend
        plotter.grid(True)

        #plotter.subplot(513)
        ## plotter.title('discharge')
        #plotter.plot(pdTimesteps, [self.Abfluss[key]["Q"] for key in keyIter])
        #plotter.xlabel('time', fontsize=13)
        #plotter.ylabel('Q value', fontsize=13)
        #plotter.xticks(rotation=45)
        #plotter.legend()  # enable the legend
        #plotter.grid(True)


        if "Sobol_t" in self.Abfluss[keyIter[0]]:
            plotter.subplot(413)
            sobol_labels = simulationNodes.nodeNames
            for i in range(len(sobol_labels)):
                if self.Abfluss[keyIter[0]]["Sobol_t"].shape[0] == len(self.timesteps):
                    plotter.plot(pdTimesteps, [(self.Abfluss[key]["Sobol_t"].T)[i] for key in keyIter],
                                 label=sobol_labels[i])
                else:
                    plotter.plot(pdTimesteps, [self.Abfluss[key]["Sobol_t"][i] for key in keyIter],
                                 label=sobol_labels[i])
            plotter.xlabel('time', fontsize=13)
            plotter.ylabel('total sobol indices', fontsize=13)
            ##plotter.xlim(0, 200)
            # ##plotter.ylim(-0.1, 1.1)
            plotter.xticks(rotation=45)
            plotter.legend()  # enable the legend
            # plotter.grid(True)

        if "Sobol_m" in self.Abfluss[keyIter[0]]:
            plotter.subplot(414)
            sobol_labels = simulationNodes.nodeNames
            for i in range(len(sobol_labels)):
                if self.Abfluss[keyIter[0]]["Sobol_m"].shape[0] == len(self.timesteps):
                    plotter.plot(pdTimesteps, [(self.Abfluss[key]["Sobol_m"].T)[i] for key in keyIter],
                                 label=sobol_labels[i])
                else:
                    plotter.plot(pdTimesteps, [self.Abfluss[key]["Sobol_m"][i] for key in keyIter],
                                 label=sobol_labels[i])
            plotter.xlabel('time', fontsize=13)
            plotter.ylabel('first order sobol indices', fontsize=13)
            ##plotter.xlim(0, 200)
            # ##plotter.ylim(-0.1, 1.1)
            plotter.xticks(rotation=45)
            plotter.legend()  # enable the legend
            # plotter.grid(True)

        if "Sobol_t" in self.Abfluss[keyIter[0]]:
            sobol_t_qoi_file = os.path.abspath(os.path.join(self.working_dir, "sobol_t_qoi_file.npy"))
            np.save(sobol_t_qoi_file, np.array([self.Abfluss[key]["Sobol_t"][i] for key in keyIter]))
        if "Sobol_m" in self.Abfluss[keyIter[0]]:
            sobol_m_qoi_file = os.path.abspath(os.path.join(self.working_dir, "sobol_m_qoi_file.npy"))
            np.save(sobol_m_qoi_file, np.array([self.Abfluss[key]["Sobol_m"][i] for key in keyIter]))


        # save figure
        pdfFileName = os.path.abspath(os.path.join(self.working_dir, paths.figureFileName + "_uq.pdf"))
        pngFileName = os.path.abspath(os.path.join(self.working_dir, paths.figureFileName + "_uq.png"))

        plotter.savefig(pdfFileName, format='pdf')
        plotter.savefig(pngFileName, format='png')

        if display:
            plotter.show()

        plotter.close()




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
        A_B.append(Y[start:end,:])

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
    variance = np.var(A, axis=0)

    out = [
        #np.mean(B*(A_B[j].T-A), -1) /
        np.mean(B*(A_B[j]-A), axis=0) /
        np.where(variance, variance, 1)
        for j in range(D)
        ]

    return np.array(out)

def _Sens_m_sample_4(Y, D, N):
    """
    First order sensitivity indices - Jensen.
    """
    A, B, A_B = _separate_output_values_2(Y, D, N)

    #variance = np.var(A, -1)
    variance = np.var(A, axis=0)

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
    variance = np.var(A, axis=0)

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
    variance = np.var(A, axis=0)

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
