import chaospy as cp
import numpy as np
import os
import pandas as pd
#import pickle
import matplotlib.pyplot as plotter
#from itertools import product
import itertools

from uqef.stat import Statistics

import paths
import LARSIM_configs as config

class Samples(object):
    """
     Samples is a collection of the sampled results of a whole UQ simulation

     """

    #collects values from every simulation for each time steps as array, saves in dictionary
    def __init__(self, rawSamples, station=None, type_of_output='Abfluss Simulation', pathsDataFormat=True):
        """

        :param rawSamples: results returned by solver. Either list of paths to diff. ergebnis files or pandas.DataFrame object containing output of all the runs
        :param pathsDataFormat: wheather or not results are in form of the path to result file or concrete data arrays
        :param station: Name of the station in which output we are interested in, if None - filter out all the stations
        :param type_of_output: can be Abfluss Simulation or ....
        """

        #self.df_simulation_result = pd.DataFrame(columns=['Index_run', 'Stationskennung', 'Type', 'TimeStamp', 'Value'])

        if pathsDataFormat: #in case rawSamples is list of path to reasults
            list_of_single_df = []
            for index_run, value in enumerate(rawSamples,): #Important that the results inside rawSamples (resulted paths) are in sorted order and correspond to parameters order
                if value is not None: #TODO IVANA What happens it it is?
                    df_single_ergebnis = config.result_parser_toPandas(value,index_run)
                    if station is not None:
                        df_single_ergebnis = df_single_ergebnis.loc[
                            (df_single_ergebnis['Stationskennung'] == station) &
                            (df_single_ergebnis['Type'] == type_of_output)]
                    else:
                        df_single_ergebnis = df_single_ergebnis.loc[
                                        df_single_ergebnis['Type'] == type_of_output]
                    list_of_single_df.append(df_single_ergebnis)
            self.df_simulation_result = pd.concat(list_of_single_df, ignore_index=True, sort=False, axis=0)

                    #self.df_simulation_result = pd.concat([self.df_simulation_result, df_single_ergebnis], ignore_index=True, sort=False)

        else: #in case rawSamples is already pandas.DataFrame object
            self.df_simulation_result = rawSamples
            if station is not None:
                self.df_simulation_result = self.df_simulation_result.loc[(self.df_simulation_result['Stationskennung'] == station) & (self.df_simulation_result['Type'] == type_of_output)]
            else:
                self.df_simulation_result = self.df_simulation_result.loc[
                                self.df_simulation_result['Type'] == type_of_output]



class LarsimStatistics(Statistics):


    def __init__(self):
        Statistics.__init__(self)


    #timesteps = self.solver.timesteps which are LarsimModel.timesteps() - duration of each simulation in discrete timesteps
    #rawSamples = self.solver.results
    def calcStatisticsForMc(self, rawSamples, timesteps,
                            simulationNodes, numEvaluations, solverTimes,
                            work_package_indexes, original_runtime_estimator, regression, saltelli, order):
        """

        :param rawSamples: simulation.solver.results
        :param timesteps: simulation.solver.timesteps
        :param simulationNodes: simulationNodes
        :param numEvaluations: simulation.numEvaluations
        :param solverTimes: simulation.solver.solverTimes
        :param work_package_indexes: simulation.solver.work_package_indexes
        :param original_runtime_estimator: simulation.original_runtime_estimator
        :return:
        """

        samples = Samples(rawSamples, station='MARI', type_of_output='Abfluss Simulation', pathsDataFormat=True)

        if regression:
            nodes = simulationNodes.distNodes
            dist = simulationNodes.joinedDists
            P = cp.orth_ttr(order, dist)

        #self.timesteps = timesteps #this is just a scalar representing total number of timesteps
        self.timesteps = samples.df_simulation_result.TimeStamp.unique()
        self.numbTimesteps = len(self.timesteps)

        # percentiles
        numPercSamples = 10 ** 5

        self.station_names = samples.df_simulation_result.Stationskennung.unique()

        self.Abfluss = {}

        grouped = samples.df_simulation_result.groupby(['Stationskennung','TimeStamp'])
        groups = grouped.groups
        for key,val in groups.items():
            discharge_values = samples.df_simulation_result.iloc[val.values].Value.values #numpy array 1xn, for sartelli it should be 1xn(2+d)
            print("Size of a single discharge array (single station  -single timestep) is: ")
            print(discharge_values.shape)

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
                standard_discharge_values = discharge_values[:numEvaluations]
                self.Abfluss[key]["Q"] = standard_discharge_values
                self.Abfluss[key]["E"] = float(np.sum(standard_discharge_values) / numEvaluations)
                self.Abfluss[key]["Var"] = float(
                    np.sum(power(standard_discharge_values)) / numEvaluations - self.Abfluss[key]["E"] ** 2)
                self.Abfluss[key]["StdDev"] = float(np.sqrt(self.Abfluss[key]["Var"]))
                self.Abfluss[key]["P10"] = float(np.percentile(standard_discharge_values, 10, axis=0))
                self.Abfluss[key]["P90"] = float(np.percentile(standard_discharge_values, 90, axis=0))
                # Calculate Sobol's Indices based on Saltelli 2010 paper!
                # Transform Discharge Values 1xn(1+d)
                dim = len(simulationNodes.nodeNames)
                A, B, AB = separate_output_values(discharge_values, dim, numEvaluations)
                print("A Matrix shape: ")
                print(A.shape)
                print("\n")
                print("B Matrix shape: ")
                print(B.shape)
                print("\n")
                print("AB Matrix shape: ")
                print(AB.shape)
                print("\n")
                si_first_orded_array = first_order(A, AB, B)
                si_total_orded_array = total_order(A, AB, B)
                for j in range(dim):
                    self.Abfluss[key]["Sobol_m"][j] = si_first_orded_array[j]
                    self.Abfluss[key]["Sobol_t"][j] = si_total_orded_array[j]
                #TODO Do this in parallel for each dimension!
            else:
                self.Abfluss[key] = {}
                self.Abfluss[key]["Q"] = discharge_values
                self.Abfluss[key]["E"] = float(np.sum(discharge_values)/ numEvaluations)
                self.Abfluss[key]["Var"] = float(np.sum(power(discharge_values)) / numEvaluations - self.Abfluss[key]["E"]**2)
                self.Abfluss[key]["StdDev"] = float(np.sqrt(self.Abfluss[key]["Var"]))
                self.Abfluss[key]["P10"] = float(np.percentile(discharge_values, 10, axis=0))
                self.Abfluss[key]["P90"] = float(np.percentile(discharge_values, 90, axis=0))


            if isinstance(self.Abfluss[key]["P10"], (list)) and len(self.Abfluss[key]["P10"]) == 1:
                self.Abfluss[key]["P10"]= self.Abfluss[key]["P10"][0]
                self.Abfluss[key]["P90"] = self.Abfluss[key]["P90"][0]

        np.save(paths.statistics_dict_path_np, self.Abfluss)
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

        samples = Samples(rawSamples, station='MARI', type_of_output='Abfluss Simulation', pathsDataFormat=True)

        #self.timesteps = timesteps #this is just a scalar representing total number of timesteps
        self.timesteps = samples.df_simulation_result.TimeStamp.unique()
        self.numbTimesteps = len(self.timesteps)

        P = cp.orth_ttr(order, dist)

        # percentiles
        numPercSamples = 10 ** 5

        self.station_names = samples.df_simulation_result.Stationskennung.unique()

        self.Abfluss = {}

        grouped = samples.df_simulation_result.groupby(['Stationskennung','TimeStamp'])
        groups = grouped.groups
        for key,val in groups.items():
            discharge_values = samples.df_simulation_result.iloc[val.values].Value.values


            if regression:
                qoi_gPCE = cp.fit_regression(P, nodes, discharge_values)
            else:
                qoi_gPCE = cp.fit_quadrature(P, nodes, weights, discharge_values) #fit_quadrature for each time step for this station over multiple runs

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

            if isinstance(self.Abfluss[key]["P10"], (list)) and len(self.Abfluss[key]["P10"]) == 1:
                self.Abfluss[key]["P10"]= self.Abfluss[key]["P10"][0]
                self.Abfluss[key]["P90"] = self.Abfluss[key]["P90"][0]


        np.save(paths.statistics_dict_path_np, self.Abfluss)
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

        plotter.subplot(411)
        # plotter.title('mean')

        keyIter = list(itertools.product([station,],pdTimesteps))
        #self.Abfluss[((station,oneTimetep) for oneTimetep in pdTimesteps)]
        #listE = [self.Abfluss[key]["E"] for key in itertools.product([station,],pdTimesteps)]

        plotter.plot(pdTimesteps, [self.Abfluss[key]["E"] for key in keyIter], 'o', label='mean')
        plotter.fill_between(pdTimesteps, [self.Abfluss[key]["P10"] for key in keyIter], [self.Abfluss[key]["P90"] for key in keyIter], facecolor='#5dcec6')
        plotter.plot(pdTimesteps, [self.Abfluss[key]["P10"] for key in keyIter], 'o', label='10th percentile')
        plotter.plot(pdTimesteps,[self.Abfluss[key]["P90"] for key in keyIter], 'o', label='90th percentile')
        plotter.xlabel('time', fontsize=13)
        plotter.ylabel('Larsim Stat. Values', fontsize=13)
        #plotter.xlim(0, 200)
        #ymin, ymax = plotter.ylim()
        #plotter.ylim(0, 20)
        plotter.xticks(rotation=45)
        plotter.legend()  # enable the legend
        plotter.grid(True)

        plotter.subplot(412)
        # plotter.title('standard deviation')
        plotter.plot(pdTimesteps, [self.Abfluss[key]["StdDev"] for key in keyIter], 'o', label='std. dev.')
        plotter.xlabel('time', fontsize=13)
        plotter.ylabel('Standard Deviation ', fontsize=13)
        #plotter.xlim(0, 200)
        #plotter.ylim(0, 20)
        plotter.xticks(rotation=45)
        plotter.legend()  # enable the legend
        plotter.grid(True)

        plotter.subplot(413)
        plotter.plot(pdTimesteps, [self.Abfluss[key]["Q"] for key in keyIter])
        plotter.xlabel('time', fontsize=13)
        plotter.ylabel('Q value', fontsize=13)
        plotter.xticks(rotation=45)
        plotter.legend()  # enable the legend
        plotter.grid(True)

        #TODO - This might differ depending on the mc vs sc run
        plotter.subplot(414)
        #sobol_labels = ["EQB", "BSF", "TGr"]
        sobol_labels = simulationNodes.nodeNames
        for i in range(len(sobol_labels)):
            plotter.plot(pdTimesteps, [self.Abfluss[key]["Sobol_t"][i] for key in keyIter], 'o', label=sobol_labels[i])
        plotter.xlabel('time', fontsize=13)
        plotter.ylabel('total sobol indices', fontsize=13)
        ##plotter.xlim(0, 200)
        ##plotter.ylim(-0.1, 1.1)
        plotter.xticks(rotation=45)
        plotter.legend()  # enable the legend
        plotter.grid(True)

        # save figure
        pdfFileName = paths.figureFileName + "_uq" + '.pdf'
        pngFileName = paths.figureFileName + "_uq" + '.png'
        plotter.savefig(pdfFileName, format='pdf')
        plotter.savefig(pngFileName, format='png')

        if display:
            plotter.show()

        plotter.close()



#helper function
def power(my_list):
    return [ x**2 for x in my_list ]


# Functions needed for calculating Sobol's indices using MC samples and Saltelli's method

def separate_output_values(Y, D, N):

    # A - function evaluations based on m2 NxD
    # B - function evaluations based on m1 NxD
    # AB - - function evaluations based on m2 with spme rows from m1 NxD

    AB = np.zeros((N, D))
    B = np.tile(Y[0:N], (D,1)) #Y[0:N]
    B = B.T
    #B = Y[0:N].T
    A = np.tile(Y[N:2*N], (D,1))
    A = A.T
    #A = Y[N:2*N].T
    for j in range(D):
        start = (j + 2)*N
        end = start + N
        AB[:, j] = (Y[start:end]).T

    return A, B, AB

def first_order(A, AB, B):
    # First order estimator following Saltelli et al. 2010 CPC, normalized by
    # sample variance
    #return np.mean(B * (AB - A), axis=0) / np.var(np.r_[A, B], axis=0)
    return np.mean(B * (AB - A), axis=0) / np.var(np.concatenate((A, B), axis=0), axis=0)

def total_order(A, AB, B):
    # Total order estimator following Saltelli et al. 2010 CPC, normalized by
    # sample variance
    #return 0.5 * np.mean((A - AB) ** 2, axis=0) / np.var(np.r_[A, B], axis=0)
    return np.mean(B * (AB - A), axis=0) / np.var(np.concatenate((A, B), axis=0), axis=0)