import os
import numpy as np
import chaospy as cp

from uqef.stat import Statistics

class Samples(object):

    #collects values from every simulation for each time steps as array, saves in dictionary
    def __init__(self, rawSamples):
        self.Abfluss = {}
        for identification in rawSamples[0]:
            self.Abfluss[identification] = []
            for i in range(0, len(rawSamples[0][identification]["Abfluss Simulation"])):
                temp = []
                for j in range(0, len(rawSamples)):
                    temp.append(rawSamples[j][identification]["Abfluss Simulation"][i])
                self.Abfluss[identification].append(temp)

class LarsimStatistics(Statistics):


    def __init__(self):
        Statistics.__init__(self)


    def calcStatisticsForMc(self, rawSamples, timesteps,
                            simulationNodes, numEvaluations, solverTimes,
                            work_package_indexes, original_runtime_estimator):
        self.timesteps = timesteps
        samples = Samples(rawSamples)
        self.station_names = []
        self.Abfluss = {}
        # calculate relevant statistical values
        for identification in samples.Abfluss:
            self.Abfluss[identification] = {}

        for identification in self.Abfluss:
            self.station_names.append(identification)
            self.Abfluss[identification]["E"] = []
            self.Abfluss[identification]["Var"] = []
            self.Abfluss[identification]["StdDev"] = []
            self.Abfluss[identification]["P10"] = []
            self.Abfluss[identification]["P90"] = []
            for i  in range(0, len( samples.Abfluss[identification])):
                self.Abfluss[identification]["E"].append(float(np.sum(samples.Abfluss[identification][i])/ numEvaluations))
                self.Abfluss[identification]["Var"].append(float(np.sum(power(samples.Abfluss[identification][i])) / numEvaluations - power(self.Abfluss[identification]["E"])[i]))
                self.Abfluss[identification]["StdDev"].append(float(np.sqrt(self.Abfluss[identification]["Var"][i])))
                self.Abfluss[identification]["P10"].append(float(np.percentile(samples.Abfluss[identification][i], 10, axis=0)))
                self.Abfluss[identification]["P90"].append(float(np.percentile(samples.Abfluss[identification][i], 90, axis=0)))

        # save dicts for later use
        if os.stat("./LARSIM_configs/master_files/dicts.npy").st_size == 0:
            np.save("./LARSIM_configs/master_files/dicts.npy", self.Abfluss)
        else:
            dict = np.load("./LARSIM_configs/master_files/dicts.npy").item()
            for identification in self.Abfluss:
                for keys in self.Abfluss[identification]:
                    for i in range(0, len(self.Abfluss[identification][keys])):
                        dict[identification][keys].append(self.Abfluss[identification][keys][i])
            np.save("./LARSIM_configs/master_files/dicts.npy", dict)


    def calcStatisticsForSc(self, rawSamples, timesteps,
                            simulationNodes, order, solverTimes,
                            work_package_indexes, original_runtime_estimator):

        nodes = simulationNodes.distNodes
        weights = simulationNodes.weights
        dist = simulationNodes.joinedDists
        self.timesteps = timesteps
        P = cp.orth_ttr(order, dist)
        numPercSamples = 10 ** 4
        samples = Samples(rawSamples)
        self.Abfluss = {}
        #fit_quadrature for each time step
        for identification in samples.Abfluss:
            self.Abfluss[identification] = {}
            for i in range(0, len(samples.Abfluss[identification])):
                samples.Abfluss[identification][i] = cp.fit_quadrature(P, nodes, weights, samples.Abfluss[identification][i])
        self.station_names = []
        #calculate relevant statistical values
        for identification in self.Abfluss:
            self.station_names.append(identification)
            self.Abfluss[identification]["E"] = []
            self.Abfluss[identification]["Var"] = []
            self.Abfluss[identification]["StdDev"] = []
            self.Abfluss[identification]["Sobol_m"] = []
            self.Abfluss[identification]["Sobol_m2"] = []
            self.Abfluss[identification]["Sobol_t"] = []
            self.Abfluss[identification]["P10"] = []
            self.Abfluss[identification]["P90"] = []

            for items in samples.Abfluss[identification]:
                self.Abfluss[identification]["E"].append(float((cp.E(items, dist))))
                self.Abfluss[identification]["Var"].append(float(cp.Var(items, dist)))
                self.Abfluss[identification]["StdDev"].append(float(cp.Var(items, dist)))
                self.Abfluss[identification]["Sobol_m"].append(cp.Sens_m(items, dist))
                self.Abfluss[identification]["Sobol_t"].append(cp.Sens_t(items, dist))
                self.Abfluss[identification]["P10"].append(float(cp.Perc(items,  10, dist, numPercSamples)))
                self.Abfluss[identification]["P90"].append(float(cp.Perc(items, 90, dist, numPercSamples)))


            if "Niederschlag" in rawSamples[0][identification]:
                self.Abfluss[identification]["Niederschlag"] = rawSamples[0][identification]["Niederschlag"]


            #save dicts for later use
        if os.stat("./LARSIM_configs/master_files/dicts.npy").st_size == 0:
            np.save("./LARSIM_configs/master_files/dicts.npy", self.Abfluss)
        else:
            dict = np.load("./LARSIM_configs/master_files/dicts.npy").item()
            for identification in self.Abfluss:
                for keys in self.Abfluss[identification]:
                    for i in range(0, len(self.Abfluss[identification][keys])):
                        dict[identification][keys].append(self.Abfluss[identification][keys][i])
            np.save("./LARSIM_configs/master_files/dicts.npy", dict)

#helper function
def power(my_list):
    return [ x**2 for x in my_list ]
