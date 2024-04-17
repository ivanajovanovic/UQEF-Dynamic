import numpy as np
import matplotlib.pyplot as plotter
import pathlib
import pickle

import chaospy as cp
import os

from uqef.stat import Statistics

import paths
from common import saltelliSobolIndicesHelpingFunctions

class Samples(object):
    """
    Samples is a collection of the sampled results of a whole UQ simulation
    """

    def __init__(self, rawSamples):
        self.qoi = []

        self.qoi = np.array([sample for sample in rawSamples])


class IshigamiStatistics(Statistics):

    def __init__(self, configurationObject, workingDir=None, *args, **kwargs):
        Statistics.__init__(self)

        self.configurationObject = configurationObject

        self.workingDir = pathlib.Path(workingDir)
        if self.workingDir is None:
            self.workingDir = pathlib.Path(paths.workingDir)

        self.a = self.configurationObject["other_model_parameters"]["a"]
        self.b = self.configurationObject["other_model_parameters"]["b"]

        self.sampleFromStandardDist = kwargs.get('sampleFromStandardDist', False)

        self.uq_method = kwargs.get('uq_method', None)
        self._compute_Sobol_t = kwargs.get('compute_Sobol_t', True)
        self._compute_Sobol_m = kwargs.get('compute_Sobol_m', True)
        self._compute_Sobol_m2 = kwargs.get('compute_Sobol_m2', False)

        self.nodeNames = []
        try:
            list_of_parameters = self.configurationObject["parameters"]
        except KeyError as e:
            print(f"Larsim Statistics: parameters key does "
                  f"not exists in the configurationObject{e}")
            raise
        for i in list_of_parameters:
            if self.uq_method == "ensemble" or i["distribution"] != "None":
                self.nodeNames.append(i["name"])
        self.dim = len(self.nodeNames)
        self.labels = [nodeName.strip() for nodeName in self.nodeNames]

        self._is_Sobol_t_computed = False
        self._is_Sobol_m_computed = False
        self._is_Sobol_m2_computed = False

        self.sobol_m_analytical = None
        self.sobol_t_analytical = None

        self.numbTimesteps = None
        self.timesteps = None
        self.number_of_unique_index_runs = None
        self.numEvaluations = None
        self.samples = None
        # self.result_dict = None

        self.solverTimes = None
        self.work_package_indexes = None

        self.result_dict = dict()

    def get_analytical_sobol_indices(self):
        # vm1 = 0.5*(1+(self.b*np.pi**4)/5)**2
        vm1 = (self.b*np.pi**4)/5 + ((self.b**2)*np.pi**8)/50 + 0.5  # Sudret!
        vm2 = self.a**2/8
        vm3 = 0.0
        vm12 = 0.0
        vm23 = 0.0
        vm13 = 8 * self.b**2 * np.pi ** 8 / 225
        # vm13 = 19 * self.b**2 * np.pi ** 8 / 450  # Ravi!
        vm123 = 0.0

        # v = self.a**2/8 + (self.b*np.pi**4)/5 + (self.b**2*np.pi**8)/18 + 0.5
        v = vm1 + vm2 + vm13
        assert np.abs(v - (vm1 + vm2 + vm13)) < 0.001

        sm1 = vm1/v
        sm2 = vm2/v
        sm3 = 0.0

        # vt1 = 0.5*(1+(self.b*np.pi**4)/5)**2 + 8*self.b**2*np.pi**8/225
        # vt2 = self.a**2/8
        # vt3 = 8*self.b**2*np.pi**8/225
        st1 = (vm1 + vm13)/v
        st2 = vm2/v
        st3 = vm13/v

        # Sobol_m_analytical = np.array([0.3138/0.3139, 0.4424/0.4424, 0.0/0.0000], dtype=np.float64)
        self.sobol_m_analytical = np.array([sm1, sm2, sm3], dtype=np.float64)

        # Sobol_t_analytical = np.array([0.5574/0.5576, 0.4424/0.4424, 0.2436/0.2437], dtype=np.float64)
        self.sobol_t_analytical = np.array([st1, st2, st3], dtype=np.float64)

    def calcStatisticsForMc(self, rawSamples, timesteps, simulationNodes,
                            numEvaluations, order, regression, poly_normed, poly_rule, solverTimes,
                            work_package_indexes, original_runtime_estimator=None, *args, **kwargs):

        samples = Samples(rawSamples)
        self.qoi = samples.qoi

        print("STATISTICS INFO: Self.qoi:")
        print(self.qoi.shape)
        print(type(self.qoi))
        print(self.qoi)  # numpy array nxt

        if regression:
            nodes = simulationNodes.distNodes
            if self.sampleFromStandardDist:
                dist = simulationNodes.joinedStandardDists
            else:
                dist = simulationNodes.joinedDists
                # P = cp.orth_ttr(order, dist)
            polynomial_expansion = cp.generate_expansion(order, dist, rule=poly_rule, normed=poly_normed)

        self.timesteps = timesteps
        self.numbTimesteps = len(self.timesteps)
        print("STATISTICS INFO: timesteps Info")
        print(type(self.timesteps))
        print("numTimesteps is: {}".format(self.numbTimesteps))

        if regression:
            self.qoi_gPCE = cp.fit_regression(polynomial_expansion, nodes, self.qoi)
            self._calc_stats_for_gPCE(dist)
        else:
            # self.E_qoi = np.sum(self.qoi, axis=0, dtype=np.float64) / numEvaluations
            self.E_qoi = np.mean(self.qoi, 0)
            #self.Var_qoi = np.sum(self.qoi ** 2, 0) / numEvaluations - self.E_qoi ** 2  # UQQEF-Test model
            self.Var_qoi = np.sum((self.qoi - self.E_qoi) ** 2, axis=0, dtype=np.float64)/(numEvaluations-1)
            # self.StdDev_qoi = np.sqrt(self.Var_qoi, dtype=np.float64)
            self.StdDev_qoi = np.std(self.qoi, 0, ddof=1)
            self.P10_qoi = np.percentile(self.qoi, 10, axis=0)
            self.P90_qoi = np.percentile(self.qoi, 90, axis=0)

        if isinstance(self.P10_qoi, list) and len(self.P10_qoi) == 1:
            self.P10_qoi = self.P10_qoi[0]
            self.P90_qoi = self.P90_qoi[0]

        self._write_statistics_to_result_dict()
        print(f"STATISTICS INFO: calcStatisticsForMC function is done!")

    def calcStatisticsForMcSaltelli(self, rawSamples, timesteps,
                            simulationNodes, numEvaluations, order, regression, poly_normed, poly_rule, solverTimes,
                            work_package_indexes, original_runtime_estimator=None, *args, **kwargs):
        
        samples = Samples(rawSamples)
        self.qoi = samples.qoi
        
        print("STATISTICS INFO: Self.qoi:")
        print(self.qoi.shape)
        print(type(self.qoi))
        print(self.qoi)  # numpy array n(2+d)xt
        
        self.timesteps = timesteps
        self.numbTimesteps = len(self.timesteps)
        print("STATISTICS INFO: timesteps Info")
        print(type(self.timesteps))
        print("numTimesteps is: {}".format(self.numbTimesteps))

        qoi_values_saltelli = self.qoi[:, np.newaxis]
        print(f"qoi_values_saltelli.shape = {qoi_values_saltelli.shape}")
        standard_qoi = qoi_values_saltelli[:numEvaluations, :]
        extended_standard_qoi = qoi_values_saltelli[:(2*numEvaluations), :]

        #self.E_qoi = np.sum(self.qoi, axis=0, dtype=np.float64) / (2*numEvaluations)
        # self.E_qoi = np.sum(standard_qoi, axis=0, dtype=np.float64) / numEvaluations
        self.E_qoi = np.mean(self.qoi[:(numEvaluations)], 0)

        #self.Var_qoi = np.sum( (self.qoi - self.E_qoi) ** 2, axis=0, dtype=np.float64) / (2*numEvaluations-1)
        # self.Var_qoi = np.sum((standard_qoi - self.E_qoi) ** 2, axis=0, dtype=np.float64) / (numEvaluations - 1)
        self.Var_qoi = np.sum((self.qoi[:(numEvaluations)] - self.E_qoi) ** 2, axis=0, dtype=np.float64)/(numEvaluations - 1)

        # self.StdDev_qoi = np.sqrt(self.qoi, dtype=np.float64)
        self.StdDev_qoi = np.std(self.qoi[:(numEvaluations)], 0, ddof=1)

        self.P10_qoi = np.percentile(self.qoi[:(numEvaluations)], 10, axis=0)
        self.P90_qoi = np.percentile(self.qoi[:(numEvaluations)], 90, axis=0)

        if isinstance(self.P10_qoi, list) and len(self.P10_qoi) == 1:
            self.P10_qoi = self.P10_qoi[0]
            self.P90_qoi = self.P90_qoi[0]

        dim = len(simulationNodes.nodeNames)
        # dim = len(simulationNodes.distNodes[0])

        if self._compute_Sobol_m:
            self.Sobol_m_qoi = saltelliSobolIndicesHelpingFunctions._Sens_m_sample(
                qoi_values_saltelli, dim, numEvaluations, code=4)

        if self._compute_Sobol_t:
            self.Sobol_t_qoi = saltelliSobolIndicesHelpingFunctions._Sens_t_sample(
                qoi_values_saltelli, dim, numEvaluations, code=4)

        print("self.Sobol_m_qoi.shape")
        print(self.Sobol_m_qoi.shape)
        print("self.Sobol_t_qoi.shape")
        print(self.Sobol_t_qoi.shape)

        self._write_statistics_to_result_dict()
        print(f"STATISTICS INFO: calcStatisticsForSallteli function is done!")

    def calcStatisticsForSc(self, rawSamples, timesteps,
                            simulationNodes, order, regression, poly_normed, poly_rule, solverTimes,
                            work_package_indexes, original_runtime_estimator=None,  *args, **kwargs):
        samples = Samples(rawSamples)
        self.qoi = samples.qoi

        print("STATISTICS INFO: Self.qoi:")
        print(self.qoi.shape)
        print(type(self.qoi))
        print(self.qoi)  # numpy array nxt

        self.timesteps = timesteps
        self.numbTimesteps = len(self.timesteps)
        print("STATISTICS INFO: timesteps Info")
        print(type(self.timesteps))
        print("numTimesteps is: {}".format(self.numbTimesteps))

        nodes = simulationNodes.distNodes
        weights = simulationNodes.weights
        if self.sampleFromStandardDist:
            dist = simulationNodes.joinedStandardDists
        else:
            dist = simulationNodes.joinedDists
        # P = cp.orth_ttr(order, dist)
        polynomial_expansion = cp.generate_expansion(order, dist, rule=poly_rule, normed=poly_normed)

        print("STATISTICS INFO: polynomial_expansion")
        print(polynomial_expansion)

        if regression:
            self.qoi_gPCE = cp.fit_regression(polynomial_expansion, nodes, self.qoi)
        else:
            self.qoi_gPCE = cp.fit_quadrature(polynomial_expansion, nodes, weights, self.qoi)
        self._calc_stats_for_gPCE(dist)
        self._write_statistics_to_result_dict()

        print(f"STATISTICS INFO: calcStatisticsForSc function is done!")

    def _calc_stats_for_gPCE(self, dist, qoi_gPCE=None):
        if qoi_gPCE is None:
            qoi_gPCE = self.qoi_gPCE

        # percentiles
        numPercSamples = 10 ** 5

        self.E_qoi = float(cp.E(qoi_gPCE, dist))
        self.Var_qoi = float(cp.Var(qoi_gPCE, dist))
        self.StdDev_qoi = float(cp.Std(qoi_gPCE, dist))

        if self._compute_Sobol_t:
            self.Sobol_t_qoi = cp.Sens_t(qoi_gPCE, dist)
        if self._compute_Sobol_m:
            self.Sobol_m_qoi = cp.Sens_m(qoi_gPCE, dist)
        if self._compute_Sobol_m2:
            self.Sobol_m2_qoi = cp.Sens_m2(qoi_gPCE, dist)

        self.Skew = cp.Skew(qoi_gPCE, dist).round(4)
        self.Kurt = cp.Kurt(qoi_gPCE, dist)
        self.QoI_Dist = cp.QoI_Dist(qoi_gPCE, dist)

        self.P10_qoi = float(cp.Perc(qoi_gPCE, 10, dist, numPercSamples))
        self.P90_qoi = float(cp.Perc(qoi_gPCE, 90, dist, numPercSamples))
        if isinstance(self.P10_qoi, (list)) and len(self.P10_qoi) == 1:
            self.P10_qoi = self.P10_qoi[0]
            self.P90_qoi = self.P90_qoi[0]

        gpceFileName = os.path.abspath(os.path.join(str(self.workingDir), "gpce.pkl"))
        with open(gpceFileName, 'wb') as handle:
            pickle.dump(qoi_gPCE, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def _write_statistics_to_result_dict(self):
        if hasattr(self, "qoi_gPCE"):
            self.result_dict["gPCE"] = self.qoi_gPCE
        if hasattr(self, "E_qoi"):
            self.result_dict["E"] = self.E_qoi
        if hasattr(self, "Var_qoi"):
            self.result_dict["Var"] = self.Var_qoi
        if hasattr(self, "StdDev_qoi"):
            self.result_dict["StdDev"] = self.StdDev_qoi

        if hasattr(self, "Skew"):
            self.result_dict["Skew"] = self.Skew
        if hasattr(self, "Kurt"):
            self.result_dict["Kurt"] = self.Kurt
        if hasattr(self, "QoI_Dist"):
            self.result_dict["qoi_dist"] = self.QoI_Dist

        if hasattr(self, "P10_qoi"):
            self.result_dict["qoi_dist"] = self.P10_qoi
        if hasattr(self, "P10"):
            self.result_dict["P90"] = self.P90_qoi

        if hasattr(self, "Sobol_m_qoi"):
            self.result_dict["Sobol_m"] = self.Sobol_m_qoi
        if hasattr(self, "Sobol_m2_qoi"):
            self.result_dict["Sobol_m2"] = self.Sobol_m2_qoi
        if hasattr(self, "Sobol_t_qoi"):
            self.result_dict["Sobol_t"] = self.Sobol_t_qoi

    def _check_if_Sobol_t_computed(self):
        if hasattr(self, "Sobol_t_qoi"):
            self._is_Sobol_t_computed = True
        else:
            self._is_Sobol_t_computed = False

    def _check_if_Sobol_m_computed(self):
        if hasattr(self, "Sobol_m_qoi"):
            self._is_Sobol_m_computed = True
        else:
            self._is_Sobol_m_computed = False

    def _check_if_Sobol_m2_computed(self):
        if hasattr(self, "Sobol_m2_qoi"):
            self._is_Sobol_m2_computed = True
        else:
            self._is_Sobol_m2_computed = False

    def _save_qoi(self):
        qoi_file = os.path.abspath(os.path.join(str(self.workingDir), "qoi_file.npy"))
        np.save(qoi_file, self.qoi)

    def plotResults(self, timestep=-1, display=False,
                    fileName="", fileNameIdent="", directory="./",
                    fileNameIdentIsFullName=False, safe=True, ** kwargs):

        self._check_if_Sobol_t_computed()
        self._check_if_Sobol_m_computed()
        self._check_if_Sobol_m2_computed()

        print("STATISTICS INFO: E, Var, Std")
        print(f"E_qoi: {self.E_qoi}")
        print(f"Var_qoi: {self.Var_qoi}")
        print(f"StdDev_qoi: {self.StdDev_qoi}")

        print("STATISTICS INFO: Sobol' Indices")
        if self._is_Sobol_t_computed:
            print(f"Sobol_t_qoi: {self.Sobol_t_qoi}")
        if self._is_Sobol_m_computed:
            print(f"Sobol_m_qoi: {self.Sobol_m_qoi}")
        if self._is_Sobol_m2_computed:
            print(f"Sobol_m2_qoi: {self.Sobol_m2_qoi}")

        v = self.a**2/8 + (self.b*np.pi**4)/5 + (self.b**2*np.pi**8)/18 + 0.5
        # vm1 = 0.5*(1+(self.b*np.pi**4)/5)**2
        vm1 = (self.b*np.pi**4)/5 + ((self.b**2)*np.pi**8)/50 + 0.5  # Sudret!
        vm2 = self.a**2/8
        vm3 = 0
        vm13 = 8 * self.b**2 * np.pi ** 8 / 225
        v = vm1 + vm2 + vm13
        assert np.abs(v - (vm1 + vm2 + vm13)) < 0.001

        sm1 = vm1/v
        sm2 = vm2/v
        sm3 = 0.0

        # vt1 = 0.5*(1+(self.b*np.pi**4)/5)**2 + 8*self.b**2*np.pi**8/225
        # vt2 = vm2
        # vt3 = 8*self.b**2*np.pi**8/225
        st1 = (vm1 + vm13)/v
        st2 = vm2/v
        st3 = vm13/v

        # Sobol_m_analytical = np.array([0.3138/0.3139, 0.4424/0.4424, 0.0/0.0000], dtype=np.float64)
        Sobol_m_analytical = np.array([sm1, sm2, sm3], dtype=np.float64)

        # Sobol_t_analytical = np.array([0.5574/0.5576, 0.4424/0.4424, 0.2436/0.2437], dtype=np.float64)
        Sobol_t_analytical = np.array([st1, st2, st3], dtype=np.float64)

        Sobol_t_error = np.empty(len(self.labels), dtype=np.float64)
        Sobol_m_error = np.empty(len(self.labels), dtype=np.float64)

        if self._is_Sobol_t_computed:
            for i in range(len(self.labels)):
                if self.Sobol_t_qoi.shape[0] == len(self.timesteps):
                    # print(f"Sobol's Total Index for parameter {self.labels[i]} is: \n")
                    # print(f"Sobol Total Simulation = {(self.Sobol_t_qoi.T)[i]} \n")
                    # print(f"Sobol Total Analytical = {Sobol_t_analytical[i]} \n")
                    Sobol_t_error[i] = (self.Sobol_t_qoi.T)[i] - Sobol_t_analytical[i]
                else:
                    # print(f"Sobol's Total Index for parameter {self.labels[i]} is: \n")
                    # print(f"Sobol Total Simulation = {self.Sobol_t_qoi[i]:.4f} \n")
                    # print(f"Sobol Total Analytical = {Sobol_t_analytical[i]:.4f} \n")
                    Sobol_t_error[i] = self.Sobol_t_qoi[i] - Sobol_t_analytical[i]
        if self._is_Sobol_m_computed:
            for i in range(len(self.labels)):
                if self.Sobol_m_qoi.shape[0] == len(self.timesteps):
                    # print(f"Sobol's Main Index for parameter {self.labels[i]} is: \n")
                    # print(f"Sobol Main Simulation = {(self.Sobol_m_qoi.T)[i]:.4f} \n")
                    # print(f"Sobol Main Analytical = {Sobol_m_analytical[i]:.4f} \n")
                    Sobol_m_error[i] = (self.Sobol_m_qoi.T)[i] - Sobol_m_analytical[i]
                else:
                    # print(f"Sobol's Main Index for parameter {self.labels[i]} is: \n")
                    # print(f"Sobol Main Simulation = {self.Sobol_m_qoi[i]:.4f} \n")
                    # print(f"Sobol Main Analytical = {Sobol_m_analytical[i]:.4f} \n")
                    Sobol_m_error[i] = self.Sobol_m_qoi[i] - Sobol_m_analytical[i]

        print("STATISTICS INFO: Sobol' Indices Error")
        if self._is_Sobol_t_computed:
            print("Sobol_t_error: {}".format(Sobol_t_error, ".6f"))
        if self._is_Sobol_m_computed:
            print("Sobol_m_error: {}".format(Sobol_m_error, ".6f"))

        if self._is_Sobol_t_computed:
            sobol_t_qoi_file = os.path.abspath(os.path.join(str(self.workingDir), "sobol_t_qoi_file.npy"))
            np.save(sobol_t_qoi_file, self.Sobol_t_qoi)

        if self._is_Sobol_m_computed:
            sobol_m_qoi_file = os.path.abspath(os.path.join(str(self.workingDir), "sobol_m_qoi_file.npy"))
            np.save(sobol_m_qoi_file, self.Sobol_m_qoi)
