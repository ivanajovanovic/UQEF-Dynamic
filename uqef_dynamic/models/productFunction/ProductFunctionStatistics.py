import numpy as np


import chaospy as cp
import os

from uqef.stat import Statistics

from uqef_dynamic import paths

class Samples(object):
    """
    Samples is a collection of the sampled results of a whole UQ simulation
    """

    def __init__(self, rawSamples):
        self.voi = []

        self.voi = np.array([sample for sample in rawSamples])



class ProductFunctionStatistics(Statistics):


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
        :param timesteps: simulation.solver.timesteps
        :param simulationNodes: simulationNodes
        :param numEvaluations: simulation.numEvaluations
        :param solverTimes: simulation.solver.solverTimes
        :param work_package_indexes: simulation.solver.work_package_indexes
        :param original_runtime_estimator: simulation.original_runtime_estimator
        :return:
        """

        samples = Samples(rawSamples) #rawSamples = self.solver.results what model.run() return
        self.qoi = samples.voi

        print("STATISTICS INFO: Self.qoi:")
        print(self.qoi.shape)
        print(type(self.qoi))
        print(self.qoi) #numpy array nxt, for sartelli it should be n(2+d)xt

        if regression:
            nodes = simulationNodes.distNodes
            dist = simulationNodes.joinedDists
            P = cp.orth_ttr(order, dist)

        self.timesteps = timesteps #this is self.solver.timesteps which are model.timesteps()
        self.numbTimesteps = len(self.timesteps)
        assert self.numbTimesteps == (self.qoi).shape[1]

        print("timesteps Info")
        print(type(self.timesteps))
        print("numTimesteps is: {}".format(self.numbTimesteps))


        # percentiles
        numPercSamples = 10 ** 5


        if regression:
            qoi_gPCE = cp.fit_regression(P, nodes, self.qoi)
            self.E_qoi = cp.E(qoi_gPCE, dist)
            self.Var_qoi = cp.Var(qoi_gPCE, dist)
            self.StdDev_qoi = cp.Std(qoi_gPCE, dist)
            self.Sobol_m_qoi = cp.Sens_m(qoi_gPCE, dist)
            self.Sobol_m2_qoi = cp.Sens_m2(qoi_gPCE, dist)
            self.Sobol_t_qoi = cp.Sens_t(qoi_gPCE, dist)
            self.P10_qoi = cp.Perc(qoi_gPCE, 10, dist, numPercSamples)
            self.P90_qoi = cp.Perc(qoi_gPCE, 90, dist, numPercSamples)
        elif saltelli:
            standard_voi = self.qoi[:numEvaluations, :]
            #self.E_qoi = np.sum(self.qoi, axis=0, dtype=np.float64) / (2*numEvaluations)
            #self.Var_qoi = np.sum( (self.qoi - self.E_qoi) ** 2, axis=0, dtype=np.float64) / (2*numEvaluations-1)
            self.E_qoi = np.sum(standard_voi, axis=0, dtype=np.float64) / numEvaluations
            self.Var_qoi = np.sum((standard_voi - self.E_qoi) ** 2, axis=0, dtype=np.float64) / (numEvaluations - 1)
            self.StdDev_qoi = np.sqrt(self.Var_qoi, dtype=np.float64)
            self.P10_qoi = np.percentile(standard_voi, 10, axis=0)
            self.P90_qoi = np.percentile(standard_voi, 90, axis=0)

            dim = len(simulationNodes.nodeNames)
            self.Sobol_m_qoi = Sens_m_sample_3(self.qoi, dim, numEvaluations)
            self.Sobol_t_qoi = Sens_t_sample_4(self.qoi, dim, numEvaluations)
            print("self.Sobol_m_qoi.shape")
            print(self.Sobol_m_qoi.shape)
            print("self.Sobol_t_qoi.shape")
            print(self.Sobol_t_qoi.shape)

        else:
            self.E_qoi = np.sum(self.qoi, axis=0, dtype=np.float64) / numEvaluations
            #self.Var_qoi = np.sum(self.qoi ** 2, 0) / numEvaluations - self.E_qoi ** 2
            self.Var_qoi = np.sum( (self.qoi - self.E_qoi) ** 2, axis=0, dtype=np.float64) / (numEvaluations-1)
            self.StdDev_qoi = np.sqrt(self.Var_qoi, dtype=np.float64)
            self.P10_qoi = np.percentile(self.qoi, 10, axis=0)
            self.P90_qoi = np.percentile(self.qoi, 90, axis=0)

        if isinstance(self.P10_qoi, (list)) and len(self.P10_qoi) == 1:
            self.P10_qoi = self.P10_qoi[0]
            self.P90_qoi = self.P90_qoi[0]



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

        samples = Samples(rawSamples)  # rawSamples = self.solver.results what model.run() return
        self.qoi = samples.voi
        print(self.qoi)  # numpy array nxt
        print("Shape of the result is:")
        print(self.qoi.shape)

        self.timesteps = timesteps  # this is self.solver.timesteps which are model.timesteps()
        self.numbTimesteps = len(self.timesteps)

        assert self.numbTimesteps == (self.qoi).shape[1]
        print("timesteps Info")
        print(type(self.timesteps))
        print("numTimesteps is: {}".format(self.numbTimesteps))

        P = cp.orth_ttr(order, dist)

        # percentiles
        numPercSamples = 10 ** 5

        if regression:
            qoi_gPCE = cp.fit_regression(P, nodes, self.qoi)
        else:
            qoi_gPCE = cp.fit_quadrature(P, nodes, weights, self.qoi)
        print("Shape of the qoi gPCE cofficients:")
        print(type(qoi_gPCE.shape))
        print(qoi_gPCE.shape)

        self.E_qoi = cp.E(qoi_gPCE, dist)
        self.Var_qoi = cp.Var(qoi_gPCE, dist)
        self.StdDev_qoi = cp.Std(qoi_gPCE, dist)
        self.Sobol_m_qoi = cp.Sens_m(qoi_gPCE, dist)
        self.Sobol_m2_qoi = cp.Sens_m2(qoi_gPCE, dist)
        self.Sobol_t_qoi = cp.Sens_t(qoi_gPCE, dist)


        self.P10_qoi = cp.Perc(qoi_gPCE, 10, dist, numPercSamples)
        self.P90_qoi = cp.Perc(qoi_gPCE, 90, dist, numPercSamples)
        if isinstance(self.P10_qoi, (list)) and len(self.P10_qoi) == 1:
            self.P10_qoi = self.P10_qoi[0]
            self.P90_qoi = self.P90_qoi[0]


    def plotResults(self, simulationNodes, display=False,
                    fileName="", fileNameIdent="", directory="./",
                    fileNameIdentIsFullName=False, safe=True):

        sobol_labels = simulationNodes.nodeNames

        Sobol_m_analytical = np.array([0.165, 0.165, 0.041, 0.041, 0.0103, 0.0103], dtype=np.float32)

        Sobol_t_analytical = np.array([0.583, 0.583, 0.233, 0.233, 0.0685, 0.0685], dtype=np.float32)

        Sj, SjT = _calculate_sj_sjt_formula(self.configurationObject)

        if hasattr(self, "Sobol_t_qoi"):
            for i in range(len(sobol_labels)):
                if self.Sobol_t_qoi.shape[0] == len(self.timesteps):
                    print("Sobol's Total Index for parameter {} is: \n".format(sobol_labels[i]))
                    print("    Sobol Total Simulation = {} \n".format((self.Sobol_t_qoi.T)[i]))
                    print("    Sobol Total Analytical = {} \n".format(Sobol_t_analytical[i]))
                    print("    Sobol Total by Formula = {} \n".format(SjT[i]))
                else:
                    print("Sobol's Total Index for parameter {} is: \n".format(sobol_labels[i]))
                    print("    Sobol Total Simulation = {} \n".format(self.Sobol_t_qoi[i]))
                    print("    Sobol Total Analytical = {} \n".format(Sobol_t_analytical[i]))
                    print("    Sobol Total by Formula = {} \n".format(SjT[i]))

        if hasattr(self, "Sobol_m_qoi"):
            for i in range(len(sobol_labels)):
                if self.Sobol_m_qoi.shape[0] == len(self.timesteps):
                    print("Sobol's Main Index for parameter {} is: \n".format(sobol_labels[i]))
                    print("    Sobol Main Simulation = {} \n".format((self.Sobol_m_qoi.T)[i]))
                    print("    Sobol Main Analytical = {} \n".format(Sobol_m_analytical[i]))
                    print("    Sobol Main by Formula = {} \n".format(Sj[i]))
                else:
                    print("Sobol's Main Index for parameter {} is: \n".format(sobol_labels[i]))
                    print("    Sobol Main Simulation = {} \n".format(self.Sobol_m_qoi[i]))
                    print("    Sobol Main Analytical = {} \n".format(Sobol_m_analytical[i]))
                    print("    Sobol Main by Formula = {} \n".format(Sj[i]))



        if hasattr(self, "Sobol_t_qoi"):
            sobol_t_qoi_file = os.path.abspath(os.path.join(self.working_dir, "sobol_t_qoi_file.npy"))
            np.save(sobol_t_qoi_file, self.Sobol_t_qoi)
        if hasattr(self, "Sobol_m_qoi"):
            sobol_m_qoi_file = os.path.abspath(os.path.join(self.working_dir, "sobol_m_qoi_file.npy"))
            np.save(sobol_m_qoi_file, self.Sobol_m_qoi)



def _calculate_sj_sjt_formula(configurationObject):
    tau1 = configurationObject["Parameters"]["tau1"]
    tau2 = configurationObject["Parameters"]["tau2"]
    tau3 = configurationObject["Parameters"]["tau3"]
    tau4 = configurationObject["Parameters"]["tau4"]
    tau5 = configurationObject["Parameters"]["tau5"]
    tau6 = configurationObject["Parameters"]["tau6"]

    mu1 = configurationObject["Parameters"]["mu1"]
    mu2 = configurationObject["Parameters"]["mu2"]
    mu3 = configurationObject["Parameters"]["mu3"]
    mu4 = configurationObject["Parameters"]["mu4"]
    mu5 = configurationObject["Parameters"]["mu5"]
    mu6 = configurationObject["Parameters"]["mu6"]

    args_tau = np.array([tau1, tau2, tau3, tau4, tau5, tau6], dtype=np.float32)
    args_mau = np.array([mu1, mu2, mu3, mu4, mu5, mu6], dtype=np.float32)

    sum_of_squares = np.square(args_mau) + np.square(args_tau)

    array_prod_par_sum_squares = [np.prod(sum_of_squares[:i]) * np.prod(sum_of_squares[i+1:]) for i in range(sum_of_squares.size)]

    D = np.prod(sum_of_squares) - np.prod(np.square(args_mau))

    Sj = (sum_of_squares - np.prod(args_mau)) / D


    SjT = np.square(args_tau) * array_prod_par_sum_squares / D

    return Sj, SjT

#helper function
def power(my_list):
    return [ x**2 for x in my_list ]


# Functions needed for calculating Sobol's indices using MC samples and Saltelli's method

def separate_output_values_2(Y, D, N):
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

def separate_output_values_j(Y, D, N, j):
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


def Sens_m_sample_1(Y, D, N):
    """
    First order sensitivity indices - Chaospy
    """
    A, B, A_B = separate_output_values_2(Y, D, N)

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

def Sens_m_sample_2(Y, D, N):
    """
    First order sensitivity indices - Homma(1996) & Sobolo (2007)
    """
    A, B, A_B = separate_output_values_2(Y, D, N)

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

def Sens_m_sample_3(Y, D, N):
    """
    First order sensitivity indices - Saltelli 2010.
    """
    A, B, A_B = separate_output_values_2(Y, D, N)

    #variance = np.var(A, -1)
    variance = np.var(A, axis=0)

    out = [
        #np.mean(B*(A_B[j].T-A), -1) /
        np.mean(B*(A_B[j]-A), axis=0) /
        np.where(variance, variance, 1)
        for j in range(D)
        ]

    return np.array(out)

def Sens_m_sample_4(Y, D, N):
    """
    First order sensitivity indices - Jensen.
    """
    A, B, A_B = separate_output_values_2(Y, D, N)

    #variance = np.var(A, -1)
    variance = np.var(A, axis=0)

    out = [
        #1 - np.mean((A_B[j].T-B)**2, -1) /
        1 - np.mean((A_B[j]-B)**2, axis=0) /
        (2*np.where(variance, variance, 1))
        for j in range(D)
        ]

    return np.array(out)



# NOTE! - This does not give proper results
def Sens_t_sample_1(Y, D, N):
    """
    Total order sensitivity indices - Chaospy
    """
    A, B, A_B = separate_output_values_2(Y, D, N)

    #variance = np.var(A, -1)
    variance = np.var(A, axis=0)

    out = [
        #1-np.mean((A_B[j].T-B)**2, -1) /
        1-np.mean((A_B[j]-B)**2, axis=0) /
        (2*np.where(variance, variance, 1))
        for j in range(D)
        ]

    return np.array(out)

def Sens_t_sample_2(Y, D, N):
    """
    Total order sensitivity indices - Homma 96
    """
    A, B, A_B = separate_output_values_2(Y, D, N)

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

def Sens_t_sample_3(Y, D, N):
    """
    Total order sensitivity indices - Sobel 2007
    """
    A, B, A_B = separate_output_values_2(Y, D, N)

    #variance = np.var(A, -1)
    variance = np.var(A, axis=0)

    out = [
        #np.mean(A*(A-A_B[j].T), -1) /
        np.mean(A*(A-A_B[j]),  axis=0) /
        np.where(variance, variance, 1)
        for j in range(D)
        ]

    return np.array(out)


def Sens_t_sample_4(Y, D, N):
    """
    Total order sensitivity indices - Saltelli 2010 & Jensen
    """
    A, B, A_B = separate_output_values_2(Y, D, N)

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

def separate_output_values(Y, D, N):
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

def first_order(A, AB, B):
    # First order estimator following Saltelli et al. 2010 CPC, normalized by
    # sample variance
    #return np.mean(B * (AB - A), axis=0) / np.var(np.r_[A, B], axis=0)
    #return np.mean(B * (AB - A), axis=0, dtype=np.float64) / np.var(np.concatenate((A, B), axis=0), axis=0, ddof=1, dtype=np.float64)
    return np.mean(B * (AB - A), axis=0, dtype=np.float64) / np.var(np.r_[A, B], axis=0, ddof=1, dtype=np.float64)

def total_order(A, AB, B):
    # Total order estimator following Saltelli et al. 2010 CPC, normalized by
    # sample variance
    #return 0.5 * np.mean((A - AB) ** 2, axis=0) / np.var(np.r_[A, B], axis=0)
    #return np.mean(B * (AB - A), axis=0, dtype=np.float64) / np.var(np.concatenate((A, B), axis=0), axis=0, ddof=1, dtype=np.float64)
    return np.mean(B * (AB - A), axis=0, dtype=np.float64) / np.var(np.r_[A, B], axis=0, ddof=1, dtype=np.float64)
