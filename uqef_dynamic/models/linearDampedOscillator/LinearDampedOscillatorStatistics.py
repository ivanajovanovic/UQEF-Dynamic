import numpy as np
import matplotlib.pyplot as plotter
import os

import chaospy as cp

from uqef.stat import Statistics

from uqef_dynamic import paths

class Samples(object):
    """
    Samples is a collection of the sampled results of a whole UQ simulation
    """

    def __init__(self, rawSamples):
        self.voi = []

        #for sample in rawSamples:
        #    self.qoi.append(sample[0])

        #self.qoi = np.array(self.qoi)

        self.voi = np.array([sample for sample in rawSamples])



class LinearDampedOscillatorStatistics(Statistics):


    def __init__(self):
        Statistics.__init__(self)
        #self.type_of_saltetlli = type_of_saltetlli


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
            self.Sobol_m_qoi = Sens_m_sample_2(self.qoi, dim, numEvaluations)
            self.Sobol_t_qoi = Sens_t_sample_4(self.qoi, dim, numEvaluations)
            print("self.Sobol_m_qoi.shape")
            print(self.Sobol_m_qoi.shape)
            print("self.Sobol_t_qoi.shape")
            print(self.Sobol_t_qoi.shape)

            #for samplesOneTimestep in self.qoi.T:
            #    A, B, AB = separate_output_values(samplesOneTimestep, dim, numEvaluations)
            #    si_first_orded_array = first_order(A, AB, B)
            #    si_total_orded_array = total_order(A, AB, B)

            #    Sobol_m_qoi.append(si_first_orded_array)
            #    Sobol_t_qoi.append(si_total_orded_array)

            #self.Sobol_m_qoi = np.array([sample for sample in Sobol_m_qoi])
            #self.Sobol_t_qoi = np.array([sample for sample in Sobol_t_qoi])


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
        #print("Sobol_m_qoi shape: ")
        #print(self.Sobol_m_qoi.shape)
        #print("Sobol_t_qoi shape: ")
        #print(self.Sobol_t_qoi.shape)

        # needs to be numpy array for plotting
        #self.qoi = np.array([sample for sample in rawSamples])
        #self.qoi = np.array([sample for sample in rawSamples])

        self.P10_qoi = cp.Perc(qoi_gPCE, 10, dist, numPercSamples)
        self.P90_qoi = cp.Perc(qoi_gPCE, 90, dist, numPercSamples)
        if isinstance(self.P10_qoi, (list)) and len(self.P10_qoi) == 1:
            self.P10_qoi = self.P10_qoi[0]
            self.P90_qoi = self.P90_qoi[0]


        #self.E_qoi = []
        #self.Var_qoi = []
        #self.StdDev_qoi = []
        #self.Sobol_m_qoi = []
        #self.Sobol_m2_qoi = []
        #self.Sobol_t_qoi = []
        #self.P10_qoi = []
        #self.P90_qoi = []

        #for samplesOneTimestep in self.qoi.T:
        #    if regression:
        #        qoi_gPCE = cp.fit_regression(P, nodes, samplesOneTimestep)
        #    else:
        #        qoi_gPCE = cp.fit_quadrature(P, nodes, weights, samplesOneTimestep)
        #        print("Shape of the qoi gPCE cofficients:")
        #        print(type(qoi_gPCE.shape))
        #        print(qoi_gPCE.shape)

        #    self.E_qoi.append(float((cp.E(qoi_gPCE, dist))))
        #    self.Var_qoi.append(float((cp.Var(qoi_gPCE, dist))))
        #    self.StdDev_qoi.append(float((cp.Std(qoi_gPCE, dist))))
        #    self.Sobol_m_qoi.append(cp.Sens_m(qoi_gPCE, dist))
        #    self.Sobol_m2_qoi.append(cp.Sens_m2(qoi_gPCE, dist))
        #    self.Sobol_t_qoi.append(cp.Sens_t(qoi_gPCE, dist))

        #    P10_qoi_temp = float(cp.Perc(qoi_gPCE, 10, dist, numPercSamples))
        #    P90_qoi_temp = float(cp.Perc(qoi_gPCE, 90, dist, numPercSamples))
        #    if isinstance(P10_qoi_temp, (list)) and len(P10_qoi_temp) == 1:
        #        P10_qoi_temp = P10_qoi_temp[0]
        #        P90_qoi_temp = P90_qoi_temp[0]
        #    self.P10_qoi.append(P10_qoi_temp)
        #    self.P90_qoi.append(P90_qoi_temp)




    def plotResults(self, simulationNodes, display=False,
                    fileName="", fileNameIdent="", directory="./",
                    fileNameIdentIsFullName=False, safe=True):


        #####################################
        ### plot: mean + percentiles
        #####################################

        figure = plotter.figure(1, figsize=(13, 10))
        window_title = 'Linear Damped Oscillator statistics'
        figure.canvas.set_window_title(window_title)


        plotter.subplot(411)
        # plotter.title('mean')
        plotter.plot(self.timesteps, self.E_qoi, 'o', label='mean')
        plotter.fill_between(self.timesteps, self.P10_qoi, self.P90_qoi, facecolor='#5dcec6')
        plotter.plot(self.timesteps, self.P10_qoi, 'o', label='10th percentile')
        plotter.plot(self.timesteps, self.P90_qoi, 'o', label='90th percentile')
        plotter.xlabel('time', fontsize=13)
        plotter.ylabel('Oscillator Vaule', fontsize=13)
        #plotter.xlim(0, 200)
        #ymin, ymax = plotter.ylim()
        #plotter.ylim(0, 20)
        plotter.xticks(rotation=45)
        plotter.legend()  # enable the legend
        plotter.grid(True)

        plotter.subplot(412)
        # plotter.title('standard deviation')
        plotter.plot(self.timesteps, self.StdDev_qoi, 'o', label='std. dev.')
        plotter.xlabel('time', fontsize=13)
        plotter.ylabel('Standard Deviation ', fontsize=13)
        #plotter.xlim(0, 200)
        #plotter.ylim(0, 20)
        plotter.xticks(rotation=45)
        plotter.legend()  # enable the legend
        plotter.grid(True)

        if hasattr(self, "Sobol_t_qoi"):
            plotter.subplot(413)
            #sobol_labels = ["EQB", "BSF", "TGr"]
            sobol_labels = simulationNodes.nodeNames
            for i in range(len(sobol_labels)):
                if self.Sobol_m_qoi.shape[0] == len(self.timesteps):
                    plotter.plot(self.timesteps, (self.Sobol_t_qoi.T)[i], 'o', label=sobol_labels[i])
                else:
                    plotter.plot(self.timesteps, (self.Sobol_t_qoi)[i], 'o', label=sobol_labels[i])
            plotter.xlabel('time', fontsize=13)
            plotter.ylabel('total sobol indices', fontsize=13)
            ##plotter.xlim(0, 200)
            # ##plotter.ylim(-0.1, 1.1)
            plotter.xticks(rotation=45)
            plotter.legend()  # enable the legend
            # plotter.grid(True)

        if hasattr(self, "Sobol_m_qoi"):
            plotter.subplot(414)
            #sobol_labels = ["EQB", "BSF", "TGr"]
            sobol_labels = simulationNodes.nodeNames
            for i in range(len(sobol_labels)):
                if self.Sobol_m_qoi.shape[0] == len(self.timesteps):
                    plotter.plot(self.timesteps, (self.Sobol_m_qoi.T)[i], 'o', label=sobol_labels[i])
                else:
                    plotter.plot(self.timesteps, (self.Sobol_m_qoi)[i], 'o', label=sobol_labels[i])
            plotter.xlabel('time', fontsize=13)
            plotter.ylabel('first order sobol indices', fontsize=13)
            ##plotter.xlim(0, 200)
            # ##plotter.ylim(-0.1, 1.1)
            plotter.xticks(rotation=45)
            plotter.legend()  # enable the legend
            # plotter.grid(True)

        if hasattr(self, "Sobol_t_qoi"):
            sobol_t_qoi_file = os.path.abspath(os.path.join(paths.current_dir, "sobol_t_qoi_file.npy"))
            np.save(sobol_t_qoi_file, self.Sobol_t_qoi)
        if hasattr(self, "Sobol_m_qoi"):
            sobol_m_qoi_file = os.path.abspath(os.path.join(paths.current_dir, "sobol_m_qoi_file.npy"))
            np.save(sobol_m_qoi_file, self.Sobol_m_qoi)

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
