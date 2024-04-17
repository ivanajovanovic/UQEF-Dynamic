from uqef.model import Model

import numpy as np
import time
import math
#from math import pow

def g_fun(x):
    return math.sqrt(12.0)*(x - 0.5)

def g_fun_array(x_array):
    return np.array([math.sqrt(12.0)*(x - 0.5) for x in x_array], dtype=np.float32)

def model(tau_p, mu_p, x):
    f = float(np.prod(mu_p + tau_p*g_fun_array(x)))
    return [f,]

class ProductFunctionModelSetUp():
    def __init__(self, configurationObject):
        pass

class ProductFunctionModel(Model):
    def __init__(self, configurationObject):
        Model.__init__(self)

        self.configurationObject = configurationObject

        self.tau1 = self.configurationObject["Parameters"]["tau1"]
        self.tau2 = self.configurationObject["Parameters"]["tau2"]
        self.tau3 = self.configurationObject["Parameters"]["tau3"]
        self.tau4 = self.configurationObject["Parameters"]["tau4"]
        self.tau5 = self.configurationObject["Parameters"]["tau5"]
        self.tau6 = self.configurationObject["Parameters"]["tau6"]

        self.mu1 = self.configurationObject["Parameters"]["mu1"]
        self.mu2 = self.configurationObject["Parameters"]["mu2"]
        self.mu3 = self.configurationObject["Parameters"]["mu3"]
        self.mu4 = self.configurationObject["Parameters"]["mu4"]
        self.mu5 = self.configurationObject["Parameters"]["mu5"]
        self.mu6 = self.configurationObject["Parameters"]["mu6"]

        self.t = [0,]
        self.t_interest = 0.0


    def prepare(self):
        pass

    def assertParameter(self, parameter):
        pass

    def normaliseParameter(self, parameter):
        return parameter

    def run(self, i_s, parameters):

        print("{}: paramater: {}".format(i_s, parameters))

        results = []

        for ip in range(0, len(i_s)):
            start = time.time()
            i = i_s[ip]
            parameter = parameters[ip]

            args_tau = np.array([self.tau1, self.tau2, self.tau3, self.tau4, self.tau5, self.tau6], dtype=np.float32)
            args_mau = np.array([self.mu1, self.mu2, self.mu3, self.mu4, self.mu5, self.mu6],dtype=np.float32)
            self.x = np.array([parameter[0], parameter[1], parameter[2], parameter[3], parameter[4], parameter[5]], dtype=np.float32)
            f_result = model(args_tau, args_mau, self.x)

            end = time.time()
            runtime = end - start

            results.append((f_result, runtime))

        #return [value_of_interest]
        return results

    def timesteps(self):
        return self.t
