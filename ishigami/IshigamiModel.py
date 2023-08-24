import json
import numpy as np
import time
from math import pow

# from uqef.model import Model

def model(p, x):
    a, b = p
    x1, x2, x3 = x
    f = np.sin(x1) + a * pow(np.sin(x2), 2) + b * pow(x3, 4) * np.sin(x1)
    return f


def model_2d(p, x):
    a, b = p
    x1, x3 = x
    x2 = 0
    # f = np.sin(q[:, 0]) + self.a * (np.sin(q[:, 1]) ** 2) + (self.b * np.sin(q[:, 0]) * (q[:, 2] ** 4))
    f = np.sin(x1) + a * pow(np.sin(x2), 2) + b * pow(x3, 4) * np.sin(x1)
    return f


class IshigamiModelSetUp():
    def __init__(self, configurationObject):
        pass


class IshigamiModel(object):
    def __init__(self, configurationObject,  *args, **kwargs):
        # Model.__init__(self)

        if isinstance(configurationObject, dict):
            self.configurationObject = configurationObject
        else:
            with open(configurationObject) as f:
                self.configurationObject = json.load(f)

        if "a" in kwargs:
            self.a = kwargs['a']
        else:
            try:
                self.a = self.configurationObject["other_model_parameters"]["a"]
            except KeyError:
                self.a = 7

        if "b" in kwargs:
            self.b = kwargs['b']
        else:
            try:
                self.b = self.configurationObject["other_model_parameters"]["b"]
            except KeyError:
                self.b = 0.1

        self.t = [0, ]
        self.t_interest = 0.0

    def prepare(self, *args, **kwargs):
        pass

    def assertParameter(self, parameter):
        pass

    def normaliseParameter(self, parameter):
        return parameter

    def run(self, i_s, parameters, *args, **kwargs):

        # print(f"[Ishigami Model] {i_s}: paramater: {parameters}")

        results = []

        for ip in range(0, len(i_s)):
            start = time.time()
            i = i_s[ip]
            parameter = parameters[ip]

            args = self.a, self.b
            x = parameter[0], parameter[1], parameter[2]
            f_result = model(args, x)

            end = time.time()
            runtime = end - start

            results.append((f_result, runtime))

        return results

    def timesteps(self):
        return self.t

    def get_analytical_sobol_indices(self):
        vm1 = 0.5*(1+(self.b*np.pi**4)/5)**2
        vm1 = (self.b*np.pi**4)/5 + ((self.b**2)*np.pi**8)/50 + 0.5  # Sudret!
        vm2 = self.a**2/8
        vm3 = 0.0
        vm12 = 0.0
        vm23 = 0.0
        vm13 = 8 * self.b**2 * np.pi ** 8 / 225
        # vm13 = 19 * self.b**2 * np.pi ** 8 / 450  # Ravi!
        vm123 = 0.0

        v = self.a**2/8 + (self.b*np.pi**4)/5 + (self.b**2*np.pi**8)/18 + 0.5
        v = vm1 + vm2 + vm13
        assert np.abs(v - (vm1 + vm2 + vm13)) < 0.001

        sm1 = vm1/v
        sm2 = vm2/v
        sm3 = 0.0  # vm3/v

        st1 = (vm1 + vm13)/v
        st2 = vm2/v
        st3 = vm13/v

        # Sobol_m_analytical = np.array([0.3138/0.3139, 0.4424/0.4424, 0.0/0.0000], dtype=np.float64)
        sobol_m_analytical = np.array([sm1, sm2, sm3], dtype=np.float64)

        # Sobol_t_analytical = np.array([0.5574/0.5576, 0.4424/0.4424, 0.2436/0.2437], dtype=np.float64)
        sobol_t_analytical = np.array([st1, st2, st3], dtype=np.float64)

        return sobol_m_analytical, sobol_t_analytical


# Ravi's code
# class Ishigami():
#
#     def __init__(self, a, b, lower=-np.pi, upper=np.pi):
#         self.lower, self.upper, self.dim = lower, upper, 3
#         self.a, self.b = a, b
#         self.num_eval_lf, self.num_eval_hf = 0, 0
#
#     def transform_coordinates(self, x):
#         return x * (self.upper - self.lower) + self.lower
#
#     def hf(self, x): # this is the main Ishigami function
#         temp = np.atleast_2d(x)
#         if temp.shape[1] != self.dim:
#             temp = temp.T
#         q = self.transform_coordinates(temp)
#         f = np.sin(q[:, 0]) + self.a * (np.sin(q[:, 1])**2) + (self.b * np.sin(q[:, 0]) * (q[:, 2]**4))
#         self.num_eval_hf += len(temp)
#         return f
#
#     def lf(self, x):
#     	# This is just a dummy function to test multi-fidelity Ishigami toy problems :D
#         temp = np.atleast_2d(x)
#         if temp.shape[1] != self.dim:
#             temp = temp.T
#         q = self.transform_coordinates(temp)
#         a, b = self.a +0.1, self.b
#         f =(np.sin(q[:, 0]) + a * (np.sin(q[:, 1])**2) + (b * np.sin(q[:, 0]) * (q[:, 2]**4)) ) + 0.02
#         self.num_eval_lf += len(temp)
#         return f
#
#     def calculate_statistics(self):
#         mean = self.a * 0.5
#         D1 = (self.b * np.pi**4 / 5) + (self.b**2 * np.pi**8 / 50) + 0.5
#         D2 = self.a**2 / 8
#         D13 = 19 * self.b**2 * np.pi**8 / 450
#         D = D1 + D2 + D13
#         assert np.abs(D - (D1 + D2 + D13)) < 0.001
#         local_sobol = [D1/D, D2/D, 0.]
#         global_sobol = [(D1+D13)/D, D2/D, D13/D]
#         return mean, D, local_sobol, global_sobol