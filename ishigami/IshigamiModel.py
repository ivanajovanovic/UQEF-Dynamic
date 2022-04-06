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
        v = self.a**2/8 + (self.b*np.pi**4)/5 + (self.b**2*np.pi**8)/18 + 0.5
        vm1 = 0.5*(1+(self.b*np.pi**4)/5)**2
        vm1 = (self.b*np.pi**4)/5 + ((self.b**2)*np.pi**8)/50 + 0.5  # Sudret
        vm2 = self.a**2/8
        vm3 = 0.0
        vm12 = 0.0
        vm23 = 0.0
        vm13 = 8 * self.b ** 2 * np.pi ** 8 / 225
        vm123 = 0.0

        sm1 = vm1/v
        sm2 = vm2/v
        sm3 = vm3/v

        vt1 = vm1 + vm13
        vt2 = vm2
        vt3 = vm13

        st1 = vt1/v
        st2 = vt2/v
        st3 = vt3/v

        # Sobol_m_analytical = np.array([0.3138/0.3139, 0.4424/0.4424, 0.0/0.0000], dtype=np.float64)
        sobol_m_analytical = np.array([sm1, sm2, sm3], dtype=np.float64)

        # Sobol_t_analytical = np.array([0.5574/0.5576, 0.4424/0.4424, 0.2436/0.2437], dtype=np.float64)
        sobol_t_analytical = np.array([st1, st2, st3], dtype=np.float64)

        return sobol_m_analytical, sobol_t_analytical