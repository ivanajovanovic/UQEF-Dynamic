from uqef.model import Model

import numpy as np
import time
from math import pow


def model(p, x):
    a, b 		= p
    x1, x2, x3 	= x
    f = np.sin(x1) + a * pow(np.sin(x2),2) + b * pow(x3,4) * np.sin(x1)
    return [f,]

class IshigamiModelSetUp():
    def __init__(self, configurationObject):
        pass

class IshigamiModel(Model):
    def __init__(self, configurationObject):
        Model.__init__(self)

        self.configurationObject = configurationObject

        self.a = self.configurationObject["Parameters"]["a"]
        self.b = self.configurationObject["Parameters"]["b"]

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

            args = self.a, self.b
            self.x = parameter[0], parameter[1], parameter[2]
            f_result = model(args, self.x)

            end = time.time()
            runtime = end - start

            results.append((f_result, runtime))

        #return [value_of_interest]
        return results

    def timesteps(self):
        return self.t
