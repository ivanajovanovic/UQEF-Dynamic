from uqef.model import Model

import numpy as np
import time
from scipy.integrate import odeint


def model(w, t, p):
    x1, x2 		= w
    c, k, f, w 	= p
    f = [x2, f*np.cos(w*t) - k*x1 - c*x2]
    return f


def discretize_oscillator_odeint(model, atol, rtol, init_cond, args, t, t_interest):
    sol = odeint(model, init_cond, t, args=(args,), atol=atol, rtol=rtol)
    #return sol[t_interest]
    return sol[:, 0]

class LinearDampedOscillatorModelSetUp():
    def __init__(self, configurationObject):
        pass

class LinearDampedOscillatorModel(Model):
    def __init__(self, configurationObject):
        Model.__init__(self)

        self.configurationObject = configurationObject

        self.t_max = self.configurationObject["Parameters"]["t_max"]#20.
        self.dt = self.configurationObject["Parameters"]["dt"]#0.01

        grid_size = int(self.t_max / self.dt) + 1
        self.t = [i * self.dt for i in range(grid_size)]
        self.t_interest = int(len(self.t) // 2)

        self.c = self.configurationObject["Parameters"]["c"] #0.5 #u(0.08,0.12)
        self.k = self.configurationObject["Parameters"]["k"] #2.0 #u(0.03,0.04)
        self.f = self.configurationObject["Parameters"]["f"] #0.5 #u(0.08,0.12)
        self.y0 = self.configurationObject["Parameters"]["y0"] #0.5 #u(0.45,0.55)
        self.y1 = self.configurationObject["Parameters"]["y1"] #0. #u(-0.05,0.05)
        self.w = self.configurationObject["Parameters"]["w"] #1.0

        self.atol = 1e-10
        self.rtol = 1e-10

        #self.init_cond = self.y0, self.y1


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

            #args = parameter[0][0], self.k, self.f, parameter[0][1] #self.c, self.k, self.f, self.w
            #self.init_cond = self.y0, self.y1
            #value_of_interest = discretize_oscillator_odeint(model, self.atol, self.rtol, self.init_cond, args, self.t, self.t_interest)

            args = self.c, self.k, self.f, parameter[0] # self.c, self.k, self.f, self.w
            self.init_cond = self.y0, parameter[1] #parameter[1] # self.y0 self.y1
            value_of_interest = discretize_oscillator_odeint(model, self.atol, self.rtol, self.init_cond, args, self.t, self.t_interest)

            end = time.time()
            runtime = end - start

            results.append((value_of_interest, runtime))

        #return [value_of_interest]
        return results

    def timesteps(self):
        #return range(1, 2)
        return self.t
