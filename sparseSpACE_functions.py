import json
import itertools
import math
import numpy as np
import numpoly
import pathlib
import pickle
import scipy
import scipy.integrate as integrate
import time

from sparseSpACE.Function import *
from sparseSpACE.spatiallyAdaptiveSingleDimension2 import *
from sparseSpACE.ErrorCalculator import *
from sparseSpACE.GridOperation import *
from sparseSpACE.StandardCombi import *
from sparseSpACE.Integrator import *

import chaospy as cp
import uqef

from ishigami import IshigamiModel

# from larsim import LarsimModel
from LarsimUtilityFunctions import larsimDataPostProcessing
from LarsimUtilityFunctions import larsimModel

from hbv_sask import HBVSASKModel

class IshigamiFunction(Function):
    def __init__(self, configurationObject, dim):
        super().__init__()

        self.ishigamiModelObject = IshigamiModel.IshigamiModel(configurationObject=configurationObject)
        self.global_eval_counter = 0
        self.dim = dim

    def output_length(self):
        return 1

    def eval(self, coordinates):
        # assert (len(coordinates) == self.dim), len(coordinates)
        self.global_eval_counter += 1
        result_tuple = self.ishigamiModelObject.run(
            i_s=[self.global_eval_counter, ],
            parameters=[coordinates, ]
        )
        value_of_interest = result_tuple[0][0]
        return value_of_interest  # np.array(value_of_interest)

    @staticmethod
    def _get_analytical_sobol_indices(a_model_param=7, b_model_param=0.1):
        v = a_model_param ** 2 / 8 + (b_model_param * np.pi ** 4) / 5 + (b_model_param ** 2 * np.pi ** 8) / 18 + 0.5
        vm1 = 0.5 * (1 + (b_model_param * np.pi ** 4) / 5) ** 2
        vm2 = a_model_param ** 2 / 8
        vm3 = 0
        sm1 = vm1 / v
        sm2 = vm2 / v
        sm3 = vm3 / v

        vt1 = 0.5 * (1 + (b_model_param * np.pi ** 4) / 5) ** 2 + 8 * b_model_param ** 2 * np.pi ** 8 / 225
        vt2 = a_model_param ** 2 / 8
        vt3 = 8 * b_model_param ** 2 * np.pi ** 8 / 225
        st1 = vt1 / v
        st2 = vt2 / v
        st3 = vt3 / v

        # Sobol_m_analytical = np.array([0.3138/0.3139, 0.4424/0.4424, 0.0/0.0000], dtype=np.float64)
        sobol_m_analytical = np.array([sm1, sm2, sm3], dtype=np.float64)

        # Sobol_t_analytical = np.array([0.5574/0.5576, 0.4424/0.4424, 0.2436/0.2437], dtype=np.float64)
        sobol_t_analytical = np.array([st1, st2, st3], dtype=np.float64)

        return sobol_m_analytical, sobol_t_analytical


class LarsimFunction(Function):
    def __init__(self, config_file, inputModelDir, outputModelDir, param_names=None, qoi="Q", gof="calculateNSE"):
        super().__init__()
        self.larsimModelObject = larsimModel.LarsimModel(
            configurationObject=config_file,
            inputModelDir=inputModelDir,
            workingDir=outputModelDir
        )

        self.larsimModelObject.prepare(infoModel=True)

        self.qoi = qoi
        self.gof = gof

        self.param_names = []
        for i in self.larsimModelObject.larsimConfObject.configurationObject["parameters"]:
            self.param_names.append((i["type"], i["name"]))

        self.global_eval_counter = 0

    def output_length(self):
        if self.qoi == "Q":
            return len(self.larsimModelObject.larsimConfObject.t) - self.larsimModelObject.larsimConfObject.warm_up_duration
        else:
            return 1

    #     def getAnalyticSolutionIntegral(self, start, end): assert "not implemented"

    def eval(self, coordinates):
        self.global_eval_counter += 1
        params = {param_name: coord for coord, param_name in zip(coordinates, self.param_names)}

        larsim_res = self.larsimModelObject.run(
            parameters=[params, ],
            i_s=[self.global_eval_counter, ],
            take_direct_value=True,
            createNewFolder=True,
            deleteFilesAfterwards=True,
            deleteFolderAfterwards=True
        )

        if self.qoi == "Q":
            df = larsimDataPostProcessing.filterResultForStationAndTypeOfOutpu(
                resultsDataFrame=larsim_res[0][0]["result_time_series"],
                station=self.larsimModelObject.larsimConfObject.station_of_Interest,
                type_of_output=self.larsimModelObject.larsimConfObject.type_of_output_of_Interest
            )
            return np.array(df['Value'])
            # TODO take just last time-step
        elif self.qoi == "GoF":
            if self.gof == "calculateRMSE":  # TODO change this - hard-coded for now...
                temp = larsim_res[0][0]['gof_df'][self.gof].values
                temp = 1000 - temp
                return np.array(temp)
            else:
                return np.array(larsim_res[0][0]['gof_df'][self.gof].values)
        else:
            raise Exception(f"Not implemented")


class HBVSASKFunction(Function):
    def __init__(self, configurationObject, inputModelDir, workingDir, dim=None,
                 param_names=None, qoi="Q", gof="calculateNSE", **kwargs):
        super().__init__()

        self.dim = dim
        self.param_names = param_names
        self.qoi = qoi
        self.gof = gof

        self.writing_results_to_a_file = kwargs.get("writing_results_to_a_file", False)
        self.plotting = kwargs.get("plotting", False)

        self.hbvsaskModelObject = HBVSASKModel.HBVSASKModel(
            configurationObject=configurationObject,
            inputModelDir=inputModelDir,
            workingDir=workingDir,
            writing_results_to_a_file=self.writing_results_to_a_file,
            plotting=self.plotting
        )

        self.global_eval_counter = 0

    def output_length(self):
        if self.qoi == "Q":
            return len(list(self.hbvsaskModelObject.simulation_range))
        else:
            return 1

    def eval(self, coordinates):
        self.global_eval_counter += 1
        params = {param_name: coord for coord, param_name in zip(coordinates, self.param_names)}

        results_array = self.hbvsaskModelObject.run(
            parameters=[params, ],
            i_s=[self.global_eval_counter, ],
            take_direct_value=True
        )

        if self.qoi == "Q":
            return np.array(results_array[0][0]['result_time_series']['Q_cms'].values)
        elif self.qoi == "GoF":
            # TODO Still missing computation of qoi
            return None
        else:
            raise Exception(f"Not implemented")


#################
# Set of functions from SparseSpACE
#################

# g-function of Sobol: https://www.sfu.ca/~ssurjano/gfunc.html
class GFunction(Function):
    def __init__(self, dim=2, **kwargs):
        super().__init__()
        self.dim = dim
        # self.a = 0.5 * np.array(range(dim))
        # self.a = np.array([0.5 * (i - 1) for i in range(dim)])
        self.a = np.array([0.5 * i for i in range(dim)])

    def eval(self, coordinates):
        assert len(coordinates) == self.dim
        return np.prod([(abs(4.0 * coordinates[d] - 2.0) + self.a[d]) / (1.0 + self.a[d]) for d in range(self.dim)])

    def eval_vectorized(self, coordinates: Sequence[Sequence[float]]):
        value_of_interest = np.zeros(np.shape(coordinates)[:-1])  #TODO ?
        i = 0
        for coordinate in coordinates:
            value_of_interest[i] = np.prod([(abs(4.0 * coordinate[d] - 2.0) + self.a[d]) / (1.0 + self.a[d]) for d in range(self.dim)])
            i += 1
        return value_of_interest

    # Uniform distributions in [0, 1] are required for this Function.
    def get_expectation(self):
        return 1.0

    def get_variance(self):
        mom2 = np.prod([1.0 + 1.0 / (3.0 * (1.0 + a_d) ** 2) for a_d in self.a])
        return mom2 - 1.0

    # ~ def get_first_order_sobol_indices(self):
        # This seems to be wrong
        # ~ fac = 1.0 / np.prod([1.0 / (3.0 * (1.0 + a_d) ** 2) for a_d in self.a])
        # ~ return [fac * 1.0 / (3.0 * (1.0 + self.a[d]) ** 2) for d in range(self.dim)]

        # ~ ivar = 1.0 / self.get_variance()
        # ~ return [ivar * (1.0 + 1.0 / (3 * (1.0 + self.a[i]) ** 2)) for i in range(self.dim)]

    def getAnalyticSolutionIntegral(self, start, end):
        assert all([v == 0.0 for v in start])
        assert all([v == 1.0 for v in end])
        return self.get_expectation()


class FunctionUQ2D(Function):
    def __init__(self, **kwargs):
        super().__init__()

    def eval(self, coordinates):
        # print(coordinates)
        assert (len(coordinates) == 2)
        parameter1 = coordinates[0]
        parameter2 = coordinates[1]

        # Model with discontinuity
        # Nicholas Zabarras Paper: „Sparse grid collocation schemes for stochastic natural convection problems“
        # e^(-x^2 + 2*sign(y))
        value_of_interest = math.exp(-parameter1 ** 2 + 2 * np.sign(parameter2))
        return value_of_interest

    def eval_vectorized(self, coordinates: Sequence[Sequence[float]]):
        parameter1_list = coordinates[:,0]
        parameter2_list = coordinates[:,0]
        value_of_interest = np.empty_like(parameter1_list)
        i = 0
        for parameter1, parameter2 in zip(parameter1_list, parameter2_list):
            value_of_interest[i] = math.exp(-parameter1 ** 2 + 2 * np.sign(parameter2))
            i += 1
        return value_of_interest

    def getAnalyticSolutionIntegral(self, start, end):
        f = lambda x, y: self.eval([x, y])
        return integrate.dblquad(f, start[1], end[1], lambda x: start[0],
                                 lambda x: end[0])[0]


class FunctionUQ3D(Function):
    def __init__(self, **kwargs):
        super().__init__()

    def eval(self, coordinates):
        assert (len(coordinates) == 3), len(coordinates)
        parameter1 = coordinates[0]
        parameter2 = coordinates[1]
        parameter3 = coordinates[2]

        # Model with discontinuity
        # Nicholas Zabarras Paper: „Sparse grid collocation schemes for stochastic natural convection problems“
        # e^(-x^2 + 2*sign(y))
        value_of_interest = math.exp(-parameter1 ** 2 + 2 * np.sign(parameter2)) + parameter3
        return value_of_interest

    def getAnalyticSolutionIntegral(self, start, end):
        f = lambda x, y, z: self.eval([x, y, z])
        return integrate.tplquad(f, start[2], end[2], lambda x: start[1], lambda x: end[1], lambda x, y: start[0],
                                 lambda x, y: end[0])[0]


class GenzDiscontinious(Function):
    def __init__(self, coeffs, border, **kwargs):
        super().__init__()
        self.coeffs = coeffs
        self.border = border
        self.dim = len(coeffs)

    def eval(self, coordinates):
        result = 0
        for d in range(self.dim):
            if coordinates[d] >= self.border[d]:
                return 0.0
            result -= self.coeffs[d] * coordinates[d]
        return np.exp(result)

    def eval_vectorized(self, coordinates: Sequence[Sequence[float]]):
        result = np.zeros(np.shape(coordinates)[:-1])
        filter = np.all(coordinates < self.border, axis=-1)
        result[filter] = np.exp(-1 * np.inner(coordinates[filter], self.coeffs))
        self.check_vectorization(coordinates, result)
        return result

    def getAnalyticSolutionIntegral(self, start, end):
        result = 1
        end = list(end)
        for d in range(self.dim):
            if start[d] >= self.border[d]:
                return 0.0
            else:
                end[d] = min(end[d], self.border[d])
                result *= (np.exp(-self.coeffs[d] * start[d]) - np.exp(-self.coeffs[d] * end[d])) / self.coeffs[d]
        return result