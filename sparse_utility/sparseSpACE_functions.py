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
# import uqef

import os
import sys
# cwd = pathlib.Path(os.getcwd())
# sys.path.insert(0, cwd.parent.absolute())

from ishigami import IshigamiModel

# from larsim import LarsimModelUQ
from LarsimUtilityFunctions import larsimDataPostProcessing
from LarsimUtilityFunctions import larsimModel

# from hbv_sask import HBVSASKModelUQ
from hbv_sask import HBVSASKModel
from hbv_sask import hbvsask_utility

class Function_Info(object):
    def __init__(self, function, dim, a, b, distributions, distributions_for_sparSpace, path_Error,
                 mean_analytical=None, variance_analytical=None,
                 first_order_sobol_indices=None, total_order_sobol_indices=None,
                 ppfs=None, function_unitCube=None):
        """
        ppf - point percentile function - inverse cumulative distribution function
        """
        self.function = function
        self.dim = dim
        self.a = a
        self.b = b
        self.distributions = distributions
        self.distributions_for_sparSpace = distributions_for_sparSpace
        self.joint_distributions = cp.J(*distributions)
        self.path_Error = path_Error
        self.mean_analytical = mean_analytical
        self.variance_analytical = variance_analytical
        self.first_order_sobol_indices = first_order_sobol_indices
        self.total_order_sobol_indices = total_order_sobol_indices
        self.ppfs = ppfs
        self.function_unitCube = function_unitCube


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

    # @staticmethod
    def get_analytical_sobol_indices(self, **kwargs):
        sobol_m_analytical, sobol_t_analytical = self.ishigamiModelObject.get_analytical_sobol_indices()
        return sobol_m_analytical, sobol_t_analytical


class LarsimFunction(Function):
    def __init__(self, configurationObject, inputModelDir, workingDir, param_names=None, qoi="Q", gof="calculateNSE"):
        super().__init__()

        self.qoi = qoi
        self.gof = gof

        self.larsimModelObject = larsimModel.LarsimModel(
            configurationObject=configurationObject,
            inputModelDir=inputModelDir,
            workingDir=workingDir
        )

        self.larsimModelObject.prepare(infoModel=True)

        self.param_names = []
        for i in self.larsimModelObject.larsimConfObject.configurationObject["parameters"]:
            self.param_names.append((i["type"], i["name"]))
        self.dim = len(self.param_names)

        output_length = self.output_length()
        if output_length > 1:
            self.vector_output = True
        else:
            self.vector_output = False

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

        results_array = self.larsimModelObject.run(
            parameters=[params, ],
            i_s=[self.global_eval_counter, ],
            take_direct_value=True,
            createNewFolder=True,
            deleteFilesAfterwards=True,
            deleteFolderAfterwards=True
        )

        if self.qoi == "Q":
            df = larsimDataPostProcessing.filterResultForStationAndTypeOfOutpu(
                resultsDataFrame=results_array[0][0]["result_time_series"],
                station=self.larsimModelObject.larsimConfObject.station_of_Interest,
                type_of_output=self.larsimModelObject.larsimConfObject.type_of_output_of_Interest
            )
            return np.array(df['Value'])
            # TODO take just last time-step
        elif self.qoi == "GoF":
            if self.gof in results_array[0][0]['gof_df'].columns:
                if self.gof == "calculateRMSE":  # TODO change this - hard-coded for now...
                    temp = results_array[0][0]['gof_df'][self.gof].values
                    temp = 1000 - temp
                    return np.array(temp)
                else:
                    return np.array(results_array[0][0]['gof_df'][self.gof].values)
            else:
                return None
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

        if self.param_names is None:
            self.param_names = []
            for i in self.hbvsaskModelObject.self.configurationObject["parameters"]:
                self.param_names.append(i["name"])
            self.dim = len(self.param_names)

        output_length = self.output_length()
        if output_length > 1:
            self.vector_output = True
        else:
            self.vector_output = False

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
            if self.gof in results_array[0][0]['gof_df'].columns:
                # return np.array(results_array[0][0]['gof_df'][self.gof].values)
                return results_array[0][0]['gof_df'][self.gof].values[0]
            else:
                return None
        else:
            raise Exception(f"Not implemented")


#################
# Set of functions from SparseSpACE
#################
# Markus functions -
# corner_peak-GenzCornerPeak, product_peak-GenzProductPeak, discontinuous-GenzDiscontinious,
# gfunction, ishigami, functionUQ-functionUQ3D

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
        parameter1_list = coordinates[:, 0]
        parameter2_list = coordinates[:, 0]
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

# discontinuous
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


class FunctionScaledUnitCube(Function):
    def __init__(self, function, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.function = function

    def eval(self, coordinates):
        result = self.function.eval([self.a[d] + (self.b[d] - self.a[d]) * coordinates[d] for d in range(len(self.a))])
        return result


class FunctionNonlinearScaledUnitCube(Function):
    def __init__(self, function, a=None, b=None, distribution_q=None, distribution_r=None):
        super().__init__()
        self.a = a
        self.b = b
        self.function = function
        if distribution_q is None:
            if a is None or b is None:
                raise Exception()
            dists = []
            for d in range(len(self.a)):
                dists.append(cp.Uniform(lower=self.a[d], upper=self.b[d]))
            self.distribution_q = cp.J(*dists)
        else:
            self.distribution_q = distribution_q
        if distribution_r is None:
            dists = []
            for d in range(len(self.a)):
                dists.append(cp.Uniform(lower=0, upper=1))
            self.distribution_r = cp.J(*dists)
        else:
            self.distribution_r = distribution_r

    def eval(self, coordinates):
        coordinates_transformed = FunctionNonlinearScaledUnitCube.transformSamples(
            coordinates, self.distribution_q, self.distribution_r)
        result = self.function.eval(coordinates_transformed)
        return result

    def eval_vectorized(self, coordinates):
        coordinates_transformed = FunctionNonlinearScaledUnitCube.transformSamples(
            coordinates, self.distribution_q, self.distribution_r)
        result = self.function.eval_vectorized(coordinates_transformed)
        return result

    @staticmethod
    def transformSamples(samples, distribution_q, distribution_r=None):
        if distribution_r is None:
            # default is Uniform[0,1]^d
            return distribution_q.inv(samples)
        else:
            return distribution_q.inv(distribution_r.fwd(samples))


def initiate_function_infos():
    dict_function_infos = {}

    configurationObject = pathlib.Path("/work/ga45met/mnt/linux_cluster_2/UQEFPP/configurations/configuration_ishigami.json")
    a = np.array([-math.pi] * 3)
    b = np.array([math.pi] * 3)
    function_object = IshigamiFunction(configurationObject, dim=3)
    mean = 3.5
    variance = 13.844587940719254
    distributions = [cp.Uniform(-math.pi, math.pi) for _ in range(3)]
    distributionsForSparseSpace = [("Uniform", -math.pi, math.pi) for _ in range(3)]
    function_info_ishigami = Function_Info(
        function=function_object, dim=3, a=a, b=b,
        distributions=distributions,
        distributions_for_sparSpace=distributionsForSparseSpace,
        path_Error='Ishigami_Variance_errors_to_plot.pickle',
        mean_analytical=mean,
        variance_analytical=variance,
        first_order_sobol_indices=[0.31390519114781146, 0.4424111447900409, 0.0],
        ppfs=[(lambda x: 2 * math.pi * x - math.pi) for _ in range(3)],
        function_unitCube=FunctionScaledUnitCube(function=function_object, a=a, b=b)
    )
    dict_function_infos['ishigami'] = function_info_ishigami

    a = np.array([0] * 3)
    b = np.array([1] * 3)
    function_object = GFunction(dim=3)
    mean = function_object.get_expectation()
    variance = function_object.get_variance()
    distributions = [cp.Uniform(0, 1) for _ in range(3)]
    distributionsForSparseSpace = [("Uniform", 0, 1) for _ in range(3)]
    function_info_gfunction = Function_Info(
        function=function_object, dim=3, a=a, b=b,
        distributions=distributions,
        distributions_for_sparSpace=distributionsForSparseSpace,
        path_Error='errors_to_plot_FunctionG.pickle',
        mean_analytical=mean,
        variance_analytical=variance,
        first_order_sobol_indices=[0.506250, 0.225033, 0.1265625],
        ppfs=[(lambda x: x) for _ in range(3)],
        function_unitCube=function_object
    )
    dict_function_infos['gfunction'] = function_info_gfunction

    a = np.array([-2.5, -2, 5])
    b = np.array([2.5, 2, 15])
    function_object = FunctionUQ3D()
    mean = function_object.get_expectation(a, b)
    variance = function_object.get_variance(a, b)
    distributions = [cp.Uniform(-2.5, 2.5), cp.Uniform(-2, 2), cp.Uniform(5, 15)]
    distributionsForSparseSpace = [("Uniform", -2.5, 2.5), ("Uniform", -2, 2), ("Uniform", 5, 15)]
    function_info_functionUQ = Function_Info(
        function=function_object, dim=3, a=a, b=b,
        distributions=distributions,
        distributions_for_sparSpace=distributionsForSparseSpace,
        path_Error='errors_to_plot_FunctionUQ3D.pickle',
        mean_analytical=mean,
        variance_analytical=variance,
        first_order_sobol_indices=[0.6606445, 37.0142899, 6.21831285],
        ppfs=[(lambda x: -2.5 + 5*x), lambda x: -2 + 4*x, lambda x: 5 + 10*x],
        function_unitCube=FunctionScaledUnitCube(function=function_object, a=a, b=b)
    )
    dict_function_infos['functionUQ'] = function_info_functionUQ

    #################
    # genzFunctions
    #################

    coeffs = [float(1) for _ in range(3)]
    a = np.array([0]*3)
    b = np.array([1]*3)
    function_object = GenzCornerPeak(coeffs)
    mean = function_object.get_expectation()
    variance = function_object.get_variance()
    distributions = [cp.Uniform(0, 1) for _ in range(3)]
    distributionsForSparseSpace = [("Uniform", 0, 1) for _ in range(3)]
    function_info_corner_peak = Function_Info(
        function=function_object, dim=3, a=a, b=b,
        distributions=distributions,
        distributions_for_sparSpace=distributionsForSparseSpace,
        path_Error='errors_to_plot_CornerPeak_.pickle',
        mean_analytical=mean,
        variance_analytical=variance,
        ppfs=[(lambda x: x) for _ in range(3)],
        function_unitCube=function_object
    )
    dict_function_infos['corner_peak'] = function_info_corner_peak

    coeffs = [float(3) for _ in range(3)]
    midpoint = [0.5 for _ in range(3)]
    a = np.array([0]*3)
    b = np.array([1]*3)
    function_object = GenzProductPeak(coeffs, midpoint)
    mean = function_object.get_expectation()
    variance = function_object.get_variance()
    distributions = [cp.Uniform(0, 1) for _ in range(3)]
    distributionsForSparseSpace = [("Uniform", 0, 1) for _ in range(3)]
    function_info_product_peak = Function_Info(
        function=function_object, dim=3, a=a, b=b,
        distributions=distributions,
        distributions_for_sparSpace=distributionsForSparseSpace,
        path_Error='errors_to_plot_CornerProduct_.pickle',
        mean_analytical=mean,
        variance_analytical=variance,
        ppfs=[(lambda x: x) for _ in range(3)],
        function_unitCube=function_object
    )
    dict_function_infos['product_peak'] = function_info_product_peak

    coeffs = [float(1) for _ in range(3)]
    midpoint = [0.5 for _ in range(3)]
    a = np.array([0]*3)
    b = np.array([1]*3)
    function_object = GenzDiscontinious(coeffs, midpoint)
    mean = function_object.get_expectation()
    variance = function_object.get_variance()
    distributions = [cp.Uniform(0, 1) for _ in range(3)]
    distributionsForSparseSpace = [("Uniform", 0, 1) for _ in range(3)]
    function_info_discontinuous = Function_Info(
        function=function_object, dim=3, a=a, b=b,
        distributions=distributions,
        distributions_for_sparSpace=distributionsForSparseSpace,
        path_Error='errors_to_plot_Discontinuous_.pickle',
        mean_analytical=mean,
        variance_analytical=variance,
        ppfs=[(lambda x: x) for _ in range(3)],
        function_unitCube=function_object
    )
    dict_function_infos['discontinuous'] = function_info_discontinuous

    return dict_function_infos