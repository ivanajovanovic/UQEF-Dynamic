import math
import pickle
import pathlib
import json
import numpy as np
import sparseSpACE
import numpoly
import chaospy
from sparseSpACE.spatiallyAdaptiveSingleDimension2 import *
from sparseSpACE.Function import *
from sparseSpACE.GridOperation import *
from sparseSpACE.spatiallyAdaptiveSingleDimension2 import *
from sparseSpACE.ErrorCalculator import *
from sparseSpACE.GridOperation import *
from matplotlib import pyplot
import importlib.machinery
HBVSASKModelSparseSpACE = importlib.machinery.SourceFileLoader('HBVSASKModelSparseSpACE', '/home/markus/studium/BA/thesis_ME/Bachelorthesis_Markus_Englberger/hbv')
import sys
sys.path.append('/home/markus/studium/BA/thesis_ME/Bachelorthesis_Markus_Englberger/hbv')
from HBVSASKModelSparseSpACE import HBVSASKFunction

#used methods and abbreviations
namesL = ['PSP using Gaussian quadrature',
             'PSP with surrogate: Grid = Standard Combi, not adaptive',
             'PSP with surrogate: Grid = Trapezoidal, adaptive',
             'PSP with surrogate, Grid = Leja, not adaptive',
             'PSP with surrogate, using Bsplines with degree 13, adaptive',
             'analytical Integration with Surrogate: Grid = Standard Combi, not adaptive',
             'analytical Integration with Surrogate: Grid = Trapezoidal, adaptive',
             'weighted adaptive trapezoidal grid, only mean and variance',
             'PSP using sparse Leja quadrature',
             'PSP using Gaussian quadrature, including number of previous grids',
             'PSP, using adaptive CB quadrature for every inner product']
names = ['A1', 'B1', 'B2', 'B3', 'B4', 'C1', 'C2', 'D', 'A2', 'A1, including points of previous grids', 'A3']

#store relevant infos about a model
class Function_Info(object):
    def __init__(self, function, dim, a, b, distributions, distributions_for_sparSpace, path_Error, mean_analytical, variance_analytical, first_order_sobol_indices=None, ppfs=None, function_unitCube=None):
        self.function = function
        self.dim = dim
        self.a = a
        self.b = b
        self.distributions = distributions
        self.distributions_for_sparSpace = distributions_for_sparSpace
        self.joint_distributions = chaospy.J(*distributions)
        self.path_Error = path_Error
        self.mean_analytical = mean_analytical
        self.variance_analytical = variance_analytical
        if first_order_sobol_indices is None:
            self.first_order_sobol_indices = [0 for _ in range(self.dim)]
        else:
            self.first_order_sobol_indices = first_order_sobol_indices
        if ppfs is None:
            self.ppfs = []
            for d in range(dim):
                m = b[d] - a[d]
                ppff = lambda x: (b[d]-a[d])*x + a[d]
                self.ppfs.append(ppff)
        else:
            self.ppfs = ppfs
        if function_unitCube is None:
            self.function_unitCube = UnitcubeUniform(function, a, b)

#transform a model with uniform input distributions in the unitcube
class UnitcubeUniform(Function):
    def __init__(self, function, a, b):
        super().__init__()
        self.function = function
        self.a = a
        self.b = b
    def eval(self, coordinates):
        coordinatesUnitcube = [(self.b[d] - self.a[d])*coordinates[d] + self.a[d] for d in range(len(self.a))]
        return self.function.eval(coordinatesUnitcube)
    def output_length(self):
        return self.function.output_length()

#test function
class testFunction(Function):
    def eval(self, coordinates):
        return 20+np.sum(coordinates)**3

#ishigami function
def ishigami(coordinates):
    x1, x2, x3 = coordinates
    return math.sin(x1) + 7 * (math.sin(x2)) ** 2 + 0.1 * x3 ** 4 * math.sin(x1)
class modelSolverFUNC(Function):
    def __init__(self):
        super().__init__()
    def output_length(self) -> int:
        return 1
    def eval(self, coordinates):
        return ishigami(coordinates)

#store result for results
def storeResult(entry, index, function_info):
    pickle_errors_to_plot_in = open(function_info.path_Error, "rb")
    errors_to_plot = pickle.load(pickle_errors_to_plot_in)
    errors_to_plot[index].append(entry)
    pickle_errors_to_plot_out = open(function_info.path_Error, "wb")
    pickle.dump(errors_to_plot, pickle_errors_to_plot_out)
   # print("errors to plot for index", index, ': ', errors_to_plot[index])
    pickle_errors_to_plot_out.close()

#delete results for some method
def delete_entries(index, function_info):
    pickle_errors_to_plot_in = open(function_info.path_Error, "rb")
    errors_to_plot = pickle.load(pickle_errors_to_plot_in)
    errors_to_plot[index] = []
    pickle_errors_to_plot_out = open(function_info.path_Error, "wb")
    pickle.dump(errors_to_plot, pickle_errors_to_plot_out)
    pickle_errors_to_plot_out.close()#

#calculate mean, variance, sobol indices from gPCE coefficients
def calculate_MeanVarianceSobol(gPCE_coefficients, polynomial_degrees, dim, totalSobol = False):
    vari = 0
    for i in range(1, len(gPCE_coefficients)):
        vari += gPCE_coefficients[i] ** 2
   # print('vari: ', vari)
    mean = gPCE_coefficients[0]
   # print('mean: ', mean)
    indices = numpoly.glexindex(start=0, stop=polynomial_degrees + 1, dimensions=dim,
                                graded=True, reverse=True,
                                cross_truncation=1.0)
    first_order_sobol = [0 for _ in range(dim)]
    for d in range(dim):
        for i, ind in enumerate(indices):
            correct = True
            if ind[d] == 0:
                correct = False
            for d_other in range(dim):
                if d_other != d and ind[d_other] != 0:
                    correct = False
            if correct:
                first_order_sobol[d] += gPCE_coefficients[i]**2 / vari

    total_order_sobol = [0 for _ in range(dim)]
    for d in range(dim):
        for i, ind in enumerate(indices):
            if not ind[d] == 0:
                total_order_sobol[d] += gPCE_coefficients[i] ** 2 / vari
    print("expected: ", mean, ", variance: ", vari, ", first order sobol indices: \n", first_order_sobol,
          ", total order sobol indices: ", total_order_sobol)
    if totalSobol:
        return (mean, vari, total_order_sobol)
    else:
        return (mean, vari, first_order_sobol)


#perform PSP directly on the model with chaospy
def chaospy_directly(gridname, order, function_info, store_result=False, count_previous_grids=False, time_series=False):
    number_points = 0
    if count_previous_grids: #only for Gauss Grid
        for order_previous in range(1, order):
            nodesPr, weightsPr = chaospy.generate_quadrature(order_previous, function_info.joint_distributions,
                                                             rule='gaussian')
            number_points += len(nodesPr.T)

    if gridname == 'gauss':
        quads = chaospy.generate_quadrature(order, function_info.joint_distributions, rule='gaussian')
    elif gridname == 'leja_direct':
        quads = chaospy.generate_quadrature(2*order+1, function_info.joint_distributions, rule="leja", sparse=True, growth=False)
        #quads = chaospy.generate_quadrature(2 * order, function_info.joint_distributions, rule="clenshaw_curtis", sparse=True,
        #                                    growth=False)
    nodes, weights = quads
    if gridname == 'leja_direct':
        number_points = len(nodes.T)
        print("number points as in nodes: ", len(nodes.T))
    else:
        number_points += len(nodes.T)
    gauss_evals = np.array([function_info.function.eval(node) for node in nodes.T])
    print("number of points", number_points)
    expansion = chaospy.generate_expansion(order, function_info.joint_distributions, normed=True)
    gauss_model_approx, gPCE_coefficients = chaospy.fit_quadrature(expansion, nodes, weights, gauss_evals, retall=True)
    expected, variance, sobol_indices = calculate_MeanVarianceSobol(gPCE_coefficients, order, function_info.dim)
    print("Expected: ", expected, ",  Variance: ", variance)
    if time_series:
        plot_Times_series(function_info, variance, expected, sobol_indices)
        std = chaospy.Std(gauss_model_approx, function_info.joint_distributions)
        coordinates = np.arange(0, 366, step=1)
        pyplot.rc("figure", figsize=[6, 4])
        pyplot.xlabel("coordinates")
        pyplot.ylabel("model approximation")
        pyplot.fill_between(
            coordinates, expected - 2 * std, expected + 2 * std, alpha=0.3)
        pyplot.plot(coordinates, expected)
        pyplot.show()
        return
    entry = (number_points, float(expected), float(variance), *sobol_indices)
    if store_result:
        if count_previous_grids:
            storeResult(entry, 9, function_info)
        elif gridname == 'gauss':
            storeResult(entry, 0, function_info)
        elif gridname == 'leja_direct':
            storeResult(entry, 8, function_info)

#use weighted trapezoidal rule to compute mean and variance, no gPCE
def only_mean_and_variance(max_evals, function_info, store_result=False):
    distributions = function_info.distributions_for_sparSpace
    op = UncertaintyQuantification(function_info.function, distributions, function_info.a, function_info.b)
    grid = GlobalTrapezoidalGridWeighted(function_info.a, function_info.b, op, modified_basis=True, boundary=False)
    op.set_grid(grid)
    op.set_expectation_variance_Function()
    combiObject = SpatiallyAdaptiveSingleDimensions2(function_info.a, function_info.b, margin=0.8, operation=op, norm=2, grid_surplusses=grid)
    lmax = 3
    error_operator = ErrorCalculatorSingleDimVolumeGuided()
    combiObject.performSpatiallyAdaptiv(1, lmax,
                                          error_operator, tol=0, max_evaluations=max_evals, do_plot=False)
    (E,), (Var,) = op.calculate_expectation_and_variance(combiObject)
    print(f"E: {E}, Var: {Var}")
    if store_result:
        number_points = combiObject.get_total_num_points()
        entry = (number_points, float(E), float(Var), 0, 0, 0)
        storeResult(entry, 7, function_info)

#perform PSP directly on the model with weighted adaptive trapezoidal rule
def Pseudo_Spectral_adative_CB_for_PCEcoefficients(max_evals, function_info, store_result=False, total=False, polynomial_degree_max = 10, boundary=True):
    problem_function = function_info.function
    distributions = function_info.distributions_for_sparSpace
    a = function_info.a
    b = function_info.b
    op = UncertaintyQuantification(problem_function, distributions, a, b)
    grid = GlobalTrapezoidalGridWeighted(a, b, op, modified_basis=not boundary, boundary=boundary)
    op.set_grid(grid)
    # The grid needs to be refined for the PCE coefficient calculation
    op.set_PCE_Function(polynomial_degree_max)
    combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, margin=0.8, operation=op, norm=2, grid_surplusses=grid)
    if function_info.dim > 3:
        lmax = 2
    else:
        lmax = 3
    error_operator = ErrorCalculatorSingleDimVolumeGuided()
    combiinstance.performSpatiallyAdaptiv(1, lmax,
                                          error_operator, tol=0, max_evaluations=max_evals, do_plot=False)
    # Create the PCE approximation; it is saved internally in the operation
    op.calculate_PCE(None, combiinstance)
    # Calculate the expectation, variance and sobol indices with the PCE coefficients
    (E,), (Var,) = op.get_expectation_and_variance_PCE()
    first_order_sobol_indices = op.get_first_order_sobol_indices()
    total_order_sobol_indices = op.get_total_order_sobol_indices()
    total_sum = np.sum(total_order_sobol_indices)
    total_order_sobol = [sobol / total_sum for sobol in total_order_sobol_indices]
    print(f"E: {E}, PCE Var: {Var}")
    print("First order Sobol indices: ", first_order_sobol_indices)
    number_points = combiinstance.get_total_num_points()
    if total:
        sobols = total_order_sobol
    else:
        sobols = first_order_sobol_indices
    entry = (number_points, float(E), float(Var), *sobols)
    if store_result:
            storeResult(entry, 10, function_info)

#perform PSP on an interpolant constructed with sparseSpACE
def Pseudo_Spectral_with_CombinationTechnique(function_info, gridName, adaptive, max_evals=None, maximum_level=None, store_result=False, time_series=False, boundary=True, polynomial_degrees=10):
    errorOperator = ErrorCalculatorSingleDimVolumeGuided()
    if adaptive:
        if gridName == 'BSpline_p3':
            grid = GlobalBSplineGrid(a=function_info.a, b=function_info.b, modified_basis=not boundary, boundary=boundary, p=9)
        else:
            grid = GlobalTrapezoidalGrid(a=function_info.a, b=function_info.b, modified_basis=not boundary, boundary=boundary)
        operation = Integration(f=function_info.function, grid=grid, dim=function_info.dim)
        combiObject = SpatiallyAdaptiveSingleDimensions2(np.ones(function_info.dim) * function_info.a, np.ones(function_info.dim) * function_info.b,
                                                                            margin=0.8,
                                                                            operation=operation)
        tolerance = 10 ** -20
        plotting = False
        if function_info.dim > 3:
            maximum_level = 2
        else:
            maximum_level = 3
        combiObject.performSpatiallyAdaptiv(1, maximum_level, errorOperator, tol=tolerance, do_plot=plotting,
                                                               max_evaluations=max_evals)
        #combiObject.print_resulting_sparsegrid(markersize=10)
        print("integral: ", operation.integral)
    else:
        if gridName == 'Trapezoidal':
            grid = TrapezoidalGrid(a=function_info.a, b=function_info.b, modified_basis=not boundary, boundary=boundary)
        elif gridName == 'Leja' or gridName == 'LejaNormal':  #for actual comparisons use LejaNormal
            grid = LejaGrid(a=function_info.a, b=function_info.b, boundary=boundary)
        elif gridName == 'BSpline_p3':
            grid = BSplineGrid(a =function_info.a, b=function_info.b, boundary=boundary, p=9)
        operation = Integration(f=function_info.function, grid=grid, dim=function_info.dim)
        minimum_level = 1
        combiObject = StandardCombi(function_info.a, function_info.b, operation=operation)
        combiObject.perform_operation(minimum_level, maximum_level+1, function_info.function)
        #combiObject.print_resulting_combi_scheme()
        #combiObject.print_resulting_sparsegrid()

    number_points = combiObject.get_total_num_points()

    if time_series:
        # TODO Ivana-note: Why always order 1?
        gauss_quads = chaospy.generate_quadrature(1, function_info.joint_distributions, rule='gaussian', sparse=False)
        nodes, weights = gauss_quads
        print("number quadrature points: ", len(nodes[0]))
        gauss_evals_Interpolation = combiObject(nodes.T)
        # TODO Ivana-note: Why always order 1?
        expansion = chaospy.generate_expansion(1, function_info.joint_distributions)
        gauss_model_approx = chaospy.fit_quadrature(expansion, nodes, weights, gauss_evals_Interpolation)

        expected = chaospy.E(gauss_model_approx, function_info.joint_distributions)
        variance = chaospy.Var(gauss_model_approx, function_info.joint_distributions)
        std = chaospy.Std(gauss_model_approx, function_info.joint_distributions)
        coordinates = np.arange(0, 366, step=1)
        pyplot.rc("figure", figsize=[6, 4])
        pyplot.xlabel("coordinates")
        pyplot.ylabel("model approximation")
        pyplot.fill_between(
            coordinates, expected - 2 * std, expected + 2 * std, alpha=0.3)
        pyplot.plot(coordinates, expected)
        pyplot.show()
        return

    if gridName == 'Leja': #link Leja interpoation order and gPCE truncation
        quads = chaospy.generate_quadrature(2*2*order, function_info.joint_distributions, rule="leja", sparse=True,
                                            growth=False)
        nodes, weights = quads
        print("number quadrature points: ", len(nodes.T))
        gauss_evals_Interpolation = combiObject(nodes.T)
        expansion = chaospy.generate_expansion(2*order-1, function_info.joint_distributions, normed=True)
        Leja_model_approx, gpCE_coefficients = chaospy.fit_quadrature(expansion, nodes, weights, gauss_evals_Interpolation, retall=True)
     #   print(gpCE_coefficients)
        expected, variance, first_order_sobol_indices = calculate_MeanVarianceSobol(gpCE_coefficients, 2*order-1, function_info.dim)
    else:
        quads = chaospy.generate_quadrature(polynomial_degrees, function_info.joint_distributions, rule='gaussian', sparse=False)
     #   quads = chaospy.generate_quadrature(2*(polynomial_degrees+1), function_info.joint_distributions, rule='leja', sparse=True, growth=False)
        nodes, weights =quads
        print("number quadrature points: ", len(nodes[0]))
        gauss_evals_Interpolation = combiObject(nodes.T)
        expansion = chaospy.generate_expansion(polynomial_degrees, function_info.joint_distributions, normed=True)
        gauss_model_approx, gPCE_coefficients = chaospy.fit_quadrature(expansion, nodes, weights, gauss_evals_Interpolation, retall=True)
        expected, variance, first_order_sobol_indices = calculate_MeanVarianceSobol(gPCE_coefficients, polynomial_degrees, function_info.dim)
    if store_result:
        entry = (number_points, float(expected), float(variance), *first_order_sobol_indices)
        if adaptive:
            if gridName == 'BSpline_p3':
                index = 4
            else:
                index = 2
        elif gridName == 'Trapezoidal':
            index = 1
        elif gridName == 'Leja' or gridName == 'LejaNormal':
            index = 3
        elif gridName == 'BSpline_p3':
            index = 4
        storeResult(entry, index, function_info)

#add place to store for a new function
def initiate_Variance_Errors(function_info):
    #pickle_errors_to_plot_in = open(function_info.path_Error, "rb")
    errors_to_plot = [[] for _ in range(10)]
    pickle_errors_to_plot_out = open(function_info.path_Error, "wb")
    pickle.dump(errors_to_plot, pickle_errors_to_plot_out)
    pickle_errors_to_plot_out.close()

#compute gPCE coefficients analytically on a piecewise linear interpolant constructed with standard CT or the single dimension refinement strategy
def analytica_integration_with_surrogate(function_info, adaptive=False, store_result=False, max_evals=None, maximum_level=None, boundary=True, time_Series=False, polynomial_degrees = 3):

   #compute the one-dimensional integral
    def computeIntegral(point_d, neighbours, distributions_d, pce_poly_1d, d):
        if point_d <= 0 or max(0, neighbours[0]) >= min(1, point_d):
            integralLeft = 0
        elif (not boundary) and neighbours[0] == 0:
            hatFunctionLeft = lambda x: 1
            integrandLeft = lambda x: pce_poly_1d(function_info.ppfs[d](x)) * hatFunctionLeft(x)
            integralLeft = integrate.fixed_quad(integrandLeft, max(0, neighbours[0]), min(1, point_d), n=7)[0]
        else:
            hatFunctionLeft = lambda x: (x - neighbours[0]) / (point_d - neighbours[0])
            integrandLeft = lambda x: pce_poly_1d(function_info.ppfs[d](x)) * hatFunctionLeft(x)
            integralLeft = integrate.fixed_quad(integrandLeft, max(0, neighbours[0]), min(1, point_d), n=7)[0]
        if point_d >= 1 or max(0, point_d) >= min(1, neighbours[1]):
            integralRight = 0
        elif (not boundary) and neighbours[1] == 1:
            hatFunctionRight = lambda x: 1
            integrandRight = lambda x: pce_poly_1d(function_info.ppfs[d](x)) * hatFunctionRight(x)
            integralRight = integrate.fixed_quad(integrandRight, max(0, point_d), min(1, neighbours[1]), n=7)[0]
        else:
            hatFunctionRight = lambda x: max(0,((x - neighbours[1]) / (point_d - neighbours[1]))[0])
            hatFunctionRight = lambda x: (x - neighbours[1]) / (point_d - neighbours[1])
            integrandRight = lambda x: pce_poly_1d(function_info.ppfs[d](x)) * hatFunctionRight(x)
            integralRight = integrate.fixed_quad(integrandRight, max(0, point_d), min(1, neighbours[1]), n=7)[0]
        return integralLeft + integralRight

    def standard_hatfunction1D(u):
        return [max(1 - abs(ui), 0) for ui in u]

    def hatfunction_level1D_position(u, l, x):
        return standard_hatfunction1D((u - x) / float(2) ** (-l))

    f = function_info.function_unitCube
    a = np.array([0 for d in range(function_info.dim)])
    b = np.array([1 for d in range(function_info.dim)])
    if adaptive:
        errorOperator = ErrorCalculatorSingleDimVolumeGuided()
        grid = GlobalTrapezoidalGrid(a=a, b=b, modified_basis = not boundary, boundary=boundary)
        operation = Integration(f=f, grid=grid, dim=function_info.dim)
        combiObject = StandardCombi(a, b, operation=operation)
        combiObject = SpatiallyAdaptiveSingleDimensions2(np.ones(function_info.dim) * a, np.ones(function_info.dim) * b, operation=operation, margin=0.8)
        tolerance = 0
        plotting = False
        if function_info.dim > 3:
            maxim_level = 2
        else:
            maxim_level = 3
        combiObject.performSpatiallyAdaptiv(1, maxim_level, errorOperator, tol=tolerance, do_plot=plotting,
                                            max_evaluations=max_evals)
        #combiObject.draw_refinement()
        combiObject.print_resulting_combi_scheme(markersize=5)
        print(operation.integral)
        dictIntegrals_adaptive = {}
        dictEvaluations = {}

        #for adaptive
        def getNeighbours(combiObject, coordinates1d, x):
            if coordinates1d[0] > 0:
                print("leftest: ", coordinates1d[0])
            left_right = [0, 1]
            index = np.where(coordinates1d == x)[0][0]
            if not (index == 0):
                left_right[0] = coordinates1d[index - 1]
            if not (index == len(coordinates1d) - 1):
                left_right[1] = coordinates1d[index + 1]
            return left_right

    else:
        grid = TrapezoidalGrid(a=np.array([0 for d in range(function_info.dim)]), b=np.array([1 for d in range(function_info.dim)]), modified_basis=not boundary, boundary=boundary)
        operation = Integration(f=f, grid=grid, dim=function_info.dim)
        combiObject = StandardCombi(a=np.array([0 for d in range(function_info.dim)]), b=np.array([1 for d in range(function_info.dim)]), operation=operation)
        minimum_level = 1
        combiObject.perform_operation(minimum_level, maximum_level)
        print("expectation: ", operation.integral)
        dictIntegrals_not_adaptive = {}
        dictEvaluations = {}

    # extract the onedimensional orthogonal polynomials and order them in the same way chaospy does
    number_points = combiObject.get_total_num_points()
    expansion = chaospy.generate_expansion(polynomial_degrees, function_info.joint_distributions, normed=True)
    pce_polys_1D = [None] * function_info.dim
    for d in range(function_info.dim):
        pce_polys_1D[d] = chaospy.expansion.stieltjes(polynomial_degrees, function_info.distributions[d], normed=True)
    indices = numpoly.glexindex(start=0, stop=polynomial_degrees + 1, dimensions=function_info.dim,
                                graded=True, reverse=True,
                                cross_truncation=1.0)
    polys = [None] * len(indices)
    for i in range(len(indices)):
        polys[i] = [pce_polys_1D[d][indices[i][d]] for d in range(function_info.dim)]

    cn = [np.zeros(function_info.function.output_length()) for _ in polys]
    for n, pce_poly in enumerate(polys):
        for component_grid in combiObject.scheme:
            if adaptive:
                gridPointCoordsAsStripes, grid_point_levels, children_indices = combiObject.get_point_coord_for_each_dim(
                    component_grid.levelvector)
                points = combiObject.get_points_component_grid(component_grid.levelvector)
                keyLevelvector = component_grid.levelvector
                if keyLevelvector in dictEvaluations:
                    evals = dictEvaluations[keyLevelvector]
                else:
                    evals = [function_info.function_unitCube(poin) for poin in points]
                    dictEvaluations[keyLevelvector] = evals
                integralCompGrid = 0
            else:
                points = combiObject.get_points_component_grid(component_grid.levelvector)
                keyLevelvector = tuple(component_grid.levelvector.tolist())
                if keyLevelvector in dictEvaluations:
                    evals = dictEvaluations[keyLevelvector]
                else:
                    evals = combiObject(points)
                    dictEvaluations[keyLevelvector] = evals
                integralCompGrid = 0
            for i, point in enumerate(points):
                product = 1
                for d in range(0, function_info.dim):
                    if adaptive:
                        neighbours = getNeighbours(combiObject, gridPointCoordsAsStripes[d], point[d])
                        if (point[d], tuple(neighbours), indices[n][
                            d], d) in dictIntegrals_adaptive:
                            onedimensionalIntegral = dictIntegrals_adaptive[(point[d], tuple(neighbours), indices[n][d], d)]
                        else:
                            onedimensionalIntegral = computeIntegral(point[d],
                                                                     neighbours, function_info.distributions[d],
                                                                     pce_poly[d], d)
                            dictIntegrals_adaptive[(point[d], tuple(neighbours), indices[n][d], d)] = onedimensionalIntegral
                    else:
                        if (point[d], component_grid.levelvector[d], indices[n][
                            d], d) in dictIntegrals_not_adaptive:
                            onedimensionalIntegral = dictIntegrals_not_adaptive[
                                (point[d], component_grid.levelvector[d], indices[n][d], d)]
                        else:
                            neighbours = [max(0, point[d]-float(2)**(-component_grid.levelvector[d])), min(1, point[d]+float(2)**(-component_grid.levelvector[d]))]
                            onedimensionalIntegral = computeIntegral(point[d], neighbours, function_info.distributions[d], pce_poly[d], d)
                            dictIntegrals_not_adaptive[
                                (point[d], component_grid.levelvector[d], indices[n][d], d)] = onedimensionalIntegral
                    product = product * onedimensionalIntegral
                integralCompGrid = integralCompGrid + product * evals[i]
            cn[n] = cn[n] + component_grid.coefficient * integralCompGrid

    expected, variance, first_order_sobol_indices = calculate_MeanVarianceSobol(cn, polynomial_degrees, function_info.dim)
    print("variance: ", variance)
    if store_result and not time_Series:
        entry = (number_points, expected, variance, *first_order_sobol_indices)
        storeResult(entry, 6 if adaptive else 5, function_info)
    if time_Series:
        plot_Times_series(function_info, variance, expected, first_order_sobol_indices)

#plot mean, variance, and sobol indices
def plot_Errors_general(indices, function_info, absolute=False):
    colors = ['blue', 'orange', 'red', 'black', 'pink', 'cyan', 'brown', 'gray', 'green', 'blue', 'dodgerblue']
    linestyles = ['solid' for _ in range(9)] + ['dotted']+['solid']
    pickle_errors_to_plot_in = open(function_info.path_Error, "rb")
    errors_to_plot = pickle.load(pickle_errors_to_plot_in)
    # mean
    legend = []
    for i in indices:
        xValues = []
        yValues = []
        errors = sorted(errors_to_plot[i], key=lambda x: x[0])
        if len(errors) == 0:
            continue
        legend.append(i)
        for entry in errors:
            if entry[0] < 20:
                continue
            xValues.append(entry[0])
            if absolute:
                to_plot = entry[1]
            else:
                to_plot = abs((entry[1] - function_info.mean_analytical) / function_info.mean_analytical)
            yValues.append(to_plot)
        print('mean: xValues for index ', i, ': ', xValues, ', yValues: ', yValues)
        pyplot.plot(xValues, yValues, colors[i], linestyle=linestyles[i])
        pyplot.yscale('log')
        pyplot.xscale('log')
    pyplot.legend([names[i] for i in legend])
    pyplot.xlabel('function evaluations')
    if absolute:
        pyplot.ylabel('values')
        pyplot.title('mean')
    else:
        pyplot.ylabel('relative error')
        pyplot.title('mean- relative error')
    pyplot.show()

    # variance
    legend = []
    for i in indices:
        xValues = []
        yValues = []
        errors = sorted(errors_to_plot[i], key=lambda x: x[0])
        if len(errors) == 0:
            continue
        legend.append(i)
        for entry in errors:
            if entry[0] < 15:
                continue
            xValues.append(entry[0])
            if absolute:
               to_plot = entry[2]
            else:
               to_plot = abs((entry[2] - function_info.variance_analytical) / function_info.variance_analytical)
            yValues.append(to_plot)
        print('variance: xValues for index ', i, ': ', xValues, ', yValues: ', yValues)
        pyplot.plot(xValues, yValues, colors[i], linestyle=linestyles[i])
        pyplot.yscale('log')
        pyplot.xscale('log')
    pyplot.legend([names[i] for i in legend])
    pyplot.xlabel('function evaluations')
    if absolute:
        pyplot.ylabel('value')
        pyplot.ylabel('variance')
    else:
        pyplot.ylabel('relative error')
        pyplot.title('variance- relative error')
    pyplot.show()

    # first order sobol indices
    indices = [i for i in indices if i != 7]
    for d in range(function_info.dim):
        legend = []
        for i in indices:
            xValues = []
            yValues = []
            errors = sorted(errors_to_plot[i], key=lambda x: x[0])
            if len(errors) == 0:
                continue
            legend.append(i)
            for entry in errors:
                xValues.append(entry[0])
                if absolute:
                    to_plot = entry[3+d]
                    yValues.append(to_plot)
                elif function_info.first_order_sobol_indices[d] < 10**(-20):
                    yValues.append(entry[3+d])
                else:
                    relError = abs(entry[3+d] - function_info.first_order_sobol_indices[d]) / function_info.first_order_sobol_indices[d]
                    yValues.append(relError)
            pyplot.plot(xValues, yValues, colors[i], linestyle=linestyles[i])
            pyplot.yscale('log')
            pyplot.xscale('log')
        pyplot.legend([names[i] for i in legend])
        pyplot.xlabel('function evaluations')
        if absolute:
            pyplot.ylabel('value')
            pyplot.title('first order sobol index S' + str(d + 1))
        elif function_info.first_order_sobol_indices[d] < 10**(-20):
            pyplot.ylabel('absolute error')
            pyplot.title('first order sobol index S' + str(d+1) + ', absolute error')
        else:
            pyplot.ylabel('relative error')
            pyplot.title('first order sobol index S' + str(d+1) +', relative error')
        pyplot.show()

#add a new method
def add_category(function_info):
    pickle_errors_to_plot_in = open(function_info.path_Error, "rb")
    errors_to_plot = pickle.load(pickle_errors_to_plot_in)
    errors_to_plot.append([])
    pickle_errors_to_plot_out = open(function_info.path_Error, "wb")
    pickle.dump(errors_to_plot, pickle_errors_to_plot_out)
    pickle_errors_to_plot_out.close()

#plot computed sobol indices
def plot_Sobol_indices(indices, function_info):
    colors = ['blue', 'orange', 'red', 'black', 'pink', 'green']
    pickle_errors_to_plot_in = open(function_info.path_Error, "rb")
    errors_to_plot = pickle.load(pickle_errors_to_plot_in)
    for i in indices:
        errors = sorted(errors_to_plot[i], key=lambda x: x[0])
        for d in range(function_info.dim):
            xValues = []
            yValues_d=[]
            for entry in errors:
                xValues.append(entry[0])
                yValues_d.append(entry[3+d])
            pyplot.plot(xValues, yValues_d, colors[d])
            pyplot.xscale('log')
        pyplot.xlabel('model evaluations')
        pyplot.ylabel('simulated total order sobol indices')
        pyplot.title(names[i])
        pyplot.show()

#create the function_info object for a chosen function
def initiate_function_info(function_name):
    if function_name == 'corner_peak':
        coeffs = [float(1) for _ in range(3)]
        return Function_Info(GenzCornerPeak(coeffs), 3, np.array([0] * 3), np.array([1] * 3),
                                                  [chaospy.Uniform(0, 1) for d in range(3)],
                                                  [("Uniform", 0, 1) for _ in range(3)],
                                                  'errors_to_plot_CornerPeak_.pickle',
                                                  0.041666666666666664, 0.00263350372942387,
                                                  ppfs=[(lambda x: x) for d in range(3)],
                                                  function_unitCube=GenzCornerPeak(coeffs))

    elif function_name == 'product_peak':
        coeffsProd = [float(3) for _ in range(3)]
        midpoint = [0.5 for _ in range(3)]
        return Function_Info(GenzProductPeak(coeffsProd, midpoint), 3, np.array([0] * 3),
                                                   np.array([1] * 3), [chaospy.Uniform(0, 1) for d in range(3)],
                                                   [("Uniform", 0, 1) for _ in range(3)],
                                                   'errors_to_plot_CornerProduct_.pickle',
                                                   0.20504107661766025, 0.017263327472119228,
                                                   ppfs=[(lambda x: x) for d in range(3)],
                                                   function_unitCube=GenzProductPeak(coeffsProd, midpoint))

    elif function_name == 'discontinuous':
        midpoint = [0.5 for _ in range(3)]
        coeffs = [float(1) for _ in range(3)]
        return Function_Info(GenzDiscontinious(coeffs, midpoint), 3, np.array([0] * 3), np.array([1] * 3),
                                  [chaospy.Uniform(0, 1) for d in range(3)], [("Uniform", 0, 1) for _ in range(3)],
                                  'errors_to_plot_Discontinuous_.pickle',
                                  0.06091618422799687, 0.02786177572755664,
                                  ppfs=[(lambda x: x) for d in range(3)], function_unitCube=GenzDiscontinious(coeffs, midpoint))

    elif function_name == 'gfunction':
        return Function_Info(FunctionG(3), 3, np.array([0] * 3), np.array([1] * 3),
                                                [chaospy.Uniform(0, 1) for d in range(3)],
                                                [("Uniform", 0, 1) for _ in range(3)],
                                                'errors_to_plot_FunctionG.pickle', FunctionG(3).get_expectation(),
                                                FunctionG(3).get_variance(),
                                                first_order_sobol_indices=[0.506250, 0.225033, 0.1265625],
                                                ppfs=[(lambda x: x) for _ in range(3)], function_unitCube=FunctionG(3))

    elif function_name == 'ishigami':
        return Function_Info(modelSolverFUNC(), 3, np.array([-math.pi] * 3), np.array([math.pi] * 3),
                                               [chaospy.Uniform(-math.pi, math.pi) for d in range(3)],
                                               [("Uniform", -math.pi, math.pi) for _ in range(3)],
                                               '../test/Ishigami_Variance_errors_to_plot.pickle',
                                               3.5, 13.844587940719254,
                                               first_order_sobol_indices=[0.31390519114781146, 0.4424111447900409, 0.0])

    elif function_name == 'functionUQ':
        a = np.array([-2.5, -2, 5])
        b = np.array([2.5, 2, 15])
        dissesForSparseSpace = [("Uniform", -2.5, 2.5), ("Uniform", -2, 2), ("Uniform", 5, 15)]
        return Function_Info(FunctionUQ(), 3, a, b,
                                                 [chaospy.Uniform(-2.5, 2.5), chaospy.Uniform(-2, 2),
                                                  chaospy.Uniform(5, 15)],
                                                 dissesForSparseSpace, 'errors_to_plot_FunctionUQ.pickle',
                                                 11.333120910992074, 13.401276901354493,
                                                 first_order_sobol_indices=[0.6606445, 37.0142899, 6.21831285],
                                                 ppfs=[(lambda x: a[0] + (b[0]-a[0]) * x), lambda x: a[1] + (b[1]-a[1]) * x,
                                                       lambda x: a[2] + (b[2]-a[2]) * x])

    elif function_name == 'test':
        dim = 4
        a2d = [0 for _ in range(dim)]
        b2d = [1 for _ in range(dim)]
        function_info_test = Function_Info(testFunction(), dim, a2d, b2d, [chaospy.Uniform(0, 1) for _ in range(dim)],
                                           [("Uniform", 0, 1) for d in range(dim)], None, 0, 0,
                                           ppfs = [(lambda x: x) for _ in range(dim)])
        return function_info_test

    elif function_name == 'hbv':
        inputModelDir = pathlib.Path("/home/markus/studium/BA/thesis_ME/Bachelorthesis_Markus_Englberger/hbv")
        outputModelDir = pathlib.Path('/home/markus/studium/BA/thesis_ME/Bachelorthesis_Markus_Englberger/hbv')
        config_file = pathlib.Path(
            '/home/markus/studium/BA/thesis_ME/Bachelorthesis_Markus_Englberger/hbv/configuration_hbv_6D.json')
        writing_results_to_a_file = False
        plotting = False
        with open(config_file) as f:
            configuration_object = json.load(f)
        dim = 0
        distributions = []
        a = []
        b = []
        param_names = []
        for single_param in configuration_object["parameters"]:
            param_names.append(single_param["name"])
            if single_param["distribution"] != "None":
                dim += 1
                distributions.append((single_param["distribution"], single_param["lower"], single_param["upper"]))
                a.append(single_param["lower"])
                b.append(single_param["upper"])
        qoi = "GoF"  # "Q" "GoF"
        gof = "RMSE"  # "RMSE" "NSE"  "None"
        operation = "UncertaintyQuantification"  # "Interpolation"
        problem_function = HBVSASKFunction(
            configurationObject=configuration_object,
            inputModelDir=inputModelDir,
            workingDir=outputModelDir,
            dim=dim,
            param_names=param_names,
            qoi=qoi,
            gof=gof,
            writing_results_to_a_file=writing_results_to_a_file,
            plotting=plotting
        )
        function_info_HBV = Function_Info(problem_function, 6, a, b, [chaospy.Uniform(a[d], b[d]) for d in range(6)],
                                          [("Uniform", a[d], b[d]) for d in range(6)], 'errors_to_plot_HBV.pickle',
                                          20.884443614844272, 45.83736482816192, [0.20023067810656728, 0.004344397732323057, 0.7253897084658435, 0.006251645502896026, 0.01838260129644447, 0.006158532124937321],
                                          ppfs = [(lambda x: a[0] + (b[0] - a[0]) * x), lambda x: a[1] + (b[1] - a[1]) * x, lambda x: a[2] + (b[2] - a[2]) * x,
                lambda x: a[3] + (b[3] - a[3]) * x,
               lambda x: a[4] + (b[4] - a[4]) * x, lambda x: a[5] + (b[5] - a[5])*x])
        return function_info_HBV

    elif function_name == 'hbvTime_series':
        inputModelDir = pathlib.Path("/home/markus/studium/BA/thesis_ME/Bachelorthesis_Markus_Englberger/hbv")
        outputModelDir = pathlib.Path('/home/markus/studium/BA/thesis_ME/Bachelorthesis_Markus_Englberger/hbv')
        config_file = pathlib.Path(
            '/home/markus/studium/BA/thesis_ME/Bachelorthesis_Markus_Englberger/hbv/configuration_hbv_6D.json')
        writing_results_to_a_file = False
        plotting = False
        with open(config_file) as f:
            configuration_object = json.load(f)
        dim = 0
        distributions = []
        a = []
        b = []
        param_names = []
        for single_param in configuration_object["parameters"]:
            param_names.append(single_param["name"])
            if single_param["distribution"] != "None":
                dim += 1
                distributions.append((single_param["distribution"], single_param["lower"], single_param["upper"]))
                a.append(single_param["lower"])
                b.append(single_param["upper"])
        qoi = "Q"  # "Q" "GoF"
        gof = "RMSE"  # "RMSE" "NSE"  "None"
        problem_function = HBVSASKFunction(
            configurationObject=configuration_object,
            inputModelDir=inputModelDir,
            workingDir=outputModelDir,
            dim=dim,
            param_names=param_names,
            qoi=qoi,
            gof=gof,
            writing_results_to_a_file=writing_results_to_a_file,
            plotting=plotting
        )
        function_info_HBV = Function_Info(problem_function, 6, a, b, [chaospy.Uniform(a[d], b[d]) for d in range(6)],
                                          [("Uniform", a[d], b[d]) for d in range(6)], 'errors_to_plot_HBVTime.pickle', -0.5733524031438637,
                                          0.7734849425297574,[0.031187601887663192, 0.05686180301399113, 0.6364034951648563, 0.010538720205043327, 0.004477443764674821, 0.14101106642723688], ppfs = [(lambda x: a[0] + (b[0] - a[0]) * x), lambda x: a[1] + (b[1] - a[1]) * x, lambda x: a[2] + (b[2] - a[2]) * x,
                lambda x: a[3] + (b[3] - a[3]) * x,
               lambda x: a[4] + (b[4] - a[4]) * x, lambda x: a[5] + (b[5] - a[5])*x])
        return function_info_HBV

#compute mean and variance with quasi monte_carlo
def monte_carlo(f_info,number_nodes):
    distribution = f_info.joint_distributions
    nodes = distribution.sample(number_nodes, "halton")
    evals = np.array([function_info.function.eval(node) for node in nodes.T])
    expected = np.mean(evals, 0)
    variance = np.var(evals, 0)
    print("mean: ", expected, ", variance: ", variance)
    plot_Times_series(f_info, variance, expected)

#plot mean, measured data, standard devation for time series with HBV
def plot_Times_series(function_info, variances, expected, total_Sobol_indices):
    std = [math.sqrt(variance) for variance in variances]
    coordinates = np.arange(0, len(variances), step=1)
    pyplot.rc("figure", figsize=[6, 4])
    pyplot.xlabel("days")
    pyplot.ylabel("streamflow")
    lower = [(e-2*s) for e,s in zip(expected, std)]
    upper = [(e+2*s) for e,s in zip(expected, std)]
    pyplot.fill_between(
        coordinates, lower, upper, alpha=0.3)
    pyplot.plot(coordinates, expected)
    pyplot.plot(coordinates, function_info.function.hbvsaskModelObject.measured)
    pyplot.legend(["simulated mean +/- 2 standard deviations", "simulated mean", "measured data"])
    pyplot.show()

    coordinates = np.arange(0, 366, step=1)
    for d in range(function_info.dim):
        pyplot.plot(coordinates, total_Sobol_indices[d])
    pyplot.xlabel("days")
    pyplot.ylabel("total order sobol indices")
    pyplot.legend(['FC', 'beta', 'FRAC', 'K1', 'alpha', 'K2'])
    pyplot.title("total order sobol indices with gaussian")
    pyplot.show()
    pickle_totalSobol = open('totalSobol.pickle', "wb")
    pickle.dump([expected, variances, total_Sobol_indices], pickle_totalSobol)
    pickle_totalSobol.close()
    return



if __name__ == "__main__":

    function_info = initiate_function_info('ishigami') #chose function
    boundary = True       #chose if boundary is used, if not boundary then modified basis
    store_Result = True   #chose if results are stored
    time_Series = False  #chose if times series, only for hbv_Timeseries

    #initiate_Variance_Errors(function_info) #add new function
    #add_category(function_info)             #ad new method
    #delete_entries(2, function_info)        #delete results for some method

    # 0 = A1, Pseudo_with_Chaospy_directly, Gauss
    # 1 = B1, Pseudo Spectral with Combination Technique: Grid = Standard Combi, not adaptive
    # 2 = B2, Pseudo Spectral with Combination Technique: Grid = Trapezoidal, adaptive
    # 3 = Pseudo_Spectral with_Combination_Technique, Grid = Leja, not adaptive
    # 4 = Pseudo Spectral with Combination Technique using Bsplines with degree 3, adaptive
    # 5 = analytical Integration with Surrogate: Grid = Standard Combi, not adaptive, nonlinear transformation
    # 6 = analytical Integration with Surrogate: Grid = Trapezoidal, adaptive, nonlinear transformation
    # 7 = weighted adaptive trapezoidal grid, only mean and variance
    # 8 = Psuedo_Spectral with sparse Leja
    # 9 = PSP with Gauss, include number or previous grids
    # 10 = PSP using adaptive CB for all inner products
    #create results, methods ordered in same way as above
   # for order in range(1,2):
   #      chaospy_directly('gauss', order, function_info , store_result=True)
    #for order in range(2,4):
    #    Pseudo_Spectral_with_CombinationTechnique(function_info, gridName='Trapezoidal', adaptive=False, maximum_level=order, store_result=False, time_series=False, boundary=False, polynomial_degrees=3)
    #for max_evals in [400,600,100,2000]:
    #   Pseudo_Spectral_with_CombinationTechnique(function_info, gridName='Trapezoidal', adaptive=True, max_evals=max_evals, store_result=True, boundary=boundary)
    #for order in range(2,8):
    #    Pseudo_Spectral_with_CombinationTechnique(function_info, gridName='LejaNormal', adaptive=False, maximum_level=order, store_result=store_Result, boundary=boundary, polynomial_degrees=10)
    #for max_evals in [600,1300, 2000]:
    #    Pseudo_Spectral_with_CombinationTechnique(function_info, gridName='BSpline_p3', adaptive=True, max_evals=max_evals, store_result=True)
    #for order in range(1,6):
    #    analytica_integration_with_surrogate(function_info, adaptive=False, store_result=store_Result, maximum_level=order, boundary=boundary)
    #for max_evals in [550]:#, 1200, 1500]:
    #    analytica_integration_with_surrogate(function_info, adaptive=True, store_result=store_Result, max_evals=max_evals, boundary=boundary, time_Series=time_Series,polynomial_degrees=10)
    #(300, function_info, store_result=True)
    #for order in range(1,4):
    #    chaospy_directly(('leja_direct'), order, function_info, store_result=True)
    #for order in range(2,5):
    #    chaospy_directly('gauss', order, function_info , store_result=True, count_previous_grids=True)
    #
  #  for max_evals in [900,1200,1500,2000]:
  #      Pseudo_Spectral_adative_CB_for_PCEcoefficients(max_evals, function_info, store_result=store_Result, polynomial_degree_max=10)
    #for max_evals in [600,800,1300]:
    #    only_mean_and_variance(max_evals,function_info,store_result=store_Result)
    #chaospy_directly('gauss', 3, function_info, store_result=False, count_previous_grids=False, time_series=True)

    indices = [0, 9, 8, 10, 1, 2, 3, 5, 6, 7] #chose for which methods to plot results
    plot_Errors_general(indices, function_info, absolute=False)
    plot_Sobol_indices(indices, function_info)







    
    
