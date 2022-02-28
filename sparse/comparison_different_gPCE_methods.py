"""
Markus E.
"""
import math
import pickle
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
        self.ppfs = ppfs
        self.function_unitCube = function_unitCube

#information for ishigami
class testFunction(Function):
    def eval(self, coordinates):
        x1, x2 = coordinates
        return 1
class testFunctionScaled(Function):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.functionOriginal = testFunction()
    def eval(self, coordinates):
        result = self.functionOriginal.eval([self.a[d]+(self.b[d]- self.a[d])*coordinates[d] for d in range(len(self.a))])
        return result

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
ppfUniform = lambda x: 2 * math.pi * x - math.pi
def model_solver_nonlinear(coordinates):
    x1, x2, x3 = coordinates
    return math.sin(ppfUniform(x1)) + 7 * (math.sin(ppfUniform(x2))) ** 2 + 0.1 * ppfUniform(x3) ** 4 * math.sin(ppfUniform(x1))
class modelSolverFUNC_nonlinear(Function):
    def __init__(self):
        super().__init__()
    def output_length(self) -> int:
        return 1
    def eval(self, coordinates):
        return model_solver_nonlinear(coordinates)

class FunctionUQscaledIntoUnitcube(Function):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.functionUQ = FunctionUQ()
    def eval(self, coordinates):
        result = self.functionUQ.eval([self.a[d]+(self.b[d]- self.a[d])*coordinates[d] for d in range(len(self.a))])
        return result

def storeResult(entry, index, function_info):
    pickle_errors_to_plot_in = open(function_info.path_Error, "rb")
    errors_to_plot = pickle.load(pickle_errors_to_plot_in)
    errors_to_plot[index].append(entry)
    pickle_errors_to_plot_out = open(function_info.path_Error, "wb")
    pickle.dump(errors_to_plot, pickle_errors_to_plot_out)
    print("errors to plot for index", index, ': ', errors_to_plot[index])
    pickle_errors_to_plot_out.close()

def delete_entries(index, function_info):
    pickle_errors_to_plot_in = open(function_info.path_Error, "rb")
    errors_to_plot = pickle.load(pickle_errors_to_plot_in)
    errors_to_plot[index] = []
    pickle_errors_to_plot_out = open(function_info.path_Error, "wb")
    pickle.dump(errors_to_plot, pickle_errors_to_plot_out)
    pickle_errors_to_plot_out.close()

def chaospy_directly(gridname, order, function_info, store_result=False, count_previous_grids=False):
    number_points = 0
    if count_previous_grids: #only for Gauss Grid
        for order_previous in range(1, order):
            nodesPr, weightsPr = chaospy.generate_quadrature(order_previous, function_info.joint_distributions,
                                                             rule='gaussian')
            #print('order_previous: ', order_previous, ', number points: ', len(nodesPr.T))
            number_points += len(nodesPr.T)
            #print(number_points)

    if gridname == 'gauss':
        quads = chaospy.generate_quadrature(order, function_info.joint_distributions, rule='gaussian')
    elif gridname == 'clenshaw_curtis':
        quads = chaospy.generate_quadrature(order, function_info.joint_distributions, rule="clenshaw_curtis", sparse=True)
    nodes, weights = quads
    number_points += len(nodes.T)

    gauss_evals = np.array([function_info.function.eval(node) for node in nodes.T])
    print("number of points", number_points)

    expansion = chaospy.generate_expansion(order, function_info.joint_distributions)

    # gauss_model_approxRet = chaospy.fit_quadrature(expansion, nodes, weights, gauss_evals, retall=True)
    gauss_model_approx = chaospy.fit_quadrature(expansion, nodes, weights, gauss_evals)

    expected = chaospy.E(gauss_model_approx, function_info.joint_distributions)
    variance = chaospy.Var(gauss_model_approx, function_info.joint_distributions)
    # print("coefficients: ", gauss_model_approxRet[1])
    print("Expected: ", expected, ",  Variance: ", variance)

    #first_order_sobol_indices = chaospy.Sens_m(gauss_model_approx, function_info.joint_distributions)
    first_order_sobol_indices = [0,0,0]
    #total_order_sobol_indices = chaospy.Sens_t(gauss_model_approx, joint)
    print("First order Sobol indices: ", first_order_sobol_indices)
    #print("Total order Sobol indices: ", total_order_sobol_indices)
    entry = (number_points, float(expected), float(variance), *first_order_sobol_indices)
    if store_result:
        if count_previous_grids:
            storeResult(entry, 9, function_info)
        elif gridname == 'gauss':
            storeResult(entry, 0, function_info)
        elif gridname == 'clenshaw_curtis':
            storeResult(entry, 8, function_info)

def only_mean_and_variance(max_evals, function_info, store_result=False):
    distributions = function_info.distributions_for_sparSpace
    op = UncertaintyQuantification(function_info.function, distributions, function_info.a, function_info.b)
    grid = GlobalTrapezoidalGridWeighted(function_info.a, function_info.b, op, boundary=True)
    op.set_grid(grid)
    op.set_expectation_variance_Function()
    combiObject = SpatiallyAdaptiveSingleDimensions2(function_info.a, function_info.b, operation=op, norm=2, grid_surplusses=grid)
    lmax = 4
    error_operator = ErrorCalculatorSingleDimVolumeGuided()
    combiObject.performSpatiallyAdaptiv(1, lmax,
                                          error_operator, tol=0, max_evaluations=max_evals, do_plot=False)
    (E,), (Var,) = op.calculate_expectation_and_variance(combiObject)
    print(f"E: {E}, Var: {Var}")

    if store_result:
        number_points = combiObject.get_total_num_points()
        entry = (number_points, float(E), float(Var), 0, 0, 0)
        storeResult(entry, 7, function_info)

def Pseudo_Spectral_with_CombinationTechnique(function_info, gridName, adaptive, max_evals=None, maximum_level=None, store_result=False):
    errorOperator = ErrorCalculatorSingleDimVolumeGuided()
    if adaptive:
        if gridName == 'BSpline_p3':
            grid = GlobalBSplineGrid(a=function_info.a, b=function_info.b, modified_basis=False, boundary=True, p=13)
        else:
            grid = GlobalTrapezoidalGrid(a=function_info.a, b=function_info.b, modified_basis=False, boundary=True)
        operation = Integration(f=function_info.function, grid=grid, dim=function_info.dim)
        combiObject = SpatiallyAdaptiveSingleDimensions2(np.ones(function_info.dim) * function_info.a, np.ones(function_info.dim) * function_info.b,
                                                                            margin=0.5,
                                                                            operation=operation)
        tolerance = 10 ** -5
        plotting = False
        combiObject.performSpatiallyAdaptiv(1, 4, errorOperator, tol=tolerance, do_plot=plotting,
                                                               max_evaluations=max_evals)
        #combiObject.print_resulting_sparsegrid(markersize=10)
    else:
        if gridName == 'Trapezoidal':
            grid = TrapezoidalGrid(a=function_info.a, b=function_info.b, modified_basis=False, boundary=True)
        elif gridName == 'Leja':
            grid = LejaGrid(a=function_info.a, b=function_info.b, boundary=True)
        elif gridName == 'BSpline_p3':
            grid = BSplineGrid(a =function_info.a, b=function_info.b, boundary=True, p=13)
        operation = Integration(f=function_info.function, grid=grid, dim=function_info.dim)
        minimum_level = 1
        combiObject = StandardCombi(function_info.a, function_info.b, operation=operation)
        combiObject.perform_operation(minimum_level, maximum_level, function_info.function)
        #combiObject.print_resulting_combi_scheme()
        #combiObject.print_resulting_sparsegrid()

    number_points = combiObject.get_total_num_points()

    #gauss_quads = chaospy.generate_quadrature(11, joint, rule='clenshaw_curtis', sparse=True)
    gauss_quads = chaospy.generate_quadrature(10, function_info.joint_distributions, rule='gaussian', sparse=False)
    nodes, weights = gauss_quads
    print("number quadrature points: ", len(nodes[0]))
    gauss_evals_Interpolation = combiObject(nodes.T)
    expansion = chaospy.generate_expansion(10, function_info.joint_distributions)
    gauss_model_approx = chaospy.fit_quadrature(expansion, nodes, weights, gauss_evals_Interpolation)

    expected = chaospy.E(gauss_model_approx, function_info.joint_distributions)
    variance = chaospy.Var(gauss_model_approx, function_info.joint_distributions)
    print("expectation = ", expected, ", variance = ", variance)
    first_order_sobol_indices = [0,0,0]
    #first_order_sobol_indices = chaospy.Sens_m(gauss_model_approx, function_info.joint_distributions)
    print("First order Sobol indices: ", first_order_sobol_indices)
    #total_order_sobol_indices = chaospy.Sens_t(gauss_model_approx, joint)
    #print("Total order Sobol indices: ", total_order_sobol_indices)
    if store_result:
        entry = (number_points, float(expected), float(variance), *first_order_sobol_indices)
        if adaptive:
            if gridName == 'BSpline_p3':
                index = 4
            else:
                index = 2
        elif gridName == 'Trapezoidal':
            print("hier1")
            index = 1
        elif gridName == 'Leja':
            index = 3
        elif gridName == 'BSpline_p3':
            index = 4
        storeResult(entry, index, function_info)

def initiate_Variance_Errors(function_info):
    #pickle_errors_to_plot_in = open(function_info.path_Error, "rb")
    errors_to_plot = [[] for _ in range(9)]
    pickle_errors_to_plot_out = open(function_info.path_Error, "wb")
    pickle.dump(errors_to_plot, pickle_errors_to_plot_out)
    pickle_errors_to_plot_out.close()

def analytica_integration_with_surrogate(function_info, adaptive=False, store_result=False, max_evals=None, maximum_level=None):

    def standard_hatfunction1D(u):
        return [max(1 - abs(ui), 0) for ui in u]

    def hatfunction_level1D_position(u, l, x):
        return standard_hatfunction1D((u - x) / float(2) ** (-l))

    f = function_info.function_unitCube
    a = np.array([0 for d in range(function_info.dim)])
    b = np.array([1 for d in range(function_info.dim)])
    if adaptive:
        errorOperator = ErrorCalculatorSingleDimVolumeGuided()
        grid = GlobalTrapezoidalGrid(a=a, b=b, modified_basis=False, boundary=True)
        operation = Integration(f=f, grid=grid, dim=function_info.dim)
        combiObject = StandardCombi(a, b, operation=operation)
        combiObject = SpatiallyAdaptiveSingleDimensions2(np.ones(function_info.dim) * a, np.ones(function_info.dim) * b, operation=operation)
        tolerance = 0
        plotting = False
        combiObject.performSpatiallyAdaptiv(1, 4, errorOperator, tol=tolerance, do_plot=plotting,
                                            max_evaluations=max_evals)
        dictIntegrals_adaptive = {}
        dictEvaluations = {}
        # store the one dimensional integrals and the function evaluations
        # pickle_integrals_in = open("dict1D_integralsAdapt.pickle", "rb")
        # pickle_evaluations_in = open("dict_evaluationsAdapt.pickle", "rb")
        # dictIntegrals_adaptive = pickle.load(pickle_integrals_in)
        # dictEvaluations = pickle.load(pickle_evaluations_in)

        def getNeighbours(combiObject, coordinates1d, x):
            left_right = [0, 1]
            index = np.where(coordinates1d == x)[0][0]
            if not (index == 0):
                left_right[0] = coordinates1d[index - 1]
            if not (index == len(coordinates1d) - 1):
                left_right[1] = coordinates1d[index + 1]
            return left_right

        def computeIntegral(point_d, neighbours, distributions_d, pce_poly_1d, d):
            # print("left: ", neighbours[0], ", point: ", point_d, ", right:", neighbours[1])
            if point_d <= 0 or max(0, neighbours[0]) >= min(1, point_d):
                integralLeft = 0
            else:
                hatFunctionLeft = lambda x: (x - neighbours[0]) / (point_d - neighbours[0])
                integrandLeft = lambda x: pce_poly_1d(function_info.ppfs[d](x)) * hatFunctionLeft(x)
                integralLeft = integrate.fixed_quad(integrandLeft, max(0, neighbours[0]), min(1, point_d), n=6)[0]
            if point_d >= 1 or max(0, point_d) >= min(1, neighbours[1]):
                integralRight = 0
            else:
                hatFunctionRight = lambda x: (x - neighbours[1]) / (point_d - neighbours[1])
                integrandRight = lambda x: pce_poly_1d(function_info.ppfs[d](x)) * hatFunctionRight(x)
                integralRight = integrate.fixed_quad(integrandRight, max(0, point_d), min(1, neighbours[1]), n=6)[0]
            return integralLeft + integralRight
    else:
        grid = TrapezoidalGrid(a=np.array([0 for d in range(function_info.dim)]), b=np.array([1 for d in range(function_info.dim)]), modified_basis=False, boundary=True)
        operation = Integration(f=f, grid=grid, dim=function_info.dim)
        combiObject = StandardCombi(a=np.array([0 for d in range(function_info.dim)]), b=np.array([1 for d in range(function_info.dim)]), operation=operation)
        minimum_level = 1
        combiObject.perform_operation(minimum_level, maximum_level)
        dictIntegrals_not_adaptive = {}
        dictEvaluations = {}
        #pickle_dict1D_integrals_in = open("dict1D_integrals.pickle", "rb")
        #dictIntegrals_not_adaptive = pickle.load(pickle_dict1D_integrals_in)
        #pickle_dict_evaluations_in = open("dict_evaluations.pickle", "rb")
        #dictEvaluations = pickle.load(pickle_dict_evaluations_in)
    #print("Sparse Grid:")
    #combiObject.print_resulting_sparsegrid(markersize=10)

    # extract the onedimensional orthogonal polynomials and order them in the same way as chaospy does
    # has to be modified if the distributions are not the same for all dimensions
    number_points = combiObject.get_total_num_points()
    polynomial_degrees = 10
    expansion = chaospy.generate_expansion(polynomial_degrees, function_info.joint_distributions)
    #print("order: ", polynomial_degrees)
    pce_polys_1D, pce_polys_1D_norms = [None] * function_info.dim, [None] * function_info.dim
    for d in range(function_info.dim):
        pce_polys_1D[d], pce_polys_1D_norms[d] = chaospy.orth_ttr(polynomial_degrees, function_info.distributions[d], retall=True)

    pce_polys_d, pce_polys_norms_d = chaospy.orth_ttr(polynomial_degrees, function_info.joint_distributions, retall=True)
    indices = numpoly.glexindex(start=0, stop=polynomial_degrees + 1, dimensions=function_info.dim,
                                graded=True, reverse=True,
                                cross_truncation=1.0)
    norms = [None] * len(indices)
    polys = [None] * len(indices)
    for i in range(len(indices)):
        polys[i] = [pce_polys_1D[d][indices[i][d]] for d in range(function_info.dim)]
        norms[i] = [pce_polys_1D_norms[d][indices[i][d]] for d in range(function_info.dim)]

    cn = np.zeros(len(polys))

    for n, pce_poly in enumerate(polys):
        for component_grid in combiObject.scheme:
            if adaptive:
                gridPointCoordsAsStripes, grid_point_levels, children_indices = combiObject.get_point_coord_for_each_dim(
                    component_grid.levelvector)
                # print(gridPointCoordsAsStripes)
                points = combiObject.get_points_component_grid(component_grid.levelvector)
                evals = []
                keyLevelvector = component_grid.levelvector
                if keyLevelvector in dictEvaluations:
                    evals = dictEvaluations[keyLevelvector]
                else:
                    evals = combiObject(points)
                    dictEvaluations[keyLevelvector] = evals
                integralCompGrid = 0
            else:
                points = combiObject.get_points_component_grid(component_grid.levelvector)
                evals = []
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
                            d], d) in dictIntegrals_adaptive:  # if distributions are not the same in all dimensions, the dimension d has to be included
                            onedimensionalIntegral = dictIntegrals_adaptive[(point[d], tuple(neighbours), indices[n][d], d)]
                        else:
                            onedimensionalIntegral = computeIntegral(point[d],
                                                                     getNeighbours(combiObject,
                                                                                   gridPointCoordsAsStripes[d],
                                                                                   point[d]), function_info.distributions[d],
                                                                     pce_poly[d], d)
                            dictIntegrals_adaptive[(point[d], tuple(neighbours), indices[n][d], d)] = onedimensionalIntegral
                            # print("integral: ", onedimensionalIntegral)
                    else:
                        if (point[d], component_grid.levelvector[d], indices[n][
                            d], d) in dictIntegrals_not_adaptive:  # if distributions are not the same in all dimensions, the dimension d has to be included
                            onedimensionalIntegral = dictIntegrals_not_adaptive[
                                (point[d], component_grid.levelvector[d], indices[n][d], d)]
                        else:
                            integrand = lambda x: pce_poly[d](function_info.ppfs[d](x)) * hatfunction_level1D_position(x,
                                                                                                            component_grid.levelvector[
                                                                                                                d],
                                                                                                            point[d])


                            #onedimensionalIntegral = integrate.quad(integrand, max(point[d] - float(2) ** (-component_grid.levelvector[d]), 0),
                            #               min(1, point[d] + float(2) ** (-component_grid.levelvector[d])),
                            #               epsabs=1e-8)[0]
                            onedimensionalIntegral = integrate.fixed_quad(integrand, max(point[d] - float(2) ** (-component_grid.levelvector[d]), 0), point[d], n=6)[0] + integrate.fixed_quad(integrand, point[d], min(1, point[d] + float(2) ** (-component_grid.levelvector[d])))[0],
                                           #epsabs=1e-8)[0]
                            dictIntegrals_not_adaptive[
                                (point[d], component_grid.levelvector[d], indices[n][d], d)] = onedimensionalIntegral
                    product = product * onedimensionalIntegral / norms[n][d]
                integralCompGrid = integralCompGrid + product * evals[i]
            cn[n] = cn[n] + component_grid.coefficient * integralCompGrid
        print("cn for n = ", n, " is ", cn[n])
    gPCE = np.transpose(np.sum(pce_polys_d * cn.T, -1))
    expected = chaospy.E(gPCE, function_info.joint_distributions)
    variance = chaospy.Var(gPCE, function_info.joint_distributions)
    print("expected: ", expected, ", variance: ", variance)

    #first_order_sobol_indices = chaospy.Sens_m(gPCE, function_info.joint_distributions)
    first_order_sobol_indices = [0,0,0]
    #total_order_sobol_indices = chaospy.Sens_t(gPCE, joint)
    print("First order Sobol indices: ", first_order_sobol_indices)
    #print("Total order Sobol indices: ", total_order_sobol_indices)
    if store_result:
        entry = (number_points, float(expected), float(variance), *first_order_sobol_indices)
        storeResult(entry, 6 if adaptive else 5, function_info)

def plot_Errors_general(indices, function_info):
    colors = ['b-', 'g-', 'r-', 'c-', 'm-', 'y-', 'k-', 'b', 'b-']
    colors = ['blue', 'orange', 'red', 'black', 'pink', 'cyan', 'brown', 'gray', 'green', 'blue']
    linestyles = ['solid' for _ in range(9)] + ['dotted']
    pickle_errors_to_plot_in = open(function_info.path_Error, "rb")
    errors_to_plot = pickle.load(pickle_errors_to_plot_in)
    # mean
    for i in indices:
        xValues = []
        yValues = []
        errors = sorted(errors_to_plot[i], key=lambda x: x[0])
        for entry in errors:
         #   print("entry: ", entry)
            xValues.append(entry[0])
            relError = abs(entry[1] - function_info.mean_analytical) / function_info.mean_analytical
            yValues.append(relError)
        #print('xValues for index ', i, ': ', xValues, ', yValues: ', yValues)
        pyplot.plot(xValues, yValues, colors[i], linestyle=linestyles[i])
        pyplot.yscale('log')
        pyplot.xscale('log')
    names = ['PSP using Gaussian quadrature',
             'PSP with Combination Technique: Grid = Standard Combi, not adaptive',
             'PSP with Combination Technique: Grid = Trapezoidal, adaptive',
             'PSP with Combination Technique, Grid = Leja, not adaptive',
             'PSP with Combination Technique using Bsplines with degree 13, adaptive',
             'analytical Integration with Surrogate: Grid = Standard Combi, not adaptive',
             'analytical Integration with Surrogate: Grid = Trapezoidal, adaptive',
             'weighted adaptive trapezoidal grid, only mean and variance',
             'PSP using sparse Clenshaw Curtis quadrature',
             'PSP using Gaussian quadrature, including number of previous grids']
    pyplot.legend([names[i] for i in indices])
    pyplot.xlabel('function evaluations')
    pyplot.ylabel('relative error')
    pyplot.title('mean- relative error')
    pyplot.show()

    # variance
    for i in indices:
        xValues = []
        yValues = []
        errors = sorted(errors_to_plot[i], key=lambda x: x[0])
        for entry in errors:
         #   print("entry: ", entry)
            xValues.append(entry[0])
            relError = abs(entry[2] - function_info.variance_analytical) / function_info.variance_analytical
            yValues.append(relError)
        #print('xValues for index ', i, ': ', xValues, ', yValues: ', yValues)
        pyplot.plot(xValues, yValues, colors[i], linestyle=linestyles[i])
        pyplot.yscale('log')
        pyplot.xscale('log')
    pyplot.legend([names[i] for i in indices])
    pyplot.xlabel('function evaluations')
    pyplot.ylabel('relative error')
    pyplot.title('variance- relative error')
    pyplot.show()

    # first order sobol index S1
    indices = [i for i in indices if i != 7]
    for d in range(function_info.dim):
        for i in indices:
            xValues = []
            yValues = []
            errors = sorted(errors_to_plot[i], key=lambda x: x[0])
            for entry in errors:
                #print("entry: ", entry)
                xValues.append(entry[0])
                if function_info.first_order_sobol_indices[d] < 10**(-20):
                    yValues.append(entry[3+d])
                else:
                    relError = abs(entry[3+d] - function_info.first_order_sobol_indices[d]) / function_info.first_order_sobol_indices[d]
                    yValues.append(relError)
            print('xValues for index ', i, ': ', xValues, ', yValues: ', yValues)
            pyplot.plot(xValues, yValues, colors[i], linestyle=linestyles[i])
            pyplot.yscale('log')
            pyplot.xscale('log')
        pyplot.legend([names[i] for i in indices])
        pyplot.xlabel('function evaluations')
        if function_info.first_order_sobol_indices[d] < 10**(-20):
            pyplot.ylabel('absolute error')
            pyplot.title('first order sobol index S' + str(d+1) + ', absolute error')
        else:
            pyplot.ylabel('relative error')
            pyplot.title('first order sobol index S' + str(d+1) +', relative error')
        pyplot.show()


def add_category(function_info):
    pickle_errors_to_plot_in = open(function_info.path_Error, "rb")
    errors_to_plot = pickle.load(pickle_errors_to_plot_in)
    errors_to_plot.append([])
    pickle_errors_to_plot_out = open(function_info.path_Error, "wb")
    pickle.dump(errors_to_plot, pickle_errors_to_plot_out)
    pickle_errors_to_plot_out.close()

def initiate_function_infos():
    dict_function_infos = {}

    # genzFunctions
    coeffs = [float(1) for _ in range(3)]
    function_info_corner_peak = Function_Info(GenzCornerPeak(coeffs), 3, np.array([0]*3), np.array([1]*3), [chaospy.Uniform(0, 1) for d in range(3)], [("Uniform", 0, 1) for _ in range(3)], 'errors_to_plot_CornerPeak_.pickle',
                     GenzCornerPeak(coeffs).get_expectation(), GenzCornerPeak(coeffs).get_variance(), ppfs=[(lambda x: x) for d in range(3)], function_unitCube=GenzCornerPeak(coeffs))
    dict_function_infos['corner_peak'] = function_info_corner_peak


    coeffsProd = [float(3) for _ in range(3)]
    midpoint = [0.5 for _ in range(3)]
    function_info_product_peak = Function_Info(GenzProductPeak(coeffsProd, midpoint), 3, np.array([0]*3), np.array([1]*3), [chaospy.Uniform(0, 1) for d in range(3)], [("Uniform", 0, 1) for _ in range(3)], 'errors_to_plot_CornerProduct_.pickle',
                     GenzProductPeak(coeffsProd, midpoint).get_expectation(), GenzProductPeak(coeffsProd, midpoint).get_variance(), ppfs=[(lambda x: x) for d in range(3)], function_unitCube=GenzProductPeak(coeffsProd, midpoint))
    dict_function_infos['product_peak'] = function_info_product_peak

    function_info_discontinuous = Function_Info(GenzDiscontinious(coeffs, midpoint), 3, np.array([0] * 3), np.array([1] * 3),
                                  [chaospy.Uniform(0, 1) for d in range(3)], [("Uniform", 0, 1) for _ in range(3)],
                                  'errors_to_plot_Discontinuous_.pickle',
                                  GenzDiscontinious(coeffs, midpoint).get_expectation(), GenzDiscontinious(coeffs, midpoint).get_variance(),
                                  ppfs=[(lambda x: x) for d in range(3)], function_unitCube=GenzDiscontinious(coeffs, midpoint))
    dict_function_infos['discontinuous'] = function_info_discontinuous

    function_info_gfunction = Function_Info(FunctionG(3), 3, np.array([0] * 3), np.array([1] * 3),
                                  [chaospy.Uniform(0, 1) for d in range(3)], [("Uniform", 0, 1) for _ in range(3)],
                                  'errors_to_plot_FunctionG.pickle', FunctionG(3).get_expectation(),
                                  FunctionG(3).get_variance(),
                                  first_order_sobol_indices=[0.506250, 0.225033, 0.1265625],
                                  ppfs=[(lambda x: x) for _ in range(3)], function_unitCube=FunctionG(3))
    dict_function_infos['gfunction'] = function_info_gfunction

    function_info_ishigami = Function_Info(modelSolverFUNC(), 3, np.array([-math.pi] * 3), np.array([math.pi] * 3),
                                           [chaospy.Uniform(-math.pi, math.pi) for d in range(3)], [("Uniform", -math.pi, math.pi) for _ in range(3)], '../test/Ishigami_Variance_errors_to_plot.pickle',
                                           3.5, 13.844587940719254, first_order_sobol_indices=[0.31390519114781146, 0.4424111447900409, 0.0],
                                           ppfs=[(lambda x: 2 * math.pi * x - math.pi)for _ in range(3)], function_unitCube=modelSolverFUNC_nonlinear())
    dict_function_infos['ishigami'] = function_info_ishigami

    a = np.array([-2.5, -2, 5])
    b = np.array([2.5, 2, 15])
    mean = FunctionUQ().get_expectation(a, b)
    variance = FunctionUQ().get_variance(a, b)
    # print("mean FunctionG: ", function_info.mean_analytical, ", variance: ", function_info.variance_analytical)
    dissesForSparseSpace = [("Uniform", -2.5, 2.5), ("Uniform", -2, 2), ("Uniform", 5, 15)]
    function_info_functionUQ = Function_Info(FunctionUQ(), 3, a, b, [chaospy.Uniform(-2.5,2.5), chaospy.Uniform(-2,2), chaospy.Uniform(5, 15)],
                                  dissesForSparseSpace, 'errors_to_plot_FunctionUQ.pickle', mean, variance, first_order_sobol_indices=[0.6606445, 37.0142899, 6.21831285],
                                  ppfs=[(lambda x : -2.5 + 5*x), lambda x: -2 + 4*x, lambda x: 5 + 10 *x], function_unitCube=FunctionUQscaledIntoUnitcube(a,b))
    dict_function_infos['functionUQ'] = function_info_functionUQ

    #a2d = [0, 10]
    #b2d = [1, 1000]
    #function_info = Function_Info(testFunction(), 2, a2d, b2d, [chaospy.Uniform(0, 1), chaospy.Uniform(10, 1000)], [("Uniform", 0, math.pi) for d in range(3)], None, 0, 0, ppfs=[lambda x : (b2d[0] - a2d[0])*x, lambda x: a2d[1] + (b2d[1] - a2d[1])*x],
    #                              function_unitCube=testFunctionScaled(a2d, b2d))

    return dict_function_infos



if __name__ == "__main__":
    #0 = Pseudo_with_Chaospy_directly, Gauss
    #1 = Pseudo Spectral with Combination Technique: Grid = Standard Combi, not adaptive
    #2 = Pseudo Spectral with Combination Technique: Grid = Trapezoidal, adaptive
    #3 = Pseudo_Spectral with_Combination_Technique, Grid = Leja, not adaptive
    #4 = Pseudo Spectral with Combination Technique using Bsplines with degree 3, adaptive
    #5 = analytical Integration with Surrogate: Grid = Standard Combi, not adaptive, nonlinear transformation
    #6 = analytical Integration with Surrogate: Grid = Trapezoidal, adaptive, nonlinear transformation
    #7 = weighted adaptive trapezoidal grid, only mean and variance
    #8 = Psuedo_Spectral with Clensahw Curtis
    #9 = PSP with Gauss, include number or previous grids

    dict_function_infos = initiate_function_infos()
    function_info = dict_function_infos['discontinuous']

    initiate_Variance_Errors(function_info)
    add_category(function_info)
    delete_entries(3, function_info)
    for order in range(2,11):
        chaospy_directly('gauss', order, function_info , store_result=True)
    for order in range(2,7):
        Pseudo_Spectral_with_CombinationTechnique(function_info, gridName='Trapezoidal', adaptive=False, maximum_level=order, store_result=True)
    Pseudo_Spectral_with_CombinationTechnique(function_info, gridName='Trapezoidal', adaptive=True, max_evals=3000, store_result=True)
    for order in range(10, 11):
        Pseudo_Spectral_with_CombinationTechnique(function_info, gridName='Leja', adaptive=False, maximum_level=order, store_result=True)
    Pseudo_Spectral_with_CombinationTechnique(function_info, gridName='BSpline_p3', adaptive=True, max_evals=2000,
                                              store_result=True)
    for order in range(2,7):
        analytica_integration_with_surrogate(function_info, adaptive=False, store_result=True, maximum_level=order)
    analytica_integration_with_surrogate(function_info, adaptive=True, store_result=True, max_evals=600)
    only_mean_and_variance(3000, function_info, store_result=True)
    for order in range(2,4):
        chaospy_directly(('clenshaw_curtis'), order, function_info, store_result=True)
    for order in range(2,11):
        chaospy_directly('gauss', order, function_info , store_result=True, count_previous_grids=True)
    plot_Errors_general([0,1,2,3,5,6,7,8,9], function_info)