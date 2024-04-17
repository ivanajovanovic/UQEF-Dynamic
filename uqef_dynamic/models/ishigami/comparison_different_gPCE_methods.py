import math
import pickle
import numpy as np
import numpoly
import sparseSpACE
import chaospy
from sparseSpACE.spatiallyAdaptiveSingleDimension2 import *
from sparseSpACE.Function import *
from sparseSpACE.GridOperation import *
from sparseSpACE.spatiallyAdaptiveSingleDimension2 import *
from sparseSpACE.ErrorCalculator import *
from sparseSpACE.GridOperation import *
from matplotlib import pyplot


dim = 3
a_ishigami = 7
b_ishigami = 0.1
alpha = chaospy.Uniform(-math.pi, math.pi)
beta = chaospy.Uniform(-math.pi, math.pi)
gamma = chaospy.Uniform(-math.pi, math.pi)
distributions = [alpha, beta, gamma]
joint = chaospy.J(alpha, beta, gamma)

def ishigami(coordinates):
    x1, x2, x3 = coordinates
    return math.sin(x1) + a_ishigami * (math.sin(x2)) ** 2 + b_ishigami * x3 ** 4 * math.sin(x1)
class modelSolverFUNC(Function):
    def __init__(self):
        super().__init__()
    def output_length(self) -> int:
        return 1
    def eval(self, coordinates):
        return ishigami(coordinates)

def storeResult(entry, index):
    pickle_errors_to_plot_in = open("Ishigami_Variance_errors_to_plot.pickle", "rb")
    errors_to_plot = pickle.load(pickle_errors_to_plot_in)
    errors_to_plot[index].append(entry)
    pickle_errors_to_plot_out = open("Ishigami_Variance_errors_to_plot.pickle", "wb")
    pickle.dump(errors_to_plot, pickle_errors_to_plot_out)
    print("errors to plot for index", index, ': ', errors_to_plot[index])
    pickle_errors_to_plot_out.close()

def delete_entries(index):
    pickle_errors_to_plot_in = open("Ishigami_Variance_errors_to_plot.pickle", "rb")
    errors_to_plot = pickle.load(pickle_errors_to_plot_in)
    errors_to_plot[index] = []
    pickle_errors_to_plot_out = open("Ishigami_Variance_errors_to_plot.pickle", "wb")
    pickle.dump(errors_to_plot, pickle_errors_to_plot_out)
    pickle_errors_to_plot_out.close()

def chaospy_directly(order, store_result=False):

    gauss_quads = chaospy.generate_quadrature(order + 1, joint, rule='gaussian')
    #sparse_quads = chaospy.generate_quadrature(8, joint_isghigami, rule="clenshaw_curtis", sparse_utility=True)
    nodes, weights = gauss_quads

    gauss_evals = np.array([ishigami(node) for node in nodes.T])
    print("number of points", len(gauss_evals))

    number_points = len(gauss_evals)
    expansion = chaospy.generate_expansion(order, joint)

    # gauss_model_approxRet = chaospy.fit_quadrature(expansion, nodes, weights, gauss_evals, retall=True)
    gauss_model_approx = chaospy.fit_quadrature(expansion, nodes, weights, gauss_evals)

    expected = chaospy.E(gauss_model_approx, joint)
    variance = chaospy.Var(gauss_model_approx, joint)
    # print("coefficients: ", gauss_model_approxRet[1])
    print("Expected: ", expected, ",  Variance: ", variance)
    first_order_sobol_indices = chaospy.Sens_m(gauss_model_approx, joint)
    #total_order_sobol_indices = chaospy.Sens_t(gauss_model_approx, joint)
    print("First order Sobol indices: ", first_order_sobol_indices)
    #print("Total order Sobol indices: ", total_order_sobol_indices)
    entry = (number_points, float(expected), float(variance), *first_order_sobol_indices)
    if store_result:
        storeResult(entry, 0)

def Pseudo_Spectral_adative_CB_for_PCEcoefficients():
    a_ishigami = 7
    b_ishigami = 0.1
    def ishigamiH(coordinates):
        x1, x2, x3 = coordinates
        return math.sin(x1) + a_ishigami * (math.sin(x2)) ** 2 + b_ishigami * x3 ** 4 * math.sin(x1)
    import numpy as np
    class modelSolverFUNC(Function):
        def __init__(self):
            super().__init__()
        def output_length(self) -> int:
            return 1
        def eval(self, coordinates):
            return ishigamiH(coordinates)

    problem_function = modelSolverFUNC()
    dim = 3
    # distributions = [("Normal", 0.2, 1.0) for _ in range(dim)]
    distributions = [("Uniform", -math.pi, math.pi) for _ in range(dim)]
    a = np.array([-math.pi] * dim)
    b = np.array([math.pi] * dim)
    op = UncertaintyQuantification(problem_function, distributions, a, b)
    grid = GlobalTrapezoidalGridWeighted(a, b, op, boundary=True)
    op.set_grid(grid)

    polynomial_degree_max = 6

    # The grid needs to be refined for the PCE coefficient calculation
    op.set_PCE_Function(polynomial_degree_max)
    # op.pce_polys[1] = lambda x1,x2,x3: x1+x2+x3

    combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, margin=0.1, operation=op, norm=2, grid_surplusses=grid)
    lmax = 3
    error_operator = ErrorCalculatorSingleDimVolumeGuided()
    combiinstance.performSpatiallyAdaptiv(1, lmax,
                                          error_operator, tol=0, max_evaluations=800, do_plot=True)

    # Create the PCE approximation; it is saved internally in the operation
    op.calculate_PCE(None, combiinstance)

    # Calculate the expectation, variance and sobol indices with the PCE coefficients
    (E,), (Var,) = op.get_expectation_and_variance_PCE()
    print(f"E: {E}, PCE Var: {Var}")
    #print("First order Sobol indices:", op.get_first_order_sobol_indices())
    #print("Total order Sobol indices:", op.get_total_order_sobol_indices())

def Pseudo_Spectral_with_CombinationTechnique(gridName, adaptive, max_evals=None, maximum_level=None, store_result=False):
    f = modelSolverFUNC()
    a = np.empty(dim)
    a.fill(-math.pi)
    b = np.empty(dim)
    b.fill(math.pi)
    errorOperator = ErrorCalculatorSingleDimVolumeGuided()
    if adaptive:
        if gridName == 'BSpline_p3':
            grid = GlobalBSplineGrid(a=a, b=b, modified_basis=False, boundary=True, p=3)
        else:
            grid = GlobalTrapezoidalGrid(a=a, b=b, modified_basis=False, boundary=True)
        operation = Integration(f=f, grid=grid, dim=dim)
        combiObject = SpatiallyAdaptiveSingleDimensions2(np.ones(dim) * a, np.ones(dim) * b,
                                                                            margin=0.5,
                                                                            operation=operation)
        tolerance = 10 ** -5
        plotting = False
        combiObject.performSpatiallyAdaptiv(1, 4, errorOperator, tol=tolerance, do_plot=plotting,
                                                               max_evaluations=max_evals)
        combiObject.print_resulting_sparsegrid(markersize=10)
    else:
        if gridName == 'Trapezoidal':
            grid = TrapezoidalGrid(a=a, b=b, modified_basis=False, boundary=True)
        elif gridName == 'Leja':
            grid = LejaGrid(a=a, b=b, boundary=True)
        elif gridName == 'BSpline_p3':
            grid = BSplineGrid(a =a, b=b, boundary=True, p=3)
        operation = Integration(f=f, grid=grid, dim=dim)
        minimum_level = 1
        combiObject = StandardCombi(a, b, operation=operation)
        combiObject.perform_operation(minimum_level, maximum_level, f)
        #combiObject.print_resulting_combi_scheme()
        #combiObject.print_resulting_sparsegrid()

    number_points = combiObject.get_total_num_points()

    #gauss_quads = chaospy.generate_quadrature(11, joint, rule='clenshaw_curtis', sparse_utility=True)
    gauss_quads = chaospy.generate_quadrature(10, joint, rule='gaussian', sparse=False)
    nodes, weights = gauss_quads
    print("number quadrature points: ", len(nodes[0]))
    gauss_evals_Interpolation = combiObject(nodes.T)
    expansion = chaospy.generate_expansion(9, joint)
    gauss_model_approx = chaospy.fit_quadrature(expansion, nodes, weights, gauss_evals_Interpolation)

    expected = chaospy.E(gauss_model_approx, joint)
    variance = chaospy.Var(gauss_model_approx, joint)
    print("expectation = ", expected, ", variance = ", variance)
    first_order_sobol_indices = chaospy.Sens_m(gauss_model_approx, joint)
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
        storeResult(entry, index)


def initiate_Variance_Errors():
    #pickle_errors_to_plot_in = open("Ishigami_Variance_errors_to_plot.pickle", "rb")
    errors_to_plot = [[] for _ in range(7)]
    pickle_errors_to_plot_out = open("Ishigami_Variance_errors_to_plot.pickle", "wb")
    pickle.dump(errors_to_plot, pickle_errors_to_plot_out)
    pickle_errors_to_plot_out.close()

def analytica_integration_with_surrogate(adaptive=False, store_result=False, max_evals=None, maximum_level=None):
    ppfUniform = lambda x: 2 * math.pi * x - math.pi

    def model_solver_nonlinear(coordinates):
        x1, x2, x3 = coordinates
        return math.sin(ppfUniform(x1)) + a_ishigami * (math.sin(ppfUniform(x2))) ** 2 + b_ishigami * ppfUniform(
            x3) ** 4 * math.sin(alpha.ppf(x1))

    def standard_hatfunction1D(u):
        return max(1 - abs(u), 0)

    def hatfunction_level1D_position(u, l, x):
        return standard_hatfunction1D((u - x) / float(2) ** (-l))

    class modelSolverFUNC_nonlinear(Function):
        def __init__(self):
            super().__init__()

        def output_length(self) -> int:
            return 1

        def eval(self, coordinates):
            return model_solver_nonlinear(coordinates)

    f = modelSolverFUNC_nonlinear()
    a = np.empty(3)
    a.fill(0.0)
    b = np.empty(3)
    b.fill(1)

    if adaptive:
        errorOperator = ErrorCalculatorSingleDimVolumeGuided()
        grid = GlobalTrapezoidalGrid(a=a, b=b, modified_basis=False, boundary=True)
        operation = Integration(f=f, grid=grid, dim=dim)
        combiObject = StandardCombi(a, b, operation=operation)
        combiObject = SpatiallyAdaptiveSingleDimensions2(np.ones(dim) * a, np.ones(dim) * b, operation=operation)
        tolerance = 10 ** (-20)
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

        def computeIntegral(point_d, neighbours, distributions_d, pce_poly_1d):
            # print("left: ", neighbours[0], ", point: ", point_d, ", right:", neighbours[1])
            if point_d <= 0 or max(0, neighbours[0]) >= min(1, point_d):
                integralLeft = 0
            else:
                hatFunctionLeft = lambda x: (x - neighbours[0]) / (point_d - neighbours[0])
                integrandLeft = lambda x: pce_poly_1d(ppfUniform(x)) * hatFunctionLeft(x)
                integralLeft = integrate.quad(integrandLeft, max(0, neighbours[0]), min(1, point_d))[0]
            if point_d >= 1 or max(0, point_d) >= min(1, neighbours[1]):
                integralRight = 0
            else:
                hatFunctionRight = lambda x: (x - neighbours[1]) / (point_d - neighbours[1])
                integrandRight = lambda x: pce_poly_1d(ppfUniform(x)) * hatFunctionRight(x)
                integralRight = integrate.quad(integrandRight, max(0, point_d), min(1, neighbours[1]))[0]
            return integralLeft + integralRight
    else:
        grid = TrapezoidalGrid(a=a, b=b, modified_basis=False, boundary=True)
        operation = Integration(f=f, grid=grid, dim=dim)
        combiObject = StandardCombi(a, b, operation=operation)
        minimum_level = 1
        combiObject.perform_operation(minimum_level, maximum_level)
        dictIntegrals_not_adaptive = {}
        dictEvaluations = {}
        pickle_dict1D_integrals_in = open("dict1D_integrals.pickle", "rb")
        dictIntegrals_not_adaptive = pickle.load(pickle_dict1D_integrals_in)
        pickle_dict_evaluations_in = open("dict_evaluations.pickle", "rb")
        dictEvaluations = pickle.load(pickle_dict_evaluations_in)
    print("Sparse Grid:")
    combiObject.print_resulting_sparsegrid(markersize=10)

    # extract the onedimensional orthogonal polynomials and order them in the same way as chaospy does
    # has to be modified if the distributions are not the same for all dimensions
    number_points = combiObject.get_total_num_points()
    polynomial_degrees = 9
    expansion = chaospy.generate_expansion(polynomial_degrees, joint)
    #while len(expansion) < 0.04 * number_points:
    #    polynomial_degrees = polynomial_degrees + 1
    #    expansion = chaospy.generate_expansion(polynomial_degrees, joint)
    #    print("order: ", polynomial_degrees, ", len(expansion): ", len(expansion))
    #print("order: ", polynomial_degrees)
    pce_polys_d, pce_polys_norms_d = chaospy.orth_ttr(polynomial_degrees, joint, retall=True)
    pce_polys1d, pce_polys1d_norms = chaospy.orth_ttr(polynomial_degrees, alpha, retall=True)
    indices = numpoly.glexindex(start=0, stop=polynomial_degrees + 1, dimensions=len(joint),
                                graded=True, reverse=True,
                                cross_truncation=1.0)
    norms = [None] * len(indices)
    polys = [None] * len(indices)
    for i in range(len(indices)):
        polys[i] = [pce_polys1d[indices[i][d]] for d in range(dim)]
        norms[i] = [pce_polys1d_norms[indices[i][d]] for d in range(dim)]

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
                for d in range(0, dim):
                    if adaptive:
                        neighbours = getNeighbours(combiObject, gridPointCoordsAsStripes[d], point[d])
                        if (point[d], tuple(neighbours), indices[n][
                            d]) in dictIntegrals_adaptive:  # if distributions are not the same in all dimensions, the dimension d has to be included
                            onedimensionalIntegral = dictIntegrals_adaptive[(point[d], tuple(neighbours), indices[n][d])]
                        else:
                            onedimensionalIntegral = computeIntegral(point[d],
                                                                     getNeighbours(combiObject,
                                                                                   gridPointCoordsAsStripes[d],
                                                                                   point[d]), distributions[d],
                                                                     pce_poly[d])
                            dictIntegrals_adaptive[(point[d], tuple(neighbours), indices[n][d])] = onedimensionalIntegral
                            # print("integral: ", onedimensionalIntegral)
                    else:
                        if (point[d], component_grid.levelvector[d], indices[n][
                            d]) in dictIntegrals_not_adaptive:  # if distributions are not the same in all dimensions, the dimension d has to be included
                            onedimensionalIntegral = dictIntegrals_not_adaptive[
                                (point[d], component_grid.levelvector[d], indices[n][d])]
                        else:
                            integrand = lambda x: pce_poly[d](ppfUniform(x)) * hatfunction_level1D_position(x,
                                                                                                            component_grid.levelvector[
                                                                                                                d],
                                                                                                            point[d])
                            onedimensionalIntegral = \
                            integrate.quad(integrand, max(point[d] - float(2) ** (-component_grid.levelvector[d]), 0),
                                           min(1, point[d] + float(2) ** (-component_grid.levelvector[d])),
                                           epsabs=1e-8)[0]
                            dictIntegrals_not_adaptive[
                                (point[d], component_grid.levelvector[d], indices[n][d])] = onedimensionalIntegral
                    product = product * onedimensionalIntegral / norms[n][d]
                integralCompGrid = integralCompGrid + product * evals[i]
            cn[n] = cn[n] + component_grid.coefficient * integralCompGrid
        print("cn for n = ", n, " is ", cn[n])
    gPCE = np.transpose(np.sum(pce_polys_d * cn.T, -1))
    expected = chaospy.E(gPCE, joint)
    variance = chaospy.Var(gPCE, joint)
    print("expected: ", expected, ", variance: ", variance)

    first_order_sobol_indices = chaospy.Sens_m(gPCE, joint)
    #total_order_sobol_indices = chaospy.Sens_t(gPCE, joint)
    print("First order Sobol indices: ", first_order_sobol_indices)
    #print("Total order Sobol indices: ", total_order_sobol_indices)
    if store_result:
        entry = (number_points, float(expected), float(variance), *first_order_sobol_indices)
        storeResult(entry, 6 if adaptive else 5)


def plot_Errors(indexes):
    mean = 3.5
    variance_actual = (a_ishigami ** 2) / 8 + (b_ishigami * math.pi ** 4) / 5 + (b_ishigami ** 2 * math.pi ** 8) / 18 + 1 / 2
    print("Variance_actual", variance_actual)

    # first order sobol indices
    V1 = 0.5 * (1 + (b_ishigami * math.pi ** 4) / 5) ** 2
    V2 = (a_ishigami**2) / 9
    V3 = 0
    S1 = V1 / variance_actual
    S2 = V2 / variance_actual
    S3 = V3 / variance_actual
    print("first order Sobol indices: ", S1, S2, S3)

    # total effect sobol indices
    Vt1 = 0.5 * (1 + (b_ishigami * math.pi ** 4) / 5) ** 2 + (8 * b_ishigami ** 2 * math.pi ** 8) / 225
    Vt2 = a_ishigami ** 2 / 8
    Vt3 = (8 * b_ishigami ** 2 * math.pi ** 8) / 225
    St1 = Vt1 / variance_actual
    St2 = Vt2 / variance_actual
    St3 = Vt3 / variance_actual
    print("total effect Sobol indices: ", St1, St2, St3)

    pickle_errors_to_plot_in = open("Ishigami_Variance_errors_to_plot.pickle", "rb")
    errors_to_plot = pickle.load(pickle_errors_to_plot_in)

    #mean
    for i in indexes:
        xValues = []
        yValues = []
        errors = sorted(errors_to_plot[i], key=lambda x: x[0])
        for entry in errors:
            print("entry: ", entry)
            xValues.append(entry[0])
            relError = abs(entry[1] - mean) / mean
            yValues.append(relError)
        print('xValues for index ', i,': ', xValues, ', yValues: ', yValues)
        pyplot.plot(xValues, yValues)
        pyplot.yscale('log')
        pyplot.xscale('log')
    names = ['Pseudo_with_Chaospy_directly',
             'Pseudo Spectral with Combination Technique: Grid = Standard Combi, not adaptive',
             'Pseudo Spectral with Combination Technique: Grid = Trapezoidal, adaptive',
             'Pseudo_Spectral_with_Combination_Technique, Grid = Leja, not adaptive',
             'Pseudo_Spectral_with_Combination_Technique using Bsplines with degree 3, adaptive',
             'analytical Integration with Surrogate: Grid = Standard Combi, not adaptive',
             'analytical Integration with Surrogate: Grid = Trapezoidal, adaptive']

    pyplot.legend([names[i] for i in indexes])
    pyplot.xlabel('function evaluations')
    pyplot.ylabel('relative error')
    pyplot.title('mean ishigami - relative error')
    pyplot.show()

    #variance
    for i in indexes:
        xValues = []
        yValues = []
        errors = sorted(errors_to_plot[i], key=lambda x: x[0])
        for entry in errors:
            print("entry: ", entry)
            xValues.append(entry[0])
            relError = abs(entry[2] - variance_actual) / variance_actual
            yValues.append(relError)
        print('xValues for index ', i,': ', xValues, ', yValues: ', yValues)
        pyplot.plot(xValues, yValues)
        pyplot.yscale('log')
        pyplot.xscale('log')
    pyplot.xlabel('function evaluations')
    pyplot.ylabel('relative error')
    pyplot.title('vatiance ishigami - relative error')
    pyplot.show()

    # first order sobol index S1
    for i in indexes:
        xValues = []
        yValues = []
        errors = sorted(errors_to_plot[i], key=lambda x: x[0])
        for entry in errors:
            print("entry: ", entry)
            xValues.append(entry[0])
            relError = abs(entry[3] - S1) / S1
            yValues.append(relError)
        print('xValues for index ', i, ': ', xValues, ', yValues: ', yValues)
        pyplot.plot(xValues, yValues)
        pyplot.yscale('log')
        pyplot.xscale('log')
    pyplot.xlabel('function evaluations')
    pyplot.ylabel('relative error')
    pyplot.title('first order sobol index S1,  ishigami - relative error')
    pyplot.show()

    # first order sobol index S2
    for i in indexes:
        xValues = []
        yValues = []
        errors = sorted(errors_to_plot[i], key=lambda x: x[0])
        for entry in errors:
            print("entry: ", entry)
            xValues.append(entry[0])
            relError = abs(entry[4] - S2) / S2
            yValues.append(relError)
        print('xValues for index ', i, ': ', xValues, ', yValues: ', yValues)
        pyplot.plot(xValues, yValues)
        pyplot.yscale('log')
        pyplot.xscale('log')
    pyplot.xlabel('function evaluations')
    pyplot.ylabel('relative error')
    pyplot.title('first order sobol index S2,  ishigami - relative error')
    pyplot.show()

    # first order sobol index S3
    for i in indexes:
        xValues = []
        yValues = []
        errors = sorted(errors_to_plot[i], key=lambda x: x[0])
        for entry in errors:
            print("entry: ", entry)
            xValues.append(entry[0])
            relError = abs(entry[5] - S3)
            yValues.append(relError)
        print('xValues for index ', i, ': ', xValues, ', yValues: ', yValues)
        pyplot.plot(xValues, yValues)
        #pyplot.yscale('log')
        #pyplot.xscale('log')
    pyplot.xlabel('function evaluations')
    pyplot.ylabel('absolute error')
    pyplot.title('first order sobol index S3,  ishigami - absolute error')
    pyplot.show()




if __name__ == "__main__":
    #0 = Pseudo_with_Chaospy_directly
    #1 = Pseudo Spectral with Combination Technique: Grid = Standard Combi, not adaptive
    #2 = Pseudo Spectral with Combination Technique: Grid = Trapezoidal, adaptive
    #3 = Pseudo_Spectral with_Combination_Technique, Grid = Leja, not adaptive
    #4 = Pseudo_Spectral with_Combination_Technique using Bsplines with degree 3, adaptive
    #5 = analytical Integration with Surrogate: Grid = Standard Combi, not adaptive, nonlinear transformation
    #6 = analytical Integration with Surrogate: Grid = Trapezoidal, adaptive, nonlinear transformation

    initiate_Variance_Errors()
    delete_entries(2)
    chaospy_directly(10, store_result=True)
    Pseudo_Spectral_with_CombinationTechnique(gridName='Trapezoidal', adaptive=False, maximum_level=7, store_result=True)
    Pseudo_Spectral_with_CombinationTechnique(gridName='Trapezoidal', adaptive=True, max_evals=700, store_result=True)
    Pseudo_Spectral_with_CombinationTechnique(gridName='Leja', adaptive=False, maximum_level=6, store_result=True)
    analytica_integration_with_surrogate(adaptive=False, store_result=True, maximum_level=7)
    analytica_integration_with_surrogate(adaptive=True, store_result=True, max_evals=3500)
    plot_Errors([0, 1, 2, 3, 4, 5, 6])

