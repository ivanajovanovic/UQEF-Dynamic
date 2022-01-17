import math
import json
import numpy as np
import numpoly
import pathlib
import pickle
import scipy
import scipy.integrate as integrate

import itertools
# time measure
import time

from sparseSpACE.Function import *
from sparseSpACE.spatiallyAdaptiveSingleDimension2 import *
from sparseSpACE.ErrorCalculator import *
from sparseSpACE.GridOperation import *
from sparseSpACE.StandardCombi import *
from sparseSpACE.Integrator import *

import chaospy as cp
import uqef

import IshigamiModel

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()


def ishigami(coordinates, a_model_param=7, b_model_param=0.1):
    x1, x2, x3 = coordinates
    return math.sin(x1) + a_model_param * (math.sin(x2)) ** 2 + b_model_param * x3 ** 4 * math.sin(x1)


class IshigamiFunctionSimple(Function):
    def __init__(self):
        super().__init__()

    def output_length(self) -> int:
        return 1

    def eval(self, coordinates):
        # TODO make it accept a_model_param and b_model_param
        return ishigami(coordinates)


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


def get_analytical_sobol_indices(a_model_param=7, b_model_param=0.1):
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


def _compute_gpce_chaospy_ishigami(a_model_param, b_model_param, labels, my_model, joint, q, p, rule, sparse, outputModelDir,
                                   can_model_evaluate_all_vector_nodes=True):
    # TODO Introduction of this function has slowed down the execution!
    quads = cp.generate_quadrature(q, joint, rule=rule, sparse=sparse)
    nodes, weights = quads

    # evaluate surrogate
    if can_model_evaluate_all_vector_nodes:
        evaluations = my_model(nodes.T)
    else:
        evaluations = np.array([my_model(node) for node in nodes.T])

    expansion = cp.generate_expansion(p, joint)
    gPCE = cp.fit_quadrature(expansion, nodes, weights, evaluations)
    # TODO add point collocation with regression!

    expectedInterp = cp.E(gPCE, joint)
    varianceInterp = cp.Var(gPCE, joint)
    first_order_sobol_indices = cp.Sens_m(gPCE, joint)
    total_order_sobol_indices = cp.Sens_t(gPCE, joint)

    print("expectation = ", expectedInterp, ", variance = ", varianceInterp)
    print("First order Sobol indices: ", first_order_sobol_indices)
    print("Total order Sobol indices: ", total_order_sobol_indices)

    sobol_m_analytical, sobol_t_analytical = get_analytical_sobol_indices(a_model_param, b_model_param)

    # Sobol_t_error = sobol_t_analytical - total_order_sobol_indices
    # Sobol_m_error = sobol_m_analytical - first_order_sobol_indices
    # print("Sobol_t_error: {}".format(Sobol_t_error, ".6f"))
    # # print(f"Sobol Total Error = {Sobol_t_error:.6f} \n")
    # print("Sobol_m_error: {}".format(Sobol_m_error, ".6f"))
    # # print(f"Sobol Main Error = {Sobol_m_error:.6f} \n")

    sobol_t_error = np.empty(len(labels), dtype=np.float64)
    for i in range(len(labels)):
        # print(f"Sobol Total Simulation = {total_order_sobol_indices[i][0]} \n")
        # print(f"Sobol Total Analytical = {sobol_t_analytical[i]:.6f} \n")
        sobol_t_error[i] = sobol_t_analytical[i] - total_order_sobol_indices[i][0]
        print(f"Sobol Total Error = {sobol_t_error[i]:.6f} \n")
    #
    sobol_m_error = np.empty(len(labels), dtype=np.float64)
    for i in range(len(labels)):
        # print(f"Sobol Main Simulation = {first_order_sobol_indices[i][0]} \n")
        # print(f"Sobol Main Analytical = {sobol_m_analytical[i]:.6f} \n")
        sobol_m_error[i] = sobol_m_analytical[i] - first_order_sobol_indices[i][0]
        print(f"Sobol Main Error = {sobol_m_error[i]:.6f} \n")

    #####################################
    # Saving
    #####################################
    # gPCE = op.get_gPCE()
    fileName = f"gpce.pkl"
    gpceFileName = str(outputModelDir / fileName)
    with open(gpceFileName, 'wb') as handle:
        pickle.dump(gPCE, handle, protocol=pickle.HIGHEST_PROTOCOL)

    fileName = f"pce_polys.pkl"
    pcePolysFileName = str(outputModelDir / fileName)
    with open(pcePolysFileName, 'wb') as handle:
        pickle.dump(expansion, handle, protocol=pickle.HIGHEST_PROTOCOL)

    temp = f"results.txt"
    save_file = outputModelDir / temp
    fp = open(save_file, "w")
    fp.write(f'E: {expectedInterp},\n Var: {varianceInterp}, \n '
             f'First order Sobol indices: {first_order_sobol_indices} \n; '
             f'Total order Sobol indices: {total_order_sobol_indices} \n')
    fp.close()


def compute_gpce_chaospy(a_model_param=7, b_model_param=0.1, outputModelDir="./", rule='gaussian', sparse=True, q=7, p=6):
    x1 = cp.Uniform(-math.pi, math.pi)
    x2 = cp.Uniform(-math.pi, math.pi)
    x3 = cp.Uniform(-math.pi, math.pi)
    joint_isghigami = cp.J(x1, x2, x3)

    labels = [param_name.strip() for param_name in ["x1", "x2", "x3"]]

    _compute_gpce_chaospy_ishigami(a_model_param=a_model_param, b_model_param=b_model_param, labels=labels,
                                   my_model=ishigami, joint=joint_isghigami,
                                   q=q, p=p, rule=rule, sparse=sparse,
                                   outputModelDir=outputModelDir, can_model_evaluate_all_vector_nodes=False)


def compute_surrogate_sparsespace_and_gpce(a_model_param=7, b_model_param=0.1, modified_basis=False,
                                           spatiallyAdaptive=True,
                                           plotting=True, outputModelDir="./",
                                           lmax=2, max_evals=2000, tolerance=10 ** -5,
                                           rule='gaussian', sparse=False, q=7, p=6):
    """
    Var 2 - Compute gPCE coefficients by integrating the (SG) surrogate
    """
    x1 = cp.Uniform(-math.pi, math.pi)
    x2 = cp.Uniform(-math.pi, math.pi)
    x3 = cp.Uniform(-math.pi, math.pi)
    joint = cp.J(x1, x2, x3)

    f = IshigamiFunctionSimple()
    dim = 3
    a = np.array([-3.2, -3.2, -3.2])
    b = np.array([3.2, 3.2, 3.2])

    labels = [param_name.strip() for param_name in ["x1", "x2", "x3"]]
    #####################################
    if modified_basis:
        grid = GlobalTrapezoidalGrid(a=a, b=b, modified_basis=True, boundary=False)
        # grid = GlobalTrapezoidalGridWeighted(a, b, boundary=False, modified_basis=True)
    else:
        grid = GlobalTrapezoidalGrid(a=a, b=b, modified_basis=False, boundary=True)
        # grid = GlobalTrapezoidalGridWeighted(a, b, boundary=True)

    operation = Integration(f=f, grid=grid, dim=dim)

    if spatiallyAdaptive:
        errorOperator = ErrorCalculatorSingleDimVolumeGuided()
        combiinstance = SpatiallyAdaptiveSingleDimensions2(np.ones(dim) * a, np.ones(dim) * b,
                                                                            operation=operation)
    else:
        combiinstance = StandardCombi(np.ones(dim) * a, np.ones(dim) * b, operation=operation, norm=2)

    if plotting:
        plot_file = str(outputModelDir / "output.png")
        filename_contour_plot = str(outputModelDir / "output_contour_plot.png")
        filename_combi_scheme_plot = str(outputModelDir / "output_combi_scheme.png")
        filename_refinement_graph = str(outputModelDir / "output_refinement_graph.png")
        filename_sparse_grid_plot = str(outputModelDir / "output_sg_graph.png")
        combiinstance.filename_contour_plot = filename_contour_plot
        combiinstance.filename_refinement_graph = filename_refinement_graph
        combiinstance.filename_combi_scheme_plot = filename_combi_scheme_plot
        combiinstance.filename_sparse_grid_plot = filename_sparse_grid_plot

    if spatiallyAdaptive:
        combiinstance.performSpatiallyAdaptiv(1, lmax, errorOperator, tol=tolerance, do_plot=plotting,
                                                               max_evaluations=max_evals)
    else:
        combiinstance.perform_operation(1, lmax)

    if plotting:
        combiinstance.print_resulting_sparsegrid(markersize=10)

    #####################################
    # TODO From this point on SparseSpACE can again come to play, instead of chaospy quad - think if it makes sense!?
    # _compute_gpce_chaospy_ishigami(a_model_param=a_model_param, b_model_param=b_model_param, labels=labels,
    #                                my_model=combiinstance, joint=joint,
    #                                q=q, p=p, rule=rule, sparse=sparse,
    #                                outputModelDir=outputModelDir, can_model_evaluate_all_vector_nodes=True)

    quads = cp.generate_quadrature(q, joint, rule=rule, sparse=sparse)
    nodes, weights = quads

    # evaluate surrogate
    evaluations = combiinstance(nodes.T)

    expansion = cp.generate_expansion(p, joint)
    gPCE = cp.fit_quadrature(expansion, nodes, weights, evaluations)
    # TODO add point collocation with regression!

    expectedInterp = cp.E(gPCE, joint)
    varianceInterp = cp.Var(gPCE, joint)
    first_order_sobol_indices = cp.Sens_m(gPCE, joint)
    total_order_sobol_indices = cp.Sens_t(gPCE, joint)

    print("expectation = ", expectedInterp, ", variance = ", varianceInterp)
    print("First order Sobol indices: ", first_order_sobol_indices)
    print("Total order Sobol indices: ", total_order_sobol_indices)

    sobol_m_analytical, sobol_t_analytical = get_analytical_sobol_indices(a_model_param, b_model_param)

    # Sobol_t_error = sobol_t_analytical - total_order_sobol_indices
    # Sobol_m_error = sobol_m_analytical - first_order_sobol_indices
    # print("Sobol_t_error: {}".format(Sobol_t_error, ".6f"))
    # # print(f"Sobol Total Error = {Sobol_t_error:.6f} \n")
    # print("Sobol_m_error: {}".format(Sobol_m_error, ".6f"))
    # # print(f"Sobol Main Error = {Sobol_m_error:.6f} \n")

    sobol_t_error = np.empty(len(labels), dtype=np.float64)
    for i in range(len(labels)):
        # print(f"Sobol Total Simulation = {total_order_sobol_indices[i][0]} \n")
        # print(f"Sobol Total Analytical = {sobol_t_analytical[i]:.6f} \n")
        sobol_t_error[i] = sobol_t_analytical[i] - total_order_sobol_indices[i][0]
        print(f"Sobol Total Error = {sobol_t_error[i]:.6f} \n")
    #
    sobol_m_error = np.empty(len(labels), dtype=np.float64)
    for i in range(len(labels)):
        # print(f"Sobol Main Simulation = {first_order_sobol_indices[i][0]} \n")
        # print(f"Sobol Main Analytical = {sobol_m_analytical[i]:.6f} \n")
        sobol_m_error[i] = sobol_m_analytical[i] - first_order_sobol_indices[i][0]
        print(f"Sobol Main Error = {sobol_m_error[i]:.6f} \n")

    #####################################
    # Saving
    #####################################
    # gPCE = op.get_gPCE()
    fileName = f"gpce.pkl"
    gpceFileName = str(outputModelDir / fileName)
    with open(gpceFileName, 'wb') as handle:
        pickle.dump(gPCE, handle, protocol=pickle.HIGHEST_PROTOCOL)

    fileName = f"pce_polys.pkl"
    pcePolysFileName = str(outputModelDir / fileName)
    with open(pcePolysFileName, 'wb') as handle:
        pickle.dump(expansion, handle, protocol=pickle.HIGHEST_PROTOCOL)

    temp = f"results.txt"
    save_file = outputModelDir / temp
    fp = open(save_file, "w")
    fp.write(f'E: {expectedInterp},\n Var: {varianceInterp}, \n '
             f'First order Sobol indices: {first_order_sobol_indices} \n; '
             f'Total order Sobol indices: {total_order_sobol_indices} \n')
    fp.close()


def compute_surrogate_sparsespace_and_gpce_analytically(a_model_param=7, b_model_param=0.1, modified_basis=False,
                                           spatiallyAdaptive=True,
                                           plotting=True, outputModelDir="./",
                                           lmax=2, max_evals=2000, tolerance=10 ** -5,
                                           rule='gaussian', sparse=False, q=7, p=6):
    """
    Var 2.2 (For Markus this is var 3) - Compute gPCE coefficients by integrating the (SG) surrogate analytically
    The gPCE coefficients are calulcated as follows:
    $c_n = \int_{(0,1)^d} f_{interp}^{nonlin}(T_{nonlin}(u)\phi_n(T_{nonlin}(u))du =
     \sum_{l, i} \alpha_{l,i}\prod_{j=1}^{d}\int_0^1\phi_j(F^{-1}(u_j))\psi_{l_j, i_j}(u_j)du_j$
     where $T_{nonlin} = (F^{-1}(u_1),...,F^{-1}(u_d))$
    So far this only works with the Standard Combination Technique.
    """
    # TODO
    def standard_hatfunction1D(u):
        return max(1 - abs(u), 0)

    def hatfunction_level1D_position(u, l, x):
        return standard_hatfunction1D((u - x) / float(2) ** (-l))

    x1 = cp.Uniform(-math.pi, math.pi)
    x2 = cp.Uniform(-math.pi, math.pi)
    x3 = cp.Uniform(-math.pi, math.pi)
    distributions = [x1, x2, x3]
    joint = cp.J(x1, x2, x3)

    f = IshigamiFunctionSimple()
    dim = 3
    a = np.array([-3.2, -3.2, -3.2])
    b = np.array([3.2, 3.2, 3.2])
    # a = np.array([0.0, 0.0, 0.0])
    # b = np.array([1, 1, 1])

    labels = [param_name.strip() for param_name in ["x1", "x2", "x3"]]
    #####################################
    if modified_basis:
        grid = GlobalTrapezoidalGrid(a=a, b=b, modified_basis=True, boundary=False)
        # grid = GlobalTrapezoidalGridWeighted(a, b, boundary=False, modified_basis=True)
    else:
        grid = GlobalTrapezoidalGrid(a=a, b=b, modified_basis=False, boundary=True)
        # grid = GlobalTrapezoidalGridWeighted(a, b, boundary=True)

    operation = Integration(f=f, grid=grid, dim=dim)

    if spatiallyAdaptive:
        combiinstance = SpatiallyAdaptiveSingleDimensions2(np.ones(dim) * a, np.ones(dim) * b,
                                                                            operation=operation)
    else:
        combiinstance = StandardCombi(a, b, operation=operation, norm=2)

    if plotting:
        plot_file = str(outputModelDir / "output.png")
        filename_contour_plot = str(outputModelDir / "output_contour_plot.png")
        filename_combi_scheme_plot = str(outputModelDir / "output_combi_scheme.png")
        filename_refinement_graph = str(outputModelDir / "output_refinement_graph.png")
        filename_sparse_grid_plot = str(outputModelDir / "output_sg_graph.png")
        combiinstance.filename_contour_plot = filename_contour_plot
        combiinstance.filename_refinement_graph = filename_refinement_graph
        combiinstance.filename_combi_scheme_plot = filename_combi_scheme_plot
        combiinstance.filename_sparse_grid_plot = filename_sparse_grid_plot

    if spatiallyAdaptive:
        errorOperator = ErrorCalculatorSingleDimVolumeGuided()
        combiinstance.performSpatiallyAdaptiv(1, lmax, errorOperator, tol=tolerance, do_plot=plotting,
                                                               max_evaluations=max_evals)
    else:
        # minimum_level = 1
        # maximum_level = 7
        # combiObject.perform_operation(minimum_level, maximum_level)
        combiinstance.perform_operation(1, lmax)

    if plotting:
        print("Sparse Grid:")
        combiinstance.print_resulting_sparsegrid(markersize=10)

    # extract the one-dimensional orthogonal polynomials and order them in the same way as chaospy does
    # has to be modified if the distributions are not the same for all dimensions
    pce_polys3d, pce_polys_norms3d = cp.orth_ttr(p, joint, retall=True)
    pce_polys1d, pce_polys1d_norms = cp.orth_ttr(p, x1, retall=True)
    indices = numpoly.glexindex(start=0, stop=p + 1, dimensions=len(joint),
                                graded=True, reverse=True,
                                cross_truncation=1.0)
    norms = [None] * len(indices)
    polys = [None] * len(indices)
    for i in range(len(indices)):
        polys[i] = [pce_polys1d[indices[i][d]] for d in range(dim)]
        norms[i] = [pce_polys1d_norms[indices[i][d]] for d in range(dim)]

    # TODO - store the one dimensional integrals and the function evaluations
    fileName = f"dict1D_integrals.pickle"
    pickle_integrals_in = str(outputModelDir / fileName)
    with open(pickle_integrals_in, 'rb') as handle:
        dictIntegrals = pickle.load(pickle_integrals_in)
    fileName = f"dict_evaluations.pickle"
    pickle_evaluations_in = str(outputModelDir / fileName)
    with open(pickle_evaluations_in, 'rb') as handle:
        dictEvaluations = pickle.load(pickle_evaluations_in)

    # compute the coefficients cn, takes long for the first coefficient
    counterIntegrals = 0
    cn = np.zeros(len(polys))
    for n, pce_poly in enumerate(polys):
        for component_grid in combiinstance.scheme:
            points = combiinstance.get_points_component_grid(component_grid.levelvector)
            evals = []
            keyLevelvector = tuple(component_grid.levelvector.tolist())
            if keyLevelvector in dictEvaluations:
                evals = dictEvaluations[keyLevelvector]
            else:
                evals = combiinstance(points)
                dictEvaluations[keyLevelvector] = evals
            integralCompGrid = 0
            for i, point in enumerate(points):
                product = 1
                for d in range(0, dim):
                    if (point[d], component_grid.levelvector[d], indices[n][d]) in dict:
                        # if distributions are not the same in all dimensions, the dimension d has to be included
                        onedimensionalIntegral = dictIntegrals[(point[d], component_grid.levelvector[d], indices[n][d])]
                    else:
                        integrand = lambda x: pce_poly[d](distributions[d].ppf(x)) * \
                                              hatfunction_level1D_position(x, component_grid.levelvector[d], point[d])
                        onedimensionalIntegral = \
                        integrate.quad(integrand, max(point[d] - float(2) ** (-component_grid.levelvector[d]), 0),
                                       min(1, point[d] + float(2) ** (-component_grid.levelvector[d])), epsabs=1e-8)[0]
                        counterIntegrals += 1
                        dictIntegrals[(point[d], component_grid.levelvector[d], indices[n][d])] = onedimensionalIntegral
                    product = product * onedimensionalIntegral / norms[n][d]
                integralCompGrid = integralCompGrid + evals[i] * product
            # print("integralCompgrid for grid ", component_grid.levelvector, " is ", integralCompGrid)
            cn[n] = cn[n] + component_grid.coefficient * integralCompGrid
            # print("CounterIntegrals: ", counterIntegrals)
            counterIntegrals = 0
        print("cn for n = ", n, " is ", cn[n])

    gPCE = np.transpose(np.sum(pce_polys3d * cn.T, -1))
    exp = cp.E(gPCE, joint)
    var = cp.Var(gPCE, joint)
    print("expected: ", exp, ", variance: ", var)
    first_order_sobol_indices = cp.Sens_m(gPCE, joint)
    total_order_sobol_indices = cp.Sens_t(gPCE, joint)
    print("First order Sobol indices: ", first_order_sobol_indices)
    print("Total order Sobol indices: ", total_order_sobol_indices)

    pickle_out_integrals = open("dict1D_integrals.pickle", "wb")
    pickle.dump(dictIntegrals, pickle_out_integrals)
    pickle_out_integrals.close()
    pickle_out_evaluations = open("dict_evaluations.pickle", "wb")
    pickle.dump(dictEvaluations, pickle_out_evaluations)
    pickle_out_evaluations.close()


def compute_gpce_sparsespace(build_sg_for_e_and_var=True, modified_basis=False,
                             parallelIntegrator=False, spatiallyAdaptive=True,
                             plotting=False, outputModelDir="./", lmax=2, max_evals=2000, tolerance=10 ** -5, p=6):
    """
    If build_sg_for_e_and_var == True
    ---> Build one SG surrogate to approximate E and Var
    Else
    --> Build one SG surrogate to approximate all N coefficients of the gPCE expansion
    """
    config_file = pathlib.Path(
        '/work/ga45met/mnt/linux_cluster_2/Larsim-UQ/configurations/configuration_ishigami.json')

    with open(config_file) as f:
        configuration_object = json.load(f)
    #####################################
    try:
        params = configuration_object["parameters"]
    except KeyError as e:
        print(f"Ishigami Statistics: parameters key does "
              f"not exists in the configurationObject{e}")
        raise
    param_names = [param["name"] for param in params]
    labels = [param_name.strip() for param_name in param_names]
    dim = len(params)  # TODO take only uncertain paramters into account
    distributions = [(param["distribution"], param["lower"], param["upper"]) for param in params]

    a = np.array([param["lower"] for param in params])
    b = np.array([param["upper"] for param in params])
    #####################################
    polynomial_degree_max = p

    problem_function = IshigamiFunction(configurationObject=configuration_object, dim=dim)
    timesteps = 1
    op = UncertaintyQuantification(problem_function, distributions, a, b)

    if modified_basis:
        grid = GlobalTrapezoidalGridWeighted(a, b, op, boundary=False, modified_basis=True)
    else:
        grid = GlobalTrapezoidalGridWeighted(a, b, op, boundary=True)

    if parallelIntegrator:
        grid.integrator = IntegratorParallelArbitraryGrid(grid)  # TODO

    # The grid initialization requires the weight functions from the
    # operation; since currently the adaptive refinement takes the grid from
    # the operation, it has to be passed here
    op.set_grid(grid)

    # Select the function for which the grid is refined
    if build_sg_for_e_and_var:
        # the expectation and variance calculation via the moments
        op.set_expectation_variance_Function()
    else:
        op.set_PCE_Function(polynomial_degree_max)

    if spatiallyAdaptive:
        # combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, operation=op, norm=2)
        combiinstance = SpatiallyAdaptiveSingleDimensions2(
            a, b, operation=op, norm=2, grid_surplusses=grid)
    else:
        combiinstance = StandardCombi(a, b, operation=op, norm=2)

    if plotting:
        plot_file = str(outputModelDir / "output.png")
        filename_contour_plot = str(outputModelDir / "output_contour_plot.png")
        filename_combi_scheme_plot = str(outputModelDir / "output_combi_scheme.png")
        filename_refinement_graph = str(outputModelDir / "output_refinement_graph.png")
        filename_sparse_grid_plot = str(outputModelDir / "output_sg_graph.png")
        combiinstance.filename_contour_plot = filename_contour_plot
        combiinstance.filename_refinement_graph = filename_refinement_graph
        combiinstance.filename_combi_scheme_plot = filename_combi_scheme_plot
        combiinstance.filename_sparse_grid_plot = filename_sparse_grid_plot

    if spatiallyAdaptive:
        error_operator = ErrorCalculatorSingleDimVolumeGuided()
        combiinstance.performSpatiallyAdaptiv(1, lmax, error_operator, tol=tolerance, max_evaluations=max_evals,
                                              do_plot=plotting)
    else:
        combiinstance.perform_operation(1, lmax)

    if plotting:
        combiinstance.print_resulting_sparsegrid(markersize=10)

    #####################################

    if build_sg_for_e_and_var:
        (E,), (Var,) = op.calculate_expectation_and_variance(combiinstance)
        print(f"E: {E}, Var: {Var}")
    else:
        # Create the PCE approximation; it is saved internally in the operation
        op.calculate_PCE(polynomial_degrees=polynomial_degree_max, combiinstance=combiinstance)  # restrict_degrees

        # Calculate the expectation, variance with the PCE coefficients
        # (E,), (Var,) = op.calculate_expectation_and_variance(combiinstance, use_combiinstance_solution=False)
        (E,), (Var,) = op.get_expectation_and_variance_PCE()
        print(f"E: {E}, Var: {Var}")

        # Calculate sobol indices with the PCE coefficients
        si_first = op.get_first_order_sobol_indices()
        si_total = op.get_total_order_sobol_indices()

        #####################################
        # Saving and plotting
        #####################################

        # gPCE = op.get_gPCE()
        fileName = f"gpce.pkl"
        gpceFileName = str(outputModelDir / fileName)
        with open(gpceFileName, 'wb') as handle:
            pickle.dump(op.gPCE, handle, protocol=pickle.HIGHEST_PROTOCOL)

        fileName = f"pce_polys.pkl"
        pcePolysFileName = str(outputModelDir / fileName)
        with open(pcePolysFileName, 'wb') as handle:
            pickle.dump(op.pce_polys, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # print(f"First order Sobol indices: {si_first.shape} \n {si_first}")
        # print(f"Total order Sobol indices: {si_total.shape} \n {si_total}")

        sobol_m_analytical, sobol_t_analytical = problem_function.ishigamiModelObject.get_analytical_sobol_indices()
        for i in range(len(labels)):
            print(f"Sobol's Total Index for parameter {labels[i]} is: \n")
            print(f"Sobol Total Simulation = {si_total[i][0]} \n")
            print(f"Sobol Total Analytical = {sobol_t_analytical[i]:.6f} \n")
            error = sobol_t_analytical[i] - si_total[i][0]
            print(f"Sobol Total Error = {error:.6f} \n")

        for i in range(len(labels)):
            print(f"Sobol's Main Index for parameter {labels[i]} is: \n")
            print(f"Sobol Main Simulation = {si_first[i][0]} \n")
            print(f"Sobol Main Analytical = {sobol_m_analytical[i]:.6f} \n")
            error = sobol_m_analytical[i] - si_first[i][0]
            print(f"Sobol Main Error = {error:.6f} \n")

        temp = f"results.txt"
        save_file = outputModelDir / temp
        fp = open(save_file, "w")
        fp.write(f'E: {E},\n Var: {Var}, \n '
                 f'First order Sobol indices: {si_first} \n; '
                 f'Total order Sobol indices: {si_total} \n')
        fp.close()


def var4_compute_leja_sg_surrogate_and_gpce(a_model_param=7, b_model_param=0.1, modified_basis=False,
                                            spatiallyAdaptive=True, plotting=True, outputModelDir="./",
                                            lmax=2, max_evals=2000, tolerance=10 ** -5,
                                            rule='gaussian', sparse=False, q=7, p=6):
    # TODO finish this!
    x1 = cp.Uniform(-math.pi, math.pi)
    x2 = cp.Uniform(-math.pi, math.pi)
    x3 = cp.Uniform(-math.pi, math.pi)
    joint = cp.J(x1, x2, x3)
    labels = [param_name.strip() for param_name in ["x1", "x2", "x3"]]

    leja_surrogate = None
    _compute_gpce_chaospy_ishigami(a_model_param=a_model_param, b_model_param=b_model_param,
                                   labels=labels, my_model=leja_surrogate, joint=joint,
                                   q=q, p=p, rule=rule, sparse=sparse,
                                   outputModelDir=outputModelDir, can_model_evaluate_all_vector_nodes=True)


if __name__ == "__main__":

    plotting = False
    scratch_dir = pathlib.Path("/work/ga45met")
    # outputModelDir = scratch_dir / "ishigami_runs" / "ishigami_sg_ss_ct_nonmodified_var2_l_2_p_8_q_11_max_2000"
    # outputModelDir = scratch_dir / "ishigami_runs" / "ishigami_sg_ss_adaptive_nonmodified_var2_l_2_p_8_q_11_max_2000"
    outputModelDir = scratch_dir / "ishigami_runs" / "ishigami_sg_ss_adaptive_nonmodified_parallel_gpce_var3_l_2_p_8_max_2000"
    # outputModelDir = scratch_dir / "ishigami_runs" / "ishigami_sg_var2_l_2_p_6_max_2000"
    outputModelDir.mkdir(parents=True, exist_ok=True)

    simulation_time_start = time.time()

    rule = 'gaussian'
    sparse = False
    q = 11
    p = 8

    # Var 1 #
    # compute_gpce_chaospy(a=7, b=0.1, outputModelDir=outputModelDir, rule=rule, sparse=sparse, q=q, p=p)

    lmax = 2  # 4
    max_evals = 2000  # 4000
    tolerance = 10 ** -5

    # Var 2 #
    # compute_surrogate_sparsespace_and_gpce(a_model_param=7, b_model_param=0.1, modified_basis=False, spatiallyAdaptive=True,
    #                                        plotting=plotting, outputModelDir=outputModelDir,
    #                                        lmax=lmax, max_evals=max_evals, tolerance=tolerance,
    #                                        rule=rule, sparse=sparse, q=q, p=p)

    # compute_surrogate_sparsespace_and_gpce(a_model_param=7, b_model_param=0.1, modified_basis=True, spatiallyAdaptive=False,
    #                                        plotting=True, outputModelDir=outputModelDir,
    #                                        lmax=lmax, max_evals=max_evals, tolerance=tolerance,
    #                                        rule=rule, sparse=sparse, q=q, p=p)

    # Var 2.2 #
    # compute_surrogate_sparsespace_and_gpce_analytically(
    # a_model_param=7, b_model_param=0.1, modified_basis=True, spatiallyAdaptive=False,
    # plotting=True, outputModelDir=outputModelDir,
    # lmax=lmax, max_evals=max_evals, tolerance=tolerance,
    # rule=rule, sparse=sparse, q=q, p=p)

    # Var 3#
    compute_gpce_sparsespace(build_sg_for_e_and_var=False, modified_basis=False, parallelIntegrator=True,
                             spatiallyAdaptive=True, plotting=plotting, outputModelDir=outputModelDir,
                             lmax=lmax, max_evals=max_evals, tolerance=tolerance, p=p)

    # compute_gpce_sparsespace(build_sg_for_e_and_var=True, modified_basis=True, parallelIntegrator=False,
    #                          spatiallyAdaptive=False, plotting=plotting, outputModelDir=outputModelDir,
    #                          lmax=lmax, max_evals=max_evals, tolerance=tolerance)

    # TODO Seems as this is not working - when building SG surrogate to fit computaton of c_n!
    # compute_gpce_sparsespace(build_sg_for_e_and_var=False, modified_basis=True, parallelIntegrator=False,
    #                          spatiallyAdaptive=True, plotting=plotting, outputModelDir=outputModelDir,
    #                          lmax=lmax, max_evals=max_evals, tolerance=tolerance)
    # compute_gpce_sparsespace(build_sg_for_e_and_var=False, modified_basis=True, parallelIntegrator=False,
    #                          spatiallyAdaptive=False, plotting=plotting, outputModelDir=outputModelDir,
    #                          lmax=lmax, max_evals=max_evals, tolerance=tolerance)

    simulation_time_end = time.time()
    simulation_time = simulation_time_end - simulation_time_start
    print("simulation time: {} sec".format(simulation_time))