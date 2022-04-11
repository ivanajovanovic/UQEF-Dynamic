import json
import inspect
import itertools
import math
from mpi4py import MPI
import numpy as np
import numpoly
import pathlib
import pickle
import scipy
import scipy.integrate as integrate
import time

import chaospy as cp
import uqef

import os
import sys
cwd = pathlib.Path(os.getcwd())
# print(cwd)
parent = cwd.parent.absolute()
# print(parent)
# sys.path.insert(0, parent)
sys.path.insert(0, os.getcwd())

from sparseSpACE.Function import *
from sparseSpACE.spatiallyAdaptiveSingleDimension2 import *
from sparseSpACE.ErrorCalculator import *
from sparseSpACE.GridOperation import *
from sparseSpACE.StandardCombi import *
from sparseSpACE.Integrator import *

from sparse_utility import sparseSpACE_functions
from sparse_utility import Sparse_Quadrature
# from sparseSpACE_functions import *
# from Sparse_Quadrature import *

comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()

#####################################
# Set of Utility Functions
#####################################

# utility for Genz Family of Functions
b_1 = 1.5  # 9.0
b_2 = 7.2
b_3 = 1.85
b_4 = 7.03
b_5 = 20.4
b_6 = 4.3
#
genz_dict = {
    "oscillatory":b_1, "product_peak":b_2, "corner_peak":b_3, "gaussian":b_4, "continous":b_5, "discontinuous":b_6
}

def generate_and_scale_coeff_and_weights(dim, b, w_norm=1, anisotropic=False):
    coeffs = cp.Uniform(0, 1).sample(dim)  # TODO Think of using some quasiMC method
    l1 = np.linalg.norm(coeffs, 1)
    coeffs = coeffs * b / l1
    if anisotropic:
        coeffs = np.array([coeff*np.exp(i/dim) for coeff, i in zip(coeffs, range(1,dim+1))])  # less isotropic
    weights = cp.Uniform(0,1).sample(dim)
    l1 = np.linalg.norm(weights, 1)
    weights = weights * w_norm / l1
    return coeffs, weights

# TODO Check validity of this code
#calculate mean, variance, sobol indices from gPCE coefficients
def calculate_MeanVarianceSobol(gPCE_coefficients, polynomial_degrees, dim):
    vari = 0
    for i in range(1, len(gPCE_coefficients)):
        vari += gPCE_coefficients[i] ** 2  # under assumption that polynomials were normalized
   # print('vari: ', vari)
    mean = gPCE_coefficients[0]
   # print('mean: ', mean)

    first_order_sobol = [0 for _ in range(dim)]
    total_order_sobol = [0 for _ in range(dim)]

    # TODO Validate this code!
    # indices = numpoly.glexindex(start=0, stop=polynomial_degrees + 1, dimensions=dim,
    #                             graded=True, reverse=True,
    #                             cross_truncation=1.0)
    # for d in range(dim):
    #     for i, ind in enumerate(indices):
    #         correct = True
    #         if ind[d] == 0:
    #             correct = False
    #         for d_other in range(dim):
    #             if d_other != d and ind[d_other] != 0:
    #                 correct = False
    #         if correct:
    #             first_order_sobol[d] += gPCE_coefficients[i]**2 / vari
    #
    # for d in range(dim):
    #     for i, ind in enumerate(indices):
    #         if not ind[d] == 0:
    #             total_order_sobol[d] += gPCE_coefficients[i] ** 2 / vari
    # print("expected: ", mean, ", variance: ", vari, ", first order sobol indices: \n", first_order_sobol,
    #       ", total order sobol indices: ", total_order_sobol)

    return mean, vari, first_order_sobol, total_order_sobol


def transformSamples(samples, distribution_r, distribution_q):
    return distribution_q.inv(distribution_r.fwd(samples))


def transformSamples_lin_or_nonlin(samples, distribution_r, distribution_q, linear=True):
    if linear:
        return Sparse_Quadrature.transformation_of_parameters_var2(samples, distribution_r, distribution_q)
    else:
        return Sparse_Quadrature.transformation_of_parameters_var1(samples, distribution_r, distribution_q)


def _read_nodes_weights_dist_from_file(parameters_file_name, parameters_setup_file_name, stochastic_dim):
    if parameters_file_name is None:
        raise Exception
    nodes_and_weights_array = np.loadtxt(parameters_file_name, delimiter=',')
    nodes = nodes_and_weights_array[:, :stochastic_dim].T
    weights = nodes_and_weights_array[:, stochastic_dim]
    print(f"shape of read nodes {nodes.shape}; min {nodes.min()}; max {nodes.max()}")
    print(f"shape of read weights {weights.shape}; min {weights.min()}; max {weights.max()}")

    if parameters_setup_file_name is not None:
        # This branch ensures that nodes come from dist!!!
        # TODO - move this to a functions
        with open(parameters_setup_file_name) as f:
            parameters_configuration_object = json.load(f)
        distsOfNodesFromFile = []
        for parameter_config in parameters_configuration_object["parameters"]:
            # node values and distributions -> automatically maps dists and their parameters by reflection mechanisms
            cp_dist_signature = inspect.signature(getattr(cp, parameter_config["distribution"]))
            dist_parameters_values = []
            for p in cp_dist_signature.parameters:
                dist_parameters_values.append(parameter_config[p])
            distsOfNodesFromFile.append(getattr(cp, parameter_config["distribution"])(*dist_parameters_values))
        jointDistOfNodesFromFile = cp.J(*distsOfNodesFromFile)
    else:
        distsOfNodesFromFile = []
        # Hard-coded by default, assumption is that read nodes have Uniform(0,1) distribution
        for _ in range(dim):
            distsOfNodesFromFile.append(cp.Uniform())
            # distsOfNodesFromFile.append(cp.Uniform(lower=0.0, upper=1.0))
        jointDistOfNodesFromFile = cp.J(*distsOfNodesFromFile)
    # nodes = transformSamples(nodes, jointDistOfNodesFromFile, dist)
    return nodes, weights, jointDistOfNodesFromFile


#####################################
# Set methods under consideration
#####################################


# Var 0 - MC method for computing reference values for mean and/or variance...
def compute_mc_quantity(model, param_names, dists, joint, jointStandard, dim, a, b, numSamples, rule="R",
                        sampleFromStandardDist=False, can_model_evaluate_all_vector_nodes=False,
                        read_nodes_from_file=False, **kwargs):
    print(f"\n==VAR0: MC Computation Chaospy==")
    labels = [param_name.strip() for param_name in param_names]
    stochastic_dim = len(param_names)

    if sampleFromStandardDist:
        dist = jointStandard
    else:
        dist = joint

    if read_nodes_from_file:
        parameters_file_name = kwargs.get('parameters_file_name', None)
        if parameters_file_name is None:
            raise
        nodes_and_weights_array = np.loadtxt(parameters_file_name, delimiter=',')
        # TODO what if nodes_and_weights_array do not correspond to dist! However, this is not important for MC
        nodes = nodes_and_weights_array[:, :stochastic_dim].T
    else:
        nodes = dist.sample(size=numSamples, rule=rule).round(4)
    nodes = np.array(nodes)

    if sampleFromStandardDist:
        # parameters = transformSamples(nodes, jointStandard, joint)
        parameters = transformSamples_lin_or_nonlin(nodes, jointStandard, joint, linear=False)
    else:
        parameters = nodes

    start_time_model_evaluations = time.time()
    if can_model_evaluate_all_vector_nodes:
        evaluations = model(parameters.T)
    else:
        evaluations = np.array([model(parameter) for parameter in parameters.T])
    end_time_model_evaluations = time.time()

    time_model_evaluations = end_time_model_evaluations - start_time_model_evaluations
    numEvaluations = len(parameters.T)  # len(evaluations)

    print(f"Needed time for model evaluations is: {time_model_evaluations} \n"
          f"for {numEvaluations} number of full model runs;")

    compute_mean = kwargs.get('compute_mean', True)
    compute_var = kwargs.get('compute_var', True)

    start_time_computing_statistics = time.time()
    expectedInterp = None
    varianceInterp = None
    if compute_mean:
        expectedInterp = np.mean(evaluations)
        print(f"MC_expectation = {expectedInterp}")
    if compute_var:
        varianceInterp = np.sum((evaluations - expectedInterp) ** 2, axis=0, dtype=np.float64) / (numEvaluations - 1)
        print(f"MC_variance = {varianceInterp}")
    end_time_computing_statistics = time.time()
    time_computing_statistics = end_time_computing_statistics - start_time_computing_statistics
    print(f"Needed time for computing statistics is: {time_computing_statistics};\n")

    return expectedInterp, varianceInterp


def compute_gpce_chaospy_uqefpp(model, param_names, dists, joint, jointStandard, dim, a, b, plotting=False,
                         writing_results_to_a_file=False, outputModelDir="./", rule='gaussian', sparse=True, q=7, p=6,
                         poly_rule="three_terms_recurrence", poly_normed=False, sampleFromStandardDist=False,
                         can_model_evaluate_all_vector_nodes=False, vector_model_output=False,
                         read_nodes_from_file=False, **kwargs):
    """
    This method created a UQsim object and relays on routines from UQEF and UQEFPP
    input model: name of the model in string format
    """
    # TODO Finish this function
    pass


# Var 1
def compute_gpce_chaospy(model, param_names, dists, joint, jointStandard, dim, a, b, plotting=False,
                         writing_results_to_a_file=False, outputModelDir="./", rule='gaussian', sparse=True, q=7, p=6,
                         poly_rule="three_terms_recurrence", poly_normed=False, sampleFromStandardDist=False,
                         can_model_evaluate_all_vector_nodes=False, vector_model_output=False,
                         read_nodes_from_file=False, **kwargs):
    """
    This is all that UQEF+UQEFPP do in short, when uq_method=sc
    - sparse_utility vs. non-sparse_utility
    - model output single vs. vector
    - transformation vs. not
    - question weather there is analytical values
    """
    print(f"\n==VAR1: gPCE Chaospy==")

    labels = [param_name.strip() for param_name in param_names]
    stochastic_dim = len(param_names)

    growth = True if (rule == "c" and not sparse) else False

    compute_mean = kwargs.get('compute_mean', True)
    compute_var = kwargs.get('compute_var', True)
    compute_Sobol_m = kwargs.get('compute_Sobol_m', False)
    compute_Sobol_t = kwargs.get('compute_Sobol_t', False)

    # get_analytical_mean = kwargs.get('get_analytical_mean', False)
    # get_analytical_var = kwargs.get('get_analytical_var', False)
    # get_analytical_Sobol_m = kwargs.get('get_analytical_Sobol_m', False)
    # get_analytical_Sobol_t = kwargs.get('get_analytical_Sobol_t', False)

    if sampleFromStandardDist:
        dist = jointStandard
    else:
        dist = joint

    if read_nodes_from_file:
        parameters_file_name = kwargs.get('parameters_file_name', None)
        parameters_setup_file_name = kwargs.get('parameters_setup_file_name', None)
        nodes, weights, jointDistOfNodesFromFile = _read_nodes_weights_dist_from_file(
            parameters_file_name, parameters_setup_file_name, stochastic_dim)
        nodes = transformSamples_lin_or_nonlin(nodes, jointDistOfNodesFromFile, dist, linear=True)
        print(f"jointDistOfNodesFromFile {jointDistOfNodesFromFile}")
        print(f"dist {dist}")
        print(f"shape of nodes after transform {nodes.shape}; min {nodes.min()}; max {nodes.max()}")
        print(f"shape of weights after transform {weights.shape}; min {weights.min()}; max {weights.max()}")
    else:
        # TODO what if rule requires some special dist=jointStandard, e.g., c-c?
        nodes, weights = cp.generate_quadrature(q, dist, rule=rule, growth=growth, sparse=sparse)
        print(f"shape of quadrature nodes {nodes.shape}; min {nodes.min()}; max {nodes.max()}")
        print(f"shape of quadrature weights {weights.shape}; min {weights.min()}; max {weights.max()}")

    # __restore__cpu_affinity()
    nodes = np.array(nodes)
    weights = np.array(weights)

    if sampleFromStandardDist:
        # parameters = transformSamples(nodes, jointStandard, joint)
        parameters = transformSamples_lin_or_nonlin(nodes, jointStandard, joint, linear=False)
    else:
        parameters = nodes

    # Option to save nodes, weights, parameters
    if writing_results_to_a_file:
        fileName = "nodes.npy"
        simulationNodesFileName = str(outputModelDir / fileName)
        with open(simulationNodesFileName, 'wb') as f:
            np.save(f, nodes)
        fileName = "weights.npy"
        simulationWeightsFileName = str(outputModelDir / fileName)
        with open(simulationWeightsFileName, 'wb') as f:
            np.save(f, weights)
        if sampleFromStandardDist:
            fileName = "parameters.npy"
            simulationParametersFileName = str(outputModelDir / fileName)
            with open(simulationParametersFileName, 'wb') as f:
                np.save(f, parameters)

    print(f"Model will be evaluated in the following parameters\n: {parameters};")

    # Note: nodes and parameters are of shape dxq
    # evaluate model
    # TODO Parallelization of this part is the main benefit of UQEF for complex models
    start_time_model_evaluations = time.time()
    if can_model_evaluate_all_vector_nodes:
        evaluations = model(parameters.T)
    else:
        evaluations = np.array([model(parameter) for parameter in parameters.T])  # TODO maybe this is problematic!!!
    end_time_model_evaluations = time.time()

    time_model_evaluations = end_time_model_evaluations - start_time_model_evaluations
    number_full_model_evaluations = len(parameters.T)

    print(f"Needed time for model evaluations is: {time_model_evaluations} \n"
          f"for {number_full_model_evaluations} number of full model runs;")

    start_time_producing_gpce = time.time()
    expansion = cp.generate_expansion(order=p, dist=dist, rule=poly_rule, normed=poly_normed)
    gPCE = cp.fit_quadrature(expansion, nodes, weights, evaluations)
    end_time_producing_gpce = time.time()

    time_producing_gpce = end_time_producing_gpce - start_time_producing_gpce
    print(f"Needed time for producing gPCE is: {time_producing_gpce};")

    # TODO Think of computing these quantities on your own!
    start_time_computing_statistics = time.time()
    expectedInterp = None
    varianceInterp = None
    first_order_sobol_indices = None
    total_order_sobol_indices = None
    if compute_mean:
        expectedInterp = cp.E(gPCE, dist)
        print(f"gPCE_mean = {expectedInterp}")
    if compute_var:
        varianceInterp = cp.Var(gPCE, dist)
        print(f"gPCE_variance = {varianceInterp}")
    if compute_Sobol_m:
        first_order_sobol_indices = cp.Sens_m(gPCE, dist)
        print("gPCE First order Sobol indices: ", first_order_sobol_indices)
        first_order_sobol_indices = np.squeeze(first_order_sobol_indices)
    if compute_Sobol_t:
        total_order_sobol_indices = cp.Sens_t(gPCE, dist)
        print("gPCE Total order Sobol indices: ", total_order_sobol_indices)
        total_order_sobol_indices = np.squeeze(total_order_sobol_indices)
    end_time_computing_statistics = time.time()
    time_computing_statistics = end_time_computing_statistics - start_time_computing_statistics

    # print(f'time_model_evaluations: {time_model_evaluations}')
    # print(f'time_producing_gpce: {time_producing_gpce}')
    # print(f'number_full_model_evaluations: {number_full_model_evaluations}')
    print(f"Needed time for computing statistics is: {time_computing_statistics};\n")

    #####################################
    # Saving
    #####################################
    if writing_results_to_a_file:
        # gPCE = op.get_gPCE()
        fileName = f"gpce.pkl"
        gpceFileName = str(outputModelDir / fileName)
        with open(gpceFileName, 'wb') as handle:
            pickle.dump(gPCE, handle, protocol=pickle.HIGHEST_PROTOCOL)

        fileName = f"pce_polys.pkl"
        pcePolysFileName = str(outputModelDir / fileName)
        with open(pcePolysFileName, 'wb') as handle:
            pickle.dump(expansion, handle, protocol=pickle.HIGHEST_PROTOCOL)

        fileName = f"results.txt"
        statFileName = str(outputModelDir / fileName)
        fp = open(statFileName, "w")
        fp.write(f'E: {expectedInterp}\n')
        fp.write(f'Var: {varianceInterp}\n')
        fp.write(f'First order Sobol indices: {first_order_sobol_indices}\n')
        fp.write(f'Total order Sobol indices: {total_order_sobol_indices}\n')
        fp.write(f'number_full_model_evaluations: {number_full_model_evaluations}\n')
        fp.write(f'time_model_evaluations: {time_model_evaluations}\n')
        fp.write(f'time_producing_gpce: {time_producing_gpce}\n')
        fp.write(f'time_computing_statistics: {time_computing_statistics}')
        fp.close()

    return gPCE, expectedInterp, varianceInterp, first_order_sobol_indices, total_order_sobol_indices


# Var 2
def compute_surrogate_sparsespace_and_gpce(model, param_names, dists, joint, jointStandard, dim, a, b,
                                           surrogate_model_of_interest="gPCE", plotting=False,
                                           writing_results_to_a_file=False, outputModelDir="./", gridName="Trapezoidal",
                                           lmin=1, lmax=2, max_evals=10000, tolerance=10 ** -5, modified_basis=False,
                                           boundary_points=True, spatiallyAdaptive=True, grid_surplusses=None,
                                           norm_spatiallyAdaptive=np.inf, rebalancing=True, rule='gaussian',
                                           sparse=False, q=7, p=6, poly_rule="three_terms_recurrence",
                                           poly_normed=False, sampleFromStandardDist=False,
                                           can_model_evaluate_all_vector_nodes=True, vector_model_output=False,
                                           read_nodes_from_file=False, **kwargs):
    """
    Var 2 - Compute gPCE coefficients by integrating the (SG) surrogate
    SG surrogate computed based on SparseSpACE
      - adaptive vs. non-adaptive
      - different grids
      - e.g.,'PSP with surrogate: Grid = Standard Combi, not adaptive',
             'PSP with surrogate: Grid = Trapezoidal, adaptive',
             'PSP with surrogate, Grid = Leja, not adaptive',
             'PSP with surrogate, using Bsplines with degree 13, adaptive'
    gPCE coefficients - chaospy
      - the same set of options for chaospy.Quadrature as in compute_gpce_chaospy
    can_model_evaluate_all_vector_nodes - refers to combiinstance, in the second step when building gPCE, seems as hast to be set to True
    """
    print(f"\n==VAR2: gPCE with Sparse surrogate==")

    labels = [param_name.strip() for param_name in param_names]
    stochastic_dim = len(param_names)
    # dim = stochastic_dim

    compute_mean = kwargs.get('compute_mean', True)
    compute_var = kwargs.get('compute_var', True)
    compute_Sobol_m = kwargs.get('compute_Sobol_m', False)
    compute_Sobol_t = kwargs.get('compute_Sobol_t', False)

    #####################################
    if spatiallyAdaptive:
        if gridName == 'BSpline_p3':
            p_bsplines = kwargs.get('p_bsplines', 3)
            grid = GlobalBSplineGrid(a=a, b=b, modified_basis=modified_basis, boundary=boundary_points, p=p_bsplines)
        # elif gridName == 'TrapezoidalWeighted':
        #     distributionsForSparseSpace =
        #     operation_uq = UncertaintyQuantification(f=model, distributions=distributionsForSparseSpace, a=a, b=b, dim=dim)
        #     grid = GlobalTrapezoidalGridWeighted(a=a, b=b, uq_operation=operation_uq,
        #                                          modified_basis=modified_basis, boundary=boundary_points)
        else:
            grid = GlobalTrapezoidalGrid(a=a, b=b, modified_basis=modified_basis, boundary=boundary_points)
    else:
        if gridName == 'Trapezoidal':
            grid = TrapezoidalGrid(a=a, b=b, modified_basis=modified_basis, boundary=boundary_points)
        elif gridName == 'Leja' or gridName == 'LejaNormal':   # for actual comparisons use LejaNormal
            grid = LejaGrid(a=a, b=b, boundary=boundary_points)
        elif gridName == 'BSpline_p3':
            p_bsplines = kwargs.get('p_bsplines', 3)
            grid = BSplineGrid(a=a, b=b, boundary=boundary_points, p=p_bsplines)

    # TODO Ask Obi - Integration is the same as Interpolation???
    operation = Integration(f=model, grid=grid, dim=dim)  # there is Interpolation(Integration)
    # operation_uq.set_grid(grid)

    if spatiallyAdaptive:
        errorOperator = ErrorCalculatorSingleDimVolumeGuided()
        # combiinstance = SpatiallyAdaptiveSingleDimensions2(
        #     np.ones(dim) * a, np.ones(dim) * b, margin=0.8, operation=operation)
        # TODO or?
        # TODO Additional parameters norm=2, grid_surplusses=grid
        # TODO if grid_surplusses=None (default does that leads to GlobalTrapezoidalGrid)
        if grid_surplusses is None:
            # default grid_surplusses = GlobalTrapezoidalGrid
            combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, margin=0.8, operation=operation,
                                                               norm=norm_spatiallyAdaptive, rebalancing=rebalancing)
        else:
            combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, margin=0.8, operation=operation,
                                                               norm=norm_spatiallyAdaptive, rebalancing=rebalancing, grid_surplusses=grid)

        # combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, margin=0.8, operation=operation, norm=2, grid_surplusses=grid)
        # combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, margin=0.8, operation=operation, norm=np.inf, grid_surplusses=grid)
    else:
        # combiinstance = StandardCombi(np.ones(dim) * a, np.ones(dim) * b, operation=operation, norm=2)
        combiinstance = StandardCombi(a, b, operation=operation)  # Markus

    if plotting:
        plot_file = kwargs.get('plot_file', str(outputModelDir / "output.png"))
        filename_contour_plot = kwargs.get('filename_contour_plot', str(outputModelDir / "output_contour_plot.png"))
        filename_combi_scheme_plot = kwargs.get('filename_combi_scheme_plot', str(outputModelDir / "output_combi_scheme.png"))
        filename_refinement_graph = kwargs.get('filename_refinement_graph', str(outputModelDir / "output_refinement_graph.png"))
        filename_sparse_grid_plot = kwargs.get('filename_sparse_grid_plot', str(outputModelDir / "output_sg_graph.png"))
        combiinstance.filename_contour_plot = filename_contour_plot
        combiinstance.filename_refinement_graph = filename_refinement_graph
        combiinstance.filename_combi_scheme_plot = filename_combi_scheme_plot
        combiinstance.filename_sparse_grid_plot = filename_sparse_grid_plot

    start_time_building_sg_surrogate = time.time()
    if spatiallyAdaptive:
        # Markus
        # if dim > 3:
        #     lmax = 2
        # else:
        #     lmax = 3
        combiinstance.performSpatiallyAdaptiv(lmin, lmax, errorOperator, tol=tolerance, do_plot=plotting,
                                                               max_evaluations=max_evals)
        # print("integral: ", operation.integral)
    else:
        minimum_level = 1  # Markus
        combiinstance.perform_operation(lmin, lmax)
        # combiinstance.perform_operation(minimum_level, lmax+1, model)  # Markus
    end_time_building_sg_surrogate = time.time()
    time_building_sg_surrogate = end_time_building_sg_surrogate - start_time_building_sg_surrogate

    if plotting:
        combiinstance.print_resulting_sparsegrid(markersize=10)

    number_full_model_evaluations = combiinstance.get_total_num_points()
    print(f"Needed time for building SG surrogate is: {time_building_sg_surrogate} \n"
          f"for {number_full_model_evaluations} number of full model runs;")

    if writing_results_to_a_file:
        fileName = f"results.txt"
        statFileName = str(outputModelDir / fileName)
        fp = open(statFileName, "w")
        fp.write(f"time_building_sg_surrogate: {time_building_sg_surrogate}\n")
        fp.write(f'number_full_model_evaluations: {number_full_model_evaluations}\n')
        fp.close()

    if surrogate_model_of_interest.lower() != "gpce":
        # This is varaiante when only the computation of SG surrogate is of importance
        # TODO check if operation.calculate_expectation_and_variance(combiinstance) exists for operation==Integration
        # TODO Check the structure of combiinstance for vector_model_output=True
        return combiinstance, None, None, None, None, None
    else:
        #####################################
        # # Markus
        # if rule == "leja":
        #     q = 2*2*order
        #     p = 2*order-1
        #
        # gPCE, expectedInterp, varianceInterp, first_order_sobol_indices, total_order_sobol_indices = \
        #   compute_gpce_chaospy(model=combiinstance, param_names=param_names, dists=dists, joint=joint,
        #                        jointStandard=jointStandard, dim=dim, a=a, b=b, plotting=plotting,
        #                        writing_results_to_a_file=writing_results_to_a_file, outputModelDir=outputModelDir,
        #                        rule=rule, sparse=sparse, q=q, p=p, poly_rule=poly_rule, poly_normed=poly_normed,
        #                        sampleFromStandardDist=True, can_model_evaluate_all_vector_nodes=True,
        #                        vector_model_output=vector_model_output, read_nodes_from_file=read_nodes_from_file,
        #                        **kwargs)

        #####################################
        growth = True if (rule == "c" and not sparse) else False

        if sampleFromStandardDist:
            dist = jointStandard
        else:
            dist = joint

        if read_nodes_from_file:
            parameters_file_name = kwargs.get('parameters_file_name', None)
            parameters_setup_file_name = kwargs.get('parameters_setup_file_name', None)
            nodes, weights, jointDistOfNodesFromFile = _read_nodes_weights_dist_from_file(
                parameters_file_name, parameters_setup_file_name, stochastic_dim)
            nodes = transformSamples_lin_or_nonlin(nodes, jointDistOfNodesFromFile, dist, linear=True)
        else:
            # TODO what if rule requires some special dist=jointStandard, e.g., c-c?
            nodes, weights = cp.generate_quadrature(q, dist, rule=rule, growth=growth, sparse=sparse)

        # __restore__cpu_affinity()
        nodes = np.array(nodes)
        weights = np.array(weights)

        if sampleFromStandardDist:
            parameters = transformSamples(nodes, jointStandard, joint)
        else:
            parameters = nodes

        # Option to save nodes, weights, parameters
        if writing_results_to_a_file:
            fileName = "nodes.npy"
            simulationNodesFileName = str(outputModelDir / fileName)
            with open(simulationNodesFileName, 'wb') as f:
                np.save(f, nodes)
            fileName = "weights.npy"
            simulationWeightsFileName = str(outputModelDir / fileName)
            with open(simulationWeightsFileName, 'wb') as f:
                np.save(f, weights)
            if sampleFromStandardDist:
                fileName = "parameters.npy"
                simulationParametersFileName = str(outputModelDir / fileName)
                with open(simulationParametersFileName, 'wb') as f:
                    np.save(f, parameters)

        # Note: nodes and parameters are of shape dxq
        # evaluate model
        # TODO Parallelization of this part is the main benefit of UQEF for complex models
        # TODO This is different from Ionut's paper where SG surrogate model is build for [0,1]^d
        start_time_model_evaluations = time.time()
        can_model_evaluate_all_vector_nodes = True
        if can_model_evaluate_all_vector_nodes:
            evaluations = combiinstance(parameters.T)
        else:
            # TODO This won't work!
            evaluations = np.array([combiinstance(parameter) for parameter in parameters.T])
        end_time_model_evaluations = time.time()

        time_model_evaluations = end_time_model_evaluations - start_time_model_evaluations
        number_surrogate_model_evaluations = len(parameters.T)

        print(f"Needed time for surrogate model evaluations is: {time_model_evaluations} \n"
              f"for {number_surrogate_model_evaluations} number of surrogate model  runs;")

        start_time_producing_gpce = time.time()
        # TODO Markus - in Leja case p=2*p-1
        expansion = cp.generate_expansion(order=p, dist=dist, rule=poly_rule, normed=poly_normed)
        gPCE = cp.fit_quadrature(expansion, nodes, weights, evaluations)
        end_time_producing_gpce = time.time()

        time_producing_gpce = end_time_producing_gpce - start_time_producing_gpce
        print(f"Needed time for producing gPCE of a surrogate model is: {time_producing_gpce};")

        start_time_computing_statistics = time.time()
        expectedInterp = None
        varianceInterp = None
        first_order_sobol_indices = None
        total_order_sobol_indices = None
        if compute_mean:
            expectedInterp = cp.E(gPCE, dist)
            print(f"SG_Surrogate + gPCE_mean = {expectedInterp}")
        if compute_var:
            varianceInterp = cp.Var(gPCE, dist)
            print(f"SG_Surrogate + gPCE_variance = {varianceInterp}")
        if compute_Sobol_m:
            first_order_sobol_indices = cp.Sens_m(gPCE, dist)
            print("SG_Surrogate + gPCE First order Sobol indices: ", first_order_sobol_indices)
            first_order_sobol_indices = np.squeeze(first_order_sobol_indices)
        if compute_Sobol_t:
            total_order_sobol_indices = cp.Sens_t(gPCE, dist)
            print("SG_Surrogate + gPCE Total order Sobol indices: ", total_order_sobol_indices)
            total_order_sobol_indices = np.squeeze(total_order_sobol_indices)
        end_time_computing_statistics = time.time()
        time_computing_statistics = end_time_computing_statistics - start_time_computing_statistics

        if writing_results_to_a_file:
            fileName = f"gpce.pkl"
            gpceFileName = str(outputModelDir / fileName)
            with open(gpceFileName, 'wb') as handle:
                pickle.dump(gPCE, handle, protocol=pickle.HIGHEST_PROTOCOL)

            fileName = f"pce_polys.pkl"
            pcePolysFileName = str(outputModelDir / fileName)
            with open(pcePolysFileName, 'wb') as handle:
                pickle.dump(expansion, handle, protocol=pickle.HIGHEST_PROTOCOL)

            fileName = f"results.txt"
            statFileName = str(outputModelDir / fileName)
            fp = open(statFileName, "w")
            fp.write(f'E: {expectedInterp}\n')
            fp.write(f'Var: {varianceInterp}\n')
            fp.write(f'First order Sobol indices: {first_order_sobol_indices}\n')
            fp.write(f'Total order Sobol indices: {total_order_sobol_indices}\n')
            fp.write(f'number_surrogate_model_evaluations: {number_surrogate_model_evaluations}\n')
            fp.write(f'surrogate time_model_evaluations: {time_model_evaluations}\n')
            fp.write(f'time_producing_gpce: {time_producing_gpce}\n')
            fp.write(f'time_computing_statistics: {time_computing_statistics}')
            fp.close()
        #####################################
        return combiinstance, gPCE, expectedInterp, varianceInterp, first_order_sobol_indices, total_order_sobol_indices


# # Var 3
# #compute gPCE coefficients analytically on a piecewise linear interpolant constructed with standard CT or the single dimension refinement strategy
# def analytica_integration_with_surrogate(model, param_names, dists, dim, a, b,
#                              surrogate_model_of_interest="gPCE", plotting=False,
#                              writing_results_to_a_file=False, outputModelDir="./", gridName="Trapezoidal",
#                              lmax=2, max_evals=2000, tolerance=10 ** -5, modified_basis=False,
#                              boundary_points=False, spatiallyAdaptive=False, p=6,
#                              build_sg_for_e_and_var=True, parallelIntegrator=False, **kwargs):
#
#    #compute the one-dimensional integral
#     def computeIntegral(point_d, neighbours, distributions_d, pce_poly_1d, d):
#         if point_d <= 0 or max(0, neighbours[0]) >= min(1, point_d):
#             integralLeft = 0
#         elif (not boundary) and neighbours[0] == 0:
#             hatFunctionLeft = lambda x: 1
#             integrandLeft = lambda x: pce_poly_1d(function_info.ppfs[d](x)) * hatFunctionLeft(x)
#             integralLeft = integrate.fixed_quad(integrandLeft, max(0, neighbours[0]), min(1, point_d), n=7)[0]
#         else:
#             hatFunctionLeft = lambda x: (x - neighbours[0]) / (point_d - neighbours[0])
#             integrandLeft = lambda x: pce_poly_1d(function_info.ppfs[d](x)) * hatFunctionLeft(x)
#             integralLeft = integrate.fixed_quad(integrandLeft, max(0, neighbours[0]), min(1, point_d), n=7)[0]
#         if point_d >= 1 or max(0, point_d) >= min(1, neighbours[1]):
#             integralRight = 0
#         elif (not boundary) and neighbours[1] == 1:
#             hatFunctionRight = lambda x: 1
#             integrandRight = lambda x: pce_poly_1d(function_info.ppfs[d](x)) * hatFunctionRight(x)
#             integralRight = integrate.fixed_quad(integrandRight, max(0, point_d), min(1, neighbours[1]), n=7)[0]
#         else:
#             hatFunctionRight = lambda x: max(0,((x - neighbours[1]) / (point_d - neighbours[1]))[0])
#             hatFunctionRight = lambda x: (x - neighbours[1]) / (point_d - neighbours[1])
#             integrandRight = lambda x: pce_poly_1d(function_info.ppfs[d](x)) * hatFunctionRight(x)
#             integralRight = integrate.fixed_quad(integrandRight, max(0, point_d), min(1, neighbours[1]), n=7)[0]
#         return integralLeft + integralRight
#
#     def standard_hatfunction1D(u):
#         return [max(1 - abs(ui), 0) for ui in u]
#
#     def hatfunction_level1D_position(u, l, x):
#         return standard_hatfunction1D((u - x) / float(2) ** (-l))
#
#     f = function_info.function_unitCube
#     a = np.array([0 for d in range(function_info.dim)])
#     b = np.array([1 for d in range(function_info.dim)])
#     if adaptive:
#         errorOperator = ErrorCalculatorSingleDimVolumeGuided()
#         grid = GlobalTrapezoidalGrid(a=a, b=b, modified_basis = not boundary, boundary=boundary)
#         operation = Integration(f=f, grid=grid, dim=function_info.dim)
#         combiObject = StandardCombi(a, b, operation=operation)
#         combiObject = SpatiallyAdaptiveSingleDimensions2(np.ones(function_info.dim) * a, np.ones(function_info.dim) * b, operation=operation, margin=0.8)
#         tolerance = 0
#         plotting = False
#         if function_info.dim > 3:
#             maxim_level = 2
#         else:
#             maxim_level = 3
#         combiObject.performSpatiallyAdaptiv(1, maxim_level, errorOperator, tol=tolerance, do_plot=plotting,
#                                             max_evaluations=max_evals)
#         #combiObject.draw_refinement()
#         combiObject.print_resulting_combi_scheme(markersize=5)
#         print(operation.integral)
#         dictIntegrals_adaptive = {}
#         dictEvaluations = {}
#
#         #for adaptive
#         def getNeighbours(combiObject, coordinates1d, x):
#             if coordinates1d[0] > 0:
#                 print("leftest: ", coordinates1d[0])
#             left_right = [0, 1]
#             index = np.where(coordinates1d == x)[0][0]
#             if not (index == 0):
#                 left_right[0] = coordinates1d[index - 1]
#             if not (index == len(coordinates1d) - 1):
#                 left_right[1] = coordinates1d[index + 1]
#             return left_right
#
#     else:
#         grid = TrapezoidalGrid(a=np.array([0 for d in range(function_info.dim)]), b=np.array([1 for d in range(function_info.dim)]), modified_basis=not boundary, boundary=boundary)
#         operation = Integration(f=f, grid=grid, dim=function_info.dim)
#         combiObject = StandardCombi(a=np.array([0 for d in range(function_info.dim)]), b=np.array([1 for d in range(function_info.dim)]), operation=operation)
#         minimum_level = 1
#         combiObject.perform_operation(minimum_level, maximum_level)
#         print("expectation: ", operation.integral)
#         dictIntegrals_not_adaptive = {}
#         dictEvaluations = {}
#
#     # extract the onedimensional orthogonal polynomials and order them in the same way chaospy does
#     number_points = combiObject.get_total_num_points()
#     expansion = chaospy.generate_expansion(polynomial_degrees, function_info.joint_distributions, normed=True)
#     pce_polys_1D = [None] * function_info.dim
#     for d in range(function_info.dim):
#         pce_polys_1D[d] = chaospy.expansion.stieltjes(polynomial_degrees, function_info.distributions[d], normed=True)
#     indices = numpoly.glexindex(start=0, stop=polynomial_degrees + 1, dimensions=function_info.dim,
#                                 graded=True, reverse=True,
#                                 cross_truncation=1.0)
#     polys = [None] * len(indices)
#     for i in range(len(indices)):
#         polys[i] = [pce_polys_1D[d][indices[i][d]] for d in range(function_info.dim)]
#
#     cn = [np.zeros(function_info.function.output_length()) for _ in polys]
#     for n, pce_poly in enumerate(polys):
#         for component_grid in combiObject.scheme:
#             if adaptive:
#                 gridPointCoordsAsStripes, grid_point_levels, children_indices = combiObject.get_point_coord_for_each_dim(
#                     component_grid.levelvector)
#                 points = combiObject.get_points_component_grid(component_grid.levelvector)
#                 keyLevelvector = component_grid.levelvector
#                 if keyLevelvector in dictEvaluations:
#                     evals = dictEvaluations[keyLevelvector]
#                 else:
#                     evals = [function_info.function_unitCube(poin) for poin in points]
#                     dictEvaluations[keyLevelvector] = evals
#                 integralCompGrid = 0
#             else:
#                 points = combiObject.get_points_component_grid(component_grid.levelvector)
#                 keyLevelvector = tuple(component_grid.levelvector.tolist())
#                 if keyLevelvector in dictEvaluations:
#                     evals = dictEvaluations[keyLevelvector]
#                 else:
#                     evals = combiObject(points)
#                     dictEvaluations[keyLevelvector] = evals
#                 integralCompGrid = 0
#             for i, point in enumerate(points):
#                 product = 1
#                 for d in range(0, function_info.dim):
#                     if adaptive:
#                         neighbours = getNeighbours(combiObject, gridPointCoordsAsStripes[d], point[d])
#                         if (point[d], tuple(neighbours), indices[n][
#                             d], d) in dictIntegrals_adaptive:
#                             onedimensionalIntegral = dictIntegrals_adaptive[(point[d], tuple(neighbours), indices[n][d], d)]
#                         else:
#                             onedimensionalIntegral = computeIntegral(point[d],
#                                                                      neighbours, function_info.distributions[d],
#                                                                      pce_poly[d], d)
#                             dictIntegrals_adaptive[(point[d], tuple(neighbours), indices[n][d], d)] = onedimensionalIntegral
#                     else:
#                         if (point[d], component_grid.levelvector[d], indices[n][
#                             d], d) in dictIntegrals_not_adaptive:
#                             onedimensionalIntegral = dictIntegrals_not_adaptive[
#                                 (point[d], component_grid.levelvector[d], indices[n][d], d)]
#                         else:
#                             neighbours = [max(0, point[d]-float(2)**(-component_grid.levelvector[d])), min(1, point[d]+float(2)**(-component_grid.levelvector[d]))]
#                             onedimensionalIntegral = computeIntegral(point[d], neighbours, function_info.distributions[d], pce_poly[d], d)
#                             dictIntegrals_not_adaptive[
#                                 (point[d], component_grid.levelvector[d], indices[n][d], d)] = onedimensionalIntegral
#                     product = product * onedimensionalIntegral
#                 integralCompGrid = integralCompGrid + product * evals[i]
#             cn[n] = cn[n] + component_grid.coefficient * integralCompGrid
#
#     expected, variance, first_order_sobol_indices = calculate_MeanVarianceSobol(cn, polynomial_degrees, function_info.dim)
#     print("variance: ", variance)
#     if store_result and not time_Series:
#         entry = (number_points, expected, variance, *first_order_sobol_indices)
#         storeResult(entry, 6 if adaptive else 5, function_info)
#     if time_Series:
#         plot_Times_series(function_info, variance, expected, first_order_sobol_indices)
#

# Var 4
def compute_gpce_sparsespace(model, param_names, dists, dim, a, b, surrogate_model_of_interest="gPCE", plotting=False,
                             writing_results_to_a_file=False, outputModelDir="./", gridName="Trapezoidal", lmin=1,
                             lmax=2, max_evals=10000, tolerance=10 ** -5, modified_basis=False, boundary_points=True,
                             spatiallyAdaptive=True, grid_surplusses=None, norm_spatiallyAdaptive=np.inf,
                             rebalancing=True, p=6, build_sg_for_e_and_var=True, parallelIntegrator=False, **kwargs):
    """
    Var 4 - This method relies on current implementation of UncertaintyQuantification Operation in SparseSpACE
    If build_sg_for_e_and_var == True
    ---> Build one SG surrogate to approximate E and Var
    Else
    --> Build one SG surrogate to approximate all N coefficients of the gPCE expansion - Erroneous!
    """
    print(f"\n==VAR4: gPCE computed directly using SG integration methods from SparseSpACE==")

    labels = [param_name.strip() for param_name in param_names]
    stochastic_dim = len(param_names)
    # dim = stochastic_dim

    if build_sg_for_e_and_var:
        surrogate_model_of_interest = "sg"

    compute_mean = kwargs.get('compute_mean', True)
    compute_var = kwargs.get('compute_var', True)
    compute_Sobol_m = kwargs.get('compute_Sobol_m', False)
    compute_Sobol_t = kwargs.get('compute_Sobol_t', False)

    # distributions = dists = distributionsForSparseSpace
    # problem_function = model
    # polynomial_degree_max = p

    operation = UncertaintyQuantification(model, dists, a, b)

    if spatiallyAdaptive:
        if gridName == 'BSpline_p3':
            p_bsplines = kwargs.get('p_bsplines', 3)
            grid = GlobalBSplineGrid(a=a, b=b, modified_basis=modified_basis, boundary=boundary_points, p=p_bsplines)
        # elif gridName == 'TrapezoidalWeighted':
        #     grid = GlobalTrapezoidalGridWeighted(a=a, b=b, uq_operation=operation, modified_basis=modified_basis, boundary=boundary_points)
        else:
            grid = GlobalTrapezoidalGrid(a=a, b=b, modified_basis=modified_basis, boundary=boundary_points)
    else:
        if gridName == 'Trapezoidal':
            grid = TrapezoidalGrid(a=a, b=b, modified_basis=modified_basis, boundary=boundary_points)
        elif gridName == 'Leja' or gridName == 'LejaNormal':   # for actual comparisons use LejaNormal
            grid = LejaGrid(a=a, b=b, boundary=boundary_points)
        elif gridName == 'BSpline_p3':
            p_bsplines = kwargs.get('p_bsplines', 3)
            grid = BSplineGrid(a=a, b=b, boundary=boundary_points, p=p_bsplines)

    if parallelIntegrator:
        print(f"==VAR4: Parallel Integration is being used")
        grid.integrator = IntegratorParallelArbitraryGrid(grid)

    # The grid initialization requires the weight functions from the
    # operation; since currently the adaptive refinement takes the grid from
    # the operation, it has to be passed here, though ambiguous what to create first: grid or operation
    operation.set_grid(grid)

    # Select the function for which the grid is refined
    if build_sg_for_e_and_var:
        # the expectation and variance calculation via the moments
        operation.set_expectation_variance_Function()
    else:
        operation.set_PCE_Function(p)

    if spatiallyAdaptive:
        errorOperator = ErrorCalculatorSingleDimVolumeGuided()
        if grid_surplusses is None:
            # default grid_surplusses = GlobalTrapezoidalGrid
            combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, margin=0.8, operation=operation,
                                                               norm=norm_spatiallyAdaptive, rebalancing=rebalancing)
        else:
            combiinstance = SpatiallyAdaptiveSingleDimensions2(a, b, margin=0.8, operation=operation,
                                                               norm=norm_spatiallyAdaptive, rebalancing=rebalancing,
                                                               grid_surplusses=grid)
    else:
        # combiinstance = StandardCombi(np.ones(dim) * a, np.ones(dim) * b, operation=operation, norm=2)
        combiinstance = StandardCombi(a, b, operation=operation, norm=2)  # Markus

    if plotting:
        plot_file = kwargs.get('plot_file', str(outputModelDir / "output.png"))
        filename_contour_plot = kwargs.get('filename_contour_plot', str(outputModelDir / "output_contour_plot.png"))
        filename_combi_scheme_plot = kwargs.get('filename_combi_scheme_plot', str(outputModelDir / "output_combi_scheme.png"))
        filename_refinement_graph = kwargs.get('filename_refinement_graph', str(outputModelDir / "output_refinement_graph.png"))
        filename_sparse_grid_plot = kwargs.get('filename_sparse_grid_plot', str(outputModelDir / "output_sg_graph.png"))
        combiinstance.filename_contour_plot = filename_contour_plot
        combiinstance.filename_refinement_graph = filename_refinement_graph
        combiinstance.filename_combi_scheme_plot = filename_combi_scheme_plot
        combiinstance.filename_sparse_grid_plot = filename_sparse_grid_plot

    start_time_building_sg_surrogate = time.time()
    if spatiallyAdaptive:
        # Markus
        # if dim > 3:
        #     lmax = 2
        # else:
        #     lmax = 3
        combiinstance.performSpatiallyAdaptiv(lmin, lmax, errorOperator, tol=tolerance, do_plot=plotting,
                                                               max_evaluations=max_evals)
        # print("integral: ", operation.integral)
    else:
        minimum_level = 1  # Markus
        combiinstance.perform_operation(lmin, lmax)
        # combiinstance.perform_operation(minimum_level, lmax+1, model)  # Markus
    end_time_building_sg_surrogate = time.time()
    time_building_sg_surrogate = end_time_building_sg_surrogate - start_time_building_sg_surrogate

    if plotting:
        combiinstance.print_resulting_sparsegrid(markersize=10)

    number_full_model_evaluations = combiinstance.get_total_num_points()
    print(f"Needed time for building SG surrogate is: {time_building_sg_surrogate} \n"
          f"for {number_full_model_evaluations} number of full model runs;")

    if writing_results_to_a_file:
        fileName = f"results.txt"
        statFileName = str(outputModelDir / fileName)
        fp = open(statFileName, "w")
        fp.write(f"time_building_sg_surrogate: {time_building_sg_surrogate}\n")
        fp.write(f'number_full_model_evaluations: {number_full_model_evaluations}\n')
        fp.close()

    #####################################
    expectedInterp = None
    varianceInterp = None
    first_order_sobol_indices = None
    total_order_sobol_indices = None

    start_time_computing_statistics = time.time()
    if build_sg_for_e_and_var:
        if compute_mean or compute_var:
            (expectedInterp,), (varianceInterp,) = operation.calculate_expectation_and_variance(combiinstance)
            print(f"E estimated via SparseSpACE SG Integration = {expectedInterp}")
            print(f"Var estimated via SparseSpACE SG Integration = {varianceInterp}")
    else:
        # Create the PCE approximation; it is saved internally in the operation
        start_time_producing_gpce = time.time()
        operation.calculate_PCE(polynomial_degrees=p, combiinstance=combiinstance)  # restrict_degrees
        end_time_producing_gpce = time.time()
        time_producing_gpce = end_time_producing_gpce - start_time_producing_gpce
        print(f"Needed time for approximating all integral when computing gPCE: {time_producing_gpce};")
        # Calculate the expectation, variance, and Sobol indices with the PCE coefficients
        if compute_mean or compute_var:
            # (expectedInterp,), (varianceInterp,) = op.calculate_expectation_and_variance(combiinstance, use_combiinstance_solution=False)
            (expectedInterp,), (varianceInterp,) = operation.get_expectation_and_variance_PCE()
            print(f"E estimated via SparseSpACE SG Integration + gPCE = {expectedInterp}")
            print(f"Var estimated via SparseSpACE SG Integration + gPCE = {varianceInterp}")
        if compute_Sobol_m:
            first_order_sobol_indices = operation.get_first_order_sobol_indices()
            first_order_sobol_indices = np.squeeze(first_order_sobol_indices)
            print("SG Integration when computing gPCE indices - First order Sobol indices: ", first_order_sobol_indices)
        if compute_Sobol_t:
            total_order_sobol_indices = operation.get_total_order_sobol_indices()
            # MARKUS
            total_sum = np.sum(total_order_sobol_indices)
            total_order_sobol = [sobol / total_sum for sobol in total_order_sobol_indices]
            total_order_sobol_indices = np.squeeze(total_order_sobol_indices)
            total_order_sobol = np.squeeze(total_order_sobol)
            print("SG Integration when computing gPCE indices - Total order Sobol indices: ", total_order_sobol_indices)
            print("SG Integration when computing gPCE indices - Total order Sobol indices Normalized: ", total_order_sobol)

    end_time_computing_statistics = time.time()
    time_computing_statistics = end_time_computing_statistics - start_time_computing_statistics

    #####################################
    # Saving and plotting
    #####################################
    if writing_results_to_a_file:
        if not build_sg_for_e_and_var:
            # gPCE = operation.get_gPCE()
            fileName = f"gpce.pkl"
            gpceFileName = str(outputModelDir / fileName)
            with open(gpceFileName, 'wb') as handle:
                pickle.dump(operation.gPCE, handle, protocol=pickle.HIGHEST_PROTOCOL)

            fileName = f"pce_polys.pkl"
            pcePolysFileName = str(outputModelDir / fileName)
            with open(pcePolysFileName, 'wb') as handle:
                pickle.dump(operation.pce_polys, handle, protocol=pickle.HIGHEST_PROTOCOL)

        fileName = f"results.txt"
        statFileName = str(outputModelDir / fileName)
        fp = open(statFileName, "w")
        fp.write(f'E: {expectedInterp}\n')
        fp.write(f'Var: {varianceInterp}\n')
        fp.write(f'First order Sobol indices: {first_order_sobol_indices}\n')
        fp.write(f'Total order Sobol indices: {total_order_sobol_indices}\n')
        if not build_sg_for_e_and_var:
            fp.write(f'time_producing_gpce: {time_producing_gpce}\n')
        fp.write(f'time_computing_statistics: {time_computing_statistics}')
        fp.close()

    if surrogate_model_of_interest.lower() != "gpce":
        # This is varaiante when only the computation of SG surrogate is of importance
        # TODO check if operation.calculate_expectation_and_variance(combiinstance) exists for operation==Integration
        return combiinstance, None, expectedInterp, varianceInterp, None, None
    else:
        return combiinstance, operation.gPCE, expectedInterp, varianceInterp, first_order_sobol_indices, total_order_sobol_indices

#####################################
# Main part
#####################################


def main_routine(model, current_output_folder, **kwargs):
    dictionary_with_inf_about_the_run = dict()

    dictionary_with_inf_about_the_run["model"] = model

    scratch_dir = cwd  # pathlib.Path("/work/ga45met")

    # default values, most likely will be overwritten later on based on settings for each model
    # TODO Experiment with this!
    can_model_evaluate_all_vector_nodes = True  # set to True if eval_vectorized is implemented,
    # though it is always inherited from sparseSpACE.Function Base class
    inputModelDir = None
    outputModelDir = None
    config_file = None
    parameters_setup_file_name = None

    if model == "larsim":
        inputModelDir = pathlib.Path("/work/ga45met/Larsim-data")
        outputModelDir = scratch_dir / "sg_anaysis" / "siam_uq" /"larsim_runs" / current_output_folder
        config_file = scratch_dir / "configurations_Larsim" / 'configurations_larsim_high_flow_small_sg.json'
        parameters_setup_file_name = scratch_dir /"configurations_Larsim"/ f"KPU_Larsim_d5.json"
    elif model == "hbvsask":
        inputModelDir = pathlib.Path("/work/ga45met/Hydro_Models/HBV-SASK-data")
        outputModelDir = scratch_dir / "sg_anaysis" / "siam_uq" / "hbvsask_runs" / current_output_folder
        config_file = scratch_dir / "configurations" / 'configuration_hbv_6D.json'
        parameters_setup_file_name = scratch_dir / "configurations" / f"KPU_HBV_d6.json"
    elif model == "ishigami":
        outputModelDir = scratch_dir / "sg_anaysis" / "siam_uq" / "ishigami_runs" / current_output_folder
        config_file = scratch_dir / "configurations" / 'configuration_ishigami.json'
        parameters_setup_file_name = scratch_dir /"configurations"/ f"KPU_ishigami_d3.json"
    else:
        outputModelDir = scratch_dir / "sg_anaysis" / "siam_uq" / model / current_output_folder
    # elif model == "gfunction":
    #     outputModelDir = scratch_dir / "sg_anaysis" / "gfunction_runs" / current_output_folder
    # elif model == "zabarras2d":
    #     outputModelDir = scratch_dir / "sg_anaysis" / "zabarras2d_runs" / current_output_folder
    # elif model == "zabarras3d":
    #     outputModelDir = scratch_dir / "sg_anaysis" / "zabarras3d_runs" / current_output_folder
    # elif model == "corner_peak":
    #     outputModelDir = scratch_dir / "sg_anaysis" / "corner_peak" / current_output_folder
    # elif model == "product_peak":
    #     outputModelDir = scratch_dir / "sg_anaysis" / "product_peak" / current_output_folder
    # elif model == "discontinuous":
    #     outputModelDir = scratch_dir / "sg_anaysis" / "discontinuous" / current_output_folder
    # elif model=="gaussian":
    #     outputModelDir = scratch_dir / "sg_anaysis" / "gaussian" / current_output_folder

    outputModelDir.mkdir(parents=True, exist_ok=True)

    dictionary_with_inf_about_the_run["config_file"] = str(config_file)
    dictionary_with_inf_about_the_run["outputModelDir"] = str(outputModelDir)

    #####################################
    # Setting the 'stochastic part'
    #####################################

    # default values, most likely will be overwritten later on based on settings for each model
    dim = 0
    param_names = []
    distributions_list_of_dicts = []
    distributionsForSparseSpace = []
    a = []
    b = []

    if config_file is not None:
        with open(config_file) as f:
            configurationObject = json.load(f)
    else:
        configurationObject = None

    # in case the set-up of the model is done via some configuration_file
    if configurationObject is not None \
            and isinstance(configurationObject, dict) and "parameters" in configurationObject:
        for single_param in configurationObject["parameters"]:
            if single_param["distribution"] != "None":
                if model == "larsim":
                    param_names.append((single_param["type"], single_param["name"]))
                else:
                    param_names.append(single_param["name"])
                dim += 1
                distributions_list_of_dicts.append(single_param)
                distributionsForSparseSpace.append((single_param["distribution"], single_param["lower"], single_param["upper"]))
                a.append(single_param["lower"])
                b.append(single_param["upper"])
    else:
        # TODO manual setup of params, param_names, dim, distributions, a, b; change this eventually. Hard-coded!
        if model == "ishigami":
            param_names = ["x0", "x1", "x2"]
            a = [-math.pi, -math.pi, -math.pi]
            b = [math.pi, math.pi, math.pi]
        elif model == "gfunction":
            param_names = ["x0", "x1", "x2"]
            a = [0.0, 0.0, 0.0]
            b = [1.0, 1.0, 1.0]
            can_model_evaluate_all_vector_nodes = True
        elif model == "zabarras2d":
            param_names = ["x0", "x1"]
            a = [-1.0, -1.0]  # [-2.5, -2]
            b = [1.0, 1.0]  # [2.5, 2]
            can_model_evaluate_all_vector_nodes = True
        elif model == "zabarras3d":
            param_names = ["x0", "x1", "x2"]
            a = [-1.0, -1.0, -1.0]  # [-2.5, -2, 5]
            b = [1.0, 1.0, 1.0]  # [2.5, 2, 15]
        elif model in ["corner_peak", "product_peak", "oscillatory", "gaussian", "discontinuous"]:
            # param_names = ["x0", "x1", "x2"]
            # a = [0.0, 0.0, 0.0]
            # b = [1.0, 1.0, 1.0]
            # dim = 3
            # coeffs, _ = generate_and_scale_coeff_and_weights(dim, b_3)
            param_names = ["x0", "x1", "x2", "x3", "x4"]
            a = [0.0, 0.0, 0.0, 0.0, 0.0]
            b = [1.0, 1.0, 1.0, 1.0, 1.0]
            dim = 5
            anisotropic = True
            if "coeffs" in kwargs:
                coeffs = kwargs["coeffs"]
                weights = None
                if "weights" in kwargs:
                    weights = kwargs["weights"]
            else:
                coeffs, weights = generate_and_scale_coeff_and_weights(dim=dim, b=b_3, anisotropic=anisotropic)
            # coeffs = [float(1) for _ in range(dim)]
            can_model_evaluate_all_vector_nodes = True
            dictionary_with_inf_about_the_run["coeffs"] = coeffs
            dictionary_with_inf_about_the_run["weights"] = weights

        dim = len(param_names)
        distributions_list_of_dicts = [{"distribution": "Uniform", "lower": a[i], "upper": b[i]} for i in range(dim)]
        distributionsForSparseSpace = [("Uniform", a[i], b[i]) for i in range(dim)]

    a = np.array(a)
    b = np.array(b)

    # setup of dist - chaospy
    dists = []
    standardDists = []
    standardDists_min_one_one = []
    standardDists_zero_one = []
    # joinedDists = None
    # joinedStandardDists = None
    for single_param_dist_config_dict in distributions_list_of_dicts:
        cp_dist_signature = inspect.signature(getattr(cp, single_param_dist_config_dict["distribution"]))
        dist_parameters_values = []
        for p in cp_dist_signature.parameters:
            dist_parameters_values.append(single_param_dist_config_dict[p])
        dists.append(getattr(cp, single_param_dist_config_dict["distribution"])(*dist_parameters_values))
        standardDists.append(getattr(cp, single_param_dist_config_dict["distribution"])())
        standardDists_min_one_one.append(getattr(cp, single_param_dist_config_dict["distribution"])(lower=-1, upper=1))
        standardDists_zero_one.append(getattr(cp, single_param_dist_config_dict["distribution"])(lower=0, upper=1))
    joinedDists = cp.J(*dists)
    joinedStandardDists = cp.J(*standardDists)
    joinedStandardDists_min_one_one = cp.J(*standardDists_min_one_one)
    # Note: in chaospy by default Uniform distribution fixed on the [-1, 1] interval, important for gPCE
    # TODO Hm, seems as this is not true! think if this trick should be done for all algorithms
    joinedStandardDists = joinedStandardDists_min_one_one

    dictionary_with_inf_about_the_run["param_names"] = param_names
    dictionary_with_inf_about_the_run["a"] = a
    dictionary_with_inf_about_the_run["b"] = b

    #####################################
    # Creation of Model Object and setting up the purpose of model run, i.e.,
    # building surrogate model or doing UQ analysis
    #####################################
    # setting default values for some 'configuration' parameters
    compute_mean = kwargs.get('compute_mean', True)
    compute_var = kwargs.get('compute_var', True)
    compute_Sobol_m = kwargs.get('compute_Sobol_m', False)
    compute_Sobol_t = kwargs.get('compute_Sobol_t', False)

    has_analyitic_mean = False
    has_analyitic_var = False
    has_analyitic_first_sobol = False
    has_analyitic_total_sobol = False

    # TODO This is probably unnecessary
    get_analytical_mean = False
    get_analytical_var = False
    get_analytical_Sobol_m = False
    get_analytical_Sobol_t = False

    surrogate_model = None

    approximated_mean = None
    approximated_var = None
    first_order_sobol_indices = None
    total_order_sobol_indices = None

    qoi = "model_output"
    operation = "UncertaintyQuantification"  # "Interpolation" "both"

    dictionary_with_inf_about_the_run["operation"] = operation

    if model == "larsim":
        qoi = "Q"  # "Q" "GoF"
        gof = "calculateLogNSE"  # "calculateRMSE" "calculateNSE"  "None"
        problem_function = sparseSpACE_functions.LarsimFunction(
            configurationObject=configurationObject,
            inputModelDir=inputModelDir,
            workingDir=outputModelDir,
            qoi=qoi,
            gof=gof
        )
        dictionary_with_inf_about_the_run["qoi"] = qoi
        dictionary_with_inf_about_the_run["gof"] = gof
        if operation == "UncertaintyQuantification":
            compute_Sobol_m = True
            compute_Sobol_t = True
    elif model == "hbvsask":
        qoi = "Q"  # "Q" "GoF"
        gof = "calculateLogNSE"  # "calculateRMSE" "calculateNSE"  "None"
        problem_function = sparseSpACE_functions.HBVSASKFunction(
            configurationObject=configurationObject,
            inputModelDir=inputModelDir,
            workingDir=outputModelDir,
            dim=dim,
            param_names=param_names,
            qoi=qoi,
            gof=gof,
            writing_results_to_a_file=False,
            plotting=False
        )
        dictionary_with_inf_about_the_run["qoi"] = qoi
        dictionary_with_inf_about_the_run["gof"] = gof
        if operation == "UncertaintyQuantification":
            compute_Sobol_m = True
            compute_Sobol_t = True
    elif model == "ishigami":
        problem_function = sparseSpACE_functions.IshigamiFunction(
            configurationObject=configurationObject, dim=dim
        )
        if operation == "UncertaintyQuantification":
            compute_Sobol_m = True
            compute_Sobol_t = True
            has_analyitic_first_sobol = True
            has_analyitic_total_sobol = True
    elif model == "gfunction":
        problem_function = sparseSpACE_functions.GFunction(dim=3)
        if operation == "UncertaintyQuantification":
            compute_Sobol_m = True
            compute_Sobol_t = True
            has_analyitic_first_sobol = True
            has_analyitic_total_sobol = True
    elif model == "zabarras2d":
        problem_function = sparseSpACE_functions.FunctionUQ2D()
    elif model == "zabarras3d":
        problem_function = sparseSpACE_functions.FunctionUQ3D()
    elif model == "product_peak":
        problem_function = GenzProductPeak(coefficients=coeffs, midpoint=weights)
    elif model == "oscillatory":
        problem_function = GenzOszillatory(coeffs=coeffs, offset=weights[0])
    elif model == "corner_peak":
        problem_function = GenzCornerPeak(coeffs=coeffs)
    elif model == "gaussian":
        problem_function = sparseSpACE_functions.GenzGaussian(midpoint=weights, coefficients=coeffs)  # TODO ubiquitous
    elif model == "discontinuous":
        problem_function = sparseSpACE_functions.GenzDiscontinious(coeffs=coeffs, border=weights)  # TODO ubiquitous

    # TODO This is probably unnecessary
    get_analytical_mean = get_analytical_mean and compute_mean and has_analyitic_mean
    get_analytical_var = get_analytical_var and compute_var and has_analyitic_var
    get_analytical_Sobol_m = get_analytical_Sobol_m and compute_Sobol_m and has_analyitic_first_sobol
    get_analytical_Sobol_t = get_analytical_Sobol_t and compute_Sobol_t and has_analyitic_total_sobol

    #####################################
    # Running the SG Simulation
    #####################################

    variant = kwargs.get('variant', 1)
    dictionary_with_inf_about_the_run["variant"] = variant

    intermediate_surrogate = None

    surrogate_model_of_interest = kwargs.get('surrogate_model_of_interest', "gpce")  # "gpce"  # "gPCE"  or "sg" this is relevant when sg surrogate is indeed computed, i.e., variant == 2 or 3 or 4
    if variant == 1:
        surrogate_model_of_interest = "gpce"
    dictionary_with_inf_about_the_run["surrogate_model_of_interest"] = surrogate_model_of_interest

    if surrogate_model_of_interest.lower() == "gpce":
        # we are always building gPCE in a 'unit'/'standard' domain
        sampleFromStandardDist_when_evaluating_surrogate = True
    else:
        sampleFromStandardDist_when_evaluating_surrogate = False

    writing_results_to_a_file = True
    plotting = True  # True

    # parameters for chaopsy quadrature, similar setup to uqef(pp)...
    quadrature_rule = kwargs.get('quadrature_rule', 'g')  #'c'
    sparse = False # True
    q_order = kwargs.get('q_order', 9) #9 #5
    p_order = kwargs.get('p_order', 4) #4 #4  # 7
    poly_rule = "three_terms_recurrence"  # "gram_schmidt" | "three_terms_recurrence" | "cholesky"
    poly_normed = kwargs.get('poly_normed', False)   # True
    sparse_quadrature = kwargs.get('sparse_quadrature', False)  #True  # False
    sampling_rule = "random"  # | "sobol" | "latin_hypercube" | "halton"  | "hammersley"
    sampleFromStandardDist = True

    read_nodes_from_file = kwargs.get('read_nodes_from_file', False)  #True
    path_to_file = pathlib.Path("/work/ga45met/sparseSpACE/sparse_grid_nodes_weights")
    # path_to_file = pathlib.Path("/dss/dsshome1/lxc0C/ga45met2/Repositories/sparse_grid_nodes_weights")
    l = kwargs.get('l', 10)
    parameters_file_name = path_to_file / f"KPU_d{dim}_l{l}.asc" # f"KPU_d3_l{l}.asc"

    dictionary_with_sg_setup = dict()

    simulation_time_start = time.time()

    if variant == 1 or variant == 2:
        dictionary_with_sg_setup["quadrature_rule"] = quadrature_rule
        dictionary_with_sg_setup["sparse_quadrature"] = sparse_quadrature
        dictionary_with_sg_setup["q_order"] = q_order
        dictionary_with_sg_setup["p_order"] = p_order
        dictionary_with_sg_setup["poly_rule"] = poly_rule
        dictionary_with_sg_setup["poly_normed"] = poly_normed
        dictionary_with_sg_setup["sampleFromStandardDist"] = sampleFromStandardDist
        dictionary_with_sg_setup["read_nodes_from_file"] = read_nodes_from_file
        dictionary_with_sg_setup["l"] = l
        if read_nodes_from_file:
            dictionary_with_sg_setup["parameters_file_name"] = str(parameters_file_name)
            if parameters_setup_file_name is not None:
                dictionary_with_sg_setup["parameters_setup_file_name"] = str(parameters_setup_file_name)
    elif variant == 4:
        dictionary_with_sg_setup["p_order"] = p_order
    dictionary_with_inf_about_the_run["dictionary_with_sg_setup"] = dictionary_with_sg_setup

    if variant == 1:
        # Var 1 #
        gPCE, approximated_mean, approximated_var, first_order_sobol_indices, total_order_sobol_indices = \
            compute_gpce_chaospy(
                model=problem_function, param_names=param_names, dists=dists, joint=joinedDists,
                jointStandard=joinedStandardDists, dim=dim, a=a, b=b, plotting=plotting,
                writing_results_to_a_file=writing_results_to_a_file, outputModelDir=outputModelDir,
                rule=quadrature_rule, sparse=sparse_quadrature, q=q_order, p=p_order, poly_rule=poly_rule,
                poly_normed=poly_normed, sampleFromStandardDist=sampleFromStandardDist,
                can_model_evaluate_all_vector_nodes=can_model_evaluate_all_vector_nodes,
                read_nodes_from_file=read_nodes_from_file, parameters_file_name=parameters_file_name,
                parameters_setup_file_name=parameters_setup_file_name,
                compute_mean=compute_mean, compute_var=compute_var,
                compute_Sobol_m=compute_Sobol_m, compute_Sobol_t=compute_Sobol_t
            )

        # in this case gPCE is a surrogate you can evaluate!
        surrogate_model = gPCE

    if variant == 2 or variant == 3 or variant == 4:
        # Var 2 | Var 3 | Var 4 - parameters for SparseSpACE
        gridName = kwargs.get('gridName', "Trapezoidal")   # "Trapezoidal" | "TrapezoidalWeighted" | "BSpline_p3" | "Leja"
        lmin = kwargs.get('lmin', 1)
        lmax = kwargs.get('lmax', 2)
        max_evals = kwargs.get('max_evals', 10**5)
        tolerance = kwargs.get('tolerance', 10 ** -5)  # or tolerance = 10 ** -20
        modified_basis = kwargs.get('modified_basis', False)
        boundary_points = kwargs.get('boundary_points', True)
        spatiallyAdaptive = kwargs.get('spatiallyAdaptive', True)
        rebalancing = kwargs.get('rebalancing', True)

        # these configuration parameters make sense when spatiallyAdaptive = True
        grid_surplusses = "grid"  # None | "grid", Note: when gridName = "Trapezoidal" grid_surplusses=None is okay...
        norm_spatiallyAdaptive = kwargs.get('norm_spatiallyAdaptive', 2) # 2 | np.inf

        dictionary_with_inf_about_the_run["gridName"] = gridName
        dictionary_with_inf_about_the_run["lmin"] = lmin
        dictionary_with_inf_about_the_run["lmax"] = lmax
        dictionary_with_inf_about_the_run["max_evals"] = max_evals
        dictionary_with_inf_about_the_run["tolerance"] = tolerance
        dictionary_with_inf_about_the_run["modified_basis"] = modified_basis
        dictionary_with_inf_about_the_run["boundary_points"] = boundary_points
        dictionary_with_inf_about_the_run["spatiallyAdaptive"] = spatiallyAdaptive
        dictionary_with_inf_about_the_run["rebalancing"] = rebalancing

        plot_file = str(outputModelDir / "output.png")
        filename_contour_plot = str(outputModelDir / "output_contour_plot.png")
        filename_refinement_graph = str(outputModelDir / "output_refinement_graph.png")
        filename_combi_scheme_plot = str(outputModelDir / "output_combi_scheme.png")
        filename_sparse_grid_plot = str(outputModelDir / "output_sg_graph.png")

    if variant == 2:
        # Var 2 #
        combiinstance, gPCE, approximated_mean, approximated_var, first_order_sobol_indices, total_order_sobol_indices = \
            compute_surrogate_sparsespace_and_gpce(model=problem_function, param_names=param_names, dists=dists,
                                                   joint=joinedDists, jointStandard=joinedStandardDists, dim=dim, a=a,
                                                   b=b, surrogate_model_of_interest=surrogate_model_of_interest,
                                                   plotting=plotting,
                                                   writing_results_to_a_file=writing_results_to_a_file,
                                                   outputModelDir=outputModelDir, gridName=gridName, lmin=lmin,
                                                   lmax=lmax, max_evals=max_evals, tolerance=tolerance,
                                                   modified_basis=modified_basis, boundary_points=boundary_points,
                                                   spatiallyAdaptive=spatiallyAdaptive, grid_surplusses=grid_surplusses,
                                                   norm_spatiallyAdaptive=norm_spatiallyAdaptive,
                                                   rebalancing=rebalancing,
                                                   rule=quadrature_rule,
                                                   sparse=sparse_quadrature, q=q_order, p=p_order, poly_rule=poly_rule,
                                                   poly_normed=poly_normed,
                                                   sampleFromStandardDist=sampleFromStandardDist,
                                                   can_model_evaluate_all_vector_nodes=can_model_evaluate_all_vector_nodes,
                                                   read_nodes_from_file=read_nodes_from_file,
                                                   parameters_file_name=parameters_file_name,
                                                   parameters_setup_file_name=parameters_setup_file_name,
                                                   compute_mean=compute_mean, compute_var=compute_var,
                                                   compute_Sobol_m=compute_Sobol_m, compute_Sobol_t=compute_Sobol_t)

        if surrogate_model_of_interest.lower() == "gpce":
            surrogate_model = gPCE
            intermediate_surrogate = combiinstance
        else:
            surrogate_model = combiinstance

    # Var 3
    # compute_surrogate_sparsespace_and_gpce_analytically(
    # a_model_param=7, b_model_param=0.1, modified_basis=True, spatiallyAdaptive=False,
    # plotting=True, outputModelDir=outputModelDir,
    # lmax=lmax, max_evals=max_evals, tolerance=tolerance,
    # rule=rule, sparse_utility=sparse_utility, q=q, p=p)

    if variant == 4:
        build_sg_for_e_and_var = kwargs.get('build_sg_for_e_and_var', True)
        if build_sg_for_e_and_var:
            surrogate_model_of_interest = "sg"
        parallelIntegrator = True

        dictionary_with_inf_about_the_run["build_sg_for_e_and_var"] = build_sg_for_e_and_var
        dictionary_with_inf_about_the_run["parallelIntegrator"] = parallelIntegrator

        combiinstance, gPCE, approximated_mean, approximated_var, first_order_sobol_indices, total_order_sobol_indices = \
            compute_gpce_sparsespace(model=problem_function, param_names=param_names, dists=distributionsForSparseSpace,
                                     dim=dim, a=a, b=b, surrogate_model_of_interest=surrogate_model_of_interest,
                                     plotting=plotting, writing_results_to_a_file=writing_results_to_a_file,
                                     outputModelDir=outputModelDir, gridName=gridName, lmin=lmin, lmax=lmax,
                                     max_evals=max_evals, tolerance=tolerance, modified_basis=modified_basis,
                                     boundary_points=boundary_points, spatiallyAdaptive=spatiallyAdaptive,
                                     grid_surplusses=grid_surplusses, norm_spatiallyAdaptive=norm_spatiallyAdaptive,
                                     rebalancing=rebalancing,
                                     p=p_order, build_sg_for_e_and_var=build_sg_for_e_and_var,
                                     parallelIntegrator=parallelIntegrator, compute_mean=compute_mean,
                                     compute_var=compute_var, compute_Sobol_m=compute_Sobol_m,
                                     compute_Sobol_t=compute_Sobol_t)
        if not build_sg_for_e_and_var and surrogate_model_of_interest.lower() == "gpce":
            surrogate_model = gPCE
            intermediate_surrogate = combiinstance
        else:
            surrogate_model = combiinstance

    #####################################
    # Access the analytical values and compare them with computed
    #####################################

    #####################################
    # E = model - surrogate_model
    #####################################
    if variant!=4:
        print(f"\n==Model Error==")
        # Evaluate surrogate model and a certain number of new points, and compute error
        # Ideas come from:
        # P. Conrad and Y. Marzouk: "ADAPTIVE SMOLYAK PSEUDOSPECTRAL APPROXIMATIONS"
        # V. Barthelmann, E. Novak, and K. Ritter: "HIGH DIMENSIONAL POLYNOMIAL INTERPOLATION ON SPARSE GRIDS"
        # TODO Experiment with this
        numSamples_for_checking = 10**dim  # Note: Big Memory problem when more than 10**4 points?
        mc_rule_for_checking = "r"  # sampling_rule or "g" or grid
        error_type = "mean"  # "mean" "l2" | "max" "l1" not relevant for now...

        dictionary_with_inf_about_the_run["comparison_surrogate_vs_model_numSamples"] = numSamples_for_checking
        dictionary_with_inf_about_the_run["comparison_surrogate_vs_model_mc_rule"] = mc_rule_for_checking
        dictionary_with_inf_about_the_run["comparison_surrogate_vs_model_error_type"] = error_type

        # TODO Experiment with different sampling strategies
        if sampleFromStandardDist:
            mc_nodes = joinedStandardDists.sample(size=numSamples_for_checking, rule=mc_rule_for_checking).round(4)
            mc_nodes = np.array(mc_nodes)
            # mc_parameters = transformSamples(mc_nodes, joinedStandardDists, joinedDists)
            mc_parameters = transformSamples_lin_or_nonlin(mc_nodes, joinedStandardDists, joinedDists, linear=False)
        else:
            mc_nodes = joinedDists.sample(size=numSamples_for_checking, rule=mc_rule_for_checking).round(4)
            mc_nodes = np.array(mc_nodes)
            mc_parameters = mc_nodes

        ######Re-evaluating Surrogate Model########
        # TODO Interesting to see if it will be more memory efficient if I would evaluate surrogate_model
        #  inside of the sub-routine and not transfer it around...
        # surrogate_evaluations = surrogate_model(mc_nodes.T)  # Var 1 - surrogate_model=gPCE;
        # surrogate_evaluations = np.array([surrogate_model(nodes) for nodes in mc_nodes.T])
        reevaluation_surrogate_model_start_time = time.time()
        # TODO - This will probably change once the output is 2D (HBV, Larsim)
        # TODO - ask if surrogate_model can evaluate all vector nodes at once
        # TODO - question if it make sense to check once again sampleFromStandardDist_when_evaluating_surrogate
        if surrogate_model_of_interest.lower() == "gpce":
            surrogate_evaluations = np.empty([mc_nodes.shape[1], ])
            i = 0
            if sampleFromStandardDist_when_evaluating_surrogate:
                for sample in mc_nodes.T:
                    surrogate_evaluations[i] = surrogate_model(*sample)
                    i += 1
            else:
                for sample in mc_parameters.T:
                    surrogate_evaluations[i] = surrogate_model(*sample)
                    i += 1
            surrogate_evaluations = np.array(surrogate_evaluations)
        else:
            if sampleFromStandardDist_when_evaluating_surrogate:
                surrogate_evaluations = np.array(surrogate_model(mc_nodes.T))
            else:
                surrogate_evaluations = np.array(surrogate_model(mc_parameters.T))
                surrogate_evaluations = np.reshape(surrogate_evaluations, surrogate_evaluations.shape[
                    0])  # TODO - This will probably change once the output is 2D (HBV, Larsim)
        reevaluation_surrogate_model_end_time = time.time()
        reevaluation_surrogate_model_duration = reevaluation_surrogate_model_end_time - reevaluation_surrogate_model_start_time
        print(f"re evaluation surrogate model duration: {reevaluation_surrogate_model_duration} "
              f"in {numSamples_for_checking} new MC points")
        # print(f"mc_nodes.shape - {mc_nodes.shape}; "
        #       f"\n type(surrogate_model) - {type(surrogate_model)};  "
        #       f"\n surrogate_model.shape - {surrogate_model.shape};")
        # print(f"type surrogate_evaluations : {type(surrogate_evaluations)}; shape {surrogate_evaluations.shape}")

        ######Re-evaluating Intermediate Surrogate Model########
        intermediate_surrogate_evaluations = None
        if intermediate_surrogate is not None:
            # intermediate_surrogate is always combibinstance and is, for now, always evaluated in original nodes
            reevaluation_intermediate_surrogate_model_start_time = time.time()
            intermediate_surrogate_evaluations = np.array(intermediate_surrogate(mc_parameters.T))
            # TODO - This will probably change once the output is 2D (HBV, Larsim)
            intermediate_surrogate_evaluations = np.reshape(intermediate_surrogate_evaluations,
                                                            intermediate_surrogate_evaluations.shape[0])
            reevaluation_intermediate_surrogate_model_end_time = time.time()
            reevaluation_intermediate_surrogate_model_duration = reevaluation_intermediate_surrogate_model_end_time - reevaluation_intermediate_surrogate_model_start_time
            print(f"re evaluation intermediate surrogate model duration: {reevaluation_intermediate_surrogate_model_duration} "
                  f"in {numSamples_for_checking} new MC points")

        ######Re-evaluating True Model########
        reevaluation_model_start_time = time.time()
        if can_model_evaluate_all_vector_nodes:
            true_model_evaluations = problem_function(mc_parameters.T)
            true_model_evaluations = np.reshape(true_model_evaluations, true_model_evaluations.shape[0])  # TODO - This will probably change once the output is 2D (HBV, Larsim)
        else:
            true_model_evaluations = np.array([problem_function(parameter) for parameter in mc_parameters.T])
            # TODO Experiment with this
            # true_model_evaluations = np.squeeze(true_model_evaluations)
            true_model_evaluations = np.reshape(true_model_evaluations, true_model_evaluations.shape[0])
        reevaluation_model_end_time = time.time()
        reevaluation_model_duration = reevaluation_model_end_time - reevaluation_model_start_time
        print(f"re evaluation model duration: {reevaluation_model_duration} in {numSamples_for_checking} new MC points")
        # print(f"mc_parameters.shape - {mc_parameters.shape}; "
        #       f"\n type(problem_function) - {type(problem_function)};")
        # print(f"type true_model_evaluations : {type(true_model_evaluations)}; shape {true_model_evaluations.shape}")

        print(f"DEBUGGING: surrogate_evaluations.shape: {surrogate_evaluations.shape}")
        print(f"DEBUGGING: true_model_evaluations.shape: {true_model_evaluations.shape}")

        # when surrogate_model_of_interest == "sg" or "combiinstance" this will produce interpolation error like in Obi's paper
        # error_linf = None
        # error_l2 = None
        # if error_type == "max" or error_type == "l1" or error_type == "L1":
        error_linf = np.max(np.abs(true_model_evaluations - surrogate_evaluations))
        if intermediate_surrogate_evaluations is not None:
            error_linf_intermediate_surrogate = np.max(np.abs(true_model_evaluations - intermediate_surrogate_evaluations))
        # error_linf = np.linalg.norm(true_model_evaluations - surrogate_evaluations, ord=np.inf)
        # elif error_type == "mean" or error_type == "l2" or error_type == "L2":
        # error_l2 = np.sqrt(np.sum((true_model_evaluations - surrogate_evaluations)**2)) / math.sqrt(abs(numSamples_for_checking))
        error_l2 = np.sqrt(np.sum((true_model_evaluations - surrogate_evaluations)**2))
        if intermediate_surrogate_evaluations is not None:
            error_l2_intermediate_surrogate = np.sqrt(np.sum((true_model_evaluations - intermediate_surrogate_evaluations) ** 2))
            # error_l2_intermediate_surrogate = np.sqrt(
            #     np.sum((true_model_evaluations - intermediate_surrogate_evaluations) ** 2)) / math.sqrt(abs(numSamples_for_checking))
        # error_l2 = np.linalg.norm(true_model_evaluations - surrogate_evaluations, ord=2)

        # else:
        #     raise Exception(f"error_type-{error_type} is not supported!!!")

        dictionary_with_inf_about_the_run["reevaluation_surrogate_model_duration"] = reevaluation_surrogate_model_duration
        dictionary_with_inf_about_the_run["reevaluation_model_duration"] = reevaluation_model_duration

        dictionary_with_inf_about_the_run["error_model_linf"] = error_linf
        dictionary_with_inf_about_the_run["error_model_l2"] = error_l2

        if intermediate_surrogate is not None and intermediate_surrogate_evaluations is not None:
            dictionary_with_inf_about_the_run["reevaluation_intermediate_surrogate_model_duration"] = reevaluation_intermediate_surrogate_model_duration
            dictionary_with_inf_about_the_run["error_intermediate_surrogate_linf"] = error_linf_intermediate_surrogate
            dictionary_with_inf_about_the_run["error_intermediate_surrogate_l2"] = error_l2_intermediate_surrogate

        print(f"Max surrogate_evaluations: {max(surrogate_evaluations)}; Min surrogate_evaluations: {min(surrogate_evaluations)};")
        print(f"Max true_model_evaluations: {max(true_model_evaluations)}; Min true_model_evaluations: {min(true_model_evaluations)};")
        print(f"Linf Error = {error_linf} \nL2 Error = {error_l2}")

        if intermediate_surrogate is not None and intermediate_surrogate_evaluations is not None:
            print(
                f"Max intermediate_surrogate_evaluations: {max(intermediate_surrogate_evaluations)}; "
                f"Min intermediate_surrogate_evaluations: {min(intermediate_surrogate_evaluations)};")
            print(f"Linf intermediate_surrogate Error = {error_linf_intermediate_surrogate} \n"
                  f"L2 intermediate_surrogate Error = {error_l2_intermediate_surrogate}")

    #####################################
    # E = model_mean - approximated_mean
    # E = model_var - approximated_var
    #####################################
    if compute_mean and approximated_mean is not None:
        print(f"\n==Mean Error==")
        if has_analyitic_mean:
            analytical_mean = problem_function.getAnalyticSolutionIntegral(a, b)
        else:
            numSamples = 10 ** 5  # numSamples_for_checking
            print(f"Computing MC based mean on {numSamples} samples, sampled rule - {sampling_rule}")
            analytical_mean, analytical_var = compute_mc_quantity(
                model=problem_function, param_names=param_names, dists=dists,
                joint=joinedDists, jointStandard=joinedStandardDists, dim=dim, a=a, b=b,
                numSamples=numSamples, rule=sampling_rule, sampleFromStandardDist=sampleFromStandardDist,
                can_model_evaluate_all_vector_nodes=can_model_evaluate_all_vector_nodes,
                read_nodes_from_file=False, compute_mean=True, compute_var=compute_var
            )
            if analytical_mean is not None:
                has_analyitic_mean = True
            if compute_var and analytical_var is not None:
                has_analyitic_var = True
        print(f"analytical_mean = {analytical_mean} \n"
              f"approximated_mean = {approximated_mean} \n"
              f"Error in mean = {abs(analytical_mean - approximated_mean)} \n")
        # TODO - This will probably change once the output is 2D (HBV, Larsim)
        dictionary_with_inf_about_the_run["analytical_mean"] = analytical_mean
        dictionary_with_inf_about_the_run["error_mean"] = abs(analytical_mean - approximated_mean)

    if compute_var and approximated_var is not None:
        print(f"\n==Var Error==")
        if not has_analyitic_var:
            numSamples = 10 ** 5  # numSamples_for_checking
            print(f"Computing MC based mean on {numSamples} samples, sampled rule - {sampling_rule}")
            analytical_mean, analytical_var = compute_mc_quantity(
                model=problem_function, param_names=param_names, dists=dists,
                joint=joinedDists, jointStandard=joinedStandardDists, dim=dim, a=a, b=b,
                numSamples=numSamples, rule=sampling_rule, sampleFromStandardDist=sampleFromStandardDist,
                can_model_evaluate_all_vector_nodes=can_model_evaluate_all_vector_nodes,
                read_nodes_from_file=False, compute_mean=compute_mean, compute_var=True
            )
            if compute_mean and analytical_mean is not None:
                has_analyitic_mean = True
            if analytical_var is not None:
                has_analyitic_var = True
        print(f"analytical_var = {analytical_var} \n"
              f"approximated_var = {approximated_var} \n"
              f"Error in var = {abs(analytical_var - approximated_var)} \n")
        # TODO - This will probably change once the output is 2D (HBV, Larsim)
        dictionary_with_inf_about_the_run["analytical_var"] = analytical_var
        dictionary_with_inf_about_the_run["error_var"] = abs(analytical_var - approximated_var)

    #####################################
    if (compute_Sobol_m and first_order_sobol_indices is not None) \
            and (compute_Sobol_t and total_order_sobol_indices is not None):
        print(f"\n==Sobol Indices Error==")
        if has_analyitic_first_sobol and has_analyitic_total_sobol:
            sobol_m_analytical, sobol_t_analytical = problem_function.get_analytical_sobol_indices()
        else:
            raise Exception(f"Still not implemented - to compute approximation of analytical Sobol indices "
                            f"via Salteli method with many points")
        sobol_m_error = sobol_m_analytical - first_order_sobol_indices
        sobol_t_error = sobol_t_analytical - total_order_sobol_indices

        print(f"sobol_m_analytical = {sobol_m_analytical}")
        print(f"first_order_sobol_indices = {first_order_sobol_indices}")
        print(f"Sobol Main Error = {sobol_m_error}")
        # print("Sobol Main Error: {}".format(sobol_m_error, ".6f"))

        print(f"sobol_t_analytical = {sobol_t_analytical}")
        print(f"total_order_sobol_indices = {total_order_sobol_indices}")
        print(f"Sobol Total Error = {sobol_t_error}")

        dictionary_with_inf_about_the_run[
            "sobol_m_analytical"] = sobol_m_analytical
        dictionary_with_inf_about_the_run[
            "sobol_t_analytical"] = sobol_t_analytical
        dictionary_with_inf_about_the_run[
            "sobol_m_error"] = sobol_m_error
        dictionary_with_inf_about_the_run[
            "sobol_t_error"] = sobol_t_error

        # print("Sobol Total Error: {}".format(sobol_t_error, ".6f"))

    # # Sobol_t_error = sobol_t_analytical - total_order_sobol_indices
    # # Sobol_m_error = sobol_m_analytical - first_order_sobol_indices
    # # print("Sobol_t_error: {}".format(Sobol_t_error, ".6f"))
    # # # print(f"Sobol Total Error = {Sobol_t_error:.6f} \n")
    # # print("Sobol_m_error: {}".format(Sobol_m_error, ".6f"))
    # # # print(f"Sobol Main Error = {Sobol_m_error:.6f} \n")
    #
    # # sobol_t_error = np.empty(len(labels), dtype=np.float64)
    # sobol_t_error = sobol_t_analytical - total_order_sobol_indices
    # for i in range(len(labels)):
    #     # print(f"Sobol Total Simulation = {total_order_sobol_indices[i][0]} \n")
    #     # print(f"Sobol Total Analytical = {sobol_t_analytical[i]:.6f} \n")
    #     # sobol_t_error[i] = sobol_t_analytical[i] - total_order_sobol_indices[i][0]
    #     print(f"Sobol Total Error = {sobol_t_error[i]:.6f} \n")
    # #
    #
    # # sobol_m_error = np.empty(len(labels), dtype=np.float64)
    # sobol_m_error = sobol_m_analytical - first_order_sobol_indices
    # for i in range(len(labels)):
    #     # print(f"Sobol Main Simulation = {first_order_sobol_indices[i][0]} \n")
    #     # print(f"Sobol Main Analytical = {sobol_m_analytical[i]:.6f} \n")
    #     # sobol_m_error[i] = sobol_m_analytical[i] - first_order_sobol_indices[i][0]
    #     print(f"Sobol Main Error = {sobol_m_error[i]:.6f} \n")

    # # Sobol_t_error = sobol_t_analytical - total_order_sobol_indices
    # # Sobol_m_error = sobol_m_analytical - first_order_sobol_indices
    # # print("Sobol_t_error: {}".format(Sobol_t_error, ".6f"))
    # # # print(f"Sobol Total Error = {Sobol_t_error:.6f} \n")
    # # print("Sobol_m_error: {}".format(Sobol_m_error, ".6f"))
    # # # print(f"Sobol Main Error = {Sobol_m_error:.6f} \n")
    #
    # #
    # sobol_m_error = np.empty(len(labels), dtype=np.float64)
    # for i in range(len(labels)):
    #     # print(f"Sobol Main Simulation = {first_order_sobol_indices[i][0]} \n")
    #     # print(f"Sobol Main Analytical = {sobol_m_analytical[i]:.6f} \n")
    #     sobol_m_error[i] = sobol_m_analytical[i] - first_order_sobol_indices[i][0]
    #     print(f"Sobol Main Error = {sobol_m_error[i]:.6f} \n")
    #
    # sobol_t_error = np.empty(len(labels), dtype=np.float64)
    # for i in range(len(labels)):
    #     # print(f"Sobol Total Simulation = {total_order_sobol_indices[i][0]} \n")
    #     # print(f"Sobol Total Analytical = {sobol_t_analytical[i]:.6f} \n")
    #     sobol_t_error[i] = sobol_t_analytical[i] - total_order_sobol_indices[i][0]
    #     print(f"Sobol Total Error = {sobol_t_error[i]:.6f} \n")

    #####################################
    # Plotting graphs for convergence
    #####################################

    #####################################
    # Saving the final dictionary
    #####################################

    simulation_time_end = time.time()
    simulation_time = simulation_time_end - simulation_time_start
    print("simulation time: {} sec".format(simulation_time))

    dictionary_with_inf_about_the_run["simulation_time"] = simulation_time

    # dictionary_with_inf_about_the_run_path = str(outputModelDir / "dictionary_with_inf_about_the_run.json")
    dictionary_with_inf_about_the_run_path = str(outputModelDir / "dictionary_with_inf_about_the_run.pkl")
    with open(dictionary_with_inf_about_the_run_path, "wb") as handle:
    # with open(dictionary_with_inf_about_the_run_path, "w") as handle:
        pickle.dump(dictionary_with_inf_about_the_run, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # json.dump(dictionary_with_inf_about_the_run, handle)

    return dictionary_with_inf_about_the_run


if __name__ == "__main__":

    #####################################
    # Initial Model Setup
    #####################################
    list_of_models = ["hbvsask", "larsim", "ishigami", "gfunction", "zabarras2d", "zabarras3d",
                      "oscillatory", "product_peak", "corner_peak", "gaussian", "discontinuous"]
    # Additional Genz Options: GenzOszillatory, GenzDiscontinious2, GenzC0, GenzGaussian

    # uncomment if you want to run analysis for Genz functions...
    list_of_genz_functions = ["oscillatory", "product_peak", "corner_peak", "gaussian", "continous", "discontinuous"]
    path_to_saved_all_genz_functions = pathlib.Path("/work/ga45met/Backup/UQEFPP/sg_anaysis/genz_functions")
    read_saved_genz_functions = True
    anisotropic = True

    # current_output_folder = "sg_gaussian_3d_p9_q12_poly_normed"  # "sg_ss_ct_modified_var2_l_2_p_4_q_5_max_2000"
    # current_output_folder = "var2_sg_trap_ct_boundery_l_4_p_4_q_5_max_4000"  # "sg_ss_ct_modified_var2_l_2_p_4_q_5_max_2000"
    # current_output_folder = "var4_ct_trap_adaptive_boundary_modified_l_2_max_4000_saved_aniso"  # "sg_ss_ct_modified_var2_l_2_p_4_q_5_max_2000"
    # current_output_folder = "sg_cc_5d_l2_sparse_p4_q8_saved_aniso"
    # current_output_folder = "va2_gpce_trap_boundary_nonmodif_adaptive_norm2_lmin2_lmax_4_maxeval_105_tol105_g_q9_p7"
    # current_output_folder = "va2_combi_trap_boundary_nonmodif_adaptive_norm2_lmin1_lmax_5_maxeval_104_tol105"
    # current_output_folder = "var1_gpce_gl_p4_q9"  # q=5,7,9
    # current_output_folder = "var1_gpce_gl_p6_q7"
    # current_output_folder = "var1_gpce_gl_p8_q9"

    ######corner_peak Var 1#######
    list_of_dict_run_setups = [
        {"model": "corner_peak", "list_of_function_ids": [1, ], "current_output_folder": "var1_gpce_cc_p3_q7",
         "variant": 1, "quadrature_rule": "c", "q_order": 7, "p_order": 3, "sparse_quadrature": True,
         "read_nodes_from_file": False, 'l': 10},
    ]

    for single_setup_dict in list_of_dict_run_setups:
        model = single_setup_dict["model"]
        assert(model in list_of_models)
        list_of_function_ids = single_setup_dict.get("list_of_function_ids", None)
        if model in list_of_genz_functions and list_of_function_ids is not None:
            number_of_functions = len(list_of_function_ids)
        else:
            number_of_functions = 1

        current_output_folder = single_setup_dict["current_output_folder"]
        variant = single_setup_dict["variant"]

        quadrature_rule = single_setup_dict.get("quadrature_rule", None)
        q_order = single_setup_dict.get("q_order", None)
        p_order = single_setup_dict.get("p_order", None)
        sparse_quadrature = single_setup_dict.get("sparse_quadrature", None)
        read_nodes_from_file = single_setup_dict.get("read_nodes_from_file", None)
        l = single_setup_dict.get("l", None)

        start_time = time.time()
        # TODO Change for Genz that this is executed in this way whenever user wants that
        if model in list_of_genz_functions and list_of_function_ids is not None:
            # Hard-coded
            dim = 5
            # all_coeffs = np.empty(shape=(number_of_functions, dim))
            # all_weights = np.empty(shape=(number_of_functions, dim))
            # problem_function_list = []
            from collections import defaultdict
            dictionary_with_inf_about_the_run = defaultdict(dict)
            # for i in range(number_of_functions):
            for i in list_of_function_ids:
                if read_saved_genz_functions:
                    if anisotropic:
                        path_to_saved_genz_functions = str(path_to_saved_all_genz_functions / model / f"coeffs_weights_anisotropic_{dim}d_{i}.npy")
                    else:
                        path_to_saved_genz_functions = str(path_to_saved_all_genz_functions / model / f"coeffs_weights_{dim}d_{i}.npy")
                    with open(path_to_saved_genz_functions, 'rb') as f:
                        coeffs_weights = np.load(f)
                        single_coeffs = coeffs_weights[0]
                        single_weights = coeffs_weights[1]
                else:
                    single_coeffs, single_weights = generate_and_scale_coeff_and_weights(
                        dim=dim, b=genz_dict[model], anisotropic=anisotropic)
                # all_coeffs[i] = single_coeffs
                # all_weights[i] = single_weights
                current_output_folder_single_model = f"{current_output_folder}_model_{i}"
                dictionary_with_inf_about_the_run_single_model = main_routine(
                    model, current_output_folder_single_model, coeffs=single_coeffs, weights=single_weights,
                    variant=variant, quadrature_rule=quadrature_rule, q_order=q_order, p_order=p_order,
                    sparse_quadrature=sparse_quadrature, read_nodes_from_file=read_nodes_from_file, l=l
                )
                # dictionary_with_inf_about_the_run.append(dictionary_with_inf_about_the_run_single_model)
                dictionary_with_inf_about_the_run[i] = dictionary_with_inf_about_the_run_single_model
                # outputModelDir = cwd
                # dictionary_with_inf_about_the_run_path = str(outputModelDir / "dictionary_with_inf_about_the_multiple_corner_peak_runs.pkl")
                # with open(dictionary_with_inf_about_the_run_path, "wb") as handle:
                #     # with open(dictionary_with_inf_about_the_run_path, "w") as handle:
                #     pickle.dump(dictionary_with_inf_about_the_run, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            dictionary_with_inf_about_the_run = main_routine(
                model, current_output_folder,
                variant=variant, quadrature_rule=quadrature_rule, q_order=q_order, p_order=p_order,
                sparse_quadrature=sparse_quadrature, read_nodes_from_file=read_nodes_from_file, l=l
            )
        end_time = time.time()
        duration = end_time - start_time
        print(f"The whole run took {duration} for examing {number_of_functions} different functions")


# TODO finish this!
# def compute_leja_sg_surrogate_and_gpce(a_model_param=7, b_model_param=0.1, modified_basis=False,
#                                        spatiallyAdaptive=True, plotting=True, outputModelDir="./",
#                                        lmax=2, max_evals=2000, tolerance=10 ** -5,
#                                        rule='gaussian', sparse_utility=False, q=7, p=6):
#     x1 = cp.Uniform(-math.pi, math.pi)
#     x2 = cp.Uniform(-math.pi, math.pi)
#     x3 = cp.Uniform(-math.pi, math.pi)
#     joint = cp.J(x1, x2, x3)
#     labels = [param_name.strip() for param_name in ["x1", "x2", "x3"]]
#
#     leja_surrogate = None
#     _compute_gpce_chaospy_ishigami(a_model_param=a_model_param, b_model_param=b_model_param,
#                                    labels=labels, my_model=leja_surrogate, joint=joint,
#                                    q=q, p=p, rule=rule, sparse_utility=sparse_utility,
#                                    outputModelDir=outputModelDir, can_model_evaluate_all_vector_nodes=True)
