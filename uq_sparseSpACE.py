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
sys.path.insert(0, parent)
# sys.path.insert(0, os.getcwd())

from sparseSpACE.Function import *
from sparseSpACE.spatiallyAdaptiveSingleDimension2 import *
from sparseSpACE.ErrorCalculator import *
from sparseSpACE.GridOperation import *
from sparseSpACE.StandardCombi import *
from sparseSpACE.Integrator import *

from sparse import sparseSpACE_functions
from sparse import Sparse_Quadrature
# from sparseSpACE_functions import *
# from Sparse_Quadrature import *

comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()


def transformSamples(samples, distribution_r, distribution_q):
    return distribution_q.inv(distribution_r.fwd(samples))

# Var 0 - TODO Add MC 10^4 for computing reference value...

# Var 1
def compute_gpce_chaospy(model, param_names, dists, joint, jointStandard, dim, a, b, plotting=False,
                         writing_results_to_a_file=False, outputModelDir="./", rule='gaussian', sparse=True, q=7, p=6,
                         poly_rule="three_terms_recurrence", poly_normed=False, sampleFromStandardDist=False,
                         can_model_evaluate_all_vector_nodes=False, vector_model_output=False, **kwargs):
    """
    This is all that UQEF+UQEFPP do in short, when uq_method=sc
    - sparse vs. non-sparse
    - model output single vs. vector
    - transformation vs. not
    - question weather there is analytical values
    """
    labels = [param_name.strip() for param_name in param_names]

    growth = True if (rule == "c" and not sparse) else False

    compute_Sobol_m = kwargs.get('compute_Sobol_m', True)
    compute_Sobol_t = kwargs.get('compute_Sobol_t', True)

    get_analytical_mean = kwargs.get('get_analytical_mean', False)
    get_analytical_var = kwargs.get('get_analytical_var', False)
    get_analytical_Sobol_m = kwargs.get('get_analytical_Sobol_m', False)
    get_analytical_Sobol_t = kwargs.get('get_analytical_Sobol_t', False)

    if sampleFromStandardDist:
        dist = jointStandard
    else:
        dist = joint

    # TODO add option for reading nodes and weights from a file, e.g., sparse-grids.de
    quads = cp.generate_quadrature(q, dist, rule=rule, growth=growth, sparse=sparse)
    nodes, weights = quads

    # __restore__cpu_affinity()
    nodes = np.array(nodes)
    weights = np.array(weights)

    if sampleFromStandardDist:
        parameters = transformSamples(nodes, jointStandard, joint)
    else:
        parameters = nodes

    # evaluate model
    start_time_model_evaluations = time.time()
    if can_model_evaluate_all_vector_nodes:
        evaluations = model(parameters.T)
    else:
        evaluations = np.array([model(parameter) for parameter in parameters.T])
    end_time_model_evaluations = time.time()

    time_model_evaluations = end_time_model_evaluations - start_time_model_evaluations
    number_full_model_evaluations = len(parameters.T)

    start_time_producing_gpce = time.time()
    expansion = cp.generate_expansion(p, dist, rule=poly_rule, normed=poly_normed)
    gPCE = cp.fit_quadrature(expansion, nodes, weights, evaluations)
    end_time_producing_gpce = time.time()

    time_producing_gpce = end_time_producing_gpce - start_time_producing_gpce

    expectedInterp = cp.E(gPCE, dist)
    varianceInterp = cp.Var(gPCE, dist)
    print("expectation = ", expectedInterp, ", variance = ", varianceInterp)
    first_order_sobol_indices = None
    total_order_sobol_indices = None
    if compute_Sobol_m:
        first_order_sobol_indices = cp.Sens_m(gPCE, dist)
        print("First order Sobol indices: ", first_order_sobol_indices)
    if compute_Sobol_t:
        total_order_sobol_indices = cp.Sens_t(gPCE, dist)
        print("Total order Sobol indices: ", total_order_sobol_indices)

    print(f'time_model_evaluations: {time_model_evaluations}\n')
    print(f'time_producing_gpce: {time_producing_gpce} \n')
    print(f'number_full_model_evaluations: {number_full_model_evaluations}')

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
        fp.write(f'E: {expectedInterp},\n Var: {varianceInterp}, \n '
                 f'First order Sobol indices: {first_order_sobol_indices} \n; '
                 f'Total order Sobol indices: {total_order_sobol_indices} \n'
                 f'time_model_evaluations: {time_model_evaluations}\n'
                 f'time_producing_gpce: {time_producing_gpce} \n'
                 f'number_full_model_evaluations: {number_full_model_evaluations}')
        fp.close()
    #####################################
    # TODO access to analytical values and compare them with computed

    # sobol_m_analytical, sobol_t_analytical = get_analytical_sobol_indices(a_model_param, b_model_param)
    #
    # # Sobol_t_error = sobol_t_analytical - total_order_sobol_indices
    # # Sobol_m_error = sobol_m_analytical - first_order_sobol_indices
    # # print("Sobol_t_error: {}".format(Sobol_t_error, ".6f"))
    # # # print(f"Sobol Total Error = {Sobol_t_error:.6f} \n")
    # # print("Sobol_m_error: {}".format(Sobol_m_error, ".6f"))
    # # # print(f"Sobol Main Error = {Sobol_m_error:.6f} \n")
    #
    # sobol_t_error = np.empty(len(labels), dtype=np.float64)
    # for i in range(len(labels)):
    #     # print(f"Sobol Total Simulation = {total_order_sobol_indices[i][0]} \n")
    #     # print(f"Sobol Total Analytical = {sobol_t_analytical[i]:.6f} \n")
    #     sobol_t_error[i] = sobol_t_analytical[i] - total_order_sobol_indices[i][0]
    #     print(f"Sobol Total Error = {sobol_t_error[i]:.6f} \n")
    # #
    # sobol_m_error = np.empty(len(labels), dtype=np.float64)
    # for i in range(len(labels)):
    #     # print(f"Sobol Main Simulation = {first_order_sobol_indices[i][0]} \n")
    #     # print(f"Sobol Main Analytical = {sobol_m_analytical[i]:.6f} \n")
    #     sobol_m_error[i] = sobol_m_analytical[i] - first_order_sobol_indices[i][0]
    #     print(f"Sobol Main Error = {sobol_m_error[i]:.6f} \n")

    return gPCE, expectedInterp, varianceInterp, first_order_sobol_indices, total_order_sobol_indices


# Var 3
def compute_surrogate_sparsespace_and_gpce(model, param_names, dists, joint, jointStandard, dim, a, b,
                                           modified_basis=False, boundery_points=False, spatiallyAdaptive=True,
                                           plotting=True, outputModelDir="./",
                                           lmax=2, max_evals=2000, tolerance=10 ** -5,
                                           rule='gaussian', sparse=False, q=7, p=6, model_paramters=None,
                                           can_model_evaluate_all_vector_nodes=False, **kwargs):
    """
    Var 2 - Compute gPCE coefficients by integrating the (SG) surrogate
    SG surrogate computed based on SparseSpACE
      - adaptive vs. non-adaptive
      - different grids
    gPCE coefficients - chaospy - TODO think about transformation!
      - the same set of options for chaospy.Quadrature as in compute_gpce_chaospy
    """

    labels = [param_name.strip() for param_name in param_names]
    #####################################
    grid = GlobalTrapezoidalGrid(a=a, b=b, modified_basis=modified_basis, boundary=boundery_points)
    # grid = GlobalTrapezoidalGridWeighted(a, b, boundary=boundery_points, modified_basis=modified_basis)

    operation = Integration(f=model, grid=grid, dim=dim)

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

    # TODO - Transformation? - the same story as in _compute_gpce_chaospy_ishigami
    quads = cp.generate_quadrature(q, joint, rule=rule, sparse=sparse)
    nodes, weights = quads

    # evaluate surrogate
    evaluations = combiinstance(nodes.T)

    expansion = cp.generate_expansion(p, joint) # TODO Normalized polynomials!? - the same story as in _compute_gpce_chaospy_ishigami
    gPCE = cp.fit_quadrature(expansion, nodes, weights, evaluations)

    expectedInterp = cp.E(gPCE, joint)
    varianceInterp = cp.Var(gPCE, joint)
    first_order_sobol_indices = cp.Sens_m(gPCE, joint)
    total_order_sobol_indices = cp.Sens_t(gPCE, joint)
    first_order_sobol_indices = np.squeeze(first_order_sobol_indices)
    total_order_sobol_indices = np.squeeze(total_order_sobol_indices)

    print("expectation = ", expectedInterp, ", variance = ", varianceInterp)
    print("First order Sobol indices: ", first_order_sobol_indices)
    print("Total order Sobol indices: ", total_order_sobol_indices)

    # sobol_m_analytical, sobol_t_analytical = get_analytical_sobol_indices(a_model_param, b_model_param)
    #
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


# Var 4
def compute_gpce_sparsespace(build_sg_for_e_and_var=True, modified_basis=False,
                             parallelIntegrator=False, spatiallyAdaptive=True,
                             plotting=False, outputModelDir="./", lmax=2, max_evals=2000, tolerance=10 ** -5, p=6):
    """
    This method relies on current implementation of UncertaintyQuantification Operation in SparseSpACE
    If build_sg_for_e_and_var == True
    ---> Build one SG surrogate to approximate E and Var (Note: Markus var 2)
    Else
    --> Build one SG surrogate to approximate all N coefficients of the gPCE expansion - Erroneous!
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


# utility for genz
b_1 = 1.5 # 9.0
b_2 = 7.2
b_3 = 1.85
b_4 = 7.03
b_5 = 20.4
b_6 = 4.3
def generate_and_scale_coeff_and_weights(dim, b, w_norm=1):
    coeffs = cp.Uniform(0, 1).sample(dim)
    l1 = norm(coeffs, 1)
    coeffs = coeffs * b / l1
    # coeffs = np.array([coeff*np.exp(i/dim) for coeff, i in zip(coeffs,range(1,dim+1))])  # less isotropic - conrad and marzouk
    weights = cp.Uniform(0,1).sample(dim)
    l1 = norm(weights, 1)
    weights = weights * w_norm / l1
    return coeffs, weights


if __name__ == "__main__":

    model = "corner_peak"  #"hbvsask"
    list_of_models = ["hbvsask", "larsim", "ishigami", "gfunction", "zabarras2d", "zabarras3d",
                      "corner_peak", "product_peak", "discontinuous"]
    # Additional Genz Options: GenzOszillatory, GenzDiscontinious2, GenzC0, GenzGaussian

    assert(model in list_of_models)

    can_model_evaluate_all_vector_nodes = False
    has_analyitic_mean = False
    has_analyitic_var = False
    has_analyitic_first_sobol = False
    has_analyitic_total_sobol = False

    current_output_folder = "sg_ss_ct_modified_var2_l_2_p_4_q_5_max_2000"
    scratch_dir = pathlib.Path("/work/ga45met")
    inputModelDir = None
    outputModelDir = None
    config_file = None
    if model == "larsim":
        inputModelDir = pathlib.Path("/work/ga45met/Larsim-data")
        outputModelDir = scratch_dir / "larsim_runs" / current_output_folder
        config_file = pathlib.Path(
            '/work/ga45met/mnt/linux_cluster_2/UQEFPP/configurations_Larsim/configurations_larsim_high_flow_small_sg.json')
    elif model == "hbvsask":
        inputModelDir = pathlib.Path("/work/ga45met/Hydro_Models/HBV-SASK-data")
        outputModelDir = scratch_dir / "hbvsask_runs" / current_output_folder
        config_file = pathlib.Path(
            '/work/ga45met/mnt/linux_cluster_2/UQEFPP/configurations/configuration_hbv.json')
    elif model == "ishigami":
        outputModelDir = scratch_dir / "ishigami_runs" / current_output_folder
        config_file = pathlib.Path('/work/ga45met/mnt/linux_cluster_2/UQEFPP/configurations/configuration_ishigami.json')
    elif model == "gfunction":
        outputModelDir = scratch_dir / "gfunction_runs" / current_output_folder
    elif model == "zabarras2d":
        outputModelDir = scratch_dir / "zabarras2d_runs" / current_output_folder
    elif model == "zabarras3d":
        outputModelDir = scratch_dir / "zabarras3d_runs" / current_output_folder
    elif model == "corner_peak":
        outputModelDir = scratch_dir / "corner_peak" / current_output_folder
    elif model == "product_peak":
        outputModelDir = scratch_dir / "product_peak" / current_output_folder
    elif model == "discontinuous":
        outputModelDir = scratch_dir / "discontinuous" / current_output_folder

    outputModelDir.mkdir(parents=True, exist_ok=True)

    ##################

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
        # TODO manual setup of params, param_names, dim, distributions, a, b; change this eventually
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
        elif model == "corner_peak":
            # param_names = ["x0", "x1", "x2"]
            # a = [0.0, 0.0, 0.0]
            # b = [1.0, 1.0, 1.0]
            # dim = 3
            # coeffs, _ = generate_and_scale_coeff_and_weights(dim, b_3)
            param_names = ["x0", "x1", "x2", "x3", "x4"]
            a = [0.0, 0.0, 0.0, 0.0, 0.0]
            b = [1.0, 1.0, 1.0, 1.0, 1.0]
            dim = 5
            coeffs, _ = generate_and_scale_coeff_and_weights(dim=dim, b=b_3)
            # coeffs = [float(1) for _ in range(dim)]
        elif model == "product_peak":
            param_names = ["x0", "x1", "x2"]
            a = [0.0, 0.0, 0.0]
            b = [1.0, 1.0, 1.0]
            dim = 3
            coeffs, weights = generate_and_scale_coeff_and_weights(dim=dim, b=dim)
            # coeffs = [float(3) for _ in range(dim)]
            # midpoint = [0.5 for _ in range(dim)]
        elif model == "discontinuous":
            param_names = ["x0", "x1", "x2"]
            a = [0.0, 0.0, 0.0]
            b = [1.0, 1.0, 1.0]
            dim = 3
            coeffs, weights = generate_and_scale_coeff_and_weights(dim=dim, b=b_6)
            # coeffs = [float(1) for _ in range(dim)]
            # midpoint = [0.5 for _ in range(dim)]
            can_model_evaluate_all_vector_nodes = True
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
    joinedDists = None
    joinedStandardDists = None
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

    ##################
    # Creation of Model Object
    ##################
    qoi = "Q"  # "Q" "GoF"
    gof = "calculateLogNSE"   # "calculateRMSE" "calculateNSE"  "None"
    operation = "UncertaintyQuantification"  # "Interpolation"

    if model == "larsim":
        problem_function = sparseSpACE_functions.LarsimFunction(
            configurationObject=configurationObject,
            inputModelDir=inputModelDir,
            workingDir=outputModelDir,
            qoi=qoi,
            gof=gof
        )
    elif model == "hbvsask":
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
    elif model == "ishigami":
        problem_function = sparseSpACE_functions.IshigamiFunction(
            configurationObject=configurationObject
        )
    elif model == "gfunction":
        problem_function = sparseSpACE_functions.GFunction(dim=3)
    elif model == "zabarras2d":
        problem_function = sparseSpACE_functions.FunctionUQ2D()
    elif model == "zabarras3d":
        problem_function = sparseSpACE_functions.FunctionUQ3D()
    elif model == "corner_peak":
        problem_function = GenzCornerPeak(coeffs)
    elif model == "product_peak":
        problem_function = GenzProductPeak(coeffs, weights)
    elif model == "discontinuous":
        problem_function = GenzDiscontinious(coeffs, weights)  # TODO ubiquitous

    ##################
    # Running the SG Simulation

    writing_results_to_a_file = False
    plotting = False #True

    # parameters for chaopsy quadrature, similar setup to uqef(pp)...
    quadrature_rule = 'gaussian'
    sparse = False
    q_order = 10
    p_order = 9
    poly_rule = "three_terms_recurrence"  # "gram_schmidt" | "three_terms_recurrence" | "cholesky"
    poly_normed = False #True
    sparse_quadrature = False  # False
    sampling_rule = "random"  # | "sobol" | "latin_hypercube" | "halton"  | "hammersley"
    sampleFromStandardDist = True

    # read_nodes_from_file = False
    # path_to_file = pathlib.Path("/dss/dsshome1/lxc0C/ga45met2/Repositories/sparse_grid_nodes_weights")
    # parameters_file = path_to_file / f"KPU_d5_l{l}.asc" # f"KPU_d3_l{l}.asc"
    # parameters_setup_file = pathlib.Path("/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEFPP/configurations_Larsim/KPU_Larsim_d5.json")

    # parameters for SparseSpACE
    lmax = 2  # 4
    max_evals = 2000  # 4000
    tolerance = 10 ** -5
    modified_basis = False
    boundery_points = False
    spatiallyAdaptive = True

    plot_file = str(outputModelDir / "output.png")
    filename_contour_plot = str(outputModelDir / "output_contour_plot.png")
    filename_refinement_graph = str(outputModelDir / "output_refinement_graph.png")
    filename_combi_scheme_plot = str(outputModelDir / "output_combi_scheme.png")
    filename_sparse_grid_plot = str(outputModelDir / "output_sg_graph.png")

    simulation_time_start = time.time()

    # Var 1 #
    gPCE, expectedInterp, varianceInterp, first_order_sobol_indices, total_order_sobol_indices = \
        compute_gpce_chaospy(model=problem_function, param_names=param_names, dists=dists, joint=joinedDists,
                             jointStandard=joinedStandardDists, dim=dim, a=a, b=b, plotting=plotting,
                             writing_results_to_a_file=writing_results_to_a_file, outputModelDir=outputModelDir,
                             rule=quadrature_rule, sparse=sparse_quadrature, q=q_order, p=p_order, poly_rule=poly_rule,
                             poly_normed=poly_normed, sampleFromStandardDist=sampleFromStandardDist,
                             can_model_evaluate_all_vector_nodes=can_model_evaluate_all_vector_nodes
                             )

    # Var 2 #
    # compute_surrogate_sparsespace_and_gpce(
    #     model=problem_function,
    #     param_names=param_names,
    #     dists=dists,
    #     joint=joinedDists,
    #     jointStandard=joinedStandardDists,
    #     dim=dim,
    #     a=a,
    #     b=b,
    #     modified_basis=modified_basis,
    #     boundery_points=boundery_points,
    #     spatiallyAdaptive=spatiallyAdaptive,
    #     plotting=plotting,
    #     outputModelDir=outputModelDir,
    #     lmax=lmax,
    #     max_evals=max_evals,
    #     tolerance=tolerance,
    #     rule=quadrature_rule,
    #     sparse=sparse_quadrature,
    #     q=q_order,
    #     p=p_order,
    #     sampleFromStandardDist=sampleFromStandardDist,
    #     can_model_evaluate_all_vector_nodes=can_model_evaluate_all_vector_nodes
    # )


    # Var 3
    # compute_surrogate_sparsespace_and_gpce_analytically(
    # a_model_param=7, b_model_param=0.1, modified_basis=True, spatiallyAdaptive=False,
    # plotting=True, outputModelDir=outputModelDir,
    # lmax=lmax, max_evals=max_evals, tolerance=tolerance,
    # rule=rule, sparse=sparse, q=q, p=p)

    # Var 4
    # compute_gpce_sparsespace(build_sg_for_e_and_var=False, modified_basis=False, parallelIntegrator=True,
    #                          spatiallyAdaptive=True, plotting=plotting, outputModelDir=outputModelDir,
    #                          lmax=lmax, max_evals=max_evals, tolerance=tolerance, p=p)

    # compute_gpce_sparsespace(build_sg_for_e_and_var=True, modified_basis=True, parallelIntegrator=False,
    #                          spatiallyAdaptive=False, plotting=plotting, outputModelDir=outputModelDir,
    #                          lmax=lmax, max_evals=max_evals, tolerance=tolerance)

    # TODO Seems as this is not working - when building SG surrogate to fit computation of c_n!
    # compute_gpce_sparsespace(build_sg_for_e_and_var=False, modified_basis=True, parallelIntegrator=False,
    #                          spatiallyAdaptive=True, plotting=plotting, outputModelDir=outputModelDir,
    #                          lmax=lmax, max_evals=max_evals, tolerance=tolerance)
    # compute_gpce_sparsespace(build_sg_for_e_and_var=False, modified_basis=True, parallelIntegrator=False,
    #                          spatiallyAdaptive=False, plotting=plotting, outputModelDir=outputModelDir,
    #                          lmax=lmax, max_evals=max_evals, tolerance=tolerance)

    simulation_time_end = time.time()
    simulation_time = simulation_time_end - simulation_time_start
    print("simulation time: {} sec".format(simulation_time))


# # TODO
# def compute_surrogate_sparsespace_and_gpce_analytically(a_model_param=7, b_model_param=0.1, modified_basis=False,
#                                            spatiallyAdaptive=True,
#                                            plotting=True, outputModelDir="./",
#                                            lmax=2, max_evals=2000, tolerance=10 ** -5,
#                                            rule='gaussian', sparse=False, q=7, p=6):
#     """
#     Var 2.2 (For Markus this is var 4) - Compute gPCE coefficients by integrating the (SG) surrogate analytically
#     The gPCE coefficients are calculated as follows:
#     $c_n = \int_{(0,1)^d} f_{interp}^{nonlin}(T_{nonlin}(u)\phi_n(T_{nonlin}(u))du =
#      \sum_{l, i} \alpha_{l,i}\prod_{j=1}^{d}\int_0^1\phi_j(F^{-1}(u_j))\psi_{l_j, i_j}(u_j)du_j$
#      where $T_{nonlin} = (F^{-1}(u_1),...,F^{-1}(u_d))$
#     So far this only works with the Standard Combination Technique.
#     """
#     def standard_hatfunction1D(u):
#         return max(1 - abs(u), 0)
#
#     def hatfunction_level1D_position(u, l, x):
#         return standard_hatfunction1D((u - x) / float(2) ** (-l))
#
#     x1 = cp.Uniform(-math.pi, math.pi)
#     x2 = cp.Uniform(-math.pi, math.pi)
#     x3 = cp.Uniform(-math.pi, math.pi)
#     distributions = [x1, x2, x3]
#     joint = cp.J(x1, x2, x3)
#
#     f = IshigamiFunctionSimple()
#     dim = 3
#     a = np.array([-3.2, -3.2, -3.2])
#     b = np.array([3.2, 3.2, 3.2])
#     # a = np.array([0.0, 0.0, 0.0])
#     # b = np.array([1, 1, 1])
#
#     labels = [param_name.strip() for param_name in ["x1", "x2", "x3"]]
#     #####################################
#     if modified_basis:
#         grid = GlobalTrapezoidalGrid(a=a, b=b, modified_basis=True, boundary=False)
#         # grid = GlobalTrapezoidalGridWeighted(a, b, boundary=False, modified_basis=True)
#     else:
#         grid = GlobalTrapezoidalGrid(a=a, b=b, modified_basis=False, boundary=True)
#         # grid = GlobalTrapezoidalGridWeighted(a, b, boundary=True)
#
#     operation = Integration(f=f, grid=grid, dim=dim)
#
#     if spatiallyAdaptive:
#         combiinstance = SpatiallyAdaptiveSingleDimensions2(np.ones(dim) * a, np.ones(dim) * b,
#                                                                             operation=operation)
#     else:
#         combiinstance = StandardCombi(a, b, operation=operation, norm=2)
#
#     if plotting:
#         plot_file = str(outputModelDir / "output.png")
#         filename_contour_plot = str(outputModelDir / "output_contour_plot.png")
#         filename_combi_scheme_plot = str(outputModelDir / "output_combi_scheme.png")
#         filename_refinement_graph = str(outputModelDir / "output_refinement_graph.png")
#         filename_sparse_grid_plot = str(outputModelDir / "output_sg_graph.png")
#         combiinstance.filename_contour_plot = filename_contour_plot
#         combiinstance.filename_refinement_graph = filename_refinement_graph
#         combiinstance.filename_combi_scheme_plot = filename_combi_scheme_plot
#         combiinstance.filename_sparse_grid_plot = filename_sparse_grid_plot
#
#     if spatiallyAdaptive:
#         errorOperator = ErrorCalculatorSingleDimVolumeGuided()
#         combiinstance.performSpatiallyAdaptiv(1, lmax, errorOperator, tol=tolerance, do_plot=plotting,
#                                                                max_evaluations=max_evals)
#     else:
#         # minimum_level = 1
#         # maximum_level = 7
#         # combiObject.perform_operation(minimum_level, maximum_level)
#         combiinstance.perform_operation(1, lmax)
#
#     if plotting:
#         print("Sparse Grid:")
#         combiinstance.print_resulting_sparsegrid(markersize=10)
#
#     # extract the one-dimensional orthogonal polynomials and order them in the same way as chaospy does
#     # has to be modified if the distributions are not the same for all dimensions
#     pce_polys3d, pce_polys_norms3d = cp.orth_ttr(p, joint, retall=True)
#     pce_polys1d, pce_polys1d_norms = cp.orth_ttr(p, x1, retall=True)
#     indices = numpoly.glexindex(start=0, stop=p + 1, dimensions=len(joint),
#                                 graded=True, reverse=True,
#                                 cross_truncation=1.0)
#     norms = [None] * len(indices)
#     polys = [None] * len(indices)
#     for i in range(len(indices)):
#         polys[i] = [pce_polys1d[indices[i][d]] for d in range(dim)]
#         norms[i] = [pce_polys1d_norms[indices[i][d]] for d in range(dim)]
#
#     # TODO - store the one dimensional integrals and the function evaluations
#     fileName = f"dict1D_integrals.pickle"
#     pickle_integrals_in = str(outputModelDir / fileName)
#     with open(pickle_integrals_in, 'rb') as handle:
#         dictIntegrals = pickle.load(pickle_integrals_in)
#     fileName = f"dict_evaluations.pickle"
#     pickle_evaluations_in = str(outputModelDir / fileName)
#     with open(pickle_evaluations_in, 'rb') as handle:
#         dictEvaluations = pickle.load(pickle_evaluations_in)
#
#     # compute the coefficients cn, takes long for the first coefficient
#     counterIntegrals = 0
#     cn = np.zeros(len(polys))
#     for n, pce_poly in enumerate(polys):
#         for component_grid in combiinstance.scheme:
#             points = combiinstance.get_points_component_grid(component_grid.levelvector)
#             evals = []
#             keyLevelvector = tuple(component_grid.levelvector.tolist())
#             if keyLevelvector in dictEvaluations:
#                 evals = dictEvaluations[keyLevelvector]
#             else:
#                 evals = combiinstance(points)
#                 dictEvaluations[keyLevelvector] = evals
#             integralCompGrid = 0
#             for i, point in enumerate(points):
#                 product = 1
#                 for d in range(0, dim):
#                     if (point[d], component_grid.levelvector[d], indices[n][d]) in dict:
#                         # if distributions are not the same in all dimensions, the dimension d has to be included
#                         onedimensionalIntegral = dictIntegrals[(point[d], component_grid.levelvector[d], indices[n][d])]
#                     else:
#                         integrand = lambda x: pce_poly[d](distributions[d].ppf(x)) * \
#                                               hatfunction_level1D_position(x, component_grid.levelvector[d], point[d])
#                         onedimensionalIntegral = \
#                         integrate.quad(integrand, max(point[d] - float(2) ** (-component_grid.levelvector[d]), 0),
#                                        min(1, point[d] + float(2) ** (-component_grid.levelvector[d])), epsabs=1e-8)[0]
#                         counterIntegrals += 1
#                         dictIntegrals[(point[d], component_grid.levelvector[d], indices[n][d])] = onedimensionalIntegral
#                     product = product * onedimensionalIntegral / norms[n][d]
#                 integralCompGrid = integralCompGrid + evals[i] * product
#             # print("integralCompgrid for grid ", component_grid.levelvector, " is ", integralCompGrid)
#             cn[n] = cn[n] + component_grid.coefficient * integralCompGrid
#             # print("CounterIntegrals: ", counterIntegrals)
#             counterIntegrals = 0
#         print("cn for n = ", n, " is ", cn[n])
#
#     gPCE = np.transpose(np.sum(pce_polys3d * cn.T, -1))
#     exp = cp.E(gPCE, joint)
#     var = cp.Var(gPCE, joint)
#     print("expected: ", exp, ", variance: ", var)
#     first_order_sobol_indices = cp.Sens_m(gPCE, joint)
#     total_order_sobol_indices = cp.Sens_t(gPCE, joint)
#     print("First order Sobol indices: ", first_order_sobol_indices)
#     print("Total order Sobol indices: ", total_order_sobol_indices)
#
#     pickle_out_integrals = open("dict1D_integrals.pickle", "wb")
#     pickle.dump(dictIntegrals, pickle_out_integrals)
#     pickle_out_integrals.close()
#     pickle_out_evaluations = open("dict_evaluations.pickle", "wb")
#     pickle.dump(dictEvaluations, pickle_out_evaluations)
#     pickle_out_evaluations.close()


# TODO finish this!
# def compute_leja_sg_surrogate_and_gpce(a_model_param=7, b_model_param=0.1, modified_basis=False,
#                                        spatiallyAdaptive=True, plotting=True, outputModelDir="./",
#                                        lmax=2, max_evals=2000, tolerance=10 ** -5,
#                                        rule='gaussian', sparse=False, q=7, p=6):
#     x1 = cp.Uniform(-math.pi, math.pi)
#     x2 = cp.Uniform(-math.pi, math.pi)
#     x3 = cp.Uniform(-math.pi, math.pi)
#     joint = cp.J(x1, x2, x3)
#     labels = [param_name.strip() for param_name in ["x1", "x2", "x3"]]
#
#     leja_surrogate = None
#     _compute_gpce_chaospy_ishigami(a_model_param=a_model_param, b_model_param=b_model_param,
#                                    labels=labels, my_model=leja_surrogate, joint=joint,
#                                    q=q, p=p, rule=rule, sparse=sparse,
#                                    outputModelDir=outputModelDir, can_model_evaluate_all_vector_nodes=True)
