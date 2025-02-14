import numpy as np
import pathlib
import os
import time

import sparseSpACE
from sparseSpACE.Function import *
from sparseSpACE.StandardCombi import *
from sparseSpACE.Grid import *
from sparseSpACE.spatiallyAdaptiveSingleDimension2 import *
from sparseSpACE.ErrorCalculator import *
from sparseSpACE.GridOperation import *
from sparseSpACE.DimAdaptiveCombi import *
from sparseSpACE.Integrator import *

# ============================================================================================
# SparseSpACE Interpolation...
# ============================================================================================


# TODO - potential changes to SparseSpACE: parallelization of in-one-subgrid model evaluations;
# Sparse-PSP Operation directly in code - based on UncertaintyQuantification/Integration!!!
# Computation in parallel...
# Error modification - ErrorCalculator; 
# Extracting basis functions and computing 1D analytical integrals
# Leja points and b-splines...


def sparsespace_integration_pipeline(a, b, model=None, dim=2, 
grid_type='trapezoidal', method='standard_combi',
directory_for_saving_plots='./', do_plot=True,  **kwargs):
    """
    Var 2 - Compute gPCE coefficients by integrating the (SG) surrogate
    SG surrogate computed based on SparseSpACE 
    
    :param a: lower bounds of the integration domain
    :param b: upper bounds of the integration domain
    :param model: SparseSpACE function - an object with evaluation operator!
    :param dim: dimension of the model
    :param grid_type: type of grid to use for sparse grid construction
                    Supported grid types: 'trapezoidal', 'chebyshev', 'leja', 'bspline_p3'
                    For spetical adaptive single dimensions algorithm: 'globa_trapezoidal', 'trapezoidal' and 'bspline_p3'
    :param method: combination technique to use for sparse grid construction
                    Supported methods: 'standard_combi', 'dim_adaptive_combi', 'dim_wise_spat_adaptive_combi'
    :param directory_for_saving_plots: directory for saving plots
    :param do_plot: flag indicating whether to generate plots or not
    :param kwargs: optional keyword arguments
    
    Optional Keyword Arguments:
        minimum_level: minimum level of the grid; Default: 1
        maximum_level: maximum level of the grid; Default: 3
        modified_basis: flag indicating whether to use modified basis or not; Default: False
        boundary: flag indicating whether to include boundary points or not; Default: True
        norm: norm to use for error calculation; Default: np.inf; 2 | np.inf
        p_bsplines: degree of B-splines; Default: 3
        rebalancing: flag indicating whether to perform rebalancing or not; Default: True
        version: version of the spatially adaptive single dimensions algorithm; Default: 6; 6 | 9 | 2
        margin: margin parameter for spatially adaptive single dimensions algorithm; Default: 0.9
        grid_surplusses: grid surplusses for spatially adaptive single dimensions algorithm; If different from None grid object will 
                         be propagated to the spatially adaptive single dimensions algorithm to compute surplusses error bases on
                         ; if None then GlobalTrapezoidalGrid is used to compute surpluses; Default: None
        max_evaluations: maximum number of evaluations for spatially adaptive single dimensions algorithm; Default: 100
        tol: tolerance for spatially adaptive single dimensions algorithm; Default: 10**-5
        writing_results_to_a_file: flag indicating whether to write results to a file or not; Default: True
    
    :return: SparseSpACE combiObject, number of full model evaluations, dictionary with time information
    """
    print(f"\n== Build Sparse Surrogate==")

    outputModelDir = directory_for_saving_plots

    # Function
    if model is None:
        model = FunctionExpVar()
        dim = 2
        a = np.zeros(dim)
        b = np.ones(dim)

        # a = np.zeros(dim)
        # b = np.ones(dim)
        # midpoint = np.ones(dim) * 0.5
        # coefficients = np.array([ 10**0 * (d+1) for d in range(dim)])
        # coefficients = np.array([ 10**1 * (d+1) for d in range(dim)])
        # model = GenzDiscontinious(border=midpoint,coeffs=coefficients)
        # model = GenzC0(midpoint=midpoint, coeffs=coefficients)

    # reference integral solution for calculating errors - if available
    reference_solution = model.getAnalyticSolutionIntegral(a,b)

    # Grid
    modified_basis = kwargs.get('modified_basis', False)
    boundary = kwargs.get('boundary', True)
    if method.lower() == 'dim_wise_spat_adaptive_combi':
        if grid_type.lower() == 'bspline_p3':
            p_bsplines = kwargs.get('p_bsplines', 3)
            grid = GlobalBSplineGrid(a=a, b=b, modified_basis=modified_basis, boundary=boundary, p=p_bsplines)
        elif grid_type.lower() == 'globa_trapezoidal' or grid_type.lower() == 'trapezoidal':
            grid = GlobalTrapezoidalGrid(a=a, b=b, modified_basis=modified_basis, boundary=boundary)
        elif grid_type.lower() == 'trapezoidal_weighted':
            raise NotImplementedError
            #     distributionsForSparseSpace = ..
            #     operation_uq = UncertaintyQuantification(f=f, distributions=distributionsForSparseSpace, a=a, b=b, dim=dim)
            #     grid = GlobalTrapezoidalGridWeighted(a=a, b=b, uq_operation=operation_uq,
            #                                          modified_basis=modified_basis, boundary=boundary)
        else:
            raise Exception(f"{grid_type} is yet not supported when method={method}!")
    elif grid_type.lower() == 'trapezoidal':
        grid = TrapezoidalGrid(a=a, b=b, modified_basis=modified_basis, boundary=boundary)
    elif grid_type.lower() == 'chebyshev':
        grid = ClenshawCurtisGrid(a=a, b=b, boundary=boundary)
    elif grid_type.lower() == 'leja':
        grid = LejaGrid(a=a, b=b, boundary=boundary)
    elif grid_type.lower() == 'bspline_p3':
        p_bsplines = kwargs.get('p_bsplines', 3)
        grid = BSplineGrid(a=a, b=b, boundary=boundary, p=p_bsplines)
    else:
        raise Exception(f"{grid_type} yet not supported!")
    
    # Operation
    operation = Integration(f=model, grid=grid, dim=dim, reference_solution=reference_solution)  # there is Interpolation(Integration)
    
    # Combination Tehnique
    minimum_level = kwargs.get('minimum_level', 1)
    maximum_level = kwargs.get('maximum_level', 3)
    max_evaluations = kwargs.get('max_evaluations', 100) # 0, 22,
    tol = kwargs.get('tol', 10**-5)   # 0.3*10**-1, 10**-4
    norm = kwargs.get('norm', np.inf) # 2, np.inf
    start_time_building_sg_surrogate = time.time()
    if method.lower() == 'standard_combi':
        # combiObject = StandardCombi(np.ones(dim) * a, np.ones(dim) * b, operation=operation, norm=2)
        combiObject = StandardCombi(a=a, b=b, operation=operation)
        combiObject.set_combi_parameters(lmin=minimum_level, lmax=maximum_level)
        combi_scheme, error, combi_result = combiObject.perform_operation(lmin=minimum_level, lmax=maximum_level)
    elif method.lower() == 'dim_adaptive_combi':
        combiObject = DimAdaptiveCombi(a=a, b=b, operation=operation)
        scheme, abs_error, combiintegral, errors, num_points = combiObject.perform_combi(minv=minimum_level, maxv=maximum_level, tolerance=tol)
    elif method.lower() == 'dim_wise_spat_adaptive_combi':
        rebalancing = kwargs.get('rebalancing', True)
        version = kwargs.get('version', 6)  #9, 2
        margin = kwargs.get('margin', 0.9)  #0.5
        errorOperator = ErrorCalculatorSingleDimVolumeGuided()
        grid_surplusses = kwargs.get('grid_surplusses', None)
        if grid_surplusses is None:
            # grid used will be GlobalTrapezoidalGrid!!!
            # a = np.ones(dim) * a, b = np.ones(dim) * b ?
            combiObject = SpatiallyAdaptiveSingleDimensions2(
                a=a, b=b, 
                operation=operation, version=version, 
                norm=norm,
                rebalancing=rebalancing, margin=margin)
        else:
            combiObject = SpatiallyAdaptiveSingleDimensions2(
                a=a, b=b, 
                operation=operation, version=version, 
                norm=norm,
                rebalancing=rebalancing, margin=margin, grid_surplusses=grid)
        refinement, scheme, lmax, combi_result, number_of_evaluations, error_array, num_point_array, surplus_error_array, interpolation_error_arrayL2, interpolation_error_arrayMax = \
            combiObject.performSpatiallyAdaptiv(lmin=minimum_level, lmax=maximum_level, errorOperator=errorOperator, tol=tol, max_evaluations=max_evaluations, do_plot=False)
        # combiObject.continue_adaptive_refinement(3 * 10**-1)  # 2 * 10**-1, 1.9 * 10**-1, ...
    else:
        raise Exception(f"{method} yet not supported!")
    end_time_building_sg_surrogate = time.time()
    time_building_sg_surrogate = end_time_building_sg_surrogate - start_time_building_sg_surrogate

    number_full_model_evaluations = combiObject.get_total_num_points()
    print(f"Needed time for building SG surrogate is: {time_building_sg_surrogate} \n"
          f"for {number_full_model_evaluations} number of full model runs;")

    if do_plot:
        filename_contour_plot = kwargs.get('filename_contour_plot', str(outputModelDir / "output_contour_plot.png"))
        filename_combi_scheme_plot = kwargs.get('filename_combi_scheme_plot', str(outputModelDir / "output_combi_scheme.png"))
        filename_refinement_graph = kwargs.get('filename_refinement_graph', str(outputModelDir / "output_refinement_graph.png"))
        filename_sparse_grid_plot = kwargs.get('filename_sparse_grid_plot', str(outputModelDir / "output_sg_graph.png"))
        combiObject.filename_contour_plot = filename_contour_plot
        combiObject.filename_refinement_graph = filename_refinement_graph
        combiObject.filename_combi_scheme_plot = filename_combi_scheme_plot
        combiObject.filename_sparse_grid_plot = filename_sparse_grid_plot

        filename = os.path.abspath(os.path.join(str(directory_for_saving_plots), "hierarchical_subspaces.pdf"))
        combiObject.print_subspaces(sparse_grid_spaces=False, ticks=False, fade_full_grid=False, filename=filename)
        filename = os.path.abspath(os.path.join(str(directory_for_saving_plots), "sparsegrid_subspaces.pdf"))
        combiObject.print_subspaces(sparse_grid_spaces=False, ticks=False, fade_full_grid=True, filename=filename)
        filename = os.path.abspath(os.path.join(str(directory_for_saving_plots), "combi.pdf"))
        combiObject.print_resulting_combi_scheme(ticks=False, filename=filename, show_border=True, fill_boundary_points=True, fontsize=60)
        # filename = os.path.abspath(os.path.join(str(directory_for_saving_plots), "sparsegrid.pdf"))
        # combiObject.print_resulting_sparsegrid(ticks=False, show_border=True, filename=filename)
        if method.lower() == 'dim_wise_spat_adaptive_combi':
            filename = os.path.abspath(os.path.join(str(directory_for_saving_plots), "standard_combi.pdf"))
            combiObject.print_resulting_combi_scheme(filename=filename, show_border=True, markersize=20, ticks=False)
            # filename = os.path.abspath(os.path.join(str(directory_for_saving_plots), "standard_sparse_grid.pdf"))
            # combiObject.print_resulting_sparsegrid(filename=filename, show_border=True, markersize=40, ticks=False)
            filename = os.path.abspath(os.path.join(str(directory_for_saving_plots), "standard_combi.svg"))
            combiObject.print_resulting_combi_scheme(filename=filename, show_border=True, markersize=20, ticks=False)
            # filename = os.path.abspath(os.path.join(str(directory_for_saving_plots), "standard_sparse_grid.svg"))
            # combiObject.print_resulting_sparsegrid(filename=filename, show_border=True, markersize=40, ticks=False)
            filename = os.path.abspath(os.path.join(str(directory_for_saving_plots), "refinement_tree_start.pdf"))
            combiObject.draw_refinement_trees(filename=filename, single_dim=0, fontsize=60)
            filename = os.path.abspath(os.path.join(str(directory_for_saving_plots), "refinement_tree_start.svg"))
            combiObject.draw_refinement_trees(filename=filename, single_dim=0, fontsize=60)
            filename = os.path.abspath(os.path.join(str(directory_for_saving_plots), "refinement_start.pdf"))
            combiObject.draw_refinement(filename=filename, single_dim=0, fontsize=60)
            filename = os.path.abspath(os.path.join(str(directory_for_saving_plots), "refinement_start.svg"))
            combiObject.draw_refinement(filename=filename, single_dim=0, fontsize=60)

    writing_results_to_a_file = kwargs.get('writing_results_to_a_file', True)
    if writing_results_to_a_file:
        fileName = f"results.txt"
        statFileName = str(outputModelDir / fileName)
        fp = open(statFileName, "a")
        fp.write(f"time_building_sg_surrogate: {time_building_sg_surrogate}\n")
        fp.write(f'number_full_model_evaluations: {number_full_model_evaluations}\n')
        fp.close()

    dict_info = {}
    dict_info["number_full_model_evaluations"] = number_full_model_evaluations
    dict_info["time_building_sg_surrogate"] = time_building_sg_surrogate
    
    # TODO Can I save model evaluations somewhere...
    return combiObject, number_full_model_evaluations, dict_info


if __name__ == "__main__":
    from uqef_dynamic.models.sparsespace import sparsespace_functions

    # Example
    a = np.zeros(2)
    b = np.ones(2)
    model = FunctionExpVar()
    combiObject, number_full_model_evaluations, dict_info = sparsespace_integration_pipeline(a, b, model=model, dim=2, 
    grid_type='trapezoidal', method='standard_combi',
    directory_for_saving_plots='./', do_plot=True,  **kwargs)
    # total_points, total_weights = combiObject.get_points_and_weights()
    # total_surplusses = combiObject.get_surplusses()
    print(f"combiObject: {combiObject}, number_full_model_evaluations: {number_full_model_evaluations}, dict_info: {dict_info}")