import dill
import numpoly
import numpy as np
from numpy import linalg as LA
from math import isclose, isinf
import pickle
import pathlib
import os
import time
import warnings
import scipy.stats as sps

import chaospy as cp

import sparseSpACE
from sparseSpACE.Function import *
from sparseSpACE.StandardCombi import *
from sparseSpACE.Grid import *
from sparseSpACE.spatiallyAdaptiveSingleDimension2 import *
from sparseSpACE.ErrorCalculator import *
from sparseSpACE.GridOperation import *
from sparseSpACE.DimAdaptiveCombi import *
from sparseSpACE.Integrator import *
from sparseSpACE.BasisFunctions import *
from sparseSpACE.RefinementContainer import RefinementContainer
from sparseSpACE.RefinementObject import RefinementObject
from sparseSpACE.Utils import *

class UncertaintyQuantificationREfactored(Integration):  #UncertaintyQuantificationREfactored(UncertaintyQuantification)
    # TODO Ivana - get_result what does it return ?
    # The constructor resembles Integration's constructor;
    # it has an additional parameter:
    # distributions can be a list, tuple or string
    def __init__(self, f, distributions, a: Sequence[float], b: Sequence[float],
                 dim: int = None, grid=None, reference_solution=None,
                 print_level: int = print_levels.NONE, log_level: int = log_levels.INFO):
        dim = dim or len(a)
        super().__init__(f, grid, dim, reference_solution)
        self.f_model = f
        # If distributions is not a list, it specifies the same distribution
        # for every dimension
        if not isinstance(distributions, list):
            distributions = [distributions for _ in range(dim)]

        # Setting the distribution to a string is a short form when
        # no parameters are given
        for d in range(dim):
            if isinstance(distributions[d], str):
                distributions[d] = (distributions[d],)

        self._prepare_distributions(distributions, a, b)
        self.f_evals = None
        self.gPCE = None
        self.pce_polys = None
        self.log_util = LogUtility(log_level=log_level, print_level=print_level)
        self.log_util.set_print_prefix('UncertaintyQuantification')
        self.log_util.set_log_prefix('UncertaintyQuantification')

    def set_grid(self, grid):
        self.grid = grid

    def set_reference_solution(self, reference_solution):
        self.reference_solution = reference_solution

    # From the user provided information about distributions, this function
    # creates the distributions list which contains Chaospy distributions
    def _prepare_distributions(self, distris, a: Sequence[float],
                               b: Sequence[float]):
        self.distributions = []
        self.distribution_infos = distris
        chaospy_distributions = []
        known_distributions = dict()
        for d in range(self.dim):
            distr_info = distris[d]
            distr_known = distr_info in known_distributions
            if distr_known:
                # Reuse the same distribution objects for multiple dimensions
                d_prev = known_distributions[distr_info]
                self.distributions.append(self.distributions[d_prev])
            else:
                known_distributions[distr_info] = d

            distr_type = distr_info[0]
            if distr_type == "Uniform":
                distr = cp.Uniform(a[d], b[d])
                chaospy_distributions.append(distr)
                if not distr_known:
                    self.distributions.append(UQDistribution.from_chaospy(distr))
            elif distr_type == "Triangle":
                midpoint = distr_info[1]
                assert isinstance(midpoint, float), "invalid midpoint"
                distr = cp.Triangle(a[d], midpoint, b[d])
                chaospy_distributions.append(distr)
                if not distr_known:
                    self.distributions.append(UQDistribution.from_chaospy(distr))
            elif distr_type == "Normal":
                mu = distr_info[1]
                sigma = distr_info[2]
                cp_distr = cp.Normal(mu=mu, sigma=sigma)
                chaospy_distributions.append(cp_distr)
                if not distr_known:
                    # The chaospy normal distribution does not work with big values
                    def pdf(x, _mu=mu, _sigma=sigma):
                        return sps.norm.pdf(x, loc=_mu, scale=_sigma)

                    def cdf(x, _mu=mu, _sigma=sigma):
                        return sps.norm.cdf(x, loc=_mu, scale=_sigma)

                    def ppf(x, _mu=mu, _sigma=sigma):
                        return sps.norm.ppf(x, loc=_mu, scale=_sigma)

                    self.distributions.append(UQDistribution(pdf, cdf, ppf))
            elif distr_type == "Laplace":
                mu = distr_info[1]
                scale = distr_info[2]
                cp_distr = cp.Laplace(mu=mu, scale=scale)
                chaospy_distributions.append(cp_distr)
                if not distr_known:
                    def pdf(x, _mu=mu, _scale=scale):
                        return sps.laplace.pdf(x, loc=_mu, scale=_scale)

                    def cdf(x, _mu=mu, _scale=scale):
                        return sps.laplace.cdf(x, loc=_mu, scale=_scale)

                    def ppf(x, _mu=mu, _scale=scale):
                        return sps.laplace.ppf(x, loc=_mu, scale=_scale)

                    self.distributions.append(UQDistribution(pdf, cdf, ppf))
            else:
                assert False, "Distribution not implemented: " + distr_type
        self.distributions_chaospy = chaospy_distributions
        self.distributions_joint = cp.J(*chaospy_distributions)
        self.all_uniform = all(k[0] == "Uniform" for k in known_distributions)
        self.a = a
        self.b = b

    def get_surplus_width(self, d: int, right_parent: float, left_parent: float) -> float:
        # Approximate the width with the probability
        cdf = self.distributions[d].cdf
        return cdf(right_parent) - cdf(left_parent)

    # This function exchanges the operation's function so that the adaptive
    # refinement can use a different function than the operation's function
    def set_function(self, f=None):
        if f is None:
            self.f = self.f_actual
            self.f_actual = None
        else:
            assert self.f_actual is None
            self.f_actual = self.f
            self.f = f

    def update_function(self, f):
        self.f = f

    def get_distributions(self):
        return self.distributions

    def get_distributions_chaospy(self):
        return self.distributions_chaospy

    # This function returns boundaries for distributions which have an infinite
    # domain, such as normal distribution
    def get_boundaries(self, tol: float) -> Tuple[Sequence[float], Sequence[float]]:
        assert 1.0 - tol < 1.0, "Tolerance is too small"
        a = []
        b = []
        for d in range(self.dim):
            dist = self.distributions[d]
            a.append(dist.ppf(tol))
            b.append(dist.ppf(1.0 - tol))
        return a, b

    def _set_pce_polys(self, polynomial_degrees):
        if self.pce_polys is not None and self.polynomial_degrees == polynomial_degrees:
            return
        self.polynomial_degrees = polynomial_degrees
        if not hasattr(polynomial_degrees, "__iter__"):
            # self.pce_polys, self.pce_polys_norms = cp.orth_ttr(polynomial_degrees, self.distributions_joint,
            #                                                    retall=True)
            self.pce_polys, self.pce_polys_norms = cp.generate_expansion(
                polynomial_degrees, self.distributions_joint, retall=True)    
            # Markus
            # self.pce_polys, self.pce_polys_norms = cp.expansion.stieltjes(polynomial_degrees, self.distributions_joint, retall=True)
            # self.polys1D, self.norms1D = [None] * self.dim, [None] * self.dim
            # for d in range(self.dim):
            #     self.polys1D[d], self.norms1D[d] = cp.expansion.stieltjes(polynomial_degrees,
            #                                                                        self.distributions_chaospy[d],
            #                                                                        retall=True)
            # indices = numpoly.glexindex(start=0, stop=polynomial_degrees + 1, dimensions=self.dim,
            #                             graded=True, reverse=True,
            #                             cross_truncation=1.0)
            # self.indices = indices                                             
            return
        # Chaospy does not support different degrees for each dimension, so
        # the higher degree polynomials are removed afterwards
        # polys, norms = cp.orth_ttr(max(polynomial_degrees), self.distributions_joint, retall=True)
        polys, norms = cp.generate_expansion(max(polynomial_degrees), self.distributions_joint, retall=True)
        # self.pce_polys, self.pce_polys_norms = cp.expansion.stieltjes(polynomial_degrees, self.distributions_joint, retall=True)
        polys_filtered, norms_filtered = [], []
        for i, poly in enumerate(polys):
            max_exponents = [max(exps) for exps in poly.exponents.T]
            if any([max_exponents[d] > deg_max for d, deg_max in enumerate(polynomial_degrees)]):
                continue
            polys_filtered.append(poly)
            norms_filtered.append(norms[i])
        self.pce_polys = cp.Poly(polys_filtered)
        self.pce_polys_norms = norms_filtered
        # self.polys1D, self.norms1D = [None] * self.dim, [None] * self.dim
        # for d in range(self.dim):
        #     self.polys1D[d], self.norms1D[d] = cp.expansion.stieltjes(polynomial_degrees[d],
        #                                                                        self.distributions_chaospy[d],
        #                                                                        retall=True)
        # indices = numpoly.glexindex(start=0, stop=polynomial_degrees + 1, dimensions=self.dim,
        #                             graded=True, reverse=True,
        #                             cross_truncation=1.0)
        # self.indices = indices   

    def _scale_values(self, values):
        assert self.all_uniform, "Division by the domain volume should be used for uniform distributions only"
        div = 1.0 / np.prod([self.b[i] - v_a for i, v_a in enumerate(self.a)])
        return values * div

    def _set_nodes_weights_evals(self, combiinstance, scale_weights=False):
        self.nodes, self.weights = combiinstance.get_points_and_weights()
        assert len(self.nodes) == len(self.weights)
        if scale_weights:
            assert combiinstance.has_basis_grid(), "scale_weights should only be needed for basis grids"
            self.weights = self._scale_values(self.weights)
            # ~ self.f_evals = combiinstance.get_surplusses()
            # Surpluses are required here..
            # TODO Ivana - but don't you already have f_model evaluated at nodes!?
            self.f_evals = [self.f_model(coord) for coord in self.nodes]
        else:
            self.f_evals = [self.f_model(coord) for coord in self.nodes]

    def _get_combiintegral(self, combiinstance, scale_weights=False):
        integral = self.get_result()
        if scale_weights:
            assert combiinstance.has_basis_grid(), "scale_weights should only be needed for basis grids"
            return self._scale_values(integral)
        return integral

    def calculate_moment(self, combiinstance, k: int = None,
                         use_combiinstance_solution=True, scale_weights=False):
        if use_combiinstance_solution:
            mom = self._get_combiintegral(combiinstance, scale_weights=scale_weights)
            assert len(mom) == self.f_model.output_length()
            return mom
        self._set_nodes_weights_evals(combiinstance)
        vals = [self.f_evals[i] ** k * self.weights[i] for i in range(len(self.f_evals))]
        return sum(vals)

    def calculate_expectation(self, combiinstance, use_combiinstance_solution=True):
        return self.calculate_moment(combiinstance, k=1, use_combiinstance_solution=use_combiinstance_solution)

    @staticmethod
    def moments_to_expectation_variance(mom1: Sequence[float],
                                        mom2: Sequence[float]) -> Tuple[Sequence[float], Sequence[float]]:
        expectation = mom1
        variance = [mom2[i] - ex * ex for i, ex in enumerate(expectation)]
        for i, v in enumerate(variance):
            if v < 0.0:
                # When the variance is zero, it can be set to something negative
                # because of numerical errors
                variance[i] = -v
        return expectation, variance

    def calculate_expectation_and_variance(self, combiinstance, use_combiinstance_solution=True, scale_weights=False):
        if use_combiinstance_solution:
            integral = self._get_combiintegral(combiinstance, scale_weights=scale_weights)
            output_dim = len(integral) // 2
            expectation = integral[:output_dim]
            expectation_of_squared = integral[output_dim:]
        else:
            expectation = self.calculate_moment(combiinstance, k=1, use_combiinstance_solution=False)
            expectation_of_squared = self.calculate_moment(combiinstance, k=2, use_combiinstance_solution=False)
        return self.moments_to_expectation_variance(expectation, expectation_of_squared)

    def calculate_PCE(self, polynomial_degrees, combiinstance, restrict_degrees=False, use_combiinstance_solution=True,
                      scale_weights=False):
        if use_combiinstance_solution:
            assert self.pce_polys is not None
            assert not restrict_degrees
            integral = self._get_combiintegral(combiinstance, scale_weights=scale_weights)
            num_polys = len(self.pce_polys)
            output_dim = len(integral) // num_polys
            coefficients = integral.reshape((num_polys, output_dim))
            self.gPCE = np.transpose(np.sum(self.pce_polys * coefficients.T, -1))
            return

        self._set_nodes_weights_evals(combiinstance)

        if restrict_degrees:
            # Restrict the polynomial degrees if in some dimension not enough points
            # are available
            # For degree deg, deg+(deg-1)+1 points should be available
            num_points = combiinstance.get_num_points_each_dim()
            polynomial_degrees = [min(polynomial_degrees, num_points[d] // 2) for d in range(self.dim)]

        self._set_pce_polys(polynomial_degrees)
        self.gPCE = cp.fit_quadrature(self.pce_polys, list(zip(*self.nodes)),
                                      self.weights, np.asarray(self.f_evals), norms=self.pce_polys_norms)

    def get_gPCE(self):
        return self.gPCE

    def get_expectation_PCE(self):
        if self.gPCE is None:
            assert False, "calculatePCE must be invoked before this method"
        return cp.E(self.gPCE, self.distributions_joint)

    def get_variance_PCE(self):
        if self.gPCE is None:
            assert False, "calculatePCE must be invoked before this method"
        return cp.Var(self.gPCE, self.distributions_joint)

    def get_expectation_and_variance_PCE(self):
        return self.get_expectation_PCE(), self.get_variance_PCE()

    def get_Percentile_PCE(self, q: float, sample: int = 10000):
        if self.gPCE is None:
            assert False, "calculatePCE must be invoked before this method"
        return cp.Perc(self.gPCE, q, self.distributions_joint, sample)

    def get_first_order_sobol_indices(self):
        if self.gPCE is None:
            assert False, "calculatePCE must be invoked before this method"
        return cp.Sens_m(self.gPCE, self.distributions_joint)

    def get_total_order_sobol_indices(self):
        if self.gPCE is None:
            assert False, "calculatePCE must be invoked before this method"
        return cp.Sens_t(self.gPCE, self.distributions_joint)

    # Returns a Function which can be passed to performSpatiallyAdaptiv
    # so that adapting is optimized for the k-th moment
    def get_moment_Function(self, k: int):
        if k == 1:
            return self.f
        return FunctionPower(self.f, k)

    def set_moment_Function(self, k: int):
        self.update_function(self.get_moment_Function(k))

    # Optimizes adapting for multiple moments at once
    def get_moments_Function(self, ks: Sequence[int]):
        return FunctionConcatenate([self.get_moment_Function(k) for k in ks])

    def set_moments_Function(self, ks: Sequence[int]):
        self.update_function(self.get_moments_Function(ks))

    def get_expectation_variance_Function(self):
        return self.get_moments_Function([1, 2])

    def set_expectation_variance_Function(self):
        self.update_function(self.get_expectation_variance_Function())

    # Returns a Function which can be passed to performSpatiallyAdaptiv
    # so that adapting is optimized for the PCE
    def get_PCE_Function(self, polynomial_degrees):
        self._set_pce_polys(polynomial_degrees)
        # self.f can change, so putting it to a local variable is important
        # ~ f = self.f
        # ~ polys = self.pce_polys
        # ~ funcs = [(lambda coords: f(coords) * polys[i](coords)) for i in range(len(polys))]
        # ~ return FunctionCustom(funcs)
        return FunctionPolysPCE(self.f, self.pce_polys, self.pce_polys_norms)
        # return FunctionPolysPCE(self.f, self.pce_polys, self.pce_polys_norms, self.polys1D, self.norms1D, self.indices) Markus

    def set_PCE_Function(self, polynomial_degrees):
        self.update_function(self.get_PCE_Function(polynomial_degrees))

    def get_pdf_Function(self):
        pdf = self.distributions_joint.pdf
        return FunctionCustom(lambda coords: float(pdf(coords)))

    def set_pdf_Function(self):
        self.update_function(self.get_pdf_Function())

    # Returns a Function which applies the PPF functions before evaluating
    # the problem function; it can be integrated without weighting
    def get_inverse_transform_Function(self, func=None):
        return FunctionInverseTransform(func or self.f, self.distributions)

    def set_inverse_transform_Function(self, func=None):
        self.update_function(self.get_inverse_transform_Function(func or self.f, self.distributions))

# ============================================================================================
# PCE / PSP Code - Relying on ChaosPy and UQEF...
# ============================================================================================



# ============================================================================================
# SparseSpACE Interpolation...
# ============================================================================================


# TODO - potential changes to SparseSpACE: parallelization of in-one-subgrid model evaluations;
# Sparse-PSP Operation directly in code - based on UncertaintyQuantification/Integration!!!
# Computation in parallel...
# Error modification - ErrorCalculator; 
# Extracting basis functions and computing 1D analytical integrals
# Leja points and b-splines...

def set_up_filenames_for_printing_spatially_adaptive(combiObject, do_plot, directory_for_saving_plots, **kwargs):
    if do_plot:
        filename_contour_plot = kwargs.get('filename_contour_plot', str(directory_for_saving_plots / "output_contour_plot.png"))
        filename_combi_scheme_plot = kwargs.get('filename_combi_scheme_plot', str(directory_for_saving_plots / "output_combi_scheme.png"))
        filename_refinement_graph = kwargs.get('filename_refinement_graph', str(directory_for_saving_plots / "output_refinement_graph.png"))
        filename_sparse_grid_plot = kwargs.get('filename_sparse_grid_plot', str(directory_for_saving_plots / "output_sg_graph.png"))
        combiObject.filename_contour_plot = filename_contour_plot
        combiObject.filename_refinement_graph = filename_refinement_graph
        combiObject.filename_combi_scheme_plot = filename_combi_scheme_plot
        combiObject.filename_sparse_grid_plot = filename_sparse_grid_plot


def sparsespace_pipeline(a, b, model=None, dim=2, 
grid_type: str='trapezoidal', method: str='standard_combi', operation_str: str='integration',
directory_for_saving_plots='./', do_plot=True,  **kwargs):
    """
    Var 2 - Compute gPCE coefficients by integrating the (SG) surrogate
    SG surrogate computed based on SparseSpACE 
    
    :param a: lower bounds of the integration domain
    :param b: upper bounds of the integration domain
    :param model: SparseSpACE function - an object with evaluation operator!
    :param operation_str: operation to perform; supported operations: 'integration', 'uq'/'uncertainty quantification'
    :param dim: dimension of the model
    :param grid_type: type of grid to use for sparse grid construction
                    Supported grid types: 'trapezoidal', 'chebyshev', 'leja', 'bspline_p3', 'gauss_legendre', 'gauss_hermite'
                    For spatial adaptive single dimensions algorithm: 'globa_trapezoidal', 'trapezoidal' and 'bspline_p3',  'gauss_legendre', 'gauss_hermite'
    :param method: combination technique to use for sparse grid construction
                    Supported methods: 'standard_combi', 'dim_adaptive_combi', 'dim_wise_spat_adaptive_combi'
    :param directory_for_saving_plots: directory for saving plots
    :param do_plot: flag indicating whether to generate plots or not
    :param kwargs: optional keyword arguments
    
    Optional Keyword Arguments propagated via kwargs:
        minimum_level: minimum level of the grid; Default: 1
        maximum_level: maximum level of the grid; Default: 3
        modified_basis: flag indicating whether to use modified basis or not; Default: False
        boundary: flag indicating whether to include boundary points or not; Default: True
        norm: norm to use for error calculation; Default: 2; 2 | np.inf (Note - default in sparsespace is np.inf!)
        p_bsplines: degree of B-splines; Default: 3
        rebalancing: flag indicating whether to perform rebalancing or not; Default: True
        version: version of the spatially adaptive single dimensions algorithm; Default: 6; 6 | 9 | 2
        margin: margin parameter for spatially adaptive single dimensions algorithm; Default: 0.9
        grid_surplusses: grid surplusses for spatially adaptive single dimensions algorithm; If different from None grid object will 
                         be propagated to the spatially adaptive single dimensions algorithm to compute surplusses error bases on
                         ; if None then GlobalTrapezoidalGrid is used to compute surpluses; Options: None | 'grid'; Default: 'grid'
        max_evaluations: maximum number of evaluations for spatially adaptive single dimensions algorithm; Default: 100
        tol: tolerance for spatially adaptive single dimensions algorithm; Default: 10**-6
        distributions: Necessary for operation='uq'; distributions for uncertainty quantification; Default: None
        polynomial_degree_max: maximum polynomial degree for PCE; Default: 2
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

    # reference integral solution for calculating errors - if available
    try:
        reference_solution = model.getAnalyticSolutionIntegral(a,b)
    except NotImplementedError:
        reference_solution = None

    operation_uq = False
    if operation_str.lower() == "uq" or operation_str.lower() == "uncertainty quantification" or operation_str.lower() == "uncertaintyquantification" :
        operation_uq = True
        distributions = kwargs.get('distributions', None)
        if distributions is None:
            raise Exception(f"Distributions are required for operation={operation}!")

    # Operation
    operation = None
    if operation_uq:
        operation = UncertaintyQuantification(f=model, distributions=distributions, a=a, b=b, dim=dim, reference_solution=reference_solution)

    # Grid
    modified_basis = kwargs.get('modified_basis', False)
    boundary = kwargs.get('boundary', True)
    if method.lower() == 'dim_wise_spat_adaptive_combi':
        if grid_type.lower() == 'bspline_p3':
            p_bsplines = kwargs.get('p_bsplines', 3)
            grid = GlobalBSplineGrid(a=a, b=b, modified_basis=modified_basis, boundary=boundary, p=p_bsplines)
        elif grid_type.lower() == 'globa_trapezoidal' or grid_type.lower() == 'trapezoidal':
            grid = GlobalTrapezoidalGrid(a=a, b=b, modified_basis=modified_basis, boundary=boundary)
        elif grid_type.lower() == 'trapezoidal_weighted' or (operation_uq and (grid_type.lower() == 'trapezoidal' or grid_type.lower() == 'global_trapezoidal')):
            grid = GlobalTrapezoidalGridWeighted(a=a, b=b, uq_operation=operation, modified_basis=modified_basis, boundary=boundary)
        elif grid_type.lower() == 'gauss_legendre':
            raise Exception(f"{grid_type} is yet not supported when method={method}!")
            # boundary = False
            # grid = GaussLegendreGrid(a=a, b=b, normalize=True)  # TODO try with normalize=False what is default
        elif grid_type.lower() == 'gauss_hermite':
            raise Exception(f"{grid_type} is yet not supported when method={method}!")
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
    elif grid_type.lower() == 'gauss_legendre':
        boundary = False
        grid = GaussLegendreGrid(a=a, b=b, normalize=True)  # TODO try with normalize=False what is default
    elif grid_type.lower() == 'gauss_hermite':
        boundary = False
        expectations = kwargs.get('expectations', 1)  # TODO extract from distributions
        standard_deviations = kwargs.get('standard_deviations', 1)  # TODO extract from distributions
        grid = GaussHermiteGrid(expectations=expectations, standard_deviations=standard_deviations, integrator=None) 
    else:
        raise Exception(f"{grid_type} yet not supported!")

    # Setting Operation
    if operation_str.lower() == 'integration':
        operation = Integration(f=model, grid=grid, dim=dim, reference_solution=reference_solution)
    elif operation_str.lower() == 'interpolation':
        operation = Interpolation(f=model, grid=grid, dim=dim, reference_solution=reference_solution)
    elif operation_uq:
        operation.set_grid(grid)  # TODO do I need this, or I have already done this...!?
    else:
        raise Exception(f"{operation_str} is yet not supported!")  

    # TODO Ask extra questions for this; There is no need to do any of this!?
    # if operation_uq:
        # operation.set_expectation_variance_Function()  
        # polynomial_degree_max = kwargs.get('polynomial_degree_max', 2)
        # operation.set_PCE_Function(polynomial_degree_max)  #this is if you want to optimize for pce coeff. integrals

    scheme = None
    refinement = None

    # Combination Tehnique
    minimum_level = kwargs.get('minimum_level', 1)
    maximum_level = kwargs.get('maximum_level', 3)
    max_evaluations = kwargs.get('max_evaluations', 100) # 0, 22,
    tol = kwargs.get('tol', 10**-6)   # 0.3*10**-1, 10**-4
    norm = kwargs.get('norm', 2) # 2, np.inf
    start_time_building_sg_surrogate = time.time()
    if method.lower() == 'standard_combi':
        # combiObject = StandardCombi(np.ones(dim) * a, np.ones(dim) * b, operation=operation, norm=2)
        combiObject = StandardCombi(a=a, b=b, operation=operation)
        combiObject.set_combi_parameters(lmin=minimum_level, lmax=maximum_level)
        scheme, error, combi_result = combiObject.perform_operation(lmin=minimum_level, lmax=maximum_level)
        print(f"CT - scheme-{scheme};\n error-{error};\n combi_result-{combi_result};\n")
    elif method.lower() == 'dim_adaptive_combi':
        combiObject = DimAdaptiveCombi(a=a, b=b, operation=operation)
        scheme, abs_error, combiintegral, errors, num_points = combiObject.perform_combi(minv=minimum_level, maxv=maximum_level, tolerance=tol)
        print(f"DACT - scheme-{scheme};\n abs_error-{abs_error};\n combiintegral-{combiintegral};\n errors-{errors};\n num_points-{num_points}\n")
    elif method.lower() == 'dim_wise_spat_adaptive_combi':
        rebalancing = kwargs.get('rebalancing', True)
        version = kwargs.get('version', 6)  #9, 2
        margin = kwargs.get('margin', 0.8)  #0.5
        errorOperator = ErrorCalculatorSingleDimVolumeGuided()
        grid_surplusses = kwargs.get('grid_surplusses', 'grid')
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
                rebalancing=rebalancing, margin=margin, grid_surplusses=operation.get_grid())  #or grid_surplusses=grid
        set_up_filenames_for_printing_spatially_adaptive(combiObject, do_plot, directory_for_saving_plots)
        refinement, scheme, lmax, combi_result, number_of_evaluations, error_array, num_point_array, surplus_error_array, interpolation_error_arrayL2, interpolation_error_arrayMax = \
            combiObject.performSpatiallyAdaptiv(lmin=minimum_level, lmax=maximum_level, errorOperator=errorOperator, tol=tol, max_evaluations=max_evaluations, do_plot=False)
        print(f"SACT - refinement-{refinement};\n scheme-{scheme};\n lmax-{lmax};\n combi_result-{combi_result};\n number_of_evaluations-{number_of_evaluations};\n"
        f"error_array-{error_array};\n num_point_array-{num_point_array};\n surplus_error_array-{surplus_error_array};\n interpolation_error_arrayL2-{interpolation_error_arrayL2};\n interpolation_error_arrayMax-{interpolation_error_arrayMax};\n")
        # combiObject.continue_adaptive_refinement(3 * 10**-1)  # 2 * 10**-1, 1.9 * 10**-1, ...
    else:
        raise Exception(f"{method} yet not supported!")
    end_time_building_sg_surrogate = time.time()
    time_building_sg_surrogate = end_time_building_sg_surrogate - start_time_building_sg_surrogate

    number_full_model_evaluations = combiObject.get_total_num_points()
    print(f"Needed time for building SG surrogate is: {time_building_sg_surrogate} \n"
          f"for {number_full_model_evaluations} number of full model runs;\n")

    # # Temporary - code snippet showing functionalities with combiObject and component_grid!
    # for component_grid in combiObject.scheme:
    #     gridPointCoordsAsStripes, grid_point_levels, children_indices = combiObject.get_point_coord_for_each_dim(
    #         component_grid.levelvector)
    #     points = combiObject.get_points_component_grid(component_grid.levelvector)
    #     keyLevelvector = component_grid.levelvector
    #     cn[n] = cn[n] + component_grid.coefficient * integralCompGrid

    # TODO Some of these function should be called before adaptivity
    # Options one can do with uncertainty quantification operation
    # if operation_uq:  
    #     (E,), (Var,) = operation.calculate_expectation_and_variance(combiinstance, use_combiinstance_solution=Fals)
    #     (E,), (Var,) = operation.calculate_expectation_and_variance(combiinstance, use_combiinstance_solution=True)
    #     integral = operation.get_result()
    #     operation._set_pce_polys(polynomial_degrees)
    #     operation._set_nodes_weights_evals(combiinstance, scale_weights=False)
    #     integral = operation._get_combiintegral(combiinstance, scale_weights=False)
    #     operation.calculate_PCE(
    #         polynomial_degrees, combiinstance, restrict_degrees=False, use_combiinstance_solution=True, scale_weights=False)
    #     operation.f_evals
    #     operation.gPCE / operation.get_gPCE
    #     operation.pce_polys / operation.pce_polys_norms
    #     operation.get_total_order_sobol_indices() / operation.get_first_order_sobol_indices() / operation.get_Percentile_PCE()

    if do_plot:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if method.lower() == 'standard_combi':
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
    dict_info["operation_str"] = operation_str
    dict_info["number_full_model_evaluations"] = number_full_model_evaluations
    dict_info["time_building_sg_surrogate"] = time_building_sg_surrogate
    
    combiObject_file_name = 'combi_object.pkl'
    combiObject_file_name = os.path.abspath(os.path.join(str(directory_for_saving_plots), combiObject_file_name))
    with open(combiObject_file_name, 'wb') as handle:
        dill.dump(combiObject, handle)
        # pickle.dump(combiObject, handle, protocol=pickle.DEFAULT_PROTOCOL)  #pickle.HIGHEST_PROTOCOL
     
    # Save combiObject; scheme; refinement
    if refinement is not None:
        # print(f"DEBUGGING type(refinement) - {type(refinement)}")
        refinement_file_name = 'refinement.pkl'
        refinement_file_name = os.path.abspath(os.path.join(str(directory_for_saving_plots), refinement_file_name))
        with open(refinement_file_name, 'wb') as handle:
            dill.dump(refinement, handle)
            # pickle.dump(refinement, handle, protocol=pickle.DEFAULT_PROTOCOL)  #pickle.HIGHEST_PROTOCOL
        
    if scheme is not None:
        # print(f"DEBUGGING type(scheme) - {type(scheme)}")
        # if isinstance(scheme, list):
        #     print(f"DEBUGGING type(scheme[0]) - {type(scheme[0])}")
        scheme_file_name = 'scheme.pkl'
        scheme_file_name = os.path.abspath(os.path.join(str(directory_for_saving_plots), scheme_file_name))
        with open(scheme_file_name, 'wb') as handle:
            dill.dump(scheme, handle)
            # pickle.dump(scheme, handle, protocol=pickle.DEFAULT_PROTOCOL)  #pickle.HIGHEST_PROTOCOL
        
    # TODO Can I save model evaluations somewhere...
    # TODO Can I evaluate the scheme in new points!!!
    # TODO Can I save coefficients, c_l

    return combiObject, number_full_model_evaluations, dict_info


# ============================================================================================
# 1D Integration of basis function, i.e., utility functions for Ionut's approachv...
# ============================================================================================

# TODO Check validity of this code
#calculate mean, variance, sobol indices from gPCE coefficients step-by-step
def calculate_MeanVarianceSobol(gPCE_coefficients, polynomial_degrees, dim):
    """
    This function should compute basic statistics and Sobol SI based on the gPCE coefficients
    without relying on chaospy
    :param gPCE_coefficients:
    :param polynomial_degrees:
    :param dim:
    :return:
    """
    vari = 0
    for i in range(1, len(gPCE_coefficients)):
        vari += gPCE_coefficients[i] ** 2  # under assumption that polynomials were normalized
    mean = gPCE_coefficients[0]
    # vari = np.sum(gPCE_coefficients**2, axis=0)-mean**2  # https://github.com/sandialabs/pyapprox/blob/master/pyapprox/surrogates/polychaos/gpc.py

    first_order_sobol = [0 for _ in range(dim)]
    total_order_sobol = [0 for _ in range(dim)]

    # TODO Validate this code!
    indices = numpoly.glexindex(
        start=0, stop=polynomial_degrees + 1, dimensions=dim,
        graded=True, reverse=True, cross_truncation=1.0)
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
    
    for d in range(dim):
        for i, ind in enumerate(indices):
            if not ind[d] == 0:
                total_order_sobol[d] += gPCE_coefficients[i] ** 2 / vari
    print("expected: ", mean, ", variance: ", vari, ", first order sobol indices: \n", first_order_sobol,
          ", total order sobol indices: ", total_order_sobol)

    return mean, vari, first_order_sobol, total_order_sobol

# #compute gPCE coefficients analytically on a piecewise linear interpolant constructed with standard CT or the single
# dimension refinement strategy
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

# ============================================================================================
# 1D Integration of basis function, i.e., utility functions for Ionut's approachv...
# ============================================================================================


if __name__ == "__main__":
    from uqef_dynamic.models.sparsespace import sparsespace_functions

    # Example
    a = np.zeros(2)
    b = np.ones(2)
    dim = 2
    model = FunctionExpVar()

    # Example 2.0
    # dim = 5
    # a = np.zeros(dim)
    # b = np.ones(dim)
    # midpoint = np.ones(dim) * 0.5
    # coefficients = np.array([ 10**0 * (d+1) for d in range(dim)])
    # coefficients = np.array([ 10**1 * (d+1) for d in range(dim)])
    # model = GenzDiscontinious(border=midpoint,coeffs=coefficients)
    # model = GenzC0(midpoint=midpoint, coeffs=coefficients)

    combiObject, number_full_model_evaluations, dict_info = sparsespace_integration_pipeline(a, b, model=model, dim=dim, 
    grid_type='trapezoidal', method='standard_combi',
    directory_for_saving_plots='./', do_plot=True)
    # total_points, total_weights = combiObject.get_points_and_weights()
    # total_surplusses = combiObject.get_surplusses()
    print(f"combiObject: {combiObject}, number_full_model_evaluations: {number_full_model_evaluations}, dict_info: {dict_info}")
