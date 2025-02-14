import math
from mpi4py import MPI
import multiprocessing
import mpi4py
import numpy as np
import os
import sys
import time
import pathlib

cwd = pathlib.Path(os.getcwd())
parent = cwd.parent.absolute()
sys.path.insert(0, os.getcwd())

import chaospy as cp

linux_cluster_run = False
if linux_cluster_run:
    sys.path.insert(0, '/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic')
else:
    sys.path.insert(0, '/work/ga45met/mnt/linux_cluster_2/UQEF-Dynamic')

from uqef_dynamic.models.sparsespace import sparsespace_functions
from uqef_dynamic.utils import utility
from uqef_dynamic.utils import sparsespace_utils


def main_routine(model, current_output_folder, **kwargs):
    """

    :param model:
    :param current_output_folder:

    Optional parameters propagated through kwargs:
    :mpi: ;default: False;
    :can_model_evaluate_all_vector_nodes: ; default: False;
    :anisotropic: ; default:True
    :compute_mean: ; default:True
    :compute_var: ; default:True
    :compute_Sobol_m: ; default:False
    :compute_Sobol_t: ; default:False
    :dict_what_to_compute_stat: ; default:DEFAULT_DICT_STAT_TO_COMPUTE
    :qoi: ; default:"model_output"
    :operation: ; default:"UncertaintyQuantification"; other_options:"Interpolation" "both"
    :use_uqef: ; default:False This is relevant when var=1 is executed
    :uq_method: ; default:'sc'
    :writing_results_to_a_file_model: ; default:False
    :plotting_model: ; default:False
    :variant: ; default:1
    :surrogate_model_of_interest: ; default:"gpce"; other_options:"gpce" | "gPCE" | "sg" this is relevant when sg surrogate is indeed computed, i.e., variant == 2 or 3 or 4
    :writing_results_to_a_file: ; default:True
    :plotting: ; default:True
    :quadrature_rule: ; default:'g'; other_options: 'c'
    :q_order: ; default:9
    :p_order: ; default:4
    :poly_rule: ; default:"three_terms_recurrence"; other_options:"gram_schmidt" | "three_terms_recurrence" | "cholesky"
    :poly_normed: ; default:False
    :sparse_quadrature: ; default:False
    :sampling_rule: ; default:"random"; other_options: "random" | "sobol" | "latin_hypercube" | "halton"  | "hammersley"
    :sampleFromStandardDist: ; default:True
    :read_nodes_from_file: ; default:False
    :level_sg: ; default:10
    :mc_numevaluations: ; default:100

    These are only relevant when uqef simulation is initiated
    :instantly_save_results_for_each_time_step: ; default:False
    :disable_statistics: ; default:False
    :parallel_statistics: ; default:True
    :num_cores: ; default:1

    The following parameters are relevant when variant == 2 | Var 3 | Var 4 - parameters for SparseSpACE
    :gridName: ; default:"Trapezoidal"; other_options: "Trapezoidal" | "TrapezoidalWeighted" | "BSpline_p3" | "Leja"
    :lmin: ; default:1
    :lmax: ; default:2
    :max_evals: ; default:10**5
    :tolerance: ; default:10 ** -5  # or tolerance = 10 ** -20
    :modified_basis: ; default:False
    :boundary_points: ; default:True
    :spatiallyAdaptive: ; default:True
    :dimensionAdaptive; default:False
    :rebalancing: ; default:True
    :version: ; default:6
    :margin: ; default:0.8

    These configuration parameters make sense when spatiallyAdaptive = True
    grid_surplusses: ; default:"grid"; other_options:None | "grid", Note: when gridName = "Trapezoidal" grid_surplusses=None is okay...
    norm_spatiallyAdaptive: ; default:2; other_options:2 | np.inf

    This is only relevant when var==3
    build_sg_for_e_and_var: ; default:True

    :return:
    """
    dictionary_with_inf_about_the_run = dict()
    dict_with_time_info = None
    dictionary_with_inf_about_the_run["model"] = model
    running_on_cluster = False
    scratch_dir = cwd

    mpi = kwargs.get("mpi", False)

    can_model_evaluate_all_vector_nodes = kwargs.get("can_model_evaluate_all_vector_nodes", False)  # set to True if eval_vectorized is implemented,
    inputModelDir = None
    outputModelDir = None
    config_file = None
    parameters_setup_file_name = None
    parameters_file_name = None

    if model.lower() == "ishigami":
        sourceDir = scratch_dir
        outputModelDir = pathlib.Path('/work/ga45met/uqef_dynamic_runs/ishigami_runs') / "sg_anaysis_feb_25" / current_output_folder

    if outputModelDir is not None and not outputModelDir.exists():
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
    anisotropic = kwargs.get("anisotropic", True)

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
                if model.lower() == "larsim":
                    param_names.append((single_param["type"], single_param["name"]))
                else:
                    param_names.append(single_param["name"])
                dim += 1
                distributions_list_of_dicts.append(single_param)
                distributionsForSparseSpace.append((single_param["distribution"], single_param["lower"], single_param["upper"]))
                a.append(single_param["lower"])
                b.append(single_param["upper"])
    else:
        # TODO manual setup of params, param_names, dim, distributions, a, b; change this eventually.
        #  Hard-coded to Uniform dist!
        if model.lower() == "ishigami":
            param_names = ["x0", "x1", "x2"]
            a = [-math.pi, -math.pi, -math.pi]
            b = [math.pi, math.pi, math.pi]
        elif model.lower() == "gfunction":
            param_names = ["x0", "x1", "x2"]
            a = [0.0, 0.0, 0.0]
            b = [1.0, 1.0, 1.0]
            can_model_evaluate_all_vector_nodes = True
        elif model.lower() == "zabarras2d":
            param_names = ["x0", "x1"]
            a = [-1.0, -1.0]  # [-2.5, -2]
            b = [1.0, 1.0]  # [2.5, 2]
            can_model_evaluate_all_vector_nodes = True
        elif model.lower() == "zabarras3d":
            param_names = ["x0", "x1", "x2"]
            a = [-1.0, -1.0, -1.0]  # [-2.5, -2, 5]
            b = [1.0, 1.0, 1.0]  # [2.5, 2, 15]
        elif model.lower() in ["corner_peak", "product_peak", "oscillatory", "gaussian", "discontinuous"]:
            # param_names = ["x0", "x1", "x2"]
            # a = [0.0, 0.0, 0.0]
            # b = [1.0, 1.0, 1.0]
            # dim = 3
            # coeffs, _ = generate_and_scale_coeff_and_weights(dim, b_3)
            param_names = ["x0", "x1", "x2", "x3", "x4"]
            a = [0.0, 0.0, 0.0, 0.0, 0.0]
            b = [1.0, 1.0, 1.0, 1.0, 1.0]
            dim = 5
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
    
    dictionary_with_inf_about_the_run["param_names"] = param_names
    dictionary_with_inf_about_the_run["a"] = a
    dictionary_with_inf_about_the_run["b"] = b

    if model.lower() == "ishigami":
        problem_function = sparsespace_functions.IshigamiFunction()
    else:
        raise ValueError(f"Model {model} is not yet supported!")

    # params for SparseSpACE
    grid_type = kwargs.get("grid_type", "trapezoidal")
    method = kwargs.get("method", "standard_combi")

    # optional parameters
    minimum_level = kwargs.get("minimum_level", kwargs.get("lmin", 1))  # used to be lmin
    maximum_level = kwargs.get("maximum_level", kwargs.get("lmax", 3))  # used to be lmax
    max_evaluations = kwargs.get("max_evaluations", kwargs.get("max_evals", 100)) # 0, 22, used to be max_evals
    tol = kwargs.get("tol", kwargs.get("tolerance", 10**-5))   # 0.3*10**-1, 10**-4  # used to be tolerance
    modified_basis = kwargs.get("modified_basis", False)
    boundary = kwargs.get("boundary", kwargs.get("boundary_points", True))  # used to be boundary_points
    norm = kwargs.get("norm", np.inf)
    p_bsplines = kwargs.get("p_bsplines", 3)
    rebalancing = kwargs.get("rebalancing", True)
    version = kwargs.get("version", 6)
    margin = kwargs.get("margin", 0.9)
    grid_surplusses = kwargs.get("grid_surplusses", None)
    # Collect all the above local variables into kwargs
    kwargs_sparsespace_integration_pipeline = {
        key: value for key, value in locals().items() if key in \
            ['grid_type', 'method', 'minimum_level', 'maximum_level', 'max_evaluations', 'tol', \
                'modified_basis', 'boundary', 'norm', 'p_bsplines', 'rebalancing', 'version', 'margin', 'grid_surplusses']}

    combiObject, number_full_model_evaluations, dict_info = sparsespace_utils.sparsespace_integration_pipeline(
        a=a, b=b, model=problem_function, dim=dim, 
        directory_for_saving_plots=outputModelDir,
        do_plot=True,
        **kwargs_sparsespace_integration_pipeline
    )
    # total_points, total_weights = combiObject.get_points_and_weights()
    # total_surplusses = combiObject.get_surplusses()
    print(f"combiObject: {combiObject}, number_full_model_evaluations: {number_full_model_evaluations}, dict_info: {dict_info}")


if __name__ == "__main__":
    main_routine(
        model='ishigami', current_output_folder='standard_combi_trapez_lmin1_lmax2_tol_10-5_maxeval1000', \
            grid_type='trapezoidal', method='standard_combi', minimum_level=1, maximum_level=2, max_evaluations=1000, tol=10**-5, modified_basis=False, boundary=True, norm=2, p_bsplines=3, rebalancing=True, version=6, margin=0.9, grid_surplusses=None)

    # # Example
    # a = np.zeros(2)
    # b = np.ones(2)
    # model = FunctionExpVar()
    # combiObject, number_full_model_evaluations, dict_info = sparsespace_utils.sparsespace_integration_pipeline(
    #     a, b, model=model, dim=2, 
    #     grid_type='trapezoidal', method='standard_combi',
    #     directory_for_saving_plots='./', do_plot=True)
    # print(f"combiObject: {combiObject}, number_full_model_evaluations: {number_full_model_evaluations}, dict_info: {dict_info}")