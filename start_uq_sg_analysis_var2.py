import datetime
from collections import defaultdict
import numpy as np
import pathlib
import time
from mpi4py import MPI
import mpi4py

import os
import sys
cwd = pathlib.Path(os.getcwd())
parent = cwd.parent.absolute()
sys.path.insert(0, os.getcwd())

from uq_sparseSpACE import *

from sparseSpACE.Integrator import *

#####################################
### MPI infos:
#####################################
comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()
version = MPI.Get_library_version()
version2 = MPI.Get_version()

if rank == 0: print("MPI version: {}".format(version))
if rank == 0: print("MPI2 version: {}".format(version2))
if rank == 0: print("MPI3 version: {}".format(MPI.VERSION))
if rank == 0: print("mpi4py version: {}".format(mpi4py.__version__))

print("rank {}: starttime: {}".format(rank, datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S')))

def is_master(mpi):
    return mpi is False or (mpi is True and rank == 0)

if __name__ == "__main__":

    #####################################
    # Initial Model Setup
    #####################################
    list_of_models = ["hbvsask", "larsim", "ishigami", "gfunction", "zabarras2d", "zabarras3d",
                      "oscillatory", "product_peak", "corner_peak", "gaussian", "discontinuous"]
    # Additional Genz Options: GenzOszillatory, GenzDiscontinious2, GenzC0, GenzGaussian

    # uncomment if you want to run analysis for Genz functions...
    list_of_genz_functions = ["oscillatory", "product_peak", "corner_peak", "gaussian", "continous", "discontinuous"]
    path_to_saved_all_genz_functions = pathlib.Path("/work/ga45met/Backup/UQEF-Hydro/sg_anaysis/genz_functions")
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

    ######Ishigami Var 2&4#######
    list_of_dict_run_setups = [
        # {"model": "ishigami", "list_of_function_ids": None,
        #  "current_output_folder": "var2_gpce_kpu_p7_l12_max2000_lmax3_modified_basis_adaptive",
        #  "variant": 2, "quadrature_rule": "kpu", "q_order": 12, "p_order": 7,
        #  "poly_normed": False,"sampleFromStandardDist": True, "sparse_quadrature": True,
        #  "read_nodes_from_file": True, 'level_sg': 12,
        #  "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
        #  "max_evals": 2000, "tolerance": 10**-3,
        #  "modified_basis": True, "boundary_points": False, "spatiallyAdaptive": True,
        #  "norm_spatiallyAdaptive": 2, "rebalancing": True, "margin": 0.8,
        #  "build_sg_for_e_and_var": True,
        #  "compute_mean": True, "compute_var": True,
        #  "compute_Sobol_m": True, "compute_Sobol_t": True,
        #  "operation": "UncertaintyQuantification"
        #  },
        # {"model": "ishigami", "list_of_function_ids": None,
        #  "current_output_folder": "var2_gpce_kpu_p7_l12_max500_lmax3_boundary_adaptive",
        #  "variant": 2, "quadrature_rule": "kpu", "q_order": 12, "p_order": 7,
        #  "poly_normed": False, "sampleFromStandardDist": True, "sparse_quadrature": True,
        #  "read_nodes_from_file": True, 'level_sg': 12,
        #  "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
        #  "max_evals": 500, "tolerance": 10 ** -3,
        #  "modified_basis": False, "boundary_points": True, "spatiallyAdaptive": True,
        #  "norm_spatiallyAdaptive": 2, "rebalancing": True, "margin": 0.8,
        #  "build_sg_for_e_and_var": True,
        #  "compute_mean": True, "compute_var": True,
        #  "compute_Sobol_m": True, "compute_Sobol_t": True,
        #  "operation": "UncertaintyQuantification"
        #  },
        {"model": "ishigami", "list_of_function_ids": None,
         "current_output_folder": "var2_gpce_kpu_p7_l12_max1000_lmax3_boundary_adaptive",
         "variant": 2, "quadrature_rule": "kpu", "q_order": 12, "p_order": 7,
         "poly_normed": False, "sampleFromStandardDist": True, "sparse_quadrature": True,
         "read_nodes_from_file": True, 'level_sg': 12,
         "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
         "max_evals": 1000, "tolerance": 10 ** -3,
         "modified_basis": False, "boundary_points": True, "spatiallyAdaptive": True,
         "norm_spatiallyAdaptive": 2, "rebalancing": True, "margin": 0.8,
         "build_sg_for_e_and_var": True,
         "compute_mean": True, "compute_var": True,
         "compute_Sobol_m": True, "compute_Sobol_t": True,
         "operation": "UncertaintyQuantification"
         },
        {"model": "ishigami", "list_of_function_ids": None,
         "current_output_folder": "var2_gpce_kpu_p7_l12_max2000_lmax3_boundary_adaptive",
         "variant": 2, "quadrature_rule": "kpu", "q_order": 12, "p_order": 7,
         "poly_normed": False, "sampleFromStandardDist": True, "sparse_quadrature": True,
         "read_nodes_from_file": True, 'level_sg': 12,
         "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
         "max_evals": 2000, "tolerance": 10 ** -3,
         "modified_basis": False, "boundary_points": True, "spatiallyAdaptive": True,
         "norm_spatiallyAdaptive": 2, "rebalancing": True, "margin": 0.8,
         "build_sg_for_e_and_var": True,
         "compute_mean": True, "compute_var": True,
         "compute_Sobol_m": True, "compute_Sobol_t": True,
         "operation": "UncertaintyQuantification"
         },
        # {"model": "ishigami", "list_of_function_ids": None,
        #  "current_output_folder": "var2_gpce_kpu_p7_l12_max3000_lmax3_boundary_adaptive",
        #  "variant": 2, "quadrature_rule": "kpu", "q_order": 12, "p_order": 7,
        #  "poly_normed": False, "sampleFromStandardDist": True, "sparse_quadrature": True,
        #  "read_nodes_from_file": True, 'level_sg': 12,
        #  "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
        #  "max_evals": 3000, "tolerance": 10 ** -3,
        #  "modified_basis": False, "boundary_points": True, "spatiallyAdaptive": True,
        #  "norm_spatiallyAdaptive": 2, "rebalancing": True, "margin": 0.8,
        #  "build_sg_for_e_and_var": True,
        #  "compute_mean": True, "compute_var": True,
        #  "compute_Sobol_m": True, "compute_Sobol_t": True,
        #  "operation": "UncertaintyQuantification"
        #  },
        # {"model": "ishigami", "list_of_function_ids": None,
        #  "current_output_folder": "var2_gpce_kpu_p7_l12_max4000_lmax3_boundary_adaptive",
        #  "variant": 2, "quadrature_rule": "kpu", "q_order": 12, "p_order": 7,
        #  "poly_normed": False, "sampleFromStandardDist": True, "sparse_quadrature": True,
        #  "read_nodes_from_file": True, 'level_sg': 12,
        #  "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
        #  "max_evals": 4000, "tolerance": 10 ** -3,
        #  "modified_basis": False, "boundary_points": True, "spatiallyAdaptive": True,
        #  "norm_spatiallyAdaptive": 2, "rebalancing": True, "margin": 0.8,
        #  "build_sg_for_e_and_var": True,
        #  "compute_mean": True, "compute_var": True,
        #  "compute_Sobol_m": True, "compute_Sobol_t": True,
        #  "operation": "UncertaintyQuantification"
        #  },
    #     {"model": "ishigami", "list_of_function_ids": None,
    #      "current_output_folder": "var2_gpce_kpu_p9_l16_max103_lmax3_modified_adaptive",
    #      "variant": 2, "quadrature_rule": "kpu", "q_order": 16, "p_order": 9, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
    #      "max_evals": 1000, "tolerance": 10**-6,
    #      "modified_basis": True, "boundary_points":False, "spatiallyAdaptive":True,
    #      "norm_spatiallyAdaptive":2, "rebalancing": True
    #      },
    #     {"model": "ishigami", "list_of_function_ids": None,
    #      "current_output_folder": "var2_gpce_kpu_p9_l16_max103_lmax3_adaptive",
    #      "variant": 2, "quadrature_rule": "kpu", "q_order": 16, "p_order": 9, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
    #      "max_evals": 1000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": False, "spatiallyAdaptive": True,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True
    #      },
    #     {"model": "ishigami", "list_of_function_ids": None,
    #      "current_output_folder": "var2_gpce_kpu_p9_l16_max104_lmax3_boundary_adaptive",
    #      "variant": 2, "quadrature_rule": "kpu", "q_order": 16, "p_order": 9, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
    #      "max_evals": 10000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": True, "spatiallyAdaptive": True,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True
    #      },
    #     {"model": "ishigami", "list_of_function_ids": None,
    #      "current_output_folder": "var2_gpce_kpu_p9_l16_max105_lmax3_boundary_adaptive",
    #      "variant": 2, "quadrature_rule": "kpu", "q_order": 16, "p_order": 9, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
    #      "max_evals": 100000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": True, "spatiallyAdaptive": True,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True
    #      },
    #     {"model": "ishigami", "list_of_function_ids": None,
    #      "current_output_folder": "var2_gpce_g_p9_max105_lmax3_boundary_adaptive",
    #      "variant": 2, "quadrature_rule": "g", "q_order": 10, "p_order": 9, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
    #      "max_evals": 100000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": True, "spatiallyAdaptive": True,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True
    #      },
    #     {"model": "ishigami", "list_of_function_ids": None,
    #      "current_output_folder": "var2_gpce_kpu_p9_l16_lmax3_boundary_nonadaptive",
    #      "variant": 2, "quadrature_rule": "kpu", "q_order": 16, "p_order": 9, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
    #      "max_evals": 10000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": True, "spatiallyAdaptive": False,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True
    #      },
    #     {"model": "ishigami", "list_of_function_ids": None,
    #      "current_output_folder": "var2_gpce_kpu_p9_l16_lmax4_boundary_nonadaptive",
    #      "variant": 2, "quadrature_rule": "kpu", "q_order": 16, "p_order": 9, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 4,
    #      "max_evals": 10000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": True, "spatiallyAdaptive": False,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True
    #      },
    #     {"model": "ishigami", "list_of_function_ids": None,
    #      "current_output_folder": "var2_gpce_kpu_p9_l16_lmax3_modified_nonadaptive",
    #      "variant": 2, "quadrature_rule": "kpu", "q_order": 16, "p_order": 9, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
    #      "max_evals": 10000, "tolerance": 10 ** -6,
    #      "modified_basis": True, "boundary_points": False, "spatiallyAdaptive": False,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True
    #      },
    #     {"model": "ishigami", "list_of_function_ids": None,
    #      "current_output_folder": "var2_gpce_kpu_p9_l16_lmax3_nonadaptive",
    #      "variant": 2, "quadrature_rule": "kpu", "q_order": 16, "p_order": 9, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
    #      "max_evals": 10000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": False, "spatiallyAdaptive": False,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True
    #      },
    ]

    ######corner_peak Var 2#######
    # list_of_dict_run_setups = [
    #     {"model": "corner_peak", "list_of_function_ids": [1, ],
    #      "current_output_folder": "var2_gpce_gl_p9_q9_max103_lmax3_boundary_adaptive",
    #      "variant": 2, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
    #      "max_evals": 1000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": True, "spatiallyAdaptive": True,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True,
    #      "compute_mean": False, "compute_var": False
    #      },
    #     {"model": "corner_peak", "list_of_function_ids": [1, ],
    #      "current_output_folder": "var2_gpce_gl_p9_q9_max104_lmax3_boundary_adaptive",
    #      "variant": 2, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
    #      "max_evals": 10000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": True, "spatiallyAdaptive": True,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True,
    #      "compute_mean": False, "compute_var": False
    #      },
    #     {"model": "corner_peak", "list_of_function_ids": [1, ],
    #      "current_output_folder": "var2_gpce_gl_p9_q9_max105_lmax3_boundary_adaptive",
    #      "variant": 2, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
    #      "max_evals": 100000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": True, "spatiallyAdaptive": True,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True,
    #      "compute_mean": False, "compute_var": False
    #      },
    #     {"model": "corner_peak", "list_of_function_ids": [1, ],
    #      "current_output_folder": "var2_gpce_gl_p9_q9_max105_lmax3_modified_adaptive",
    #      "variant": 2, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
    #      "max_evals": 100000, "tolerance": 10 ** -6,
    #      "modified_basis": True, "boundary_points": False, "spatiallyAdaptive": True,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True,
    #      "compute_mean": False, "compute_var": False
    #      },
    #     # {"model": "corner_peak", "list_of_function_ids": [1, ],
    #     #  "current_output_folder": "var2_gpce_gl_p9_q9_max103_lmax3_adaptive",
    #     #  "variant": 2, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #     #  "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
    #     #  "max_evals": 1000, "tolerance": 10 ** -6,
    #     #  "modified_basis": False, "boundary_points": False, "spatiallyAdaptive": True,
    #     #  "norm_spatiallyAdaptive": 2, "rebalancing": True,
    #     #  "compute_mean": False, "compute_var": False
    #     #  },
    #     # {"model": "corner_peak", "list_of_function_ids": [1, ],
    #     #  "current_output_folder": "var2_gpce_gl_p9_q9_max104_lmax3_adaptive",
    #     #  "variant": 2, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #     #  "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
    #     #  "max_evals": 10000, "tolerance": 10 ** -6,
    #     #  "modified_basis": False, "boundary_points": False, "spatiallyAdaptive": True,
    #     #  "norm_spatiallyAdaptive": 2, "rebalancing": True,
    #     #  "compute_mean": False, "compute_var": False
    #     #  },
    #     {"model": "corner_peak", "list_of_function_ids": [1, ],
    #      "current_output_folder": "var2_gpce_gl_p9_q9_max105_lmax3_adaptive",
    #      "variant": 2, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
    #      "max_evals": 100000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": False, "spatiallyAdaptive": True,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True,
    #      "compute_mean": False, "compute_var": False
    #      },
    #     {"model": "corner_peak", "list_of_function_ids": [1, ],
    #      "current_output_folder": "var2_gpce_gl_p9_q9_max103_lmax2_boundary_adaptive",
    #      "variant": 2, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 2,
    #      "max_evals": 1000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": True, "spatiallyAdaptive": True,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True,
    #      "compute_mean": False, "compute_var": False
    #      },
    #     {"model": "corner_peak", "list_of_function_ids": [1, ],
    #      "current_output_folder": "var2_gpce_gl_p9_q9_max104_lmax2_boundary_adaptive",
    #      "variant": 2, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 2,
    #      "max_evals": 10000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": True, "spatiallyAdaptive": True,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True,
    #      "compute_mean": False, "compute_var": False
    #      },
    #     {"model": "corner_peak", "list_of_function_ids": [1, ],
    #      "current_output_folder": "var2_gpce_gl_p9_q9_max105_lmax2_boundary_adaptive",
    #      "variant": 2, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 2,
    #      "max_evals": 100000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": True, "spatiallyAdaptive": True,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True,
    #      "compute_mean": False, "compute_var": False
    #      },
    #     # {"model": "corner_peak", "list_of_function_ids": [1, ],
    #     #  "current_output_folder": "var2_gpce_gl_p9_q9_max104_lmax2_adaptive",
    #     #  "variant": 2, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #     #  "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 2,
    #     #  "max_evals": 10000, "tolerance": 10 ** -6,
    #     #  "modified_basis": False, "boundary_points": False, "spatiallyAdaptive": True,
    #     #  "norm_spatiallyAdaptive": 2, "rebalancing": True,
    #     #  "compute_mean": False, "compute_var":False
    #     #  },
    #     {"model": "corner_peak", "list_of_function_ids": [1, ],
    #      "current_output_folder": "var2_gpce_gl_p9_q9_lmax3_boundary_nonadaptive",
    #      "variant": 2, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
    #      "max_evals": 100000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": True, "spatiallyAdaptive": False,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True,
    #      "compute_mean": False, "compute_var": False
    #      },
    #     {"model": "corner_peak", "list_of_function_ids": [1, ],
    #      "current_output_folder": "var2_gpce_gl_p9_q9_lmax3_nonadaptive",
    #      "variant": 2, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
    #      "max_evals": 100000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": False, "spatiallyAdaptive": False,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True,
    #      "compute_mean": False, "compute_var": False
    #      },
    #     {"model": "corner_peak", "list_of_function_ids": [1, ],
    #      "current_output_folder": "var2_gpce_gl_p9_q9_lmax3_modified_nonadaptive",
    #      "variant": 2, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
    #      "max_evals": 100000, "tolerance": 10 ** -6,
    #      "modified_basis": True, "boundary_points": False, "spatiallyAdaptive": False,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True,
    #      "compute_mean": False, "compute_var": False
    #      },
    #     {"model": "corner_peak", "list_of_function_ids": [1, ],
    #      "current_output_folder": "var2_gpce_gl_p9_q9_lmax2_boundary_nonadaptive",
    #      "variant": 2, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 2,
    #      "max_evals": 100000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": True, "spatiallyAdaptive": False,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True,
    #      "compute_mean": False, "compute_var": False
    #      },
    #     {"model": "corner_peak", "list_of_function_ids": [1, ],
    #      "current_output_folder": "var2_gpce_gl_p9_q9_lmax2_nonadaptive",
    #      "variant": 2, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 2,
    #      "max_evals": 100000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": False, "spatiallyAdaptive": False,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True,
    #      "compute_mean": False, "compute_var": False
    #      },
    #     {"model": "corner_peak", "list_of_function_ids": [1, ],
    #      "current_output_folder": "var2_gpce_gl_p9_q9_lmax2_modified_nonadaptive",
    #      "variant": 2, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 2,
    #      "max_evals": 100000, "tolerance": 10 ** -6,
    #      "modified_basis": True, "boundary_points": False, "spatiallyAdaptive": False,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True,
    #      "compute_mean": False, "compute_var": False
    #      },
    # ]

    ######discontinuous Var 2#######
    # list_of_dict_run_setups = [
    #     {"model": "discontinuous", "list_of_function_ids": [2, ],
    #      "current_output_folder": "var2_gpce_gl_p9_q9_max103_lmax2_boundary_adaptive",
    #      "variant": 2, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 2,
    #      "max_evals": 1000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": True, "spatiallyAdaptive": True,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True,
    #      "compute_mean": False, "compute_var": False
    #      },
    #     {"model": "discontinuous", "list_of_function_ids": [2,],
    #      "current_output_folder": "var2_gpce_gl_p9_q9_max103_lmax3_boundary_adaptive",
    #      "variant": 2, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
    #      "max_evals": 1000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": True, "spatiallyAdaptive": True,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True,
    #      "compute_mean": False, "compute_var": False
    #      },
    #     {"model": "discontinuous", "list_of_function_ids": [2, ],
    #      "current_output_folder": "var2_gpce_gl_p9_q9_max104_lmax3_boundary_adaptive",
    #      "variant": 2, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
    #      "max_evals": 10000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": True, "spatiallyAdaptive": True,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True,
    #      "compute_mean": False, "compute_var": False
    #      },
    #     {"model": "discontinuous", "list_of_function_ids": [2, ],
    #      "current_output_folder": "var2_gpce_gl_p9_q9_max105_lmax3_boundary_adaptive",
    #      "variant": 2, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
    #      "max_evals": 100000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": True, "spatiallyAdaptive": True,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True,
    #      "compute_mean": False, "compute_var": False
    #      },
    #     # {"model": "discontinuous", "list_of_function_ids": [2, ],
    #     #  "current_output_folder": "var2_gpce_gl_p9_q9_max105_lmax3_modified_adaptive",
    #     #  "variant": 2, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #     #  "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
    #     #  "max_evals": 100000, "tolerance": 10 ** -6,
    #     #  "modified_basis": True, "boundary_points": False, "spatiallyAdaptive": True,
    #     #  "norm_spatiallyAdaptive": 2, "rebalancing": True,
    #     #  "compute_mean": False, "compute_var": False
    #     #  },
    #     {"model": "discontinuous", "list_of_function_ids": [2, ],
    #      "current_output_folder": "var2_gpce_gl_p9_q9_lmax2_boundary_nonadaptive",
    #      "variant": 2, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 2,
    #      "max_evals": 100000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": True, "spatiallyAdaptive": False,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True,
    #      "compute_mean": False, "compute_var": False
    #      },
    #     {"model": "discontinuous", "list_of_function_ids": [2, ],
    #      "current_output_folder": "var2_gpce_gl_p9_q9_lmax3_boundary_nonadaptive",
    #      "variant": 2, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
    #      "max_evals": 100000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": True, "spatiallyAdaptive": False,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True,
    #      "compute_mean": False, "compute_var": False
    #      },
    #     {"model": "discontinuous", "list_of_function_ids": [2, ],
    #      "current_output_folder": "var2_gpce_gl_p9_q9_lmax3_modified_nonadaptive",
    #      "variant": 2, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
    #      "max_evals": 100000, "tolerance": 10 ** -6,
    #      "modified_basis": True, "boundary_points": False, "spatiallyAdaptive": False,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True,
    #      "compute_mean": False, "compute_var": False
    #      },
    #     {"model": "discontinuous", "list_of_function_ids": [2, ],
    #      "current_output_folder": "var2_gpce_gl_p9_q9_lmax3_nonadaptive",
    #      "variant": 2, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 3,
    #      "max_evals": 100000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": False, "spatiallyAdaptive": False,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True,
    #      "compute_mean": False, "compute_var": False
    #      },
    #     {"model": "discontinuous", "list_of_function_ids": [2, ],
    #      "current_output_folder": "var2_gpce_gl_p9_q9_lmax4_boundary_nonadaptive",
    #      "variant": 2, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 4,
    #      "max_evals": 100000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": True, "spatiallyAdaptive": False,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True,
    #      "compute_mean": False, "compute_var": False
    #      },
    # ]

    # list_of_dict_run_setups = [
    #     {"model": "corner_peak", "list_of_function_ids": [1, ],
    #      "current_output_folder": "var2_gpce_gl_p9_q9_lmax4_boundary_nonadaptive",
    #      "variant": 2, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 4,
    #      "max_evals": 100000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": True, "spatiallyAdaptive": False,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True,
    #      "compute_mean": False, "compute_var": False
    #      },
    #     {"model": "corner_peak", "list_of_function_ids": [1, ],
    #      "current_output_folder": "var2_gpce_gl_p9_q9_lmax5_boundary_nonadaptive",
    #      "variant": 2, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 5,
    #      "max_evals": 100000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": True, "spatiallyAdaptive": False,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True,
    #      "compute_mean": False, "compute_var": False
    #      },
    #     {"model": "ishigami", "list_of_function_ids": None,
    #      "current_output_folder": "var2_gpce_kpu_p9_l16_lmax5_boundary_nonadaptive",
    #      "variant": 2, "quadrature_rule": "kpu", "q_order": 16, "p_order": 9, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 5,
    #      "max_evals": 10000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": True, "spatiallyAdaptive": False,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True
    #      },
    #     {"model": "ishigami", "list_of_function_ids": None,
    #      "current_output_folder": "var2_gpce_kpu_p9_l16_lmax6_boundary_nonadaptive",
    #      "variant": 2, "quadrature_rule": "kpu", "q_order": 16, "p_order": 9, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l': 16, "gridName": "Trapezoidal", "lmin": 1, "lmax": 6,
    #      "max_evals": 10000, "tolerance": 10 ** -6,
    #      "modified_basis": False, "boundary_points": True, "spatiallyAdaptive": False,
    #      "norm_spatiallyAdaptive": 2, "rebalancing": True
    #      },
    # ]

    for single_setup_dict in list_of_dict_run_setups:
        model = single_setup_dict["model"]
        assert (model in list_of_models)
        list_of_function_ids = single_setup_dict.get("list_of_function_ids", None)
        if model in list_of_genz_functions and list_of_function_ids is not None:
            number_of_functions = len(list_of_function_ids)
        else:
            number_of_functions = 1

        current_output_folder = single_setup_dict["current_output_folder"]
        variant = single_setup_dict["variant"]

        quadrature_rule = single_setup_dict["quadrature_rule"]
        q_order = single_setup_dict["q_order"]
        p_order = single_setup_dict["p_order"]
        sparse_quadrature = single_setup_dict["sparse_quadrature"]
        read_nodes_from_file = single_setup_dict["read_nodes_from_file"]
        level_sg = single_setup_dict.get("level_sg", 10)

        gridName = single_setup_dict["gridName"]
        lmin = single_setup_dict["lmin"]
        lmax = single_setup_dict["lmax"]
        max_evals = single_setup_dict["max_evals"]
        tolerance = single_setup_dict["tolerance"]
        modified_basis = single_setup_dict["modified_basis"]
        boundary_points = single_setup_dict["boundary_points"]
        spatiallyAdaptive = single_setup_dict["spatiallyAdaptive"]
        rebalancing = single_setup_dict["rebalancing"]
        norm_spatiallyAdaptive = single_setup_dict["norm_spatiallyAdaptive"]
        build_sg_for_e_and_var = single_setup_dict.get("build_sg_for_e_and_var", True)

        compute_mean = single_setup_dict.get("compute_mean", True)
        compute_var = single_setup_dict.get("compute_var", True)

        start_time = time.time()
        # TODO Change for Genz that this is executed in this way whenever user wants that
        if model in list_of_genz_functions and list_of_function_ids is not None:
            # Hard-coded
            dim = 5
            # all_coeffs = np.empty(shape=(number_of_functions, dim))
            # all_weights = np.empty(shape=(number_of_functions, dim))
            # problem_function_list = []
            dictionary_with_inf_about_the_run = defaultdict(dict)
            # for i in range(number_of_functions):
            for i in list_of_function_ids:
                if read_saved_genz_functions:
                    if anisotropic:
                        path_to_saved_genz_functions = str(
                            path_to_saved_all_genz_functions / model / f"coeffs_weights_anisotropic_{dim}d_{i}.npy")
                    else:
                        path_to_saved_genz_functions = str(
                            path_to_saved_all_genz_functions / model / f"coeffs_weights_{dim}d_{i}.npy")
                    with open(path_to_saved_genz_functions, 'rb') as f:
                        coeffs_weights = np.load(f)
                        single_coeffs = coeffs_weights[0]
                        single_weights = coeffs_weights[1]
                else:
                    single_coeffs, single_weights = generate_and_scale_coeff_and_weights(dim=dim,
                                                                                         b=genz_dict[model],
                                                                                         anisotropic=anisotropic)
                # all_coeffs[i] = single_coeffs
                # all_weights[i] = single_weights
                current_output_folder_single_model = f"{current_output_folder}_model_{i}"
                dictionary_with_inf_about_the_run_single_model = main_routine(
                coeffs=single_coeffs, weights=single_weights,
                model=model,
                current_output_folder=current_output_folder_single_model,
                variant=single_setup_dict["variant"],
                quadrature_rule=single_setup_dict["quadrature_rule"],
                q_order=single_setup_dict["q_order"], p_order=single_setup_dict["p_order"],
                poly_normed=single_setup_dict["poly_normed"],
                sampleFromStandardDist=single_setup_dict["sampleFromStandardDist"],
                sparse_quadrature=single_setup_dict["sparse_quadrature"],
                read_nodes_from_file=single_setup_dict["read_nodes_from_file"],
                level_sg=single_setup_dict["level_sg"],
                gridName=single_setup_dict["gridName"],
                lmin=single_setup_dict["lmin"],
                lmax=single_setup_dict["lmax"],
                max_evals=single_setup_dict["max_evals"],
                tolerance=single_setup_dict["tolerance"],
                modified_basis=single_setup_dict["modified_basis"],
                boundary_points=single_setup_dict["boundary_points"],
                spatiallyAdaptive=single_setup_dict["spatiallyAdaptive"],
                rebalancing=single_setup_dict["rebalancing"],
                margin=single_setup_dict["margin"],
                norm_spatiallyAdaptive=single_setup_dict["norm_spatiallyAdaptive"],
                build_sg_for_e_and_var=single_setup_dict["build_sg_for_e_and_var"],
                compute_mean=single_setup_dict["compute_mean"],
                compute_var=single_setup_dict["compute_var"],
                compute_Sobol_m=single_setup_dict["compute_Sobol_m"],
                compute_Sobol_t=single_setup_dict["compute_Sobol_t"],
                operation=single_setup_dict["operation"]

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
                model=single_setup_dict["model"],
                current_output_folder=single_setup_dict["current_output_folder"],
                variant=single_setup_dict["variant"],
                quadrature_rule=single_setup_dict["quadrature_rule"],
                q_order=single_setup_dict["q_order"], p_order=single_setup_dict["p_order"],
                poly_normed=single_setup_dict["poly_normed"],
                sampleFromStandardDist=single_setup_dict["sampleFromStandardDist"],
                sparse_quadrature=single_setup_dict["sparse_quadrature"],
                read_nodes_from_file=single_setup_dict["read_nodes_from_file"],
                level_sg=single_setup_dict["level_sg"],
                gridName=single_setup_dict["gridName"],
                lmin=single_setup_dict["lmin"],
                lmax=single_setup_dict["lmax"],
                max_evals=single_setup_dict["max_evals"],
                tolerance=single_setup_dict["tolerance"],
                modified_basis=single_setup_dict["modified_basis"],
                boundary_points=single_setup_dict["boundary_points"],
                spatiallyAdaptive=single_setup_dict["spatiallyAdaptive"],
                rebalancing=single_setup_dict["rebalancing"],
                margin=single_setup_dict["margin"],
                norm_spatiallyAdaptive=single_setup_dict["norm_spatiallyAdaptive"],
                build_sg_for_e_and_var=single_setup_dict["build_sg_for_e_and_var"],
                compute_mean=single_setup_dict["compute_mean"],
                compute_var=single_setup_dict["compute_var"],
                compute_Sobol_m=single_setup_dict["compute_Sobol_m"],
                compute_Sobol_t=single_setup_dict["compute_Sobol_t"],
                operation=single_setup_dict["operation"]
            )
        end_time = time.time()
        duration = end_time - start_time
        print(f"The whole run took {duration} for examing {number_of_functions} different functions")