from collections import defaultdict
import numpy as np
import pathlib
import time

import os
import sys
cwd = pathlib.Path(os.getcwd())
parent = cwd.parent.absolute()
sys.path.insert(0, os.getcwd())

from uq_sparseSpACE import *

from sparseSpACE.Integrator import *

comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()

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

    ######Ishigami Var 1#######
    # list_of_dict_run_setups = [
    #     {"model": "ishigami", "list_of_function_ids": None, "current_output_folder": "var1_gpce_gl_p4_q5",
    #      "variant": 1, "quadrature_rule": "g", "q_order": 5, "p_order": 4, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l':10},
    #     {"model": "ishigami", "list_of_function_ids": None, "current_output_folder": "var1_gpce_gl_p4_q7",
    #      "variant": 1, "quadrature_rule": "g", "q_order": 7, "p_order": 4, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "ishigami", "list_of_function_ids": None, "current_output_folder": "var1_gpce_gl_p4_q9",
    #      "variant": 1, "quadrature_rule": "g", "q_order": 9, "p_order": 4, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "ishigami", "list_of_function_ids": None, "current_output_folder": "var1_gpce_gl_p6_q7",
    #      "variant": 1, "quadrature_rule": "g", "q_order": 7, "p_order": 6, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "ishigami", "list_of_function_ids": None, "current_output_folder": "var1_gpce_gl_p8_q9",
    #      "variant": 1, "quadrature_rule": "g", "q_order": 9, "p_order": 8, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "ishigami", "list_of_function_ids": None, "current_output_folder": "var1_gpce_gl_p9_q9",
    #      "variant": 1, "quadrature_rule": "g", "q_order": 9, "p_order": 9, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "ishigami", "list_of_function_ids": None, "current_output_folder": "var1_gpce_gl_p9_q10",
    #      "variant": 1, "quadrature_rule": "g", "q_order": 10, "p_order": 9, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "ishigami", "list_of_function_ids": None, "current_output_folder": "var1_gpce_gl_p9_q11",
    #      "variant": 1, "quadrature_rule": "g", "q_order": 11, "p_order": 9, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 10}
    # ]

    # TODO Change this in 5D
    # list_of_dict_run_setups = [
    #     {"model": "ishigami", "list_of_function_ids": None, "current_output_folder": "var1_gpce_cc_p3_q7",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 7, "p_order": 3, "sparse_quadrature": True,
    #      "read_nodes_from_file": False, 'l':10},
    #     {"model": "ishigami", "list_of_function_ids": None, "current_output_folder": "var1_gpce_cc_p5_q11",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 11, "p_order": 5, "sparse_quadrature": True,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "ishigami", "list_of_function_ids": None, "current_output_folder": "var1_gpce_cc_p7_q7",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 7, "p_order": 7, "sparse_quadrature": True,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "ishigami", "list_of_function_ids": None, "current_output_folder": "var1_gpce_cc_p7_q11",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 11, "p_order": 7, "sparse_quadrature": True,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "ishigami", "list_of_function_ids": None, "current_output_folder": "var1_gpce_cc_p7_q15",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 15, "p_order": 7, "sparse_quadrature": True,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "ishigami", "list_of_function_ids": None, "current_output_folder": "var1_gpce_cc_p9_q9",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 9, "p_order": 9, "sparse_quadrature": True,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "ishigami", "list_of_function_ids": None, "current_output_folder": "var1_gpce_cc_p9_q19",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 19, "p_order": 9, "sparse_quadrature": True,
    #      "read_nodes_from_file": False, 'l': 10},
    # ]
    #

    # list_of_dict_run_setups = [
    #     {"model": "ishigami", "list_of_function_ids": None, "current_output_folder": "var1_gpce_kpu_p3_l7",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 9, "p_order": 3, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l':7},
    #     {"model": "ishigami", "list_of_function_ids": None, "current_output_folder": "var1_gpce_kpu_p5_l9",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 9, "p_order": 5, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l': 9},
    #     {"model": "ishigami", "list_of_function_ids": None, "current_output_folder": "var1_gpce_kpu_p7_l9",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 9, "p_order": 7, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l': 9},
    #     {"model": "ishigami", "list_of_function_ids": None, "current_output_folder": "var1_gpce_kpu_p3_l11",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 9, "p_order": 3, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l': 11},
    #     {"model": "ishigami", "list_of_function_ids": None, "current_output_folder": "var1_gpce_kpu_p5_l11",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 9, "p_order": 5, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l': 11},
    #     {"model": "ishigami", "list_of_function_ids": None, "current_output_folder": "var1_gpce_kpu_p7_l11",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 9, "p_order": 7, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l': 11},
    #     {"model": "ishigami", "list_of_function_ids": None, "current_output_folder": "var1_gpce_kpu_p9_l16",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 9, "p_order": 9, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l': 16},
    #     {"model": "ishigami", "list_of_function_ids": None, "current_output_folder": "var1_gpce_kpu_p9_l20",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 9, "p_order": 9, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l': 20},
    # ]

    ######corner_peak Var 1#######
    # TODO Is it problem that total number of nodes in one dim is always even??? superconvergence
    # list_of_dict_run_setups = [
    #     {"model": "corner_peak", "list_of_function_ids": [1,], "current_output_folder": "var1_gpce_gl_p4_q4",
    #      "variant": 1, "quadrature_rule": "g", "q_order": 4, "p_order": 4, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l':10},
    #     {"model": "corner_peak", "list_of_function_ids": [1,], "current_output_folder": "var1_gpce_gl_p4_q6",
    #      "variant": 1, "quadrature_rule": "g", "q_order": 6, "p_order": 4, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "corner_peak", "list_of_function_ids": [1,], "current_output_folder": "var1_gpce_gl_p4_q8",
    #      "variant": 1, "quadrature_rule": "g", "q_order": 8, "p_order": 4, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "corner_peak", "list_of_function_ids": [1,], "current_output_folder": "var1_gpce_gl_p6_q6",
    #      "variant": 1, "quadrature_rule": "g", "q_order": 6, "p_order": 6, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "corner_peak", "list_of_function_ids": [1,], "current_output_folder": "var1_gpce_gl_p6_q8",
    #      "variant": 1, "quadrature_rule": "g", "q_order": 8, "p_order": 6, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 10},
    # ]

    # list_of_dict_run_setups = [
    #     {"model": "corner_peak", "list_of_function_ids": [1, ], "current_output_folder": "var1_gpce_cc_p3_q7",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 7, "p_order": 3, "sparse_quadrature": True,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "corner_peak", "list_of_function_ids": [1,], "current_output_folder": "var1_gpce_cc_p4_q4",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 4, "p_order": 4, "sparse_quadrature": True,
    #      "read_nodes_from_file": False, 'l':10},
    #     {"model": "corner_peak", "list_of_function_ids": [1,], "current_output_folder": "var1_gpce_cc_p4_q6",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 6, "p_order": 4, "sparse_quadrature": True,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "corner_peak", "list_of_function_ids": [1,], "current_output_folder": "var1_gpce_cc_p4_q9",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 9, "p_order": 4, "sparse_quadrature": True,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "corner_peak", "list_of_function_ids": [1,], "current_output_folder": "var1_gpce_cc_p4_q11",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 11, "p_order": 4, "sparse_quadrature": True,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "corner_peak", "list_of_function_ids": [1,], "current_output_folder": "var1_gpce_cc_p6_q7",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 7, "p_order": 6, "sparse_quadrature": True,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "corner_peak", "list_of_function_ids": [1,], "current_output_folder": "var1_gpce_cc_p6_q11",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 11, "p_order": 6, "sparse_quadrature": True,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "corner_peak", "list_of_function_ids": [1,], "current_output_folder": "var1_gpce_cc_p6_q13",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 13, "p_order": 6, "sparse_quadrature": True,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "corner_peak", "list_of_function_ids": [1, ], "current_output_folder": "var1_gpce_kpu_p4_l7",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 9, "p_order": 4, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l':7},
    #     {"model": "corner_peak", "list_of_function_ids": [1, ], "current_output_folder": "var1_gpce_kpu_p4_l9",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 9, "p_order": 4, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l': 9},
    #     {"model": "corner_peak", "list_of_function_ids": [1, ], "current_output_folder": "var1_gpce_kpu_p4_l10",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 9, "p_order": 4, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l': 10},
    #     {"model": "corner_peak", "list_of_function_ids": [1, ], "current_output_folder": "var1_gpce_kpu_p6_l10",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 9, "p_order": 6, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l': 10},
    #     {"model": "corner_peak", "list_of_function_ids": [1, ], "current_output_folder": "var1_gpce_kpu_p6_l12",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 9, "p_order": 6, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l': 12},
    #     {"model": "corner_peak", "list_of_function_ids": [1, ], "current_output_folder": "var1_gpce_kpu_p6_l14",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 9, "p_order": 6, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l': 14}
    # ]

    # ######product_peak Var 1#######
    # # functions in the consideration: anisotropic 2 or 1
    # list_of_dict_run_setups = [
    #     {"model": "product_peak", "list_of_function_ids": [2,], "current_output_folder": "var1_gpce_gl_p4_q4",
    #      "variant": 1, "quadrature_rule": "g", "q_order": 4, "p_order": 4, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l':10},
    #     {"model": "product_peak", "list_of_function_ids": [2,], "current_output_folder": "var1_gpce_gl_p4_q6",
    #      "variant": 1, "quadrature_rule": "g", "q_order": 6, "p_order": 4, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "product_peak", "list_of_function_ids": [2,], "current_output_folder": "var1_gpce_gl_p4_q8",
    #      "variant": 1, "quadrature_rule": "g", "q_order": 8, "p_order": 4, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "product_peak", "list_of_function_ids": [2,], "current_output_folder": "var1_gpce_gl_p6_q6",
    #      "variant": 1, "quadrature_rule": "g", "q_order": 6, "p_order": 6, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "product_peak", "list_of_function_ids": [2,], "current_output_folder": "var1_gpce_gl_p6_q8",
    #      "variant": 1, "quadrature_rule": "g", "q_order": 8, "p_order": 6, "sparse_quadrature": False,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "product_peak", "list_of_function_ids": [2,], "current_output_folder": "var1_gpce_cc_p3_q7",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 7, "p_order": 3, "sparse_quadrature": True,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "product_peak", "list_of_function_ids": [2,], "current_output_folder": "var1_gpce_cc_p4_q4",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 4, "p_order": 4, "sparse_quadrature": True,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "product_peak", "list_of_function_ids": [2,], "current_output_folder": "var1_gpce_cc_p4_q6",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 6, "p_order": 4, "sparse_quadrature": True,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "product_peak", "list_of_function_ids": [2,], "current_output_folder": "var1_gpce_cc_p4_q9",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 9, "p_order": 4, "sparse_quadrature": True,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "product_peak", "list_of_function_ids": [2,], "current_output_folder": "var1_gpce_cc_p4_q11",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 11, "p_order": 4, "sparse_quadrature": True,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "product_peak", "list_of_function_ids": [2,], "current_output_folder": "var1_gpce_cc_p6_q7",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 7, "p_order": 6, "sparse_quadrature": True,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "product_peak", "list_of_function_ids": [2,], "current_output_folder": "var1_gpce_cc_p6_q11",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 11, "p_order": 6, "sparse_quadrature": True,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "product_peak", "list_of_function_ids": [2,], "current_output_folder": "var1_gpce_cc_p6_q13",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 13, "p_order": 6, "sparse_quadrature": True,
    #      "read_nodes_from_file": False, 'l': 10},
    #     {"model": "product_peak", "list_of_function_ids": [2,], "current_output_folder": "var1_gpce_kpu_p4_l7",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 9, "p_order": 4, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l': 7},
    #     {"model": "product_peak", "list_of_function_ids": [2,], "current_output_folder": "var1_gpce_kpu_p4_l9",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 9, "p_order": 4, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l': 9},
    #     {"model": "product_peak", "list_of_function_ids": [2,], "current_output_folder": "var1_gpce_kpu_p4_l10",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 9, "p_order": 4, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l': 10},
    #     {"model": "product_peak", "list_of_function_ids": [2,], "current_output_folder": "var1_gpce_kpu_p6_l10",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 9, "p_order": 6, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l': 10},
    #     {"model": "product_peak", "list_of_function_ids": [2,], "current_output_folder": "var1_gpce_kpu_p6_l12",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 9, "p_order": 6, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l': 12},
    #     {"model": "product_peak", "list_of_function_ids": [2,], "current_output_folder": "var1_gpce_kpu_p6_l14",
    #      "variant": 1, "quadrature_rule": "c", "q_order": 9, "p_order": 6, "sparse_quadrature": True,
    #      "read_nodes_from_file": True, 'l': 14}
    # ]

    ######discontinuous Var 1#######
    # functions in the consideration: non-anisotropic 2 or 3
    list_of_dict_run_setups = [
        # {"model": "discontinuous", "list_of_function_ids": [2,], "current_output_folder": "var1_gpce_gl_p4_q4",
        #  "variant": 1, "quadrature_rule": "g", "q_order": 4, "p_order": 4, "sparse_quadrature": False,
        #  "read_nodes_from_file": False, 'l':10,
        #  "compute_mean": False, "compute_var": False
        #  },
        # {"model": "discontinuous", "list_of_function_ids": [2,], "current_output_folder": "var1_gpce_gl_p6_q6",
        #  "variant": 1, "quadrature_rule": "g", "q_order": 6, "p_order": 6, "sparse_quadrature": False,
        #  "read_nodes_from_file": False, 'l': 10,
        #  "compute_mean": False, "compute_var": False
        #  },
        # {"model": "discontinuous", "list_of_function_ids": [2,], "current_output_folder": "var1_gpce_gl_p6_q9",
        #  "variant": 1, "quadrature_rule": "g", "q_order": 9, "p_order": 6, "sparse_quadrature": False,
        #  "read_nodes_from_file": False, 'l': 10,
        #  "compute_mean": False, "compute_var": False
        # #  },
        # {"model": "discontinuous", "list_of_function_ids": [2,], "current_output_folder": "var1_gpce_cc_p3_q7",
        #  "variant": 1, "quadrature_rule": "c", "q_order": 7, "p_order": 3, "sparse_quadrature": True,
        #  "read_nodes_from_file": False, 'l': 10,
        #  "compute_mean": False, "compute_var": False
        #  },
        # {"model": "discontinuous", "list_of_function_ids": [2,], "current_output_folder": "var1_gpce_cc_p4_q11",
        #  "variant": 1, "quadrature_rule": "c", "q_order": 11, "p_order": 4, "sparse_quadrature": True,
        #  "read_nodes_from_file": False, 'l': 10,
        #  "compute_mean": False, "compute_var": False
        #  },
        {"model": "discontinuous", "list_of_function_ids": [2,], "current_output_folder": "var1_gpce_cc_p6_q13",
         "variant": 1, "quadrature_rule": "c", "q_order": 13, "p_order": 6, "sparse_quadrature": True,
         "read_nodes_from_file": False, 'l': 10,
         "compute_mean": False, "compute_var": False
         },
        {"model": "discontinuous", "list_of_function_ids": [2,], "current_output_folder": "var1_gpce_kpu_p4_l7",
         "variant": 1, "quadrature_rule": "c", "q_order": 7, "p_order": 4, "sparse_quadrature": True,
         "read_nodes_from_file": True, 'l': 7,
         "compute_mean": False, "compute_var": False
         },
        {"model": "discontinuous", "list_of_function_ids": [2,], "current_output_folder": "var1_gpce_kpu_p6_l10",
         "variant": 1, "quadrature_rule": "c", "q_order": 10, "p_order": 6, "sparse_quadrature": True,
         "read_nodes_from_file": True, 'l': 10,
         "compute_mean": False, "compute_var": False
         },
        {"model": "discontinuous", "list_of_function_ids": [2,], "current_output_folder": "var1_gpce_kpu_p6_l14",
         "variant": 1, "quadrature_rule": "c", "q_order": 14, "p_order": 6, "sparse_quadrature": True,
         "read_nodes_from_file": True, 'l': 14,
         "compute_mean": False, "compute_var": False
         }
    ]

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
        l = single_setup_dict["l"]

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
                    model, current_output_folder_single_model, coeffs=single_coeffs, weights=single_weights,
                    variant=variant, quadrature_rule=quadrature_rule, q_order=q_order, p_order=p_order,
                    sparse_quadrature=sparse_quadrature, read_nodes_from_file=read_nodes_from_file, l=l,
                    compute_mean=compute_mean,
                    compute_var=compute_var
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
                sparse_quadrature=sparse_quadrature, read_nodes_from_file=read_nodes_from_file, l=l,
                compute_mean=compute_mean,
                compute_var=compute_var
            )
        end_time = time.time()
        duration = end_time - start_time
        print(f"The whole run took {duration} for examing {number_of_functions} different functions")