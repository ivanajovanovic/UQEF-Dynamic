"""
Usage of the UQEF and UQEF-Dynamic with Hydrology models, more generally, models that produce time-dependent output.
@author: Florian Kuenzner and Ivana Jovanovic
"""
import os
import subprocess
import sys
import pickle
import dill
from distutils.util import strtobool
import time

import uqef

# additionally added for the debugging of the nodes
import pandas as pd
import pathlib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

# sys.path.insert(0, os.getcwd())
sys.path.insert(0, "/work/ga45met/mnt/linux_cluster_2/UQEF-Dynamic")

from uqef_dynamic.utils import utility

from uqef_dynamic.models.larsim import LarsimModelUQ
from uqef_dynamic.models.larsim import LarsimStatistics

from uqef_dynamic.models.linearDampedOscillator import LinearDampedOscillatorModel
from uqef_dynamic.models.linearDampedOscillator import LinearDampedOscillatorStatistics

from uqef_dynamic.models.ishigami import IshigamiModel
from uqef_dynamic.models.ishigami import IshigamiStatistics

from uqef_dynamic.models.productFunction import ProductFunctionModel
from uqef_dynamic.models.productFunction import ProductFunctionStatistics

from uqef_dynamic.models.hbv_sask import HBVSASKModelUQ
from uqef_dynamic.models.hbv_sask import HBVSASKStatistics

from uqef_dynamic.models.pybamm import pybammModelUQ as pybammmodel
from uqef_dynamic.models.pybamm import pybammStatistics


# instantiate UQsim
uqsim = uqef.UQsim()

#####################################
# change args locally for testing and debugging
#####################################

local_debugging = True
if local_debugging:
    save_solver_results = False

    uqsim.args.model = "hbvsask"  # "larsim" "hbvsask" "battery"

    uqsim.args.uncertain = "all"
    uqsim.args.chunksize = 1

    uqsim.args.uq_method = "mc"  # "sc" | "saltelli" | "mc" | "ensemble"
    
    uqsim.args.mc_numevaluations = 1000
    uqsim.args.sampling_rule = "halton"  # "random" | "sobol" | "latin_hypercube" | "halton"  | "hammersley"
    
    uqsim.args.sc_q_order = 5  # 7 #10 3
    uqsim.args.sc_p_order = 2  # 4, 5, 6, 8
    uqsim.args.sc_quadrature_rule = "g"  # "p" "genz_keister_24" "leja" "clenshaw_curtis"

    uqsim.args.read_nodes_from_file = False
    l = 5  # 10
    path_to_file = pathlib.Path("/work/ga45met/sparseSpACE/sparse_grid_nodes_weights")
    uqsim.args.parameters_file = path_to_file / f"KPU_d6_l{l}.asc" # f"KPU_d7_l{l}.asc"
    uqsim.args.parameters_setup_file = None #pathlib.Path("/work/ga45met/mnt/linux_cluster_2/UQEF-Dynamic/data/configurations/KPU_HBV_d3.json")

    uqsim.args.sc_poly_rule = "three_terms_recurrence"  # "gram_schmidt" | "three_terms_recurrence" | "cholesky"
    uqsim.args.sc_poly_normed = True  # True
    uqsim.args.sc_sparse_quadrature = False  # False
    uqsim.args.regression = False
    uqsim.args.cross_truncation = 0.7

    # uqsim.args.inputModelDir = str(pathlib.Path('/work/ga45met/Larsim-data'))
    uqsim.args.inputModelDir = pathlib.Path("/work/ga45met/Hydro_Models/HBV-SASK-data")
    # uqsim.args.inputModelDir = pathlib.Path('/work/ga45met/anaconda3/envs/py3115_uq/lib/python3.11/site-packages/pybamm/input/drive_cycles')

    # uqsim.args.sourceDir = os.path.abspath(os.path.join('/work/ga45met/mnt/linux_cluster_2', 'UQEF-Dynamic'))
    uqsim.args.sourceDir = pathlib.Path("/work/ga45met/Hydro_Models/HBV-SASK-data")

    uqsim.args.outputResultDir = os.path.abspath(os.path.join("/work/ga45met/battery_results", 'sc_p2_l5_ct07'))  # mc_10000 mc_10000_terminal_voltage
    uqsim.args.outputResultDir = os.path.abspath(os.path.join('/work/ga45met/Hydro_Models/HBV-SASK-data', 'Oldman_Basin/model_runs', 'mc_10d_1000_short')) 
    uqsim.args.outputModelDir = uqsim.args.outputResultDir

    uqsim.args.config_file = '/work/ga45met/mnt/linux_cluster_2/UQEF-Dynamic/data/configurations/configuration_hbv_6D_GoF.json'
    # uqsim.args.config_file = '/work/ga45met/mnt/linux_cluster_2/UQEF-Dynamic/data/configurations/configuration_battery.json'

    uqsim.args.sampleFromStandardDist = True  # False

    uqsim.args.mpi = True
    uqsim.args.mpi_method = "MpiPoolSolver"  # "LinearSolver"

    uqsim.args.disable_statistics = False
    uqsim.args.disable_calc_statistics = True
    uqsim.args.parallel_statistics = True  # False

    uqsim.args.instantly_save_results_for_each_time_step = False #False
    uqsim.args.uqsim_store_to_file = False

    uqsim.args.compute_Sobol_t = True  # True False
    uqsim.args.compute_Sobol_m = False  # True False

    uqsim.args.num_cores = 1

    uqsim.args.save_all_simulations = True  # True for sc
    uqsim.args.store_qoi_data_in_stat_dict = False  # if set to True, the qoi_values entry is stored in the stat_dict 
    uqsim.args.store_gpce_surrogate_in_stat_dict = True
    uqsim.args.collect_and_save_state_data = False # False 

    uqsim.setup_configuration_object()

# TODO Eventually add these configurations to uqef.args
utility.DEFAULT_DICT_WHAT_TO_PLOT = {
    "E_minus_std": False, "E_plus_std": False, "P10": True, "P90": True,
    "StdDev": True, "Skew": False, "Kurt": False, "Sobol_m": False, "Sobol_m2": False, "Sobol_t": True
}
utility.DEFAULT_DICT_STAT_TO_COMPUTE = {
    "Var": True, "StdDev": False, "P10": True, "P90": True,
    "Skew": False, "Kurt": False, "Sobol_m": True, "Sobol_m2": False, "Sobol_t": True
}
dict_stat_to_compute = utility.DEFAULT_DICT_STAT_TO_COMPUTE
compute_sobol_indices_with_samples = True  # This is only relevant in the mc-saltelli's approach
if uqsim.args.uq_method == "mc" and uqsim.args.compute_Sobol_m:
    compute_sobol_indices_with_samples = True

save_gpce_surrogate = True  # if True a gpce surrogate for each QoI for each time step is saved in a separate file
compute_other_stat_besides_pce_surrogate = True  # This is relevant only when uq_method == "sc" 

compute_kl_expansion_of_qoi = False
compute_timewise_gpce_next_to_kl_expansion = False
kl_expansion_order = 10
compute_generalized_sobol_indices = False
compute_generalized_sobol_indices_over_time = False
compute_covariance_matrix_in_time = True

#####################################
# additional path settings:
#####################################

if uqsim.is_master() and not uqsim.is_restored():
    if not os.path.isdir(uqsim.args.outputResultDir):
        subprocess.run(["mkdir", "-p", uqsim.args.outputResultDir])
    print("outputResultDir: {}".format(uqsim.args.outputResultDir))

# Set the working folder where all the model runs related output and files will be written
try:
    uqsim.args.workingDir = os.path.abspath(os.path.join(uqsim.args.outputResultDir,
                                                         uqsim.configuration_object["model_paths"]["workingDir"]))
except KeyError:
    uqsim.args.workingDir = os.path.abspath(os.path.join(uqsim.args.outputResultDir, "model_runs"))

try:
    uqsim.configuration_object["model_paths"]["workingDir"] = uqsim.args.workingDir
except KeyError:
    uqsim.configuration_object["model_paths"] = {}
    uqsim.configuration_object["model_paths"]["workingDir"] = uqsim.args.workingDir

if uqsim.is_master() and not uqsim.is_restored():
    if not os.path.isdir(uqsim.configuration_object["model_paths"]["workingDir"]):
        subprocess.run(["mkdir", uqsim.configuration_object["model_paths"]["workingDir"]])

#####################################
# register model
#####################################

uqsim.models.update({"larsim"         : (lambda: LarsimModelUQ.LarsimModelUQ(
    configurationObject=uqsim.configuration_object,
    inputModelDir=uqsim.args.inputModelDir,
    workingDir=uqsim.args.workingDir,
    sourceDir=uqsim.args.sourceDir,
    disable_statistics=uqsim.args.disable_statistics,
    uq_method=uqsim.args.uq_method))})
uqsim.models.update({"oscillator"     : (lambda: LinearDampedOscillatorModel.LinearDampedOscillatorModel(uqsim.configuration_object))})
uqsim.models.update({"ishigami"       : (lambda: IshigamiModel.IshigamiModel(
    configurationObject=uqsim.configuration_object,
    workingDir=uqsim.args.workingDir
    ))})
uqsim.models.update({"productFunction": (lambda: ProductFunctionModel.ProductFunctionModel(uqsim.configuration_object))})
uqsim.models.update({"hbvsask"         : (lambda: HBVSASKModelUQ.HBVSASKModelUQ(
    configurationObject=uqsim.configuration_object,
    inputModelDir=uqsim.args.inputModelDir,
    workingDir=uqsim.args.workingDir,
    disable_statistics=uqsim.args.disable_statistics,
    uq_method=uqsim.args.uq_method
))})
uqsim.models.update({"battery"         : (lambda: pybammmodel.pybammModelUQ(
    configurationObject=uqsim.configuration_object,
    inputModelDir=uqsim.args.inputModelDir,
    workingDir=uqsim.args.workingDir,
))})

#####################################
# register statistics
#####################################

uqsim.statistics.update({"larsim"         : (lambda: LarsimStatistics.LarsimStatistics(
    configurationObject=uqsim.configuration_object,
    workingDir=uqsim.args.workingDir,
    sampleFromStandardDist=uqsim.args.sampleFromStandardDist,
    store_qoi_data_in_stat_dict=uqsim.args.store_qoi_data_in_stat_dict,
    store_gpce_surrogate=uqsim.args.store_gpce_surrogate_in_stat_dict,
    parallel_statistics=uqsim.args.parallel_statistics,
    mpi_chunksize=uqsim.args.mpi_chunksize,
    unordered=False,
    uq_method=uqsim.args.uq_method,
    compute_Sobol_t=uqsim.args.compute_Sobol_t,
    compute_Sobol_m=uqsim.args.compute_Sobol_m,
    save_gpce_surrogate=save_gpce_surrogate,
))})
uqsim.statistics.update({"oscillator"     : (lambda: LinearDampedOscillatorStatistics.LinearDampedOscillatorStatistics())})
uqsim.statistics.update({"ishigami"       : (lambda: IshigamiStatistics.IshigamiStatistics(
    configurationObject=uqsim.configuration_object,
    workingDir=uqsim.args.outputResultDir,  # .args.workingDir,
    sampleFromStandardDist=uqsim.args.sampleFromStandardDist,
    parallel_statistics=uqsim.args.parallel_statistics,
    mpi_chunksize=uqsim.args.mpi_chunksize,
    unordered=False,
    uq_method=uqsim.args.uq_method,
    compute_Sobol_t=uqsim.args.compute_Sobol_t,
    compute_Sobol_m=uqsim.args.compute_Sobol_m,
    compute_Sobol_m2=uqsim.args.compute_Sobol_m2,
    save_all_simulations=uqsim.args.save_all_simulations,
    store_qoi_data_in_stat_dict=uqsim.args.store_qoi_data_in_stat_dict,
    store_gpce_surrogate_in_stat_dict=uqsim.args.store_gpce_surrogate_in_stat_dict,
    instantly_save_results_for_each_time_step=uqsim.args.instantly_save_results_for_each_time_step,
    compute_sobol_indices_with_samples=compute_sobol_indices_with_samples,
    save_gpce_surrogate=save_gpce_surrogate,
    dict_stat_to_compute=dict_stat_to_compute,
))})
uqsim.statistics.update({"productFunction": (lambda: ProductFunctionStatistics.ProductFunctionStatistics(uqsim.configuration_object))})
uqsim.statistics.update({"hbvsask"         : (lambda: HBVSASKStatistics.HBVSASKStatistics(
    configurationObject=uqsim.configuration_object,  # uqsim.args.config_file,
    workingDir=uqsim.args.outputResultDir,  # .args.workingDir,
    inputModelDir=uqsim.args.inputModelDir,
    sampleFromStandardDist=uqsim.args.sampleFromStandardDist,
    parallel_statistics=uqsim.args.parallel_statistics,
    mpi_chunksize=uqsim.args.mpi_chunksize,
    unordered=False,
    uq_method=uqsim.args.uq_method,
    compute_Sobol_t=uqsim.args.compute_Sobol_t,
    compute_Sobol_m=uqsim.args.compute_Sobol_m,
    compute_Sobol_m2=uqsim.args.compute_Sobol_m2,
    save_all_simulations=uqsim.args.save_all_simulations,
    collect_and_save_state_data=uqsim.args.collect_and_save_state_data,
    store_qoi_data_in_stat_dict=uqsim.args.store_qoi_data_in_stat_dict,
    store_gpce_surrogate_in_stat_dict=uqsim.args.store_gpce_surrogate_in_stat_dict,
    instantly_save_results_for_each_time_step=uqsim.args.instantly_save_results_for_each_time_step,
    dict_what_to_plot=utility.DEFAULT_DICT_WHAT_TO_PLOT,
    compute_sobol_indices_with_samples=compute_sobol_indices_with_samples,
    save_gpce_surrogate=save_gpce_surrogate,
    compute_other_stat_besides_pce_surrogate=compute_other_stat_besides_pce_surrogate,
    compute_kl_expansion_of_qoi = compute_kl_expansion_of_qoi,
    compute_timewise_gpce_next_to_kl_expansion=compute_timewise_gpce_next_to_kl_expansion,
    kl_expansion_order = kl_expansion_order,
    compute_generalized_sobol_indices = compute_generalized_sobol_indices,
    compute_generalized_sobol_indices_over_time = compute_generalized_sobol_indices_over_time,
    compute_covariance_matrix_in_time = compute_covariance_matrix_in_time,
    dict_stat_to_compute=dict_stat_to_compute,
))})
uqsim.statistics.update({"battery"         : (lambda: pybammStatistics.pybammStatistics(
    configurationObject=uqsim.configuration_object,  # uqsim.args.config_file,
    workingDir=uqsim.args.outputResultDir,  # .args.workingDir,
    inputModelDir=uqsim.args.inputModelDir,
    sampleFromStandardDist=uqsim.args.sampleFromStandardDist,
    parallel_statistics=uqsim.args.parallel_statistics,
    mpi_chunksize=uqsim.args.mpi_chunksize,
    unordered=False,
    uq_method=uqsim.args.uq_method,
    compute_Sobol_t=uqsim.args.compute_Sobol_t,
    compute_Sobol_m=uqsim.args.compute_Sobol_m,
    compute_Sobol_m2=uqsim.args.compute_Sobol_m2,
    save_all_simulations=uqsim.args.save_all_simulations,
    collect_and_save_state_data=uqsim.args.collect_and_save_state_data,
    store_qoi_data_in_stat_dict=uqsim.args.store_qoi_data_in_stat_dict,
    store_gpce_surrogate_in_stat_dict=uqsim.args.store_gpce_surrogate_in_stat_dict,
    instantly_save_results_for_each_time_step=uqsim.args.instantly_save_results_for_each_time_step,
    dict_what_to_plot=utility.DEFAULT_DICT_WHAT_TO_PLOT,
    compute_sobol_indices_with_samples=compute_sobol_indices_with_samples,
    save_gpce_surrogate=save_gpce_surrogate,
    compute_other_stat_besides_pce_surrogate=compute_other_stat_besides_pce_surrogate,
    compute_kl_expansion_of_qoi = compute_kl_expansion_of_qoi,
    compute_timewise_gpce_next_to_kl_expansion=compute_timewise_gpce_next_to_kl_expansion,
    kl_expansion_order = kl_expansion_order,
    compute_generalized_sobol_indices = compute_generalized_sobol_indices,
    compute_generalized_sobol_indices_over_time = compute_generalized_sobol_indices_over_time,
    compute_covariance_matrix_in_time = compute_covariance_matrix_in_time,
    dict_stat_to_compute=dict_stat_to_compute,
))})

#####################################
# setup
#####################################

uqsim.setup()

# save simulation nodes
if uqsim.is_master():
    simulationNodes_save_file = "nodes"
    uqsim.save_simulationNodes(fileName=simulationNodes_save_file)
    number_full_model_evaluations = len(uqsim.simulationNodes.nodes.T)

# save the dictionary with the arguments - once before the simulation
if uqsim.is_master():
    argsFileName = os.path.abspath(os.path.join(uqsim.args.outputResultDir, "uqsim_args.pkl"))
    with open(argsFileName, 'wb') as handle:
        pickle.dump(uqsim.args, handle, protocol=pickle.HIGHEST_PROTOCOL)

# save initially configurationObject if program breaks during simulation
if uqsim.is_master():
    fileName = pathlib.Path(uqsim.args.outputResultDir) / "configurationObject"
    with open(fileName, 'wb') as f:
        dill.dump(uqsim.configuration_object, f)

#####################################
# start the simulation
#####################################

start_time_model_simulations = time.time()
uqsim.simulate()
end_time_model_simulations = time.time()
time_model_simulations = end_time_model_simulations - start_time_model_simulations

#####################################
#uqsim.save_simulation_parameters()
if hasattr(uqsim.simulation, 'parameters') and uqsim.simulation.parameters is not None:
    df = pd.DataFrame({'parameters': [row for row in uqsim.simulation.parameters]})
    df.to_pickle(os.path.abspath(os.path.join(uqsim.args.outputResultDir, "df_uqsim_simulation_parameters.pkl")), compression="gzip")

if hasattr(uqsim.simulation, 'nodes') and uqsim.simulation.nodes is not None:
    df = pd.DataFrame({'nodes': [row for row in uqsim.simulation.nodes]})
    df.to_pickle(os.path.abspath(os.path.join(uqsim.args.outputResultDir, "df_uqsim_simulation_nodes.pkl")), compression="gzip")

if hasattr(uqsim.simulation, 'weights') and uqsim.simulation.weights is not None:
    df = pd.DataFrame({'weights': [row for row in uqsim.simulation.weights]})
    df.to_pickle(os.path.abspath(os.path.join(uqsim.args.outputResultDir, "df_uqsim_simulation_weights.pkl")), compression="gzip")

#####################################
# re-save uqsim.configurationObject
#####################################

if uqsim.is_master():
    fileName = pathlib.Path(uqsim.args.outputResultDir) / "configurationObject"
    with open(fileName, 'wb') as f:
        dill.dump(uqsim.configuration_object, f)

#####################################
# statistics
#####################################

start_time_computing_statistics = time.time()
uqsim.prepare_statistics()
uqsim.calc_statistics()
# if uqsim.is_master():
#     uqsim.statistic.compute_covariance_matrix_in_time()
end_time_computing_statistics = time.time()
time_computing_statistics = end_time_computing_statistics - start_time_computing_statistics

uqsim.save_statistics()

# save the dictionary with the arguments once again
if uqsim.is_master():
    time_infoFileName = os.path.abspath(os.path.join(uqsim.args.outputResultDir, f"time_info.txt"))
    with open(time_infoFileName, 'w') as fp:
        fp.write(f'number_full_model_runs: {number_full_model_evaluations}\n')
        fp.write(f'time_model_simulations: {time_model_simulations}\n')
        # fp.write(f'time_producing_gpce: {time_producing_gpce}\n')
        fp.write(f'time_computing_statistics: {time_computing_statistics}')

#####################################
# tear down
#####################################

uqsim.tear_down()
