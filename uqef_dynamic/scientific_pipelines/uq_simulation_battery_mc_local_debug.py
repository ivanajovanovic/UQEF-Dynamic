"""
Local debugging script for Battery model MC analysis.
Runs 10000 MC samples and computes basic output statistics (time series).

Run with:
    mpiexec -n <num_processes> python uqef_dynamic/scientific_pipelines/uq_simulation_battery_mc_local_debug.py

Example (4 local processes):
    mpiexec -n 4 python uqef_dynamic/scientific_pipelines/uq_simulation_battery_mc_local_debug.py
"""
import os
import subprocess
import sys
import pickle
import dill
import time
import pathlib

import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

from uqef_dynamic.utils import utility
from uqef_dynamic.config import ConfigurationFactory, ExtendedUQSimArgumentParser, ExtendedUQSim

try:
    from uqef_dynamic.models.pybamm import pybammModelUQ as pybammmodel
    from uqef_dynamic.models.pybamm import pybammStatistics
    PYBAMM_AVAILABLE = True
except ImportError:
    PYBAMM_AVAILABLE = False
    print("ERROR: pybamm model not available. Install pybamm or check your conda environment.")
    sys.exit(1)

# ─────────────────────────────────────────────
# Paths — adjust if needed
# ─────────────────────────────────────────────

BASE_SOURCE_PATH = pathlib.Path(__file__).resolve().parents[2]  # UQEF-Dynamic root

CONDA_ENV        = "my_uq_env"
INPUT_MODEL_DIR  = pathlib.Path(
    f"/dss/dsshome1/lxc0C/ga45met2/.conda/envs/{CONDA_ENV}"
    "/lib/python3.11/site-packages/pybamm/input/drive_cycles"
)
CONFIG_FILE      = BASE_SOURCE_PATH / "uqef_dynamic/models/pybamm/configuration_battery_24_shot_names.json"
OUTPUT_RESULT_DIR = BASE_SOURCE_PATH / "debug_output" / "battery_mc_local_debug"

# ─────────────────────────────────────────────
# Statistics to compute
# ─────────────────────────────────────────────

dict_stat_to_compute = {
    "Var": True, "StdDev": True,
    "P10": True, "P90": True,
    "E_minus_std": False, "E_plus_std": False,
    "Skew": False, "Kurt": False,
    "Sobol_m": True, "Sobol_m2": False, "Sobol_t": True,
}

dict_what_to_plot = {
    "E_minus_std": False, "E_plus_std": False,
    "E_minus_2std": True, "E_plus_2std": True,
    "P10": True, "P90": True,
    "StdDev": True, "Skew": False, "Kurt": False,
    "Sobol_m": True, "Sobol_m2": False, "Sobol_t": True,
    "generalized_sobol_total_index": False, "generalized_sobol_main_index": False,
}

# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────

uqsim = ExtendedUQSim()
extended_parser = ExtendedUQSimArgumentParser(uqsim)

config = ConfigurationFactory.create_configuration(
    model_type="battery",
    uq_method="mc",
    mc_numevaluations=10000,
    sampling_rule="random",
    mpi=True,
    mpi_method="MpiPoolSolver",
    parallel=False,
    sampleFromStandardDist=True,
    parallel_statistics=True,
    compute_Sobol_m=True,
    compute_Sobol_t=True,
    save_all_simulations=True,
    config_file=str(CONFIG_FILE),
    inputModelDir=str(INPUT_MODEL_DIR),
    outputResultDir=str(OUTPUT_RESULT_DIR),
    sourceDir=str(BASE_SOURCE_PATH),
)

config.apply_to_uqsim(uqsim)

print(f"Model:            {config.model}")
print(f"UQ method:        {config.uq_method}")
print(f"MC evaluations:   {config.mc_numevaluations}")
print(f"Sampling rule:    {config.sampling_rule}")
print(f"Config file:      {CONFIG_FILE}")
print(f"Input model dir:  {INPUT_MODEL_DIR}")
print(f"Output dir:       {OUTPUT_RESULT_DIR}")

uqsim.setup_configuration_object()

start_time = time.time()

utility.DEFAULT_DICT_STAT_TO_COMPUTE = dict_stat_to_compute
utility.DEFAULT_DICT_WHAT_TO_PLOT    = dict_what_to_plot

# Advanced analysis — all off for basic debug run
# For pure MC (no regression), Sobol indices must be computed from samples — auto-set by
# ConfigurationFactory when uq_method="mc" and compute_Sobol_m=True, so read it from config.
compute_sobol_indices_with_samples        = getattr(config, 'compute_sobol_indices_with_samples', True)
save_gpce_surrogate                       = False
compute_other_stat_besides_pce_surrogate  = False
compute_kl_expansion_of_qoi              = False
kl_expansion_order                       = 10
compute_timewise_gpce_next_to_kl_expansion = False
compute_generalized_sobol_indices         = False
compute_generalized_sobol_indices_over_time = False
compute_covariance_matrix_in_time        = False
allow_conditioning_results_based_on_metric = False
condition_results_based_on_metric        = "NSE"
condition_results_based_on_metric_value  = 0.2
condition_results_based_on_metric_sign   = "greater_or_equal"

# ─────────────────────────────────────────────
# Create output directory
# ─────────────────────────────────────────────

os.makedirs(str(OUTPUT_RESULT_DIR), exist_ok=True)

working_dir = str(OUTPUT_RESULT_DIR)
try:
    working_dir = os.path.abspath(os.path.join(
        uqsim.args.outputResultDir,
        uqsim.configuration_object["model_paths"]["workingDir"]
    ))
except KeyError:
    working_dir = str(OUTPUT_RESULT_DIR)

try:
    uqsim.configuration_object["model_paths"]["workingDir"] = working_dir
except KeyError:
    uqsim.configuration_object["model_paths"] = {"workingDir": working_dir}

os.makedirs(working_dir, exist_ok=True)

# ─────────────────────────────────────────────
# Register model and statistics
# ─────────────────────────────────────────────

uqsim.models.update({"battery": (lambda: pybammmodel.pybammModelUQ(
    configurationObject=uqsim.configuration_object,
    inputModelDir=uqsim.args.inputModelDir,
    workingDir=working_dir,
))})

uqsim.statistics.update({"battery": (lambda: pybammStatistics.pybammStatistics(
    configurationObject=uqsim.configuration_object,
    workingDir=uqsim.args.outputResultDir,
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
    dict_what_to_plot=dict_what_to_plot,
    compute_sobol_indices_with_samples=compute_sobol_indices_with_samples,
    save_gpce_surrogate=save_gpce_surrogate,
    compute_other_stat_besides_pce_surrogate=compute_other_stat_besides_pce_surrogate,
    compute_kl_expansion_of_qoi=compute_kl_expansion_of_qoi,
    compute_timewise_gpce_next_to_kl_expansion=compute_timewise_gpce_next_to_kl_expansion,
    kl_expansion_order=kl_expansion_order,
    compute_generalized_sobol_indices=compute_generalized_sobol_indices,
    compute_generalized_sobol_indices_over_time=compute_generalized_sobol_indices_over_time,
    compute_covariance_matrix_in_time=compute_covariance_matrix_in_time,
    dict_stat_to_compute=dict_stat_to_compute,
))})

# ─────────────────────────────────────────────
# Run simulation
# ─────────────────────────────────────────────

uqsim.setup()

if uqsim.is_master():
    uqsim.save_simulationNodes(fileName="nodes")
    number_full_model_evaluations = uqsim.get_simulation_parameters_shape()[0]

if uqsim.is_master():
    argsFileName = os.path.abspath(os.path.join(uqsim.args.outputResultDir, utility.ARGS_FILE))
    with open(argsFileName, 'wb') as handle:
        pickle.dump(uqsim.args, handle, protocol=pickle.HIGHEST_PROTOCOL)

if uqsim.is_master():
    fileName = pathlib.Path(uqsim.args.outputResultDir) / utility.CONFIGURATION_OBJECT_FILE
    with open(fileName, 'wb') as f:
        dill.dump(uqsim.configuration_object, f)

print("\n---- start Battery MC simulation ----")
start_time_sim = time.time()
uqsim.simulate()
end_time_sim = time.time()
print(f"---- simulation done in {end_time_sim - start_time_sim:.1f}s ----\n")

if hasattr(uqsim.simulation, 'parameters') and uqsim.simulation.parameters is not None:
    df = pd.DataFrame({'parameters': [row for row in uqsim.simulation.parameters]})
    df.to_pickle(os.path.abspath(os.path.join(uqsim.args.outputResultDir, utility.DF_UQSIM_SIMULATION_PARAMETERS_FILE)), compression="gzip")

if hasattr(uqsim.simulation, 'nodes') and uqsim.simulation.nodes is not None:
    df = pd.DataFrame({'nodes': [row for row in uqsim.simulation.nodes]})
    df.to_pickle(os.path.abspath(os.path.join(uqsim.args.outputResultDir, utility.DF_UQSIM_SIMULATION_NODES_FILE)), compression="gzip")

if uqsim.is_master():
    fileName = pathlib.Path(uqsim.args.outputResultDir) / utility.CONFIGURATION_OBJECT_FILE
    with open(fileName, 'wb') as f:
        dill.dump(uqsim.configuration_object, f)

# ─────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────

print("---- computing statistics ----")
start_time_stats = time.time()
uqsim.prepare_statistics()
uqsim.calc_statistics()
end_time_stats = time.time()
print(f"---- statistics done in {end_time_stats - start_time_stats:.1f}s ----\n")

uqsim.save_statistics()

end_time = time.time()
total_time = end_time - start_time

uqsim.print_statistics()

if uqsim.is_master():
    time_infoFileName = os.path.abspath(os.path.join(uqsim.args.outputResultDir, utility.TIME_INFO_FILE))
    with open(time_infoFileName, 'w') as fp:
        fp.write(f'number_full_model_runs: {number_full_model_evaluations}\n')
        fp.write(f'time_model_simulations: {end_time_sim - start_time_sim}\n')
        fp.write(f'time_computing_statistics: {end_time_stats - start_time_stats}\n')
        fp.write(f'total_time: {total_time}\n')

print(f"\nResults written to: {OUTPUT_RESULT_DIR}")

uqsim.tear_down()
