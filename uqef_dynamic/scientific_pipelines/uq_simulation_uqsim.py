"""
Usage of the UQEF and UQEF-Dynamic with Hydrology models, more generally, models that produce time-dependent output.
@authors: Ivana Jovanovic Buha and Florian Kuenzner
"""
import os
import subprocess
import sys
import pickle
import dill
from distutils.util import strtobool
import time

# additionally added for the debugging of the nodes
import pandas as pd
import pathlib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

from uqef_dynamic.utils import utility

# Optional LARSIM imports
try:
    from uqef_dynamic.models.larsim import LarsimModelUQ
    from uqef_dynamic.models.larsim import LarsimStatistics
    LARSIM_AVAILABLE = True
except ImportError:
    LARSIM_AVAILABLE = False
    LarsimModelUQ = None
    LarsimStatistics = None

from uqef_dynamic.models.linearDampedOscillator import LinearDampedOscillatorModel
from uqef_dynamic.models.linearDampedOscillator import LinearDampedOscillatorStatistics

from uqef_dynamic.models.ishigami import IshigamiModel
from uqef_dynamic.models.ishigami import IshigamiStatistics

from uqef_dynamic.models.productFunction import ProductFunctionModel
from uqef_dynamic.models.productFunction import ProductFunctionStatistics

from uqef_dynamic.models.hbv_sask import HBVSASKModelUQ
from uqef_dynamic.models.hbv_sask import HBVSASKStatistics

from uqef_dynamic.models.simpleOscilator.simple_oscillator_model import simpleOscillatorUQ
from uqef_dynamic.models.simpleOscilator.simple_oscillator_statistics import simpleOscillatorStatistics

# Optional pybamm imports
try:
    from uqef_dynamic.models.pybamm import pybammModelUQ as pybammmodel
    from uqef_dynamic.models.pybamm import pybammStatistics
    PYBAMM_AVAILABLE = True
except ImportError:
    PYBAMM_AVAILABLE = False
    pybammmodel = None
    pybammStatistics = None

#####################################
# Configuration Management System
#####################################

# Import the new configuration system
from uqef_dynamic.config import ConfigurationFactory, ExtendedUQSimArgumentParser, ExtendedUQSim

#####################################

# instantiate Extended UQsim
uqsim = ExtendedUQSim()

# Create extended parser for configuration processing
extended_parser = ExtendedUQSimArgumentParser(uqsim)

local_debugging = False
if local_debugging:
    # Use the new configuration system
    try:
        # Option: Custom configuration
        config = ConfigurationFactory.create_configuration(
            model_type="hbvsask",
            uq_method="mc",
            mc_numevaluations=5000,
            sampling_rule="sobol",
            mpi=True, 
            sampleFromStandardDist=True, 
            parallel_statistics=True, 
            compute_Sobol_m=True,
            save_all_simulations=True,
            config_file='/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEF-Dynamic/data/configurations/configuration_hbv_10D_single_qoi_autoregressive.json',
            outputResultDir=os.path.abspath(os.path.join("/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/hbvsask_runs", 'mc_10d_5000_sobol_autoregressive_09_qcms_oldman_2006_2007')),
            # run_type="custom_run",
            # custom_name='mc_10d_5000_sobol_autoregressive_09_qcms_oldman_2006_2007',
        )
        
        # Apply configuration to uqsim
        config.apply_to_uqsim(uqsim)
        
        print(f"Applied configuration: {config}")
        print(f"Model: {config.model}, UQ Method: {config.uq_method}")
        print(f"Output Directory: {config.outputResultDir}")
        
    except Exception as e:
        print(f"Configuration error: {e}")
        raise

    uqsim.setup_configuration_object()
else:
    # Cluster execution mode: Use configuration from command-line arguments
    try:
        # Parse extended arguments from command line
        extended_args = extended_parser.parse_extended_args()
        
        # Create configuration from UQEF's parsed command-line arguments
        config = ConfigurationFactory.from_uqsim_args(uqsim.args)
        
        # Apply extended arguments as configuration overrides
        if extended_args:
            print(f"Applying extended configuration arguments: {list(extended_args.keys())}")
            ConfigurationFactory.apply_configuration_overrides(config, **extended_args)
        
        # Apply configuration to uqsim
        config.apply_to_uqsim(uqsim)
        
        print(f"Applied configuration from command-line args: {config}")
        print(f"Model: {config.model}, UQ Method: {config.uq_method}")
        print(f"Output Directory: {config.outputResultDir}")
        
        # Print extended configuration info if any extended args were used
        if extended_args:
            print("Extended analysis features enabled:")
            for key, value in extended_args.items():
                if value not in [False, None, {}]:  # Only show non-default values
                    print(f"  - {key}: {value}")
        
    except Exception as e:
        print(f"Configuration error when parsing command-line args: {e}")
        print("Falling back to standard UQEF configuration handling")
        raise

    uqsim.setup_configuration_object()

start_time = time.time()

#####################################
# Configuration-driven variables (previously hardcoded)
# These values are now taken from the configuration system
# If you need to override these values, use ConfigurationFactory.apply_configuration_overrides()

# Get configuration values or use defaults if not available
dict_stat_to_compute = getattr(config, 'dict_stat_to_compute', {
    "Var": True, "StdDev": True, "P10": True, "P90": True,
    "E_minus_std": False, "E_plus_std": False,
    "Skew": False, "Kurt": False, 
    "Sobol_m": True, "Sobol_m2": False, "Sobol_t": True
})

dict_what_to_plot = getattr(config, 'dict_what_to_plot', {
    "E_minus_std": False, "E_plus_std": False, "E_minus_2std": True, "E_plus_2std": True, 
    "P10": True, "P90": True,
    "StdDev": True, "Skew": False, "Kurt": False, 
    "Sobol_m": True, "Sobol_m2": False, "Sobol_t": True,
    "generalized_sobol_total_index": True, "generalized_sobol_main_index": True,
})

# for the ensemble runs we want to have different defaults
dict_stat_to_compute = getattr(config, 'dict_stat_to_compute', {
    "Var": True, "StdDev": True, "P10": True, "P90": True,
    "E_minus_std": False, "E_plus_std": False,
    "Skew": False, "Kurt": False, 
    "Sobol_m": False, "Sobol_m2": False, "Sobol_t": False
})

dict_what_to_plot = getattr(config, 'dict_what_to_plot', {
    "E_minus_std": False, "E_plus_std": False, "E_minus_2std": True, "E_plus_2std": True, 
    "P10": True, "P90": True,
    "StdDev": True, "Skew": False, "Kurt": False, 
    "Sobol_m": False, "Sobol_m2": False, "Sobol_t": False,
    "generalized_sobol_total_index": False, "generalized_sobol_main_index": False,
})

# Advanced analysis options from configuration
compute_sobol_indices_with_samples = getattr(config, 'compute_sobol_indices_with_samples', False)
save_gpce_surrogate = getattr(config, 'save_gpce_surrogate', True)
compute_other_stat_besides_pce_surrogate = getattr(config, 'compute_other_stat_besides_pce_surrogate', True)

# KL expansion settings from configuration
compute_kl_expansion_of_qoi = getattr(config, 'compute_kl_expansion_of_qoi', False)
kl_expansion_order = getattr(config, 'kl_expansion_order', 10)
compute_timewise_gpce_next_to_kl_expansion = getattr(config, 'compute_timewise_gpce_next_to_kl_expansion', False)

# Generalized Sobol indices settings from configuration
compute_generalized_sobol_indices = getattr(config, 'compute_generalized_sobol_indices', False)
compute_generalized_sobol_indices_over_time = getattr(config, 'compute_generalized_sobol_indices_over_time', False)

# Covariance matrix settings from configuration
compute_covariance_matrix_in_time = getattr(config, 'compute_covariance_matrix_in_time', False)

# Conditional analysis settings from configuration
allow_conditioning_results_based_on_metric = getattr(config, 'allow_conditioning_results_based_on_metric', False)
condition_results_based_on_metric = getattr(config, 'condition_results_based_on_metric', 'NSE')
condition_results_based_on_metric_value = getattr(config, 'condition_results_based_on_metric_value', 0.2)
condition_results_based_on_metric_sign = getattr(config, 'condition_results_based_on_metric_sign', "greater_or_equal")

# Update utility defaults for backward compatibility
utility.DEFAULT_DICT_STAT_TO_COMPUTE = dict_stat_to_compute
utility.DEFAULT_DICT_WHAT_TO_PLOT = dict_what_to_plot

#####################################
# additional path settings:
#####################################

if uqsim.is_master() and not uqsim.is_restored():
    if not os.path.isdir(uqsim.args.outputResultDir):
        subprocess.run(["mkdir", "-p", uqsim.args.outputResultDir])
        # os.makedirs(str(uqsim.args.outputResultDir), exist_ok=True)
    print("outputResultDir: {}".format(uqsim.args.outputResultDir))

# Set the working folder where all the model runs related output and files will be written
try:
    uqsim.args.workingDir = os.path.abspath(os.path.join(uqsim.args.outputResultDir,
                                                         uqsim.configuration_object["model_paths"]["workingDir"]))
except KeyError:
    # uqsim.args.workingDir = os.path.abspath(os.path.join(uqsim.args.outputResultDir, "model_runs"))
    uqsim.args.workingDir = str(uqsim.args.outputResultDir)

try:
    uqsim.configuration_object["model_paths"]["workingDir"] = uqsim.args.workingDir
except KeyError:
    uqsim.configuration_object["model_paths"] = {}
    uqsim.configuration_object["model_paths"]["workingDir"] = uqsim.args.workingDir

if uqsim.is_master() and not uqsim.is_restored():
    if not os.path.isdir(uqsim.configuration_object["model_paths"]["workingDir"]):
        subprocess.run(["mkdir", uqsim.configuration_object["model_paths"]["workingDir"]])
        # os.makedirs(uqsim.configuration_object["model_paths"]["workingDir"], exist_ok=True)

#####################################
# Register models
#####################################

if LARSIM_AVAILABLE:
    uqsim.models.update({"larsim"         : (lambda: LarsimModelUQ.LarsimModelUQ(
        configurationObject=uqsim.configuration_object,
        inputModelDir=uqsim.args.inputModelDir,
        workingDir=uqsim.args.workingDir,
        sourceDir=uqsim.args.sourceDir,
        disable_statistics=uqsim.args.disable_statistics,
        uq_method=uqsim.args.uq_method))})
uqsim.models.update({"oscillator"     : (lambda: LinearDampedOscillatorModel.LinearDampedOscillatorModel(
    configurationObject=uqsim.configuration_object,
    workingDir=uqsim.args.workingDir,
    atol=1e-10,
    rtol=1e-10,
    ))})
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
uqsim.models.update({"simple_oscillator"         : (lambda: simpleOscillatorUQ(
    configurationObject=uqsim.configuration_object,
    workingDir=uqsim.args.workingDir,
))})
if PYBAMM_AVAILABLE:
    uqsim.models.update({"battery"         : (lambda: pybammmodel.pybammModelUQ(
    configurationObject=uqsim.configuration_object,
    inputModelDir=uqsim.args.inputModelDir,
    workingDir=uqsim.args.workingDir,
    ))})

#####################################
# Register statistics
#####################################

if LARSIM_AVAILABLE:
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
    chunksize=uqsim.args.chunksize,
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
    index_column_name = utility.INDEX_COLUMN_NAME,
    allow_conditioning_results_based_on_metric=allow_conditioning_results_based_on_metric,
    condition_results_based_on_metric = condition_results_based_on_metric,
    condition_results_based_on_metric_value = condition_results_based_on_metric_value,
    condition_results_based_on_metric_sign = condition_results_based_on_metric_sign,
    compute_timewise_gpce_next_to_kl_expansion=compute_timewise_gpce_next_to_kl_expansion,
    kl_expansion_order = kl_expansion_order,
    compute_generalized_sobol_indices = compute_generalized_sobol_indices,
    compute_generalized_sobol_indices_over_time = compute_generalized_sobol_indices_over_time,
    compute_covariance_matrix_in_time = compute_covariance_matrix_in_time,
    dict_stat_to_compute=dict_stat_to_compute,
    # always_process_original_model_output=True,
))})
uqsim.statistics.update({"simple_oscillator"         : (lambda: simpleOscillatorStatistics(
    configurationObject=uqsim.configuration_object,  # uqsim.args.config_file,
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
uqsim.statistics.update({"oscillator"     : (lambda: LinearDampedOscillatorStatistics.LinearDampedOscillatorStatistics(
    configurationObject=uqsim.configuration_object,
    workingDir=uqsim.args.workingDir,
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
if PYBAMM_AVAILABLE:
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
    number_full_model_evaluations = uqsim.get_simulation_parameters_shape()[0]
    #number_full_model_evaluations = len(uqsim.simulationNodes.nodes.T)

# save the dictionary with the arguments - once before the simulation
if uqsim.is_master():
    argsFileName = os.path.abspath(os.path.join(uqsim.args.outputResultDir, utility.ARGS_FILE))
    with open(argsFileName, 'wb') as handle:
        pickle.dump(uqsim.args, handle, protocol=pickle.HIGHEST_PROTOCOL)

# save initially configurationObject if program breaks during simulation
if uqsim.is_master():
    fileName = pathlib.Path(uqsim.args.outputResultDir) / utility.CONFIGURATION_OBJECT_FILE
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
    df.to_pickle(os.path.abspath(os.path.join(uqsim.args.outputResultDir, utility.DF_UQSIM_SIMULATION_PARAMETERS_FILE)), compression="gzip")

if hasattr(uqsim.simulation, 'nodes') and uqsim.simulation.nodes is not None:
    df = pd.DataFrame({'nodes': [row for row in uqsim.simulation.nodes]})
    df.to_pickle(os.path.abspath(os.path.join(uqsim.args.outputResultDir, utility.DF_UQSIM_SIMULATION_NODES_FILE)), compression="gzip")

if hasattr(uqsim.simulation, 'weights') and uqsim.simulation.weights is not None:
    df = pd.DataFrame({'weights': [row for row in uqsim.simulation.weights]})
    df.to_pickle(os.path.abspath(os.path.join(uqsim.args.outputResultDir, utility.DF_UQSIM_SIMULATION_WEIGHTS_FILE)), compression="gzip")

#####################################
# re-save uqsim.configurationObject
#####################################

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
# if uqsim.is_master():
#     uqsim.statistic.compute_covariance_matrix_in_time()
end_time_stats = time.time()
print(f"---- statistics done in {end_time_stats - start_time_stats:.1f}s ----\n")

uqsim.save_statistics()

end_time = time.time()
total_time = end_time - start_time

uqsim.print_statistics()

# save the dictionary with the arguments once again
if uqsim.is_master():
    time_infoFileName = os.path.abspath(os.path.join(uqsim.args.outputResultDir, utility.TIME_INFO_FILE))
    with open(time_infoFileName, 'w') as fp:
        fp.write(f'number_full_model_runs: {number_full_model_evaluations}\n')
        fp.write(f'time_model_simulations: {time_model_simulations}\n')
        # fp.write(f'time_producing_gpce: {time_producing_gpce}\n')
        fp.write(f'time_computing_statistics: {end_time_stats - start_time_stats}\n')
        fp.write(f'total_time: {total_time}\n')
        if hasattr(uqsim.statistic, 'mean_model_eval_time') and uqsim.statistic.mean_model_eval_time is not None:
            fp.write(f'mean_model_eval_time: {uqsim.statistic.mean_model_eval_time}\n')
            fp.write(f'total_model_eval_time: {uqsim.statistic.total_model_eval_time}\n')

print(f"\nResults written to: {uqsim.args.outputResultDir}")
#####################################
# tear down
#####################################

uqsim.tear_down()
