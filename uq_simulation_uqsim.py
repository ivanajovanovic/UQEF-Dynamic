"""
Usage of the UQEF with a (mainly) Hydrological models.
@author: Florian Kuenzner and Ivana Jovanovic
"""
import os
import subprocess
import sys
import pickle
import dill
from distutils.util import strtobool

import uqef

# additionally added for the debugging of the nodes
import pandas as pd
import pathlib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

# sys.path.insert(0, os.getcwd())
sys.path.insert(0, '/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEFPP')

from larsim import LarsimModelUQ
from larsim import LarsimStatistics

from linearDampedOscillator import LinearDampedOscillatorModel
from linearDampedOscillator import LinearDampedOscillatorStatistics

from ishigami import IshigamiModel
from ishigami import IshigamiStatistics

from productFunction import ProductFunctionModel
from productFunction import ProductFunctionStatistics

from hbv_sask import HBVSASKModelUQ
from hbv_sask import HBVSASKStatistics

# instantiate UQsim
uqsim = uqef.UQsim()

#####################################
# change args locally for testing and debugging
#####################################

local_debugging = True
if local_debugging:
    save_solver_results = False

    uqsim.args.model = "hbvsask"  # "larsim" "hbvsask"

    uqsim.args.uncertain = "all"
    uqsim.args.chunksize = 1

    uqsim.args.uq_method = "mc"  # "sc" | "saltelli" | "mc" | "ensemble"
    uqsim.args.mc_numevaluations = 2
    uqsim.args.sampling_rule = "random"  # | "sobol" | "latin_hypercube" | "halton"  | "hammersley"
    uqsim.args.sc_q_order = 6  # 7 #10 3
    uqsim.args.sc_p_order = 3  # 4, 5, 6, 8
    uqsim.args.sc_quadrature_rule = "clenshaw_curtis"  # "p" "genz_keister_24" "leja"

    uqsim.args.read_nodes_from_file = False
    l = 6  # 10
    path_to_file = pathlib.Path("/dss/dsshome1/lxc0C/ga45met2/Repositories/sparse_grid_nodes_weights")
    uqsim.args.parameters_file = path_to_file / f"KPU_d6_l{l}.asc" # f"KPU_d3_l{l}.asc"
    uqsim.args.parameters_setup_file = pathlib.Path("/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEFPP/configurations/KPU_HBV_d6.json")
    # uqsim.args.parameters_setup_file = pathlib.Path("/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEFPP/configurations_Larsim/KPU_Larsim_d5.json")

    uqsim.args.sc_poly_rule = "three_terms_recurrence"  # "gram_schmidt" | "three_terms_recurrence" | "cholesky"
    uqsim.args.sc_poly_normed = False  # True
    uqsim.args.sc_sparse_quadrature = True  # False
    uqsim.args.regression = False

    # uqsim.args.inputModelDir = os.path.abspath(os.path.join('/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2','Larsim-data'))
    uqsim.args.inputModelDir = pathlib.Path("/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/HBV-SASK-data")
    # uqsim.args.sourceDir = os.path.abspath(os.path.join('/dss/dsshome1/lxc0C/ga45met2', 'Repositories', 'UQEFPP'))
    uqsim.args.sourceDir = pathlib.Path("/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/HBV-SASK-data")
    # uqsim.args.outputResultDir = os.path.abspath(os.path.join("/gpfs/scratch/pr63so/ga45met2", "Larsim_runs", 'larsim_run_ensemble_2013_all_tgb'))
    # uqsim.args.outputResultDir = os.path.abspath(os.path.join("/gpfs/scratch/pr63so/ga45met2", "Larsim_runs", 'larsim_run_lai_may_cc_q_6_p_4_stat_trial'))
    uqsim.args.outputResultDir = os.path.abspath(os.path.join("/gpfs/scratch/pr63so/ga45met2", "hbvsask_runs", 'gradientr_trials'))
    # uqsim.args.outputResultDir = os.path.abspath(os.path.join("/gpfs/scratch/pr63so/ga45met2", "Larsim_runs", 'larsim_run_sc_kpu_l_6_d_5_p_3_2013'))
    uqsim.args.outputModelDir = uqsim.args.outputResultDir
    # uqsim.args.config_file = '/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEFPP/configurations_Larsim/configurations_larsim_boundery_values.json'
    # uqsim.args.config_file = '/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEFPP/configurations_Larsim/configurations_larsim_4_may.json'
    uqsim.args.config_file = '/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEFPP/configurations/configuration_hbv_6D.json'
    # uqsim.args.config_file = '/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEFPP/configurations_Larsim/configurations_larsim_high_flow.json'

    uqsim.args.sampleFromStandardDist = True  # False
    # uqsim.args.transformToStandardDist = True

    uqsim.args.mpi = True
    uqsim.args.mpi_method = "MpiPoolSolver"  # "LinearSolver"

    uqsim.args.uqsim_store_to_file = False

    uqsim.args.disable_statistics = True
    uqsim.args.parallel_statistics = True  # False
    uqsim.args.compute_Sobol_t = True  # True False
    uqsim.args.compute_Sobol_m = True  # True False

    uqsim.args.num_cores = 1

    uqsim.setup_configuration_object()

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
    configurationObject=uqsim.configuration_object))})
uqsim.models.update({"productFunction": (lambda: ProductFunctionModel.ProductFunctionModel(uqsim.configuration_object))})
uqsim.models.update({"hbvsask"         : (lambda: HBVSASKModelUQ.HBVSASKModelUQ(
    configurationObject=uqsim.configuration_object,
    inputModelDir=uqsim.args.inputModelDir,
    workingDir=uqsim.args.workingDir,
    disable_statistics=uqsim.args.disable_statistics,
    uq_method=uqsim.args.uq_method,
    writing_results_to_a_file=False,
    plotting=False
))})

#####################################
# register statistics
#####################################

uqsim.statistics.update({"larsim"         : (lambda: LarsimStatistics.LarsimStatistics(
    configurationObject=uqsim.configuration_object,
    workingDir=uqsim.args.workingDir,
    sampleFromStandardDist=uqsim.args.sampleFromStandardDist,
    store_qoi_data_in_stat_dict=False,
    parallel_statistics=uqsim.args.parallel_statistics,
    mpi_chunksize=uqsim.args.mpi_chunksize,
    unordered=False,
    uq_method=uqsim.args.uq_method,
    compute_Sobol_t=uqsim.args.compute_Sobol_t,
    compute_Sobol_m=uqsim.args.compute_Sobol_m
))})
uqsim.statistics.update({"oscillator"     : (lambda: LinearDampedOscillatorStatistics.LinearDampedOscillatorStatistics())})
uqsim.statistics.update({"ishigami"       : (lambda: IshigamiStatistics.IshigamiStatistics(
    configurationObject=uqsim.configuration_object,
    workingDir=uqsim.args.workingDir,
    sampleFromStandardDist=uqsim.args.sampleFromStandardDist,
    uq_method=uqsim.args.uq_method,
    compute_Sobol_t=uqsim.args.compute_Sobol_t,
    compute_Sobol_m=uqsim.args.compute_Sobol_m
))})
uqsim.statistics.update({"productFunction": (lambda: ProductFunctionStatistics.ProductFunctionStatistics(uqsim.configuration_object))})
uqsim.statistics.update({"hbvsask"         : (lambda: HBVSASKStatistics.HBVSASKStatistics(
    configurationObject=uqsim.configuration_object,
    workingDir=uqsim.args.outputResultDir,  # .args.workingDir,
    sampleFromStandardDist=uqsim.args.sampleFromStandardDist,
    store_qoi_data_in_stat_dict=False,
    parallel_statistics=uqsim.args.parallel_statistics,
    mpi_chunksize=uqsim.args.mpi_chunksize,
    unordered=False,
    uq_method=uqsim.args.uq_method,
    compute_Sobol_t=uqsim.args.compute_Sobol_t,
    compute_Sobol_m=uqsim.args.compute_Sobol_m,
    save_samples=True,
    qoi_column="Q_cms",
    inputModelDir=uqsim.args.inputModelDir
))})

#####################################
# setup
#####################################

uqsim.setup()

# save simulation nodes
simulationNodes_save_file = "nodes"
uqsim.save_simulationNodes(fileName=simulationNodes_save_file)

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

uqsim.simulate()

#####################################
# check-up (for Larsim model)
#####################################

if uqsim.is_master():
    # TODO This will be a lot of duplicated savings in case
    #  always_save_original_model_runs=True and run_and_save_simulations=True
    if uqsim.args.model == "larsim" and uqsim.args.disable_statistics:
        processed_sample_results = LarsimStatistics.LarsimSamples(uqsim.solver.results,
                                                                  configurationObject=uqsim.configuration_object)
        processed_sample_results.save_samples_to_file(uqsim.args.outputResultDir)
        processed_sample_results.save_index_parameter_values(uqsim.args.outputResultDir)
        processed_sample_results.save_index_parameter_gof_values(uqsim.args.outputResultDir)
        if strtobool(uqsim.configuration_object["model_run_settings"]["compute_gradients"]):
            processed_sample_results.save_dict_of_approx_matrix_c(uqsim.args.outputResultDir)
            processed_sample_results.save_dict_of_matrix_c_eigen_decomposition(uqsim.args.outputResultDir)

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

uqsim.calc_statistics()
uqsim.save_statistics()

if uqsim.args.model == "larsim":
    uqsim.plot_statistics(
        display=False,
        plot_measured_timeseries=strtobool(uqsim.configuration_object["model_settings"]["get_measured_discharge"]),
        plot_unaltered_timeseries=strtobool(uqsim.configuration_object["model_settings"]["run_unaltered_sim"])
    )
elif uqsim.args.model == "hbvsask":
    # TODO This only for now - change the logic
    uqsim.plot_statistics(display=False,
                          plot_measured_timeseries=True,
                          plot_unaltered_timeseries=False,
                          plot_forcing_timeseries=True,
                          time_column_name="TimeStamp",
                          measured_df_column_to_draw="streamflow",
                          measured_df_timestamp_column="index",
                          precipitation_df_column_to_draw="precipitation",
                          precipitation_df_timestamp_column="index",
                          temperature_df_column_to_draw="temperature",
                          temperature_df_timestamp_column="index",
                          )
else:
    uqsim.plot_statistics(display=False)

# uqsim.args.uqsim_file = os.path.abspath(os.path.join(uqsim.args.outputResultDir, "uqsim.saved"))
# #uqsim.store_to_file()

#####################################
# tear down
#####################################

uqsim.tear_down()
