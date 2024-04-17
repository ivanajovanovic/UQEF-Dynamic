"""
Usage of the UQEF with a (mainly) Larsim model.
@author: Florian Kuenzner and Ivana Jovanovic
"""
import os
import subprocess
import sys
import pickle
import dill
from distutils.util import strtobool

import uqef

from uqef_dynamic.models.larsim import LarsimModelUQ
from uqef_dynamic.models.larsim import LarsimStatistics

from uqef_dynamic.models.linearDampedOscillator import LinearDampedOscillatorModel
from uqef_dynamic.models.linearDampedOscillator import LinearDampedOscillatorStatistics

from uqef_dynamic.models.ishigami import IshigamiModel
from uqef_dynamic.models.ishigami import IshigamiStatistics

from uqef_dynamic.models.productFunction import ProductFunctionModel
from uqef_dynamic.models.productFunction import ProductFunctionStatistics

from LarsimUtilityFunctions import larsimModel, larsimPaths

# additionally added for the debugging of the nodes
import chaospy as cp
import os.path as osp
import pandas as pd
import pathlib
from LarsimUtilityFunctions import larsimConfigurationSettings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

sys.path.insert(0, os.getcwd())

# instantiate UQsim
uqsim = uqef.UQsim()

#####################################
# change args locally for testing and debugging
#####################################

local_debugging = True
if local_debugging:
    local_debugging_nodes = True
    exit_after_debugging_nodes = True
    save_solver_results = True

    uqsim.args.model = "larsim"

    uqsim.args.uncertain = "all"
    uqsim.args.chunksize = 1

    uqsim.args.uq_method = "sc"  # "sc" | "saltelli" | "mc" | "ensemble"
    uqsim.args.mc_numevaluations = 50
    uqsim.args.sampling_rule = "halton"  # | "sobol" | "latin_hypercube" | "halton"  | "hammersley"
    uqsim.args.sc_q_order = 8 #7  # 10 #7
    uqsim.args.sc_p_order = 7 #6  # 6 #5
    uqsim.args.sc_quadrature_rule = "patterson"  # "clenshaw_curtis", "patterson", "G"

    uqsim.args.sc_poly_rule = "three_terms_recurrence"  # "gram_schmidt" | "three_terms_recurrence" | "cholesky"
    uqsim.args.sc_poly_normed = False
    uqsim.args.sc_sparse_quadrature = True  # False
    uqsim.args.regression = False

    uqsim.args.inputModelDir = pathlib.Path('/work/ga45met/Larsim-data')  # paths.larsim_data_path
    uqsim.args.sourceDir = pathlib.Path("/work/ga45met")  # paths.sourceDir
    scratch_dir = uqsim.args.sourceDir
    uqsim.args.outputResultDir = uqsim.args.sourceDir / "larsim_runs" / 'larsim_run_sc_sg_patterson_p7_q8'
    uqsim.args.outputResultDir = str(uqsim.args.outputResultDir)  # for now reast of the code expects path in the string
    uqsim.args.outputModelDir = uqsim.args.outputResultDir

    uqsim.args.config_file = pathlib.Path(
        '/work/ga45met/mnt/linux_cluster_2/UQEF-Dynamic/configurations_Larsim/configurations_larsim_high_flow.json')

    uqsim.args.sampleFromStandardDist = True
    uqsim.args.transformToStandardDist = True

    uqsim.args.mpi = True
    uqsim.args.mpi_method = "MpiPoolSolver"

    uqsim.args.uqsim_store_to_file = False

    uqsim.args.disable_statistics = True
    uqsim.args.parallel_statistics = True  # False
    uqsim.args.compute_Sobol_t = True
    uqsim.args.compute_Sobol_m = True

    uqsim.setup_configuration_object()

#####################################
# additional path settings:
#####################################

if uqsim.is_master() and not uqsim.is_restored():
    if not os.path.isdir(uqsim.args.outputResultDir): subprocess.run(["mkdir", "-p", uqsim.args.outputResultDir])
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
# one time initial model setup
#####################################
# put here if there is something specifically related to the model that should be done only once
# if uqsim.is_master() and not uqsim.is_restored():
#     def initialModelSetUp():
#         models = {
#             "larsim": (lambda: larsimModel.LarsimModelSetUp(configurationObject=uqsim.configurationObject,
#                                                             inputModelDir=uqsim.args.inputModelDir,
#                                                             workingDir=uqsim.args.workingDir,
#                                                             sourceDir=uqsim.args.sourceDir))
#            ,"oscillator"     : (lambda: LinearDampedOscillatorModel.LinearDampedOscillatorModelSetUp(uqsim.configurationObject))
#            ,"ishigami"       : (lambda: IshigamiModel.IshigamiModelSetUp(uqsim.configurationObject))
#            ,"productFunction": (lambda: ProductFunctionModel.ProductFunctionModelSetUp(uqsim.configurationObject))
#         }
#         models[uqsim.args.model]()
#     initialModelSetUp()

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
uqsim.models.update({"ishigami"       : (lambda: IshigamiModel.IshigamiModel(uqsim.configuration_object))})
uqsim.models.update({"productFunction": (lambda: ProductFunctionModel.ProductFunctionModel(uqsim.configuration_object))})

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
    save_samples=True,
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

#####################################
# setup
#####################################

uqsim.setup()

# save simulation nodes
simulationNodes_save_file = "nodes"
uqsim.save_simulationNodes(fileName=simulationNodes_save_file)

# print the dictionary with the arguments
if uqsim.is_master():
    uqsim_args_temp_dict = vars(uqsim.args)
    print(f"UQSIM.ARGS")
    for key, value in uqsim_args_temp_dict.items():
        print(f"{key}: {value}")

# save the dictionary with the arguments - once before the simulation
if uqsim.is_master():
    argsFileName = os.path.abspath(os.path.join(uqsim.args.outputResultDir, "uqsim_args.pkl"))
    with open(argsFileName, 'wb') as handle:
        pickle.dump(uqsim.args, handle, protocol=pickle.HIGHEST_PROTOCOL)

#####################################
# check-up (for Larsim model)
#####################################

if uqsim.is_master():
    if local_debugging_nodes:
        # # experiment by Ivana - remove
        # print(uqsim.configurationObject["tuples_parameters_info"])

        # Do the initial set-up
        larsimConfigurationsObject = larsimModel.LarsimConfigurations(configurationObject=uqsim.configuration_object)
        _larsimModelSetUpObject = larsimModel.LarsimModelSetUp(
            configurationObject=larsimConfigurationsObject,
            inputModelDir=uqsim.args.inputModelDir,
            workingDir=uqsim.args.workingDir,
            sourceDir=uqsim.args.sourceDir)
        _larsimModelSetUpObject.copy_master_folder()
        _larsimModelSetUpObject.configure_master_folder()
        # larsimPathsObject = larsimPaths.LarsimPathsClass(
        #     root_data_dir==uqsim.args.inputModelDir, **larsimConfigurationsObject.dict_params_for_defining_paths)
        if larsimConfigurationsObject.boolean_make_local_copy_of_master_dir:
            local_master_dir = pathlib.Path(uqsim.args.workingDir) / 'master_configuration'
        else:
            local_master_dir = pathlib.Path(uqsim.args.workingDir)
        tape35_path = local_master_dir / "tape35"
        lanu_path = local_master_dir / "lanu.par"

        # play with different sampling rules...

        # plot position of the nodes
        local_nodes = uqsim.simulationNodes.nodes.T
        print(f"Shape of simulationNodes.nodes.T is: {local_nodes.shape}")
        local_parameters = uqsim.simulationNodes.parameters.T
        print(f"Shape of simulationNodes.parameters.T is: {local_parameters.shape}")
        local_simulation_parameters = uqsim.simulation.parameters
        print(f"Shape of simulation.parameters is: {local_simulation_parameters.shape}")

        if uqsim.simulationNodes.distNodes.size:
            local_distNodes = uqsim.simulationNodes.distNodes.T
            print(f"Shape of simulationNodes.distNodes.T is: {local_distNodes.shape}")
        if uqsim.simulationNodes.weights.size:
            local_weights = uqsim.simulationNodes.weights
            print(f"Shape of simulationNodes.weights is: {local_weights.shape}")

        # Problem with Saltelli & MC is that uqsim.simulationNodes.parameters
        # are different from uqsim.simulation.parameters
        # plot position of the final parameters
        if "tuples_parameters_info" not in uqsim.configuration_object:
            print(f"uqsim.configurationObject was not updated together with model.configurationObject")
            larsimConfigurationSettings.update_configurationObject_with_parameters_info(uqsim.configuration_object)
        list_of_parameters_dict = []
        for parameter in local_parameters:  # local_simulation_parameters
            ordered_dict_of_all_params, unsupported_values_of_parameters_flag = \
                larsimConfigurationSettings.params_configurations(
                    parameters=parameter,
                    tape35_path=tape35_path,
                    lanu_path=lanu_path,
                    configurationObject=uqsim.configuration_object,
                    process_id=0,
                    reference_value_from_TGB=3085,
                    take_direct_value=False,
                    write_new_values_to_tape35=False,
                    write_new_values_to_lanu=False,
                    break_if_faulty_values_of_parameters=False)
            if unsupported_values_of_parameters_flag:
                ordered_dict_of_all_params["correct_parameters"] = "failed"
            else:
                ordered_dict_of_all_params["correct_parameters"] = "correct"
            list_of_parameters_dict.append(ordered_dict_of_all_params)
        df_with_final_parameters = pd.DataFrame(list_of_parameters_dict)
        temp_file_path = pathlib.Path(uqsim.args.outputResultDir) / "parameters.pkl"
        df_with_final_parameters.to_pickle(temp_file_path, compression="gzip")

        # Plot polynomials
        # polynomial_expansion = cp.orth_ttr(order, dist)
        # local_dist = uqsim.simulationNodes.joinedDists
        # local_standard_dist = None
        # polynomial_standard_expansion = None
        # if uqsim.simulationNodes.joinedStandardDists:
        #     local_standard_dist = uqsim.simulationNodes.joinedStandardDists
        # if uqsim.args.uq_method == "sc":
        #     polynomial_expansion = cp.generate_expansion(
        #         uqsim.args.sc_q_order, local_dist,
        #         rule=uqsim.args.sc_poly_rule, normed=uqsim.args.sc_poly_normed)
        #     if uqsim.simulationNodes.joinedStandardDists:
        #         polynomial_standard_expansion = cp.generate_expansion(
        #             uqsim.args.sc_q_order, local_standard_dist,
        #             rule=uqsim.args.sc_poly_rule, normed=uqsim.args.sc_poly_normed)

        # # plotting simulation nodes
        # if uqsim.simulationNodes.dists:
        #     uqsim.simulationNodes.plotDists(fileName=uqsim.args.outputResultDir + "/dists", fileNameIdentIsFullName=True)
        #     uqsim.simulationNodes.plotDistsSetup(fileName=uqsim.args.outputResultDir + "/distsSetup.pdf",
        #                                          numCollocationPointsPerDim=10)
        # # uqsim.plot_nodes()

        if exit_after_debugging_nodes:
            # return polynomial_expansion, polynomial_standard_expansion
            sys.exit()

#####################################
# start the simulation
#####################################

uqsim.simulate()

#####################################
# check-up (for Larsim model)
#####################################

if uqsim.is_master():
    if save_solver_results and uqsim.args.disable_statistics:
        # save raw results, i.e., solver results
        # assert uqsim.solver is uqsim.simulation.solver
        # assert id(uqsim.solver) == id(uqsim.simulation.solver)
        processed_sample_results = LarsimStatistics.LarsimSamples(uqsim.solver.results,
                                                                  configurationObject=uqsim.configuration_object)
        processed_sample_results.save_samples_to_file(uqsim.args.outputResultDir)
        processed_sample_results.save_index_parameter_values(uqsim.args.outputResultDir)
        processed_sample_results.save_index_parameter_gof_values(uqsim.args.outputResultDir)
        if strtobool(uqsim.configuration_object["model_run_settings"]["compute_gradients"]) :
            processed_sample_results.save_dict_of_approx_matrix_c(uqsim.args.outputResultDir)
            processed_sample_results.save_dict_of_matrix_c_eigen_decomposition(uqsim.args.outputResultDir)

#####################################
# save uqsim.configurationObject
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
uqsim.plot_statistics(display=False, plot_measured_timeseries=True, plot_unaltered_timeseries=True)

# uqsim.args.uqsim_file = os.path.abspath(os.path.join(uqsim.args.outputResultDir, "uqsim.saved"))
# uqsim.store_to_file()

#####################################
# tear down
#####################################

uqsim.tear_down()
