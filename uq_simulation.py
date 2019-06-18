import argparse
import datetime
import json
from mpi4py import MPI
import multiprocessing
import os
import os.path as osp
import pandas as pd
import sys
import subprocess
import time

import chaospy as cp

import uqef

import paths
import LARSIM_configs as config
import LarsimModel
import LarsimStatistics

#####################################
### MPI infos:
#####################################
comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()


#####################################
### parsing args:
#####################################
if rank == 0: print("parsing args...")
parser = argparse.ArgumentParser(description='Uncertainty Quantification simulation.')
parser.add_argument('--smoketest', action='store_true', default=False)

parser.add_argument('-or','--outputResultDir' , default="./saves/")

parser.add_argument('--parallel', action='store_true', default=False)
parser.add_argument('--num_cores', type=int, default=multiprocessing.cpu_count())
parser.add_argument('--mpi', action='store_true')
parser.add_argument('--mpi_method', default="new")  # new (MpiPoolSolver), old (MpiPoolSolverOld)
parser.add_argument('--mpi_combined_parallel', action='store_true', default=False)

parser.add_argument('--model', default="larsim")

parser.add_argument('--chunksize', type=int, default=1)
parser.add_argument('--mpi_chunksize', type=int, default=1)

parser.add_argument('--uncertain', default='all')  # all, uncertain_param_1, uncertain_param_2

parser.add_argument('--uncertain_1_dist', default='normal')  # normal or uniform
parser.add_argument('--uncertain_2_dist', default='normal')  # normal or uniform

parser.add_argument('--uq_method', default="sc")  # sc, mc
parser.add_argument('--regression',action='store_true', default=False)
parser.add_argument('--sparse_quadrature',action='store_true', default=False)
parser.add_argument('--saltelli',action='store_true', default=False) # compute Sobol's Indices using MC and Saltelli method, if saltelli = True then uq_method should be mc and regression = False
parser.add_argument('--mc_numevaluations', type=int, default=27)
parser.add_argument('--sc_q_order', type=int, default=3)  # number of collocation points in each direction (Q)
parser.add_argument('--sc_p_order', type=int, default=2)  # number of terms in PCE (N)

args = parser.parse_args()

#####################################
### parallelisation setup:
#####################################
# mpi
if args.mpi:
    mpi = True
else:
    mpi = False

# cores
num_cores = args.num_cores
if mpi == False or (mpi == True and rank == 0):
    print("set num cores to: {}".format(num_cores))


#####################################
### path settings:
#####################################
# each mpi processor has it's own outputResultDir
if mpi == False or (mpi == True and rank == 0):
    print("path settings...")

if args.outputResultDir:
    rootDir = args.outputResultDir
else:
    rootDir = os.getcwd()

outputResultDir = os.path.abspath(os.path.join(rootDir,datetime.datetime.now().strftime("%Y-%m-%d:%H:%M")))
args.outputResultDir = outputResultDir

if mpi == False or (mpi == True and rank == 0):
    # call(["mkdir " + args.outputResultDir], shell=True)
    if not os.path.isdir(outputResultDir): subprocess.run(["mkdir", outputResultDir])
    print("outputResultDir: {}".format(outputResultDir))

#####################################
### initialise uncertain parameters - simulationNodes:
#####################################
# each mpi processor has it's own model
model = args.model

configuration_object = None
#TODO - Ivana think if you should regarde distributions as values or error distributions
if args.mpi == False or (args.mpi == True and rank == 0):
    with open("configurations.json") as f:
        configuration_object = json.load(f)
        print(configuration_object)

if mpi == True:
    configuration_object = comm.bcast(configuration_object, root=0)
    print("broadcasting configuration object...")

if mpi == False or (mpi == True and rank == 0):
    distributions = []
    for i in configuration_object["Variables"]:
        if i["distribution"] == "normal":
            distributions.append((i["name"], cp.Normal(i["mean"], i["std"])))
        elif i["distribution"] == "uniform":
            distributions.append((i["name"], cp.Uniform(i["uniform_low"], i["uniform_high"])))
    nodeNames = []
    for items in distributions:
        nodeNames.append(items[0])

    simulationNodes = uqef.simulation.Nodes(nodeNames)

    #if args.uncertain == "all":
    for items in distributions:
        simulationNodes.setDist(items[0], items[1])

    print("model: {}".format(args.model))
    print("chunksize: {}".format(args.chunksize))
    print("nodes config: {}".format(args.uncertain))
    print(simulationNodes.printNodesSetup()) # just a setup, before really sampling the nodes

    # generates output folder for plots
    node_setup_name = outputResultDir + "/node_setup.txt"
    with open(node_setup_name, "w") as f:
        f.write(simulationNodes.printNodesSetup())

#####################################
### initialise model
#####################################
# each mpi processor has it's own modelGenerator
def modelGenerator():
    models = {
        "larsim": (lambda: LarsimModel.LarsimModel())
        #"larsim": (lambda: uqef.model.TestModel())
    }
    return models[model]()
    #return models["larsim"]()

#####################################
### initialise solver
#####################################
# each mpi processor has it's own solver
if mpi == True:
    if args.mpi_method == "new":
        solver = uqef.solver.MpiPoolSolver(modelGenerator, mpi_chunksize=args.mpi_chunksize,
                                           combinedParallel=args.mpi_combined_parallel, num_cores=num_cores)
    else:
        solver = uqef.solver.MpiSolverOld(modelGenerator, mpi_chunksize=args.mpi_chunksize,
                                              combinedParallel=args.mpi_combined_parallel, num_cores=num_cores)
elif args.parallel:
    solver = uqef.solver.ParallelSolver(modelGenerator, num_cores)
else:
    solver = uqef.solver.LinearSolver(modelGenerator)

if mpi == False or (mpi == True and rank == 0):
    print("solver-setup: {}".format(solver.getSetup()))

#####################################
### additional paths, time and configuration settings
#####################################

if mpi == False or (mpi == True and rank == 0):
    timeframe = config.datetime_parse(configuration_object) #tuple with EREIGNISBEGINN EREIGNISENDE
    timestep = configuration_object["Timeframe"]["timestep"] #how long one consecutive run should take
    #start_date_sim = timeframe[0]
    #end_date_sim = timeframe[1]
    if timestep == 0:
        pass
    else:
        timeframe[1] = timeframe[0] + datetime.timedelta(days=timestep)
    #EREIGNISBEGINN = timeframe[0]
    #EREIGNISBEGINN_MIN_3 = EREIGNISBEGINN - datetime.timedelta(days=3)
    #EREIGNISENDE = timeframe[1]
    #VORHERSAGEBEGINN = 53
    #VORHERSAGEDAUER = (EREIGNISENDE - EREIGNISBEGINN) * 24 - VORHERSAGEBEGINN

    #####################################
    ### copy configuration files
    #####################################
    # Get all the inputs

    # All the configurations needed for proper execution

    # Base on time settings change tape10_master file
    config.tape10_configurations(timeframe=timeframe, master_tape10_file=paths.master_tape10_file, new_path=paths.master_dir)

    # Filter out whm files
    config.copy_whm_files(timeframe=timeframe, all_whms_path=paths.all_whms_path, new_path=paths.master_dir)

    # Parse big lila files and create small ones
    #config.master_lila_parser_on_time_crete_new(timeframe=timeframe, master_lila_paths=paths.master_lila_paths, configured_lila_paths=paths.lila_configured_paths)
    config.master_lila_parser_on_time_crete_new(timeframe=timeframe, master_lila_paths=paths.master_lila_paths,
                                                new_lila_files = paths.lila_files, new_path = paths.master_dir)

    for one_lila_file in paths.lila_configured_paths:
        if not osp.exists(one_lila_file):
            raise IOError('File does not exist: %s. %s' % (one_lila_file, IOError.strerror))
            #OSError
            #try:
            #    with open(path) as f:
            #        pass
            #except IOError as exc:
            #    raise IOError("%s: %s" % (path, exc.strerror))
            #sys.exit(1)

    #TODO run unaltered simulation
    #####################################
    ### run unaltered simulation
    #####################################

    print("INFO: All the files have been copied to master folder! ")
    start_time = time.time()

# start new simulation
# generate nodes for the first time and save them
#for loop
# inside for loop - change tape10
# inside for loop - call solver.init()
# inside for loop - call simulation.prepareSolver()
# solver.solve()
# inside for loop - inside of the model.run mkdir, copy all the files - tape10 just once
# inside for loop - inside of the model.run delete larsim.ok, tape11, ergebnis.lila and karte
# inside for loop - inside of the model.run copy tape10
# inside for loop - inside of the model.run copy new ergebnis to safe place

#or

#inside model.run - constanlty change tape10
#inside model.run - constanlty delete larsim.ok, tape11, ergebnis.lila and karte
#inside model.run - constanlty run larsim.exe
#inside model.run - constanlty concatinate ergebnis and save to some final ergebnis file

#####################################
### initialise simulation
#####################################
if mpi == False or (mpi == True and rank == 0):
    simulations = {
        "mc": (lambda: uqef.simulation.McSimulation(solver, args.mc_numevaluations, args.regression, args.saltelli, args.sc_p_order))
       ,"sc": (lambda: uqef.simulation.ScSimulation(solver, args.sc_q_order, args.sc_p_order, "G", args.sparse_quadrature, args.regression))
    }
    simulation = simulations[args.uq_method]()

    print("simulation-setup: {}".format(simulation.getSetup()))

    print("initialise simulation...")

    simulation.generateSimulationNodes(simulationNodes) #simulation.parameters are set from now on
    print(simulationNodes.printNodes()) #this is after nodes = self.nodes.T
    # do smt. like print(simulation.parameters) print(simulationNodes.nodes)
    simulationNodes_save_file = outputResultDir + "/nodes.txt"
    with open(simulationNodes_save_file, "w") as f:
        f.write(simulationNodes.printNodes())
    #simulationNodes_save_file = outputResultDir + "/nodes"
    #simulationNodes.saveToFile(simulationNodes_save_file)

    #####################################
    ### start the simulation
    #####################################
    print("start the simulation...")
    solver.init()

    simulation.prepareSolver() #this sets self.parameters of the main solver

#####################################
### do "solving"
#####################################
solver.solve(chunksize=args.chunksize)

#####################################
### stop solvers
#####################################
if mpi == False or (mpi == True and rank == 0):

    solver.tearDown() # stop the solver

    #final_results = simulation.getterResults()
    #df_simulation_result = pd.DataFrame(columns=['Index_run', 'Stationskennung', 'Type', 'TimeStamp', 'Value'])
    #for index_run, value in enumerate(final_results):
    #    if value is not None:
    #        df_single_ergebnis = config.result_parser_toPandas(value, index_run)
    #        df_single_mariental_ergebnis = df_single_ergebnis.loc[
    #            (df_single_ergebnis['Stationskennung'] == 'MARI') & (
    #                        df_single_ergebnis['Type'] == 'Abfluss Simulation')]
    #        df_simulation_result.append(df_single_mariental_ergebnis, ignore_index=True)
    #simulation.setterResults(results)

    #####################################
    ### calculate statistics:
    #####################################
    print("calculate statistics...")
    statistics_ = {
        "larsim": ( lambda: simulation.calculateStatistics(LarsimStatistics.LarsimStatistics(), simulationNodes))
    }
    statistics = statistics_[model]()
    print("--- %s seconds ---" % (time.time() - start_time))

    #####################################
    ### print statistics:
    #####################################
    #print("print statistics...")
    #print(statistics.printResults())

    #####################################
    ### generate plots
    #####################################
    print("generate plots...")
    #fileName = simulation.name
    #statistics.plotResults(fileName=fileName, directory=outputResultDir, display=False)
    statistics.plotResults(simulationNodes)

if mpi == True:
    print("rank: {} exit".format(rank))

print("I'm successfully done")
