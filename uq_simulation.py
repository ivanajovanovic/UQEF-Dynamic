import argparse
import datetime
import json
from mpi4py import MPI
import multiprocessing

import matplotlib
matplotlib.use('Agg')

import glob, os
import pandas as pd
import numpy as np
import os.path as osp
import sys
import subprocess
import time
from tabulate import tabulate

import chaospy as cp

import uqef

import LarsimModel
import LarsimStatistics

import LinearDampedOscillatorModel
import LinearDampedOscillatorStatistics

import IshigamiModel
import IshigamiStatistics

import ProductFunctionModel
import ProductFunctionStatistics

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

parser.add_argument('-or','--outputResultDir', default="./saves/") #./oscilator/ or ./ishigami/

parser.add_argument('--configurationsFile', default="configurations.json") #configuration_oscillator.json or configuration_ishigami.json or configuration_product_function

parser.add_argument('--parallel', action='store_true', default=False)
parser.add_argument('--num_cores', type=int, default=multiprocessing.cpu_count())
parser.add_argument('--mpi', action='store_true')
parser.add_argument('--mpi_method', default="new")  # new (MpiPoolSolver), old (MpiPoolSolverOld)
parser.add_argument('--mpi_combined_parallel', action='store_true', default=False)

parser.add_argument('--model', default="larsim") #oscillator ishigami productFunction

parser.add_argument('--chunksize', type=int, default=1)
parser.add_argument('--mpi_chunksize', type=int, default=1)


parser.add_argument('--uq_method', default="sc")  # sc, mc
parser.add_argument('--regression',action='store_true', default=False)
parser.add_argument('--sparse_quadrature',action='store_true', default=False)
parser.add_argument('--saltelli',action='store_true', default=False) # compute Sobol's Indices using MC and Saltelli method, if saltelli = True then uq_method should be mc and regression = False
parser.add_argument('--mc_numevaluations', type=int, default=27)
parser.add_argument('--sc_q_order', type=int, default=3)  # number of collocation points in each direction (Q)
parser.add_argument('--sc_p_order', type=int, default=2)  # number of terms in PCE (N)

parser.add_argument('--run_statistics', action='store_true', default=False)

parser.add_argument('--transformToStandardDist', action='store_true', default=True)


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
#args.outputResultDir = outputResultDir

if mpi == False or (mpi == True and rank == 0):
    if not os.path.isdir(outputResultDir): subprocess.run(["mkdir", outputResultDir])
    print("outputResultDir: {}".format(outputResultDir))


#####################################
### read configuration setting and additional path settings:
#####################################

configuration_object = None

if mpi == False or (mpi == True and rank == 0):
    with open(args.configurationsFile) as f:
        configuration_object = json.load(f)
        print(configuration_object)

    #Set the working folder where all the model runs related output and files will be written
    try:
        configuration_object["Directories"]["working_dir"] = os.path.abspath(os.path.join(outputResultDir,
                                                                                          configuration_object["Directories"]["working_dir"]))
    except KeyError:
        try:
            configuration_object["Directories"]["working_dir"] = os.path.abspath(os.path.join(outputResultDir, "model_runs"))
        except KeyError:
            configuration_object["Directories"] = {}
            configuration_object["Directories"]["working_dir"] = os.path.abspath(
                os.path.join(outputResultDir, "model_runs"))

    if not os.path.isdir(configuration_object["Directories"]["working_dir"]):
        subprocess.run(["mkdir", configuration_object["Directories"]["working_dir"]])

#####################################
### initialise uncertain parameters - simulationNodes:
#####################################

model = args.model

if mpi == True:
    configuration_object = comm.bcast(configuration_object, root=0)
    #print("broadcasting configuration object...")

if mpi == False or (mpi == True and rank == 0):
    distributions = []
    standard_distributions = []
    transformation_param = {}
    nodeNames = []
    for i in configuration_object["Variables"]:
        if i["distribution"] == "normal":
            distributions.append((i["name"], cp.Normal(i["mean"], i["std"])))
            standard_distributions.append((i["name"], cp.Normal()))
            transformation_param[i["name"]] = (i["mean"], i["std"])
        elif i["distribution"] == "uniform":
            distributions.append((i["name"], cp.Uniform(i["uniform_low"], i["uniform_high"])))
            standard_distributions.append((i["name"], cp.Uniform(-1,1)))
            _a = (i["uniform_low"] + i["uniform_high"]) / 2
            _b = (i["uniform_high"] - i["uniform_low"]) / 2
            transformation_param[i["name"]] = (_a, _b)
        nodeNames.append(i["name"])

    simulationNodes = uqef.simulation.Nodes(nodeNames)

    #for items in distributions:
    for items in standard_distributions:
        simulationNodes.setDist(items[0], items[1])

    print("model: {}".format(args.model))
    print("chunksize: {}".format(args.chunksize))
    # Print just a node setup (before really sampling the nodes)
    print(simulationNodes.printNodesSetup())
    node_setup_name = outputResultDir + "/node_setup.txt"
    with open(node_setup_name, "w") as f:
        f.write(simulationNodes.printNodesSetup())

    transformation = lambda x, mu, std: mu + std*x

#####################################
### one time initial model setup
#####################################
# put here is there is something specifically related to the model that should be done only once
if mpi == False or (mpi == True and rank == 0):
    def initialModelSetUp():
        models = {
            "larsim": (lambda: LarsimModel.LarsimModelSetUp(configuration_object))
            ,"oscillator": (lambda: LinearDampedOscillatorModel.LinearDampedOscillatorModelSetUp(configuration_object))
            ,"ishigami": (lambda: IshigamiModel.IshigamiModelSetUp(configuration_object))
            ,"productFunction": (lambda: ProductFunctionModel.ProductFunctionModelSetUp(configuration_object))
        }
        models[model]()
    initialModelSetUp()

#####################################
### initialise model
#####################################
# each mpi processor has it's own model

def modelGenerator():
    models = {
        "larsim": (lambda: LarsimModel.LarsimModel(configuration_object))
        ,"oscillator": (lambda: LinearDampedOscillatorModel.LinearDampedOscillatorModel(configuration_object))
        ,"ishigami": (lambda: IshigamiModel.IshigamiModel(configuration_object))
        ,"productFunction": (lambda: ProductFunctionModel.ProductFunctionModel(configuration_object))
    }
    return models[model]()

#####################################
### initialise solver
#####################################

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
###
#####################################
start_time = time.time()
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

    #generate simulation nodes
    simulation.generateSimulationNodes(simulationNodes) #simulation.parameters are set from now on
    #TODO All this printing doesnt mean much if you cample from standard distributions and then do the transformation - CHANGE THIS
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

    if args.transformToStandardDist:
        #TODO Perform transformation over parameters array (parameters.shape = #totalSamples x #ofParameters)
        parameters = []
        simulation.setParameters(parameters)
        #TODO Print parameters now and save them...

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

    #####################################
    ### calculate statistics:
    #####################################
    if args.run_statistics:
        print("calculate statistics...")

        #####################################
        ### as part of statistics - simple model predictions comparioson with groung truth - collect over different runs
        #####################################
        list_gof_dataFrames = []
        path = configuration_object["Directories"]["working_dir"]
        files = [f for f in glob.glob(path + "/" + "**/goodness_of_fit_*.csv", recursive=True)]
        for single_file in files:
            list_gof_dataFrames.append(pd.read_csv(single_file))  # TODO Maybe some postreading processing will be required
        gof_dataFrame = pd.concat(list_gof_dataFrames, ignore_index=True, sort=False, axis=0)
        # Printout
        print(tabulate(gof_dataFrame, headers=gof_dataFrame.columns, floatfmt=".4f"))
        print("RMSE MEAN: {:.4f} \n".format(np.mean(gof_dataFrame.RMSE.values)))
        print("BIAS MEAN: {:.4f} \n".format(np.mean(gof_dataFrame.BIAS.values)))
        print("NSE MEAN: {:.4f} \n".format(np.mean(gof_dataFrame.NSE.values)))
        print("LogNSE MEAN: {:.4f} \n".format(np.mean(gof_dataFrame.LogNSE.values)))
        # Save to CSV file
        gof_dataFrame.to_csv(path_or_buf=os.path.abspath(os.path.join(path, "goodness_of_fit.csv")),index=True)


        statistics_ = {
            "larsim": ( lambda: simulation.calculateStatistics(LarsimStatistics.LarsimStatistics(configuration_object), simulationNodes))
            ,"oscillator": (lambda: simulation.calculateStatistics(LinearDampedOscillatorStatistics.LinearDampedOscillatorStatistics(), simulationNodes))
            ,"ishigami": (lambda: simulation.calculateStatistics(IshigamiStatistics.IshigamiStatistics(configuration_object), simulationNodes))
            ,"productFunction": (lambda: simulation.calculateStatistics(ProductFunctionStatistics.ProductFunctionStatistics(configuration_object), simulationNodes))
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
        statistics.plotResults(simulationNodes, display=True)

if mpi == True:
    print("rank: {} exit".format(rank))

print("I'm successfully done")
