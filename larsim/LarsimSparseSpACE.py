import datetime
import dill
import json
import inspect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os.path as osp
import pathlib
import pandas as pd
import pickle
import plotly.graph_objects as go
import seaborn as sns
import time

import sparseSpACE
from sparseSpACE.Function import *
from sparseSpACE.spatiallyAdaptiveSingleDimension2 import *
from sparseSpACE.ErrorCalculator import *
from sparseSpACE.GridOperation import *

from LarsimUtilityFunctions import larsimPaths as larsimPaths
from LarsimUtilityFunctions import larsimDataPostProcessing
from LarsimUtilityFunctions import larsimModel

import chaospy as cp
import uqef

# from . import LarsimModel


class LarsimFunction(Function):
    def __init__(self, config_file, param_names=None, qoi="Q", gof="calculateNSE"):
        super().__init__()
        self.larsimModelObject = larsimModel.LarsimModel(
            configurationObject=config_file,
            inputModelDir=inputModelDir,
            workingDir=outputModelDir,
            sourceDir=sourceDir
        )

        self.larsimModelObject.prepare(infoModel=True)

        self.qoi = qoi
        self.gof = gof

        self.param_names = []
        for i in self.larsimModelObject.larsimConfObject.configurationObject["parameters"]:
            self.param_names.append((i["type"], i["name"]))

        self.global_eval_counter = 0

    def output_length(self):
        if self.qoi == "Q":
            return len(self.larsimModelObject.larsimConfObject.t) - self.larsimModelObject.larsimConfObject.warm_up_duration
        else:
            return 1
    #     def getAnalyticSolutionIntegral(self, start, end): assert "not implemented"

    def eval(self, coordinates):
        self.global_eval_counter += 1
        params = {param_name: coord for coord, param_name in zip(coordinates, self.param_names)}

        larsim_res = self.larsimModelObject.run(
            parameters=[params, ],
            i_s=[self.global_eval_counter, ],
            take_direct_value=True,
            createNewFolder=True,
            deleteFilesAfterwards=True,
            deleteFolderAfterwards=True
        )

        if self.qoi == "Q":
            df = larsimDataPostProcessing.filterResultForStationAndTypeOfOutpu(
                resultsDataFrame=larsim_res[0][0]["result_time_series"],
                station=self.larsimModelObject.larsimConfObject.station_of_Interest,
                type_of_output=self.larsimModelObject.larsimConfObject.type_of_output_of_Interest
            )
            return np.array(df['Value'])
        elif self.qoi == "GoF":
            if self.gof == "calculateRMSE":  # TODO change this - hard-coded for now...
                temp = larsim_res[0][0]['gof_df'][self.gof].values
                temp = 1000 - temp
                return np.array(temp)
            else:
                return np.array(larsim_res[0][0]['gof_df'][self.gof].values)
        else:
            raise Exception(f"Not implemented")


local_debugging = True
if local_debugging:
    sourceDir = pathlib.Path('/work/ga45met/mnt/linux_cluster_2/Larsim-UQ')
    inputModelDir = pathlib.Path(osp.abspath(osp.join(larsimPaths.data_dir, 'Larsim-data'))) #larsimPaths.larsim_data_path
    scratch_dir = pathlib.Path("/work/ga45met")
    outputModelDir = outputResultDir = workingDir = scratch_dir / "larsim_runs" / 'larsim_sg_nse_2' #"Larsim_runs"
    config_file = pathlib.Path('/work/ga45met/mnt/linux_cluster_2/Larsim-UQ/configurations_Larsim/configurations_larsim_high_flow_small.json')
    # LARSIM_REGEN_DATA_FILES_PATH = pathlib.Path("/home/ga45met/Repositories/Larsim/Larsim-data/WHM Regen/data_files")
    # TODO save config_file!!!

    plot_file = str(outputModelDir / "output.png")
    filename_contour_plot = str(outputModelDir / "output_contour_plot.png")
    filename_refinement_graph = str(outputModelDir / "output_refinement_graph.png")
    filename_combi_scheme_plot = str(outputModelDir / "output_combi_scheme.png")
    filename_sparse_grid_plot = str(outputModelDir / "output_sg_graph.png")

    with open(config_file) as f:
        configuration_object = json.load(f)

    #####################################
    # node_names = []
    # for parameter_config in configuration_object["parameters"]:
    #     node_names.append(parameter_config["name"])
    # simulationNodes = uqef.nodes.Nodes(node_names)
    #
    # for parameter_config in configuration_object["parameters"]:
    #     if parameter_config["distribution"] == "None":
    #         simulationNodes.setValue(parameter_config["name"], parameter_config["default"])
    #     else:
    #         cp_dist_signature = inspect.signature(getattr(cp, parameter_config["distribution"]))
    #         dist_parameters_values = []
    #         for p in cp_dist_signature.parameters:
    #             dist_parameters_values.append(parameter_config[p])
    #         simulationNodes.setDist(parameter_config["name"],
    #                                 getattr(cp, parameter_config["distribution"])(*dist_parameters_values))
    #
    # print(f"Dictionary of the distributions: {simulationNodes.dists}")
    # orderdDists = []
    # orderdDistsNames = []
    # for i in range(0, len(simulationNodes.nodeNames)):
    #     nameOfNode = simulationNodes.nodeNames[i]
    #     if nameOfNode in simulationNodes.dists:
    #         orderdDists.append(simulationNodes.dists[nameOfNode])
    #         orderdDistsNames.append(nameOfNode)
    #
    # if len(simulationNodes.dists) > 0:
    #     simulationNodes.joinedDists = cp.J(*orderdDists)
    # print(f"Jones Dist: {simulationNodes.joinedDists}")
    #####################################

    params = configuration_object["parameters"]
    param_names = [(param["type"], param["name"]) for param in params]
    dim = len(params)
    distributions = [(param["distribution"], param["lower"], param["upper"]) for param in params]

    a = np.array([param["lower"] for param in params])
    b = np.array([param["upper"] for param in params])

    #####################################
    qoi = "GoF"  # "Q" "GoF"
    gof = "calculateNSE"   # "calculateRMSE" "calculateNSE"  "None"
    operation = "UncertaintyQuantification"  # "Interpolation"
    problem_function = LarsimFunction(config_file=config_file, qoi=qoi, gof=gof)

    op = UncertaintyQuantification(problem_function, distributions, a, b)
    # grid = GaussLegendreGrid(a, b, op)
    grid = GlobalTrapezoidalGridWeighted(a, b, op, boundary=False)  # try with modified_basis=True
    # try odther grids:
    # GlobalBSplineGrid, GlobalHierarchizationGrid, ClenshawCurtisGridGlobal,
    # GlobalLagrangeGrid, GlobalLagrangeGridWeighted
    # TODO - Add this from Jonas' branch
    # grid.integrator = IntegratorParallelArbitraryGridOptimized(grid)
    op.set_grid(grid)

    polynomial_degree_max = 3
    # op.set_expectation_variance_Function()
    op.set_PCE_Function(polynomial_degree_max)

    # the adaptive refinement - combiinstance takes the grid from the operation
    # combiinstance = StandardCombi(a, b, operation=op, norm=2)
    combiinstance = SpatiallyAdaptiveSingleDimensions2(
        a, b, operation=op, norm=2, grid_surplusses=grid)
    combiinstance.filename_contour_plot = filename_contour_plot
    combiinstance.filename_refinement_graph = filename_refinement_graph
    combiinstance.filename_combi_scheme_plot = filename_combi_scheme_plot
    combiinstance.filename_sparse_grid_plot = filename_sparse_grid_plot

    error_operator = ErrorCalculatorSingleDimVolumeGuided()

    lmax = 4  # 2, max_evaluations=50
    # combiinstance.perform_operation(1, lmax)
    combiinstance.performSpatiallyAdaptiv(1, lmax, error_operator, tol=0, max_evaluations=50,
                                          do_plot=True)

    # Create the PCE approximation; it is saved internally in the operation
    op.calculate_PCE(None, combiinstance)  # restrict_degrees

    fileName = f"gpce_{qoi}_{gof}.pkl"
    gpceFileName = str(outputModelDir / fileName)
    with open(gpceFileName, 'wb') as handle:
        pickle.dump(op.gPCE, handle, protocol=pickle.HIGHEST_PROTOCOL)

    fileName = f"pce_polys_{qoi}_{gof}.pkl"
    pcePolysFileName = str(outputModelDir / fileName)
    with open(pcePolysFileName, 'wb') as handle:
        pickle.dump(op.pce_polys, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Compare with chaospy
    # polynomial_degree_max = 4
    # num_quad_points = 10
    # op._set_pce_polys(polynomial_degree_max)
    # nodes, weights = cp.generate_quadrature(num_quad_points,
    #                                         op.distributions_joint, rule="G")
    # f_evals = [op.f(c) for c in zip(*nodes)]
    # op.gPCE = cp.fit_quadrature(op.pce_polys, nodes, weights,
    #                             np.asarray(f_evals), norms=op.pce_polys_norms)

    # TODO - Save gPCE coeff. from chaospy to compare

    # Calculate sobol indices with the PCE coefficients
    si_first = op.get_first_order_sobol_indices()
    si_total = op.get_total_order_sobol_indices()

    print("First order Sobol indices:", op.get_first_order_sobol_indices())
    print("Total order Sobol indices:", op.get_total_order_sobol_indices())

    # Calculate the expectation, variance with the PCE coefficients
    # (E,), (Var,) = op.calculate_expectation_and_variance(combiinstance)
    # (E,), (Var,) = op.calculate_expectation_and_variance(combiinstance, use_combiinstance_solution=False)
    (E,), (Var,) = op.get_expectation_and_variance_PCE()
    print(f"E: {E}, PCE Var: {Var}")

    # results_e_var = op.get_expectation_and_variance_PCE()
    # print(f"results_e_var: {results_e_var}")

    temp = f"results_{qoi}_{gof}.txt"
    save_file = outputModelDir / temp
    fp = open(save_file, "w")
    fp.write(f'E: {E},\n Var: {Var}, \n '
             f'First order Sobol indices: {si_first} \n; '
             f'Total order Sobol indices: {si_total} \n')
    # fp.write(f'results_e_var: {results_e_var} \n '
    #          f'First order Sobol indices: {si_first} \n; '
    #          f'Total order Sobol indices: {si_total} \n')
    fp.close()
