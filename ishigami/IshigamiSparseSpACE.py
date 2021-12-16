import json
import numpy as np
import pathlib
import pickle

from sparseSpACE.Function import *
from sparseSpACE.spatiallyAdaptiveSingleDimension2 import *
from sparseSpACE.ErrorCalculator import *
from sparseSpACE.GridOperation import *

import chaospy as cp
import uqef

import IshigamiModel

class IshigamiFunction(Function):
    def __init__(self, configurationObject, dim):
        super().__init__()

        self.ishigamiModelObject = IshigamiModel.IshigamiModel(configurationObject=configurationObject)
        self.global_eval_counter = 0
        self.dim = dim

    def output_length(self):
        return 1

    def eval(self, coordinates):
        # assert (len(coordinates) == self.dim), len(coordinates)

        self.global_eval_counter += 1
        result_tuple = self.ishigamiModelObject.run(
            i_s=[self.global_eval_counter, ],
            parameters=[coordinates, ]
        )
        value_of_interest = result_tuple[0][0]
        return value_of_interest  # np.array(value_of_interest)


def main(local_debugging):
    scratch_dir = pathlib.Path("/work/ga45met")
    outputModelDir = scratch_dir / "ishigami_runs" / "ishigami_sg_l_6_p_3_max_1000"
    outputModelDir.mkdir(parents=True, exist_ok=True)
    plot_file = str(outputModelDir / "output.png")
    filename_contour_plot = str(outputModelDir / "output_contour_plot.png")
    filename_combi_scheme_plot = str(outputModelDir / "output_combi_scheme.png")
    filename_refinement_graph = str(outputModelDir / "output_refinement_graph.png")
    filename_sparse_grid_plot = str(outputModelDir / "output_sg_graph.png")

    if local_debugging:
        config_file = pathlib.Path(
            '/work/ga45met/mnt/linux_cluster_2/Larsim-UQ/configurations/configuration_ishigami.json')

        with open(config_file) as f:
            configuration_object = json.load(f)
        #####################################
        try:
            params = configuration_object["parameters"]
        except KeyError as e:
            print(f"Ishigami Statistics: parameters key does "
                  f"not exists in the configurationObject{e}")
            raise
        param_names = [param["name"] for param in params]
        labels = [param_name.strip() for param_name in param_names]
        dim = len(params)  # TODO take only uncertain paramters into account
        distributions = [(param["distribution"], param["lower"], param["upper"]) for param in params]

        a = np.array([param["lower"] for param in params])
        b = np.array([param["upper"] for param in params])
        #####################################
        problem_function = IshigamiFunction(configurationObject=configuration_object, dim=dim)
        timesteps = 1
        op = UncertaintyQuantification(problem_function, distributions, a, b)
        grid = GlobalTrapezoidalGridWeighted(a, b, op, boundary=False,
                                             modified_basis=True)  # try with modified_basis=True
        # TODO - Add this from Jonas' branch
        # grid.integrator = IntegratorParallelArbitraryGridOptimized(grid)
        op.set_grid(grid)

        # Select the function for which the grid is refined
        polynomial_degree_max = 3 #8
        op.set_PCE_Function(polynomial_degree_max)
        # here it is the expectation and variance calculation via the moments
        # op.set_expectation_variance_Function()

        # combiinstance = StandardCombi(a, b, operation=op, norm=2)
        combiinstance = SpatiallyAdaptiveSingleDimensions2(
            a, b, operation=op, norm=2, grid_surplusses=grid)
        combiinstance.filename_contour_plot = filename_contour_plot
        combiinstance.filename_refinement_graph = filename_refinement_graph
        combiinstance.filename_combi_scheme_plot = filename_combi_scheme_plot
        combiinstance.filename_sparse_grid_plot = filename_sparse_grid_plot

        error_operator = ErrorCalculatorSingleDimVolumeGuided()

        lmax = 4 #6  # 2, max_evaluations=50
        # combiinstance.perform_operation(1, lmax)
        combiinstance.performSpatiallyAdaptiv(1, lmax, error_operator, tol=0, max_evaluations=1000,
                                              do_plot=True)
        #####################################
        # Create the PCE approximation; it is saved internally in the operation
        op.calculate_PCE(polynomial_degrees=polynomial_degree_max, combiinstance=combiinstance)  # restrict_degrees
        # (E,), (Var,) = op.calculate_expectation_and_variance(combiinstance)

        gPCE = op.get_gPCE()
        fileName = f"gpce.pkl"
        gpceFileName = str(outputModelDir / fileName)
        with open(gpceFileName, 'wb') as handle:
            pickle.dump(op.gPCE, handle, protocol=pickle.HIGHEST_PROTOCOL)

        fileName = f"pce_polys.pkl"
        pcePolysFileName = str(outputModelDir / fileName)
        with open(pcePolysFileName, 'wb') as handle:
            pickle.dump(op.pce_polys, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #####################################
        # Calculate the expectation, variance with the PCE coefficients
        # (E,), (Var,) = op.calculate_expectation_and_variance(combiinstance)
        # (E,), (Var,) = op.calculate_expectation_and_variance(combiinstance, use_combiinstance_solution=False)
        (E,), (Var,) = op.get_expectation_and_variance_PCE()
        print(f"E: {E}, Var: {Var}")

        # Calculate sobol indices with the PCE coefficients
        si_first = op.get_first_order_sobol_indices()
        si_total = op.get_total_order_sobol_indices()

        # print(f"First order Sobol indices: {si_first.shape} \n {si_first}")
        # print(f"Total order Sobol indices: {si_total.shape} \n {si_total}")

        sobol_m_analytical, sobol_t_analytical = problem_function.ishigamiModelObject.get_analytical_sobol_indices()
        for i in range(len(labels)):
            print(f"Sobol's Total Index for parameter {labels[i]} is: \n")
            print(f"Sobol Total Simulation = {si_total[i][0]} \n")
            print(f"Sobol Total Analytical = {sobol_t_analytical[i]:.4f} \n")

        for i in range(len(labels)):
            print(f"Sobol's Main Index for parameter {labels[i]} is: \n")
            print(f"Sobol Main Simulation = {si_first[i][0]} \n")
            print(f"Sobol Main Analytical = {sobol_m_analytical[i]:.4f} \n")

        temp = f"results.txt"
        save_file = outputModelDir / temp
        fp = open(save_file, "w")
        fp.write(f'E: {E},\n Var: {Var}, \n '
                 f'First order Sobol indices: {si_first} \n; '
                 f'Total order Sobol indices: {si_total} \n')
        fp.close()
    else:
        print("IshigamiSparseSpACE - local_debugging is False!")


if __name__ == "__main__":
    local_debugging = True
    main(local_debugging)