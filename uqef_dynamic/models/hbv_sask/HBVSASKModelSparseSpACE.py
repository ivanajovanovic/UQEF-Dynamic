import json
import pathlib
# import pandas as pd
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# from plotly.offline import plot
# from distutils.util import strtobool
# import time
# import dill

from sparseSpACE.Function import *
# from sparseSpACE.spatiallyAdaptiveSingleDimension2 import *
# from sparseSpACE.ErrorCalculator import *
# from sparseSpACE.GridOperation import *
# from sparseSpACE.StandardCombi import *
# from sparseSpACE.Integrator import *

from uqef_dynamic.models.hbv_sask import HBVSASKModel as hbvsaskmodel

class HBVSASKFunction(Function):
    def __init__(self, configurationObject, inputModelDir, workingDir, dim=None,
                 param_names=None, qoi="Q", gof="NSE", **kwargs):
        super().__init__()

        self.dim = dim
        self.param_names = param_names
        self.qoi = qoi
        self.gof = gof # likelihood function

        self.writing_results_to_a_file = kwargs.get("writing_results_to_a_file", False)
        self.plotting = kwargs.get("plotting", False)

        self.hbvsaskModelObject = hbvsaskmodel.HBVSASKModel(
            configurationObject=configurationObject,
            inputModelDir=inputModelDir,
            workingDir=workingDir,
            writing_results_to_a_file=self.writing_results_to_a_file,
            plotting=self.plotting
        )

        self.global_eval_counter = 0

    def output_length(self):
        if self.qoi == "Q":
            return len(list(self.hbvsaskModelObject.simulation_range))
        else:
            return 1

    def eval(self, coordinates):
        self.global_eval_counter += 1
        params = {param_name: coord for coord, param_name in zip(coordinates, self.param_names)}

        results_array = self.hbvsaskModelObject.run(
            parameters=[params, ],
            i_s=[self.global_eval_counter, ],
            take_direct_value=True
        )

        if self.qoi == "Q":
            return np.array(results_array[0][0]['result_time_series']['Q_cms'].values)
        elif self.qoi == "GoF":
            if self.gof in results_array[0][0]['gof_df'].columns:
                return results_array[0][0]['gof_df'][self.gof].values[0]
            else:
                return None
        else:
            raise Exception(f"Not implemented")


local_debugging = True
if local_debugging:
    # TODO Change these paths accordingly!
    inputModelDir = pathlib.Path("/dss/dssfs02/lwp-dss-0001/pr63so/pr63so-dss-0000/ga45met2/HBV-SASK-data")
    outputModelDir = pathlib.Path('/gpfs/scratch/pr63so/ga45met2/hbvsask_runs/initila_trial_run')
    config_file = pathlib.Path('/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEFPP/configurations/configuration_hbv.json')
    # config_file = pathlib.Path('/work/ga45met/mnt/linux_cluster_2/UQEFPP/configurations/configuration_hbv.json')

    writing_results_to_a_file = True
    plotting = True

    with open(config_file) as f:
        configuration_object = json.load(f)

    # This will be needed for SparseSpACE
    dim = 0
    distributions = []
    a = []
    b = []
    param_names = []
    for single_param in configuration_object["parameters"]:
        param_names.append(single_param["name"])
        if single_param["distribution"] != "None":
            dim += 1
            distributions.append((single_param["distribution"], single_param["lower"], single_param["upper"]))
            a.append(single_param["lower"])
            b.append(single_param["upper"])

    # params = configuration_object["parameters"]
    # param_names = [(param["type"], param["name"]) for param in params if param["distribution"] != "None"]
    # dim = len(params)
    # # TODO make this more advances with reading all arguments
    # distributions = [(param["distribution"], param["lower"], param["upper"]) for param in params if
    #                  param["distribution"] != "None"]
    # a = np.array([param["lower"] for param in params if param["distribution"] != "None"])
    # b = np.array([param["upper"] for param in params if param["distribution"] != "None"])

    #####################################
    qoi = "Q"  # "Q" "GoF"
    gof = "RMSE"   # "RMSE" "NSE"  "None"
    operation = "UncertaintyQuantification"  # "Interpolation"
    problem_function = HBVSASKFunction(
        configurationObject=configuration_object,
        inputModelDir=inputModelDir,
        workingDir=outputModelDir,
        dim=dim,
        param_names=param_names,
        qoi=qoi,
        gof=gof,
        writing_results_to_a_file=writing_results_to_a_file,
        plotting=plotting
    )

    coordinates = [0.0, 5.0, 0.5, 0.5, 100, 2.0, 0.5, 0.5, 2.0, 0.025]
    single_run_results = problem_function.eval(coordinates)
