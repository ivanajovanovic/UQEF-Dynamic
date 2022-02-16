import json
import pathlib
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot
from distutils.util import strtobool
import time
import dill

from sparseSpACE.Function import *
from sparseSpACE.spatiallyAdaptiveSingleDimension2 import *
from sparseSpACE.ErrorCalculator import *
from sparseSpACE.GridOperation import *
# from sparseSpACE.StandardCombi import *
# from sparseSpACE.Integrator import *

from . import HBVSASKModel as hbv

class HBVSASKFunction(Function):
    def __init__(self, configurationObject, dim, param_names=None, qoi="Q", gof="calculateNSE"):
        super().__init__()
        self.global_eval_counter = 0
        self.dim = dim

    def output_length(self):
        return 1

    def eval(self, coordinates):
        pass


local_debugging = True
if local_debugging:
    config_file = pathlib.Path('/dss/dsshome1/lxc0C/ga45met2/Repositories/UQEFPP/configurations/configuration_hbv.json')
    # config_file = pathlib.Path('/work/ga45met/mnt/linux_cluster_2/UQEFPP/configurations/configuration_hbv.json')
    with open(config_file) as f:
        configuration_object = json.load(f)

    dim = 0
    distributions = []
    a = []
    b = []
    for single_param in configuration_object["parameters"]:
        param_names.append(single_param["name"])
        if single_param["distribution"] != "None":
            dim += 1
            distributions.append((single_param["distribution"], single_param["lower"], single_param["upper"]))
            a.append(single_param["lower"])
            b.append(single_param["upper"])
