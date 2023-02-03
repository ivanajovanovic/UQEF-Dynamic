import copy
import dill
from distutils.util import strtobool
import json
import os.path as osp
import pathlib
import pandas as pd
import time

class HydroModelConfig(object):
    def __init__(self, configurationObject, deep_copy=False, *args, **kwargs):
        if configurationObject is None:
            self.configurationObject = dict()
        elif not isinstance(configurationObject, dict):
            self.configurationObject = larsimConfig.return_configuration_object(configurationObject)
        elif deep_copy:
            self.configurationObject = copy.deepcopy(configurationObject)
        else:
            self.configurationObject = configurationObject