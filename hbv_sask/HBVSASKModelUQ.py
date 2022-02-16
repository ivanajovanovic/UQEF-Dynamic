import json
import pathlib
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot
from distutils.util import strtobool
import time
import dill

from uqef.model import Model
from hbv_sask import HBVSASKModel as hbvsaskmodel

class HBVSASKModelUQ(hbvsaskmodel.HBVSASKModel, Model):
    def __init__(self, configurationObject, inputModelDir, workingDir=None, *args, **kwargs):
        super().__init__(configurationObject, inputModelDir,
                         workingDir=workingDir, *args, **kwargs)