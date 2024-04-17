from uqef.model import Model
from time_dependent_model import HydroModel

class HydroModelUQ(HydroModel.HydroModel, Model):
    def __init__(self, configurationObject, inputModelDir, workingDir=None, *args, **kwargs):
        super().__init__(configurationObject, inputModelDir,
                         workingDir=workingDir, *args, **kwargs)