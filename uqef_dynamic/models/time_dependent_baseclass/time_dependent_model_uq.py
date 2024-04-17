from uqef.model import Model
from uqef_dynamic.models.time_dependent_baseclass.time_dependent_model import TimeDependentModel

class HydroModelUQ(TimeDependentModel, Model):
    def __init__(self, configurationObject, inputModelDir, workingDir=None, *args, **kwargs):
        super().__init__(configurationObject, inputModelDir,
                         workingDir=workingDir, *args, **kwargs)