from uqef.model import Model
from LarsimUtilityFunctions import larsimModel

from LarsimUtilityFunctions import Utils as utils


class LarsimModelUQ(larsimModel.LarsimModel, Model):
    def __init__(self, configurationObject, inputModelDir, workingDir=None,
                 log_level=utils.log_levels.INFO, print_level=utils.print_levels.INFO,
                 debug=False, *args, **kwargs):
        super().__init__(configurationObject, inputModelDir,
                         workingDir=workingDir,
                         log_level=log_level,
                         print_level=print_level,
                         debug=debug, *args, **kwargs)

    def assertParameter(self, parameter):
        pass

    def normaliseParameter(self, parameter):
        return parameter



