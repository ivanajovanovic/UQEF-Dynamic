from uqef.model import Model
from uqef_dynamic.models.hbv_sask import HBVSASKModel as hbvsaskmodel

class HBVSASKModelUQ(hbvsaskmodel.HBVSASKModel, Model):
    def __init__(self, configurationObject, inputModelDir, workingDir=None, *args, **kwargs):
        super().__init__(configurationObject, inputModelDir,
                         workingDir=workingDir, *args, **kwargs)