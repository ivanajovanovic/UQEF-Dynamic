from uqef.model import Model

from ishigami import IshigamiModel as ishigamimodel

class IshigamiModelUQ(ishigamimodel.IshigamiModel, Model):
    def __init__(self, configurationObject, inputModelDir, workingDir=None, *args, **kwargs):
        super().__init__(configurationObject, *args, **kwargs)