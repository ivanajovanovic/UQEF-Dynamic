from uqef.model import Model

from uqef_dynamic.models.ishigami import IshigamiModel as ishigamimodel

class IshigamiModelUQ(ishigamimodel.IshigamiModel, Model):
    def __init__(self, configurationObject, *args, **kwargs):
        super().__init__(configurationObject, *args, **kwargs)