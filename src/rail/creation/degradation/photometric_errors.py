"""The LSST Model for photometric errors."""
from dataclasses import MISSING

from ceci.config import StageParameter as Param
from photerr import LsstErrorModel as peLsstErrorModel
from photerr import LsstErrorParams as peLsstErrorParams
from photerr import RomanErrorModel as peRomanErrorModel
from photerr import RomanErrorParams as peRomanErrorParams
from rail.creation.noisifier import Noisifier

class PhotoErrorModel(Noisifier):
    """The Base Model for photometric errors.

    This is a wrapper around the error model from PhotErr. The parameter
    docstring below is dynamically added by the installed version of PhotErr:

    """
    
    def set_params(self, peparams):
        PhotErrErrorParams = peparams
                
        config_options = Noisifier.config_options.copy()

        # Dynamically add all parameters from PhotErr
        _photerr_params = PhotErrErrorParams.__dataclass_fields__
        self._photerr_params = _photerr_params
        for key, val in _photerr_params.items():
            # Get the default value
            if val.default is MISSING:
                default = val.default_factory()
            else:
                default = val.default

            # Add this param to config_options
            self.config[key] = Param(
                None,  # Let PhotErr handle type checking
                default,  
                msg="See the main docstring for details about this parameter.",
                required=False,
            )
        
    def __init__(self, args, comm=None):
        """
        Constructor

        Does standard Degrader initialization and sets up the error model.
        """
        Noisifier.__init__(self, args, comm=comm)
        

        
    def addNoise(self, noiseModel):
        
        # Load the input catalog
        data = self.get_data("input")

        # Add photometric errors
        obsData = noiseModel(data, random_state=self.config.seed)
        
        # Return the new catalog
        
        self.add_data("output", obsData)
        
        print('addNoise is ran')
        
        
class LSSTErrorModel(PhotoErrorModel):
    
    name = "LSSTErrorModel"
#     PhotErrErrorParams = peLsstErrorParams
#     PhotoErrorModel.define_default_params(peLsstErrorParams)
    
    def __init__(self, args, comm=None):
        """
        Constructor

        Does standard Degrader initialization and sets up the error model.
        """
        PhotoErrorModel.__init__(self, args, comm=comm)
        
        
        self.set_params(peLsstErrorParams)
        self.initNoiseModel()
        
        
    def initNoiseModel(self):
        self.noiseModel = peLsstErrorModel(
            **{key: self.config[key] for key in self._photerr_params}
        )
        
        
class RomanErrorModel(PhotoErrorModel):
    
    name = "RomanErrorModel"
    # PhotoErrorModel.PhotErrErrorParams = peRomanErrorParams
    # PhotoErrorModel.define_default_params()
    
    def __init__(self, args, comm=None):
        """
        Constructor

        Does standard Degrader initialization and sets up the error model.
        """
        PhotoErrorModel.__init__(self, args, comm=comm)
        
        self.set_params(peRomanErrorParams)
        self.initNoiseModel()
        
        
    def initNoiseModel(self):
        self.noiseModel = peRomanErrorModel(
            **{key: self.config[key] for key in self._photerr_params}
        )