"""The LSST Model for photometric errors."""
from dataclasses import MISSING

from ceci.config import StageParameter as Param
from photerr import LsstErrorModel as PhotErrErrorModel
from photerr import LsstErrorParams as PhotErrErrorParams
from rail.creation.degrader import Degrader


class LSSTErrorModel(Degrader):
    """The LSST Model for photometric errors.

    This is a wrapper around the error model from PhotErr. The parameter
    docstring below is dynamically added by the installed version of PhotErr:

    """

    # Dynamically add the parameter docstring from PhotErr
    __doc__ += PhotErrErrorParams.__doc__

    name = "LSSTErrorModel"
    config_options = Degrader.config_options.copy()

    # Dynamically add all parameters from PhotErr
    _photerr_params = PhotErrErrorParams.__dataclass_fields__
    for key, val in _photerr_params.items():
        # Get the default value
        if val.default is MISSING:
            default = val.default_factory()
        else:
            default = val.default

        # Add this param to config_options
        config_options[key] = Param(
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
        Degrader.__init__(self, args, comm=comm)
        self.error_model = PhotErrErrorModel(
            **{key: self.config[key] for key in self._photerr_params}
        )

    def run(self):
        """Return pandas DataFrame with photometric errors."""
        # Load the input catalog
        data = self.get_data("input")

        # Add photometric errors
        obsData = self.error_model(data, random_state=self.config.seed)

        # Return the new catalog
        self.add_data("output", obsData)
