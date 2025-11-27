"""
The Photometric error Model LSST, Roman, and Euclid. Based on photometric
error models defined in the package photerr

Author: John Franklin Crenshaw, Tianqing Zhang
"""

from dataclasses import MISSING

import numpy as np
from ceci.config import StageParameter as Param
from photerr import EuclidDeepErrorModel as peEuclidDeepErrorModel
from photerr import EuclidDeepErrorParams as peEuclidDeepErrorParams
from photerr import EuclidErrorModel as peEuclidErrorModel
from photerr import EuclidErrorParams as peEuclidErrorParams
from photerr import EuclidWideErrorModel as peEuclidWideErrorModel
from photerr import EuclidWideErrorParams as peEuclidWideErrorParams
from photerr import LsstErrorModel as peLsstErrorModel
from photerr import LsstErrorParams as peLsstErrorParams
from photerr import RomanDeepErrorModel as peRomanDeepErrorModel
from photerr import RomanDeepErrorParams as peRomanDeepErrorParams
from photerr import RomanErrorModel as peRomanErrorModel
from photerr import RomanErrorParams as peRomanErrorParams
from photerr import RomanMediumErrorModel as peRomanMediumErrorModel
from photerr import RomanMediumErrorParams as peRomanMediumErrorParams
from photerr import RomanUltraDeepErrorModel as peRomanUltraDeepErrorModel
from photerr import RomanUltraDeepErrorParams as peRomanUltraDeepErrorParams
from photerr import RomanWideErrorModel as peRomanWideErrorModel
from photerr import RomanWideErrorParams as peRomanWideErrorParams
from rail.creation.noisifier import Noisifier


class PhotoErrorModel(Noisifier):
    """The Base Model for photometric errors.

    This is a wrapper around the error model from PhotErr. The parameter
    docstring below is dynamically added by the installed version of PhotErr:


    """

    name = "PhotoErrorModel"

    def set_params(self, peparams):
        """
        Set the photometric error parameters from photerr to
        the ceci config
        """
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
            # Use setattr() becuase ceci.StageConfig has
            # implemented __setitem__ to just set the value
            # rather than add the parameters
            param = Param(
                None,  # Let PhotErr handle type checking
                default,
                msg="See the main docstring for details about this parameter.",
                required=False,
            )
            setattr(self.config, key, param)

    def reload_pars(self, args):
        """This is needed b/c the parameters are dynamically defined,
        so we have to reload them _after_ then have been defined"""
        if isinstance(args, dict):
            # coming from python, add 'config' to the configuration
            copy_args = args.copy()
            copy_args["config"] = args
        else:  # pragma: no cover
            # coming from cli, just convert to a dict
            copy_args = vars(args).copy()
        self.load_configs(copy_args)
        self._io_checked = False
        self.check_io()

    def _initNoiseModel(self):
        """
        Initialize the noise model by the peNoiseModel
        """
        self.noiseModel = self.peNoiseModel(
            **{key: self.config[key] for key in self._photerr_params}
        )

    def _addNoise(self):
        """
        Add noise to the input catalog
        """
        # Load the input catalog
        data = self.get_data("input")

        # Add photometric errors
        if self.config.seed is not None:  # pragma: no cover
            seed = int(self.config.seed)
        else:
            seed = np.random.Generator(np.random.PCG64())
        obsData = self.noiseModel(data, random_state=seed)

        # Return the new catalog
        self.add_data("output", obsData)


class LSSTErrorModel(PhotoErrorModel):
    """
    The LSST Error model, defined by peLsstErrorParams and peLsstErrorModel
    """

    name = "LSSTErrorModel"

    def __init__(self, args, **kwargs):

        super().__init__(args, **kwargs)

        self.set_params(peLsstErrorParams)
        self.reload_pars(args)
        self.peNoiseModel = peLsstErrorModel


class RomanErrorModel(PhotoErrorModel):
    """
    The Roman Error model, defined by peRomanErrorParams and peRomanErrorModel
    """

    name = "RomanErrorModel"

    def __init__(self, args, **kwargs):

        super().__init__(args, **kwargs)

        self.set_params(peRomanErrorParams)
        self.reload_pars(args)
        self.peNoiseModel = peRomanErrorModel


class RomanWideErrorModel(PhotoErrorModel):
    """
    The Roman WideError model, defined by peRomanWideErrorParams and peRomanWideErrorModel
    """

    name = "RomanWideErrorModel"

    def __init__(self, args, **kwargs):

        super().__init__(args, **kwargs)

        self.set_params(peRomanWideErrorParams)
        self.reload_pars(args)
        self.peNoiseModel = peRomanWideErrorModel


class RomanMediumErrorModel(PhotoErrorModel):
    """
    The Roman Medium Error model, defined by peRomanMediumErrorParams and peRomanMediumErrorModel
    """

    name = "RomanMediumErrorModel"

    def __init__(self, args, **kwargs):

        super().__init__(args, **kwargs)

        self.set_params(peRomanMediumErrorParams)
        self.reload_pars(args)
        self.peNoiseModel = peRomanMediumErrorModel


class RomanDeepErrorModel(PhotoErrorModel):
    """
    The Roman Deep Error model, defined by peRomanDeepErrorParams and peRomanDeepErrorModel
    """

    name = "RomanDeepErrorModel"

    def __init__(self, args, **kwargs):

        super().__init__(args, **kwargs)

        self.set_params(peRomanDeepErrorParams)
        self.reload_pars(args)
        self.peNoiseModel = peRomanDeepErrorModel


class RomanUltraDeepErrorModel(PhotoErrorModel):
    """
    The Roman UltraDeep Error model, defined by peRomanUltraDeepErrorParams and peRomanUltraDeepErrorModel
    """

    name = "RomanUltraDeepErrorModel"

    def __init__(self, args, **kwargs):

        super().__init__(args, **kwargs)

        self.set_params(peRomanUltraDeepErrorParams)
        self.reload_pars(args)
        self.peNoiseModel = peRomanUltraDeepErrorModel


class EuclidErrorModel(PhotoErrorModel):
    """
    The Euclid Error model, defined by peEuclidErrorParams and peEuclidErrorModel
    """

    name = "EuclidErrorModel"

    def __init__(self, args, **kwargs):

        super().__init__(args, **kwargs)

        self.set_params(peEuclidErrorParams)
        self.reload_pars(args)
        self.peNoiseModel = peEuclidErrorModel


class EuclidWideErrorModel(PhotoErrorModel):
    """
    The Euclid Wide Error model, defined by peEuclidWideErrorParams and peEuclidWideErrorModel
    """

    name = "EuclidWideErrorModel"

    def __init__(self, args, **kwargs):

        super().__init__(args, **kwargs)

        self.set_params(peEuclidWideErrorParams)
        self.reload_pars(args)
        self.peNoiseModel = peEuclidWideErrorModel


class EuclidDeepErrorModel(PhotoErrorModel):
    """
    The Euclid Deep Error model, defined by peEuclidDeepErrorParams and peEuclidDeepErrorModel
    """

    name = "EuclidDeepErrorModel"

    def __init__(self, args, **kwargs):

        super().__init__(args, **kwargs)

        self.set_params(peEuclidDeepErrorParams)
        self.reload_pars(args)
        self.peNoiseModel = peEuclidDeepErrorModel
