"""The LSST Model for photometric errors."""
from dataclasses import MISSING
from typing import get_origin, get_type_hints, get_args, Union

from ceci.config import StageParameter as Param
from photerr import LsstErrorModel as PhotErrErrorModel
from photerr import LsstErrorParams as PhotErrErrorParams

from rail.creation.degrader import Degrader


def _resolve_type(val):
    """Take the input type, and resolve it to its most basic value.

    This is a helper function that helps extract the type information from
    PhotErr in a way that Ceci likes (i.e. Ceci doesn't like parameterized
    generic types).

    E.g., it transforms dict[str, int] into dict, or

    Parameters
    ----------
    val: Any
        Some kind of type object. Intended to be a generic type, a
        parameterized generic type, or a DataClass InitVar type.

    Returns
    -------
    type
        After the recursion ends, the generic resolved type is returned.
    """
    # If val has a type attribute (e.g. it's an InitVar),
    # feed this type attribute back into this function
    if hasattr(val, "type"):
        return _resolve_type(val.type)

    # If it's a Union
    elif get_origin(val) is Union:
        # Get the types inside the Union
        type_tuple = get_args(val)

        # Optional types are allowed.. but return type without None
        if len(type_tuple) == 2 and type_tuple[1] is type(None):
            return _resolve_type(type_tuple[0])
        # Unions that aren't just Optional[type] are not allowed by Ceci
        else:
            raise TypeError("Sadly Ceci doesn't allow Union types.")

    # If val is a parameterized generic type, remove the parameterization
    elif get_origin(val) is not None:
        return _resolve_type(get_origin(val))

    # If none of the above apply, it should just be a type,
    # so end the recursion and return the type
    else:
        return val


class LSSTErrorModel(Degrader):
    """The LSST Model for photometric errors.

    This is a wrapper around the error model from PhotErr. The parameter
    docstring below is dynamically added by the installed version of PhotErr:

    """

    # Dynamically add the parameter docstring from PhotErr
    __doc__ += PhotErrErrorParams.__doc__

    name = "LSSTErrorModel"
    config_options = Degrader.config_options.copy()

    # Now we want to copy all parameters from the installed PhotErr
    # First, let's get a dict of all params, including the InitVars
    _photerr_params = PhotErrErrorParams.__dataclass_fields__

    # Now get the resolved types for every parameter
    _photerr_types = {
        key: _resolve_type(val)
        for key, val in get_type_hints(PhotErrErrorParams).items()
    }

    # Finally, loop over all params to add to config_options
    for key, val in _photerr_params.items():
        # Get the default value
        if val.default is MISSING:
            default = val.default_factory()
        else:
            default = val.default

        # Add this param to config_options
        config_options[key] = Param(
            _photerr_types[key],
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
