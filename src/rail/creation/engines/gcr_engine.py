"""Stage that loads a catalog from GCRCatalogs."""

import GCRCatalogs
import pandas as pd
from ceci.config import StageParameter as Param
from rail.core.data import PqHandle
from rail.creation.engine import Creator


class GCRCreator(Creator):
    """Creator that returns a catalog from GCRCatalogs.

    For info on GCRCatalogs, see https://github.com/LSSTDESC/gcr-catalogs

    See https://github.com/LSSTDESC/gcr-catalogs/blob/master/examples/GCRCatalogs%20Demo.ipynb
    for how to get available options for catalog_name.

    See https://github.com/LSSTDESC/gcr-catalogs/blob/master/GCRCatalogs/SCHEMA.md
    for all options for quantities to query.
    """

    name = "GCRLoader"
    entrypoint_function = "sample"  # the user-facing science function for this class
    outputs = [("output", PqHandle)]

    config_options = Creator.config_options.copy()
    config_options.update(
        gcr_root_dir=Param(
            str,
            "/global/cfs/cdirs/lsst/shared",
            msg="The path to the GCR catalogs.",
        ),
        catalog_name=Param(
            str,
            "cosmoDC2_v1.1.4_small",
            msg="The name of the GCR catalog to load.",
        ),
        quantities=Param(
            list,
            ["redshift"]
            + [f"mag_{band}_lsst" for band in "ugrizy"]
            + ["size_true", "size_minor_true"],
            msg="The quantities to query from the catalog.",
        ),
        filters=Param(
            list,
            ["mag_i_lsst < 26.5"],
            msg="Filters passed to the GCR query.",
        ),
    )

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

        # Provides override for unit test
        self._catalog_override = None

    def run(self):
        """Load the GCR catalog, subsample, and return pandas DataFrame."""
        # Set the GCRCatalog path
        GCRCatalogs.set_root_dir(self.config.gcr_root_dir)

        # Load the catalog
        # In the unit test, this override is provided
        if self._catalog_override is not None:
            gc = GCRCatalogs.load_catalog(*self._catalog_override)
        # Otherwise, proceed as normal
        else:  # pragma: no cover
            gc = GCRCatalogs.load_catalog(self.config.catalog_name)

        # Query the requested quantities
        cat = gc.get_quantities(self.config.quantities, filters=self.config.filters)

        # Convert to a DataFrame
        cat = pd.DataFrame(cat)

        # Downsample
        cat = cat.sample(n=self.config.n_samples, random_state=self.config.seed)
        cat = cat[self.config.quantities]

        self.add_data("output", cat)
