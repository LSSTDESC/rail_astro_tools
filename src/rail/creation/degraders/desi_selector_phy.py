"""Degrader that applies DESI tracer selection using pre-computed redshift-dependent thresholds."""

import numpy as np
from ceci.config import StageParameter as Param
from rail.core.data import PqHandle, TableHandle
from rail.creation.selector import Selector
from scipy.interpolate import interp1d
import pandas as pd


class SpecSelection_DESI_Phy(Selector):
    """DESI tracer selector based on pre-computed redshift-dependent thresholds.

    Applies a selection to a simulation catalog by comparing a physical
    parameter column against a threshold that varies with redshift. The
    threshold table is provided externally (e.g. from abundance matching)
    and is not computed by this stage.

    All supported DESI tracer types (bgs, lrg, elg) select objects whose
    physical parameter value is *above* the redshift-interpolated threshold.

    Inputs
    ------
    input : PqHandle
        Simulation catalog containing the physical parameter column and a
        redshift column.
    threshold_table : TableHandle
        Table with two columns:
          - ``z``     : redshift bin centers
          - ``thresh``: threshold values at those redshift centers

    Output
    ------
    output : PqHandle
        Catalog after applying the DESI selection mask.
    """

    name = "SpecSelection_DESI_Phy"
    entrypoint_function = "__call__"
    interactive_function = "spec_selection_desi_phy"

    inputs = [("input", PqHandle)]
    outputs = [("output", PqHandle)]

    config_options = Selector.config_options.copy()
    config_options.update(
        desi_type=Param(
            str,
            "lrg",
            msg="DESI tracer type: 'bgs', 'lrg', or 'elg'",
        ),
        threshold_col=Param(
            str,
            "None",
            msg="Column in the input catalog used for threshold-based selection "
                "(e.g. 'log_peak_sub_halo_mass' for bgs/lrg, 'log_sfr' for elg)",
        ),
        redshift_col=Param(
            str,
            "redshift",
            msg="Column name for redshift in the input catalog",
        ),
        threshold_table=Param(str, "None", msg="Filename of the threshold file")
    )

    def __call__(self, sample, **kwargs):
        """Apply the DESI physical selection to a catalog.

        Parameters
        ----------
        sample : table-like or PqHandle
            Input simulation catalog.

        Returns
        -------
        PqHandle
            Handle to the output catalog containing only the selected objects
            (when ``drop_rows=True``, the default) or the full catalog with a
            ``flag`` column (when ``drop_rows=False``).
        """
        self.set_data("input", sample)
        self.run()
        self.finalize()
        return self.get_handle("output")

    def _select(self):
        """Compute and return the selection mask.

        Returns
        -------
        numpy.ndarray of int
            Array of 0/1 flags, one per row of the input catalog.
        """
        data = self.get_data("input", allow_missing=True)
        # thresh_data = self.get_data("threshold_table", allow_missing=True)

        threshold_col = self.config.threshold_col
        redshift_col = self.config.redshift_col

        threshold_table = self.config.threshold_table
        try:
            thresh_data = pd.read_parquet(threshold_table)
        except Exception as e: # pragma: no cover
            raise ValueError(
                f"Could not read threshold file '{threshold_table}' as a Parquet file. "
                f"Ensure the file exists and is a valid Parquet file. Original error: {e}"
            ) from e

        # --- validate input columns ---
        for col in (threshold_col, redshift_col):
            if col not in data.columns:
                raise ValueError(
                    f"Input catalog is missing required column '{col}'. "
                    f"Available columns: {list(data.columns)}"
                )
        for col in ("z", "thresh"):
            if col not in thresh_data.columns:
                raise ValueError(
                    f"Threshold table is missing required column '{col}'. "
                    f"Available columns: {list(thresh_data.columns)}"
                )

        # --- interpolate threshold as a function of redshift ---
        z_centers = np.array(thresh_data["z"])
        thresholds = np.array(thresh_data["thresh"])
        thres_of_z = interp1d(
            z_centers, thresholds, fill_value="extrapolate", bounds_error=False
        )
        threshold_all = thres_of_z(data[redshift_col].values)

        # --- apply selection: keep objects above the threshold ---
        mask = data[threshold_col].values > threshold_all

        self.log.info(
            f"SpecSelection_DESI_Phy ({self.config.desi_type}): "
            f"selected {mask.sum()} / {len(mask)} objects "
            f"using column '{threshold_col}'."
        )

        return mask.astype(int)
