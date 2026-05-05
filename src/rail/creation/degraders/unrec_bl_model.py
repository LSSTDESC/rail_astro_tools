"""Model for Creating Unrecognized Blends"""

import FoFCatalogMatching
import numpy as np
import pandas as pd
import healpy

from ceci.config import StageParameter as Param
from rail.core.common_params import SHARED_PARAMS
from rail.core.data import PqHandle
from rail.creation.degrader import Degrader

#lsst_zp_dict = {"u": 12.65, "g": 14.69, "r": 14.56, "i": 14.38, "z": 13.99, "y": 13.02}

# use arbitrary zero point
# we work independently in each band, so the zero point are not important
ZERO_POINT = 14.0


class UnrecBlModel(Degrader):
    """Model for Creating Unrecognized Blends.

    Finding objects nearby each other. Merge them into one blended
    Use Friends of Friends for matching. May implement shape matching in the future.
    Take avergaged Ra and Dec for blended source, and sum up fluxes in each band. May
    implement merged shapes in the future.

    Requires gcc, which depending on your installation, may be difficult for the caller
    (FoFCatalogMatching dependency fast3tree) to find. Conda-installed gcc seems to fix this.
    """

    name = "UnrecBlModel"
    entrypoint_function = "__call__"  # the user-facing science function for this class
    interactive_function = "unrec_bl_model"
    config_options = Degrader.config_options.copy()
    config_options.update(
        ra_label=Param(str, "ra", msg="ra column name"),
        dec_label=Param(str, "dec", msg="dec column name"),
        linking_lengths=Param(float, 1.0, msg="linking_lengths for FoF matching"),
        hpx_nside=Param(int, 128, "Healpix nside to use for parallelization"),
        bands=SHARED_PARAMS,
        ref_band=SHARED_PARAMS,
        redshift_col=SHARED_PARAMS,
        match_size=Param(bool, False, msg="consider object size for finding blends"),
        match_shape=Param(bool, False, msg="consider object shape for finding blends"),
        obj_size=Param(str, "obj_size", msg="object size column name"),
        a=Param(str, "semi_major", msg="semi major axis column name"),
        b=Param(str, "semi_minor", msg="semi minor axis column name"),
        theta=Param(str, "orientation", msg="orientation angle column name"),
    )

    outputs = [("output", PqHandle), ("compInd", PqHandle)]

    blend_info_cols = [
        "group_id",
        "n_obj",
        "brightest_flux",
        "total_flux",
        "z_brightest",
        "z_weighted",
        "z_mean",
        "z_stdev",
    ]

    def __call__(self, sample, seed: int = None, **kwargs) -> dict[str, PqHandle]:
        """The main interface method for ``Degrader``.

        Applies degradation.

        This will attach the sample to this `Degrader` (for introspection and
        provenance tracking).

        Then it will call the run() and finalize() methods, which need to be
        implemented by the sub-classes.

        The run() method will need to register the data that it creates to this
        Estimator by using ``self.add_data('output', output_data)``.

        Finally, this will return a PqHandle providing access to that output
        data.

        Parameters
        ----------
        sample : table-like
            The sample to be degraded
        seed : int, default=None
            An integer to set the numpy random seed

        Returns
        -------
        dict[str, PqHandle]
            A handle giving access to a table with degraded sample
        """
        if seed is not None:  # pragma: no cover
            self.config.seed = seed

        self.set_data("input", sample)
        self.run()
        self.finalize()

        return {
            "output": self.get_handle("output"),
            "compInd": self.get_handle("compInd"),
        }

    def __match_bl__(self, data):
        """Group sources with friends of friends"""

        ra_label, dec_label = self.config.ra_label, self.config.dec_label
        linking_lengths = self.config.linking_lengths

        results = FoFCatalogMatching.match(
            {"truth": data},
            linking_lengths=linking_lengths,
            ra_label=ra_label,
            dec_label=dec_label,
        )
        results.remove_column("catalog_key")
        results = results.to_pandas(index="row_index")
        results.sort_values(by="row_index", inplace=True)

        data['row_index'] = np.arange(len(data))
        
        ## adding the group id as the last column to data
        match_data = pd.merge(data, results, left_on='row_index', right_index=True)            
        return match_data, results

    def __merge_bl__(self, data: pd.DataFrame, which_pix: int):
        """Merge sources within a group into unrecognized blends."""
    
        # Filter to groups that contain at least one object in which_pix
        groups_in_pix = data[data['hpx_idx'] == which_pix]['group_id'].unique()
        if len(groups_in_pix) == 0:
            return pd.DataFrame(columns=self._get_merge_columns())
    
        data = data[data['group_id'].isin(groups_in_pix)].copy()
    
        ra_label, dec_label = self.config.ra_label, self.config.dec_label
        cols = self._get_merge_columns()
    
        # Pre-compute all fluxes at once
        for b in self.config.bands:
            data[f'_flux_{b}'] = 10 ** (-(data[b].values - ZERO_POINT) / 2.5)
    
        ref_band = self.config.ref_band
    
        # Use groupby for vectorized operations
        grouped = data.groupby('group_id', sort=False)
    
        # Vectorized aggregations
        agg_dict = {
            ra_label: 'mean',
            dec_label: 'mean',
            self.config.redshift_col: ['mean', 'std'],
            f'_flux_{ref_band}': ['sum', 'max'],
        }
    
        result = grouped.agg(agg_dict)
        result.columns = [
            '_'.join(col).strip('_') if isinstance(col, tuple) else col 
            for col in result.columns
        ]
    
        # Get brightest object info for each group (for hpx_idx and redshift)
        brightest_indices = grouped[f'_flux_{ref_band}'].idxmax()
        brightest_data = data.loc[brightest_indices, ['hpx_idx', self.config.redshift_col]]
        brightest_data.index = brightest_indices.index  # Match group_id index
    
        # Filter to only groups where brightest is in which_pix
        valid_groups = brightest_data[brightest_data['hpx_idx'] == which_pix].index
    
        # Apply filter to all results
        result = result.loc[valid_groups]
        brightest_data = brightest_data.loc[valid_groups]
    
        if len(result) == 0:
            return pd.DataFrame(columns=cols)
    
        # Calculate summed magnitudes for each band
        for b in self.config.bands:
            result[b] = grouped[f'_flux_{b}'].sum().loc[valid_groups].apply(
                lambda flux_sum: -2.5 * np.log10(flux_sum) + ZERO_POINT
            )
    
        # Calculate weighted redshift (only for valid groups)
        def calc_weighted_z(g):
            return np.sum(g[self.config.redshift_col].values * g[f'_flux_{ref_band}'].values) / np.sum(g[f'_flux_{ref_band}'].values)
    
        weighted_z = grouped.apply(calc_weighted_z, include_groups=False).loc[valid_groups]
    
        # Get group sizes
        group_sizes = grouped.size().loc[valid_groups]
    
        # Build final dataframe
        mergeData_df = pd.DataFrame(index=valid_groups)
        mergeData_df[ra_label] = result[f'{ra_label}_mean']
        mergeData_df[dec_label] = result[f'{dec_label}_mean']
        mergeData_df['hpx_idx'] = which_pix
    
        # Add band columns
        for b in self.config.bands:
            mergeData_df[b] = result[b]
    
        mergeData_df[self.config.redshift_col] = brightest_data[self.config.redshift_col]
        mergeData_df['group_id'] = valid_groups.astype(int)
        mergeData_df['n_obj'] = group_sizes.astype(int)
        mergeData_df['brightest_flux'] = result[f'flux_{ref_band}_max']
        mergeData_df['total_flux'] = result[f'flux_{ref_band}_sum']
        mergeData_df['z_brightest'] = brightest_data[self.config.redshift_col]
        mergeData_df['z_mean'] = result[f'{self.config.redshift_col}_mean']
        mergeData_df['z_weighted'] = weighted_z
        mergeData_df['z_stdev'] = result[f'{self.config.redshift_col}_std'].fillna(0.0)
    
        # Ensure correct column order and reset index
        mergeData_df = mergeData_df[cols].reset_index(drop=True)
    
        return mergeData_df

    def _get_merge_columns(self):
        """Helper to get column list."""
        ra_label, dec_label = self.config.ra_label, self.config.dec_label
        return (
            [ra_label, dec_label, 'hpx_idx']
            + list(self.config.bands)
            + [self.config.redshift_col]
            + self.blend_info_cols
        )
    
    def run(self):
        """Return pandas DataFrame with blending errors."""

        # Load the input catalog
        data = self.get_data("input")

        ra_label, dec_label = self.config.ra_label, self.config.dec_label

        hpx_idx = healpy.pixelfunc.ang2pix(
            self.config.hpx_nside,
            data[dec_label],
            data[ra_label],
            lonlat=True,
        )

        idx_list = np.sort(np.unique(hpx_idx))

        match_list = []
        results_list = []

        for which_pix in idx_list:            
            mask = hpx_idx == which_pix
            all_neighbours = healpy.pixelfunc.get_all_neighbours(self.config.hpx_nside, which_pix)
            for neighbour in all_neighbours:
                mask = np.bitwise_or(mask, hpx_idx == neighbour)

            sub_data = data[mask]
            central_mask = hpx_idx[mask] == which_pix
            
            # Match for close-by objects
            matchData, compInd = self.__match_bl__(sub_data)
            matchData['hpx_idx'] = hpx_idx[mask]
            
            # Merge matched objects into unrec-bl
            blData = self.__merge_bl__(matchData, which_pix)
            blData = blData[blData['hpx_idx'] == which_pix]

            compInd = compInd[central_mask.to_numpy()]
            compInd['hpx_idx'] = which_pix
            
            results_list.append(blData)
            match_list.append(compInd)
            
        blData_all = pd.concat(results_list)
        compInd_all = pd.concat(match_list)
        
        # Return the new catalog and component index in original catalog
        self.add_data("output", blData_all)
        self.add_data("compInd", compInd_all)
