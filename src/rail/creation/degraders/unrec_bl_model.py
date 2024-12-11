"""Model for Creating Unrecognized Blends"""

from ceci.config import StageParameter as Param
from rail.creation.degrader import Degrader
from rail.core.data import PqHandle
from rail.core.common_params import SHARED_PARAMS
import numpy as np, pandas as pd
import FoFCatalogMatching


lsst_zp_dict = {'u':12.65, 'g':14.69, 'r':14.56, 'i': 14.38, 'z':13.99, 'y': 13.02}

class UnrecBlModel(Degrader):
    """Model for Creating Unrecognized Blends.

    Finding objects nearby each other. Merge them into one blended
    Use Friends of Friends for matching. May implement shape matching in the future.
    Take avergaged Ra and Dec for blended source, and sum up fluxes in each band. May implement merged shapes in the future.

    """
    name = "UnrecBlModel"
    config_options = Degrader.config_options.copy()
    config_options.update(ra_label=Param(str, 'ra', msg='ra column name'),
                          dec_label=Param(str, 'dec', msg='dec column name'),
                          linking_lengths=Param(float, 1.0, msg='linking_lengths for FoF matching'),
                          bands=SHARED_PARAMS,
                          zp_dict=Param(dict, lsst_zp_dict, msg='magnitude zeropoints dictionary'),
                          ref_band=SHARED_PARAMS,
                          redshift_col=SHARED_PARAMS,
                          match_size=Param(bool, False, msg='consider object size for finding blends'),
                          match_shape=Param(bool, False, msg='consider object shape for finding blends'),
                          obj_size=Param(str, 'obj_size', msg='object size column name'),
                          a=Param(str, 'semi_major', msg='semi major axis column name'),
                          b=Param(str, 'semi_minor', msg='semi minor axis column name'),
                          theta=Param(str, 'orientation', msg='orientation angle column name'))

    outputs = [("output", PqHandle), ("compInd", PqHandle)]

    blend_info_cols = ['group_id', 'n_obj', 'brightest_flux', 'total_flux', 'z_brightest', 'z_weighted', 'z_mean', 'z_stdev']

    def __call__(self, sample, seed: int = None):
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
        output_data : PqHandle
            A handle giving access to a table with degraded sample
        """
        if seed is not None:
            self.config.seed = seed

        self.set_data("input", sample)
        self.run()
        self.finalize()

        return {'output':self.get_handle("output"), 'compInd':self.get_handle("compInd")}

    def __match_bl__(self, data):

        """Group sources with friends of friends"""

        ra_label, dec_label = self.config.ra_label, self.config.dec_label
        linking_lengths = self.config.linking_lengths

        results = FoFCatalogMatching.match({'truth': data}, linking_lengths=linking_lengths, ra_label=ra_label, dec_label=dec_label)
        results.remove_column('catalog_key')

        results = results.to_pandas(index='row_index')
        results.sort_values(by='row_index', inplace=True)

        ## adding the group id as the last column to data
        matchData = pd.merge(data, results, left_index=True, right_index=True)

        return matchData, results

    def __merge_bl__(self, data):

        """Merge sources within a group into unrecognized blends."""

        group_id = data['group_id']
        unique_id = np.unique(group_id)

        ra_label, dec_label = self.config.ra_label, self.config.dec_label
        cols = [ra_label, dec_label] + [b for b in self.config.bands] + [self.config.redshift_col] + self.blend_info_cols

        N_rows = len(unique_id)
        N_cols = len(cols)

        # compute the fluxes once for all the galaxies
        fluxes = {b:10**(-(data[b] - self.config.zp_dict[b])/2.5) for b in self.config.bands}

        # pull the column indices
        idx_ra = cols.index(ra_label)
        idx_dec = cols.index(dec_label)
        idx_redshift = cols.index(self.config.redshift_col)
        idx_n_obj = cols.index('n_obj')
        idx_brightest_flux = cols.index('brightest_flux')
        idx_total_flux = cols.index('total_flux')
        idx_z_brightest = cols.index('z_brightest')
        idx_z_mean = cols.index('z_mean')
        idx_z_weighted = cols.index('z_weighted')
        idx_z_stdev = cols.index('z_stdev')

        mergeData = np.zeros((N_rows, N_cols))
        for i, id in enumerate(unique_id):

            # Get the mask for this grouping
            mask = data['group_id'] == id

            # Get the data and fluxes for this grouping
            this_group = data[mask]
            these_fluxes = {b:fluxes[b][mask] for b in self.config.bands}

            # Pull put some useful stuff
            n_obj = len(this_group)
            ref_fluxes = these_fluxes[self.config.ref_band]
            these_redshifts = this_group[self.config.redshift_col]

            ## take the average position for the blended source
            mergeData[i, idx_ra] = this_group[ra_label].mean()
            mergeData[i, idx_dec] = this_group[dec_label].mean()

            ## sum up the fluxes into the blended source
            for b in self.config.bands:
                mergeData[i, cols.index(b)] = -2.5*np.log10(np.sum(these_fluxes[b])) + self.config.zp_dict[b]

            brighest_idx = np.argmax(ref_fluxes)
            redshifts = these_redshifts.iloc[brighest_idx]

            mergeData[i, idx_redshift] = redshifts
            mergeData[i, idx_n_obj] = n_obj
            mergeData[i, idx_brightest_flux] = ref_fluxes.max()
            mergeData[i, idx_total_flux] = np.sum(ref_fluxes)
            mergeData[i, idx_z_brightest] = redshifts
            mergeData[i, idx_z_mean] = np.mean(these_redshifts)
            mergeData[i, idx_z_weighted] = np.sum(these_redshifts*ref_fluxes)/np.sum(ref_fluxes)
            if n_obj > 1:
                mergeData[i, idx_z_stdev] = np.std(these_redshifts)
            else:
                mergeData[i, idx_z_stdev] = 0.

        mergeData[:,cols.index('group_id')] = unique_id
        mergeData_df = pd.DataFrame(data=mergeData, columns=cols)
        mergeData_df['group_id'] = mergeData_df['group_id'].astype(int)
        mergeData_df['n_obj'] = mergeData_df['n_obj'].astype(int)

        return mergeData_df

    def run(self):
        """Return pandas DataFrame with blending errors."""

        # Load the input catalog
        data = self.get_data("input")

        # Match for close-by objects
        matchData, compInd = self.__match_bl__(data)

        # Merge matched objects into unrec-bl
        blData = self.__merge_bl__(matchData)

        # Return the new catalog and component index in original catalog
        self.add_data("output", blData)
        self.add_data("compInd", compInd)
