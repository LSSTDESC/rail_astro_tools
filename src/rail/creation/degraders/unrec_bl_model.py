"""Model for Creating Unrecognized Blends"""

from ceci.config import StageParameter as Param
from rail.creation.degrader import Degrader
from rail.core.data import PqHandle
import numpy as np, pandas as pd
import FoFCatalogMatching


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
                          bands=Param(str, 'ugrizy', msg='name of filters'),
                          match_size=Param(bool, False, msg='consider object size for finding blends'),
                          match_shape=Param(bool, False, msg='consider object shape for finding blends'),
                          obj_size=Param(str, 'obj_size', msg='object size column name'),
                          a=Param(str, 'semi_major', msg='semi major axis column name'),
                          b=Param(str, 'semi_minor', msg='semi minor axis column name'),
                          theta=Param(str, 'orientation', msg='orientation angle column name'))

    outputs = [("output", PqHandle), ("compInd", PqHandle)]

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
        cols = [ra_label, dec_label] + [b for b in self.config.bands] + ['group_id']

        N_rows = len(unique_id)
        N_cols = len(cols)

        mergeData = np.zeros((N_rows, N_cols))
        for i, id in enumerate(unique_id):

            this_group = data.query(f'group_id=={id}')

            ## take the average position for the blended source
            mergeData[i, cols.index(ra_label)] = this_group[ra_label].mean()
            mergeData[i, cols.index(dec_label)] = this_group[dec_label].mean()

            ## sum up the fluxes into the blended source
            for b in self.config.bands:
                  mergeData[i, cols.index(b)] = -2.5*np.log10(np.sum(10**(-this_group[b]/2.5)))

        mergeData[:,cols.index('group_id')] = unique_id
        mergeData_df = pd.DataFrame(data=mergeData, columns=cols)
        mergeData_df['group_id'] = mergeData_df['group_id'].astype(int)

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

