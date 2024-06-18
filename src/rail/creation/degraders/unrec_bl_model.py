"""Model for Creating Unrecognized Blends"""

from ceci.config import StageParameter as Param
from rail.creation.degrader import Degrader
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

    def __init__(self, args, comm=None):
        """
        Constructor

        Does standard Degrader initialization and sets up the bl model.
        """
        Degrader.__init__(self, args, comm=comm)

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

        return matchData

    def __merge_bl__(self, data):

        """Merge sources within a group into unrecognized blends."""
        
        group_id = data['group_id']
        unique_id = np.unique(group_id)

        ra_label, dec_label = self.config.ra_label, self.config.dec_label

        cols = list(data.columns)
        ra_ind = cols.index(ra_label)
        dec_ind = cols.index(dec_label)
        bands_ind = {b:cols.index(b) for b in self.config.bands}

        N_rows = len(unique_id)
        N_cols = len(cols)

        mergeData = np.zeros((N_rows, N_cols))
        
        for i, id in enumerate(unique_id):

            this_group = data.query(f'group_id=={id}')

            ## take the average position for the blended source
            mergeData[i, ra_ind] = this_group[ra_label].mean()
            mergeData[i, dec_ind] = this_group[dec_label].mean()

            ## sum up the fluxes into the blended source
            for b in self.config.bands:
                  mergeData[i, bands_ind[b]] = -2.5*np.log10(np.sum(10**(-this_group[b]/2.5)))

        mergeData[:,-1] = unique_id
        mergeData_df = pd.DataFrame(data=mergeData, columns=cols)
        mergeData_df['group_id'] = mergeData_df['group_id'].astype(int)

        return mergeData_df

    def run(self):
        """Return pandas DataFrame with blending errors."""

        # Load the input catalog
        data = self.get_data("input")

        # Match for close-by objects
        matchData = self.__match_bl__(data)

        # Merge matched objects into unrec-bl
        blData = self.__merge_bl__(matchData)

        # Return the new catalog
        self.add_data("output", blData)
