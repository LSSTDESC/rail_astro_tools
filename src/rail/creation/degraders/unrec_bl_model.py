"""Model for Creating Unrecognized Blends"""

from dataclasses import MISSING
from ceci.config import StageParameter as Param
from rail.creation.degrader import Degrader
from astropy.coordinates import SkyCoord
import numpy as np, pandas as pd

class UnrecBlModel(Degrader):
    """Model for Creating Unrecognized Blends.

    Finding objects nearby each other. Merge them into one blended

    """
    name = "UnrecBlModel"
    config_options = Degrader.config_options.copy()

    def __init__(self, args, comm=None):
        """
        Constructor

        Does standard Degrader initialization and sets up the error model.
        """
        Degrader.__init__(self, args, comm=comm)

    def __match_bl__(self, data):

        """Find nearest neighbor of each object within certain radius."""

        N_data = len(data)
        group_id = np.linspace(0, N_data-1, N_data, dtype='int')

        coords = SkyCoord(ra=data['ra'], dec=data['dec'], unit='deg')
        idx, d2d, d3d = coords.match_to_catalog_sky(coords, nthneighbor=2)

        for i in range(N_data):
            if d2d.arcsec[i] < self.config.radius_cut:
                group_id[idx[i]] = group_id[i]

        data['bl_group_id'] = group_id

        return data

    def __merge_bl__(self, data):

        """Merge nearest neighbors into unrecognized blends."""
        
        group_id = data['bl_group_id']
        unique_id = np.unique(group_id)

        cols = list(data.columns[:-1])   ## exluding the group id
        ra_ind = cols.index('ra')
        dec_ind = cols.index('dec')
        bands_ind = {b:cols.index(b) for b in self.config.bands}

        N_rows = len(unique_id)
        N_cols = len(cols)

        mergeData = np.zeros((N_rows, N_cols))
        
        for i, id in enumerate(unique_id):

            this_group = data.query(f'bl_group_id=={id}')

            mergeData[i, dec_ind] = this_group['dec'].mean()
            mergeData[i, ra_ind] = this_group['ra'].mean()

            for b in self.config.bands:
                  mergeData[i, bands_ind[b]] = -2.5*np.log10(np.sum(10**(-this_group[b]/2.5)))

        mergeData_df = pd.DataFrame(data=mergeData, columns=cols)
        mergeData_df["bl_group_id"] = unique_id

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
