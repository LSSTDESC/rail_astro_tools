"""Degraders that emulate spectroscopic effects on photometry"""

import numpy as np
import pandas as pd
from ceci.config import StageParameter as Param
from rail.creation.selector import Selector
from rail.creation.noisifier import Noisifier



class LineConfusion(Noisifier):
    """Degrader that simulates emission line confusion.

    .. code-block:: python

       degrader = LineConfusion(true_wavelen=3727,
                                wrong_wavelen=5007,
                                frac_wrong=0.05)

    is a degrader that misidentifies 5% of OII lines (at 3727 angstroms)
    as OIII lines (at 5007 angstroms), which results in a larger
    spectroscopic redshift.

    Note that when selecting the galaxies for which the lines are confused,
    the degrader ignores galaxies for which this line confusion would result
    in a negative redshift, which can occur for low redshift galaxies when
    wrong_wavelen < true_wavelen.

    """

    name = 'LineConfusion'
    config_options = Noisifier.config_options.copy()
    config_options.update(
        true_wavelen=Param(float, required=True, msg="wavelength of the true emission line"),
        wrong_wavelen=Param(float, required=True, msg="wavelength of the wrong emission line"),
        frac_wrong=Param(float, required=True, msg="fraction of galaxies with confused emission lines"),
    )

    def __init__(self, args, **kwargs):
        """
        """
        super().__init__(args, **kwargs)
        # validate parameters
        if self.config.true_wavelen < 0:  # pragma: no cover
            raise ValueError("true_wavelen must be positive, not {self.config.true_wavelen}")
        if self.config.wrong_wavelen < 0:  # pragma: no cover
            raise ValueError("wrong_wavelen must be positive, not {self.config.wrong_wavelen}")
        if self.config.frac_wrong < 0 or self.config.frac_wrong > 1:  # pragma: no cover
            raise ValueError("frac_wrong must be between 0 and 1., not {self.config.wrong_wavelen}")
            
            
    def _initNoiseModel(self):
        self.rng = np.random.default_rng(self.config.seed)

    def _addNoise(self):
        """ Run method

        Applies line confusion

        Notes
        -----
        Get the input data from the data store under this stages 'input' tag
        Puts the data into the data store under this stages 'output' tag
        """
        data = self.get_data('input')

        # convert to an array for easy manipulation
        values, columns = data.values.copy(), data.columns.copy()

        # get the minimum redshift
        # if wrong_wavelen < true_wavelen, this is minimum the redshift for
        # which the confused redshift is still positive
        zmin = self.config.wrong_wavelen / self.config.true_wavelen - 1

        # select the random fraction of galaxies whose lines are confused
        idx = self.rng.choice(
            np.where(values[:, 0] > zmin)[0],
            size=int(self.config.frac_wrong * values.shape[0]),
            replace=False,
        )

        # transform these redshifts
        values[idx, 0] = (
            1 + values[idx, 0]
        ) * self.config.true_wavelen / self.config.wrong_wavelen - 1

        # return results in a data frame
        outData = pd.DataFrame(values, columns=columns)
        self.add_data('output', outData)


class InvRedshiftIncompleteness(Selector):
    """Degrader that simulates incompleteness with a selection function
    inversely proportional to redshift.

    The survival probability of this selection function is
    p(z) = min(1, z_p/z),
    where z_p is the pivot redshift.

    """

    name = 'InvRedshiftIncompleteness'
    config_options = Selector.config_options.copy()
    config_options.update(
        pivot_redshift=Param(float, required=True, msg="redshift at which the incompleteness begins"),
    )    

    def __init__(self, args, **kwargs):
        """
        """
        super().__init__(args, **kwargs)
        if self.config.pivot_redshift < 0:
            raise ValueError("pivot redshift must be positive, not {self.config.pivot_redshift}")

    def _select(self):
        """ Run method

        Applies incompleteness

        Notes
        -----
        Get the input data from the data store under this stages 'input' tag
        Puts the data into the data store under this stages 'output' tag
        """
        data = self.get_data('input')

        # calculate survival probability for each galaxy
        survival_prob = np.clip(self.config.pivot_redshift / data["redshift"], 0, 1)

        # probabalistically drop galaxies from the data set
        rng = np.random.default_rng(self.config.seed)
        mask = rng.random(size=data.shape[0]) <= survival_prob

        return mask
