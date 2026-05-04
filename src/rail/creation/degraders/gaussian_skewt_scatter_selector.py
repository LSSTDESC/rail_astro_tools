"""Add a bias to redshift using a Gaussian core + skewed Student-t tail error model."""


from typing import Any

import numpy as np
from ceci.config import StageParameter as Param
from scipy.stats import jf_skew_t

from rail.creation.selector import Selector

default_selector_model_dict = dict(mag_i_bin_edges = [15.5, 22. , 23. , 24. , 29. ],
                                   z_bin_edges = [0. , 0.3, 0.7, 1. , 1.5, 2. , 2.5, 3. , 4. ],
                                   bias_median_lookup_table = [[ 0.   ,  0.002, -0.002,  0.001,  0.001,  0.001,  0.001,  0.001],
                                           [ 0.   , -0.   , -0.002, -0.004,  0.01 ,  0.01 ,  0.01 ,  0.01 ],
                                           [ 0.003, -0.   , -0.001, -0.006, -0.002,  0.024,  0.007,  0.   ],
                                           [ 0.008, -0.005,  0.007, -0.015, -0.019,  0.017,  0.011,  0.   ]],
                                   bias_std_lookup_table = [[0.01 , 0.02 , 0.026, 0.038, 0.038, 0.038, 0.038, 0.038],
                                           [0.011, 0.019, 0.025, 0.036, 0.062, 0.062, 0.062, 0.062],
                                           [0.011, 0.022, 0.027, 0.044, 0.063, 0.115, 0.093, 0.074],
                                           [0.013, 0.023, 0.025, 0.051, 0.069, 0.12 , 0.103, 0.069]],
                                   f_tail_by_mag_i = [0.088 , 0.1377, 0.4312, 0.4312  ],
                                   tail_loc_by_mag_i = [-0.0055,  0.1568,  0.2   , 0.2  ],
                                   tail_scale_by_mag_i = [0.2041, 0.3522, 0.237 , 0.237  ],
                                   tail_a_by_mag_i = [ 3.7662, 10.1149,  2.    ,  2.    ],
                                   tail_b_by_mag_i = [ 4.    , 11.2095,  4.    ,  4.    ])

class GaussianSkewtScatterSelector(Selector):
    """Add a mock photometric redshift column to a dataframe with a Gaussian + skew Student-t error model"""

    name = "GaussianSkewtScatterSelector"
    entrypoint_function = "__call__"  # the user-facing science function for this class
    interactive_function = "GaussianSkewtScatterSelector"
    config_options = Selector.config_options.copy()
    config_options.update(
        col_name=Param(
            str, "photoz_mock", msg="Name of the mock photometric redshift column to make"
        ),
        col_name_mag_i=Param(
            str, "mag_i", msg="Name of the i-band magnitude column"
        ),
        col_name_z=Param(
            str, "z", msg="Name of the (true) redshift column"
        ),
        selector_model_dict=Param(
            dict, default_selector_model_dict, msg="Dictionary of model parameters for Gaussian core and skew-t tail distribution components"
        ),
    )

    def __init__(self, args: Any, **kwargs: Any) -> None:
        """
        Constructor
        Does standard Selector initialization
        """
        Selector.__init__(self, args, **kwargs)

    def _initNoiseModel(self) -> None:  # pragma: no cover
        self._rng = np.random.default_rng(self.config.seed)
        self._model = self.config.selector_model_dict

    def _addNoise(self) -> None:  # pragma: no cover
        self._addNoiseGaussianSkewtScatter()

    def _select(self) -> None:  # pragma: no cover
        # for this selector, we currently don't actually select any rows
        data = self.get_data("input")
        selection_mask = np.ones(len(data), dtype=bool)

        # for the GaussianSkewtScatter selector, emulate photo-z's
        self._initNoiseModel()
        self._addNoise()
        return selection_mask
    
    def GaussianSkewtScatterSelector(self, sample: Any, seed: int | None = None, **kwargs: Any):
        return self.__call__(sample, seed=seed, **kwargs)

    def _sample_parametric_bias_model(
        self,
        data_i: np.ndarray,
        data_z: np.ndarray,
        target_mask: np.ndarray,
    ) -> np.ndarray:
        """Sample bias using a Gaussian core + skewed Student-t tail."""
        z_bias_samples = np.zeros_like(data_z)
        if not np.any(target_mask):
            return z_bias_samples

        
        model = {k:np.array(v) for k, v in self.config.selector_model_dict.items()}

        n_target = int(np.sum(target_mask))
        target_i = data_i[target_mask]
        target_z = data_z[target_mask]

        # determine mag_i and z bin for each target galaxy
        # if the target falls outside the bin edges, assign it to the nearest bin
        target_mag_i_bin = np.digitize(target_i, model["mag_i_bin_edges"]) - 1
        target_z_bin = np.digitize(target_z, model["z_bin_edges"]) - 1
        target_mag_i_bin = np.clip(target_mag_i_bin, 0, model["bias_median_lookup_table"].shape[0] - 1)
        target_z_bin = np.clip(target_z_bin, 0, model["bias_median_lookup_table"].shape[1] - 1)
        # set component parameters for each target
        target_mean_bias_component1 = np.array(model["bias_median_lookup_table"])[target_mag_i_bin, target_z_bin]
        target_std_bias_component1 = np.array(model["bias_std_lookup_table"])[target_mag_i_bin, target_z_bin]
        target_tail_prob = model["f_tail_by_mag_i"][target_mag_i_bin]

        # Monte Carlo sampling to determine which component each galaxy belongs to.
        u = self._rng.random(n_target)
        is_tail = u < target_tail_prob
        is_core = ~is_tail

        # generate redshift bias for each galaxy based on its assigned component
        target_bias = np.zeros(n_target)
        # core component (Gaussian from Yin+25)
        target_bias[is_core] = self._rng.normal(
            loc=target_mean_bias_component1[is_core],
            scale=target_std_bias_component1[is_core],
        )
        # tail component (skewed Student-t)
        if np.any(is_tail):
            idx_tail = target_mag_i_bin[is_tail]
            tail_samples = np.empty(np.sum(is_tail))
            # iterate over unique mag_i bins
            for idx_ in np.unique(idx_tail):
                in_mag = idx_tail == idx_
                tail_samples[in_mag] = jf_skew_t.rvs(
                    a=model["tail_a_by_mag_i"][idx_],
                    b=model["tail_b_by_mag_i"][idx_],
                    loc=model["tail_loc_by_mag_i"][idx_],
                    scale=model["tail_scale_by_mag_i"][idx_],
                    size=int(np.sum(in_mag)),
                    random_state=self._rng,
                )
            target_bias[is_tail] = tail_samples
        z_bias_samples[target_mask] = target_bias
        return z_bias_samples

    def _addNoiseGaussianSkewtScatter(self) -> None:  # pragma: no cover
        data = self.get_data("input")
        data_i = np.asarray(data[self.config.col_name_mag_i])
        data_z = np.asarray(data[self.config.col_name_z])

        valid_target_mask = np.isfinite(data_i) & np.isfinite(data_z) & (data_z > 0)
        z_bias_samples = self._sample_parametric_bias_model(
            data_i=data_i,
            data_z=data_z,
            target_mask=valid_target_mask,
        )
        z_noisified = data_z + z_bias_samples

        # Re-sample out-of-bounds mock redshifts up to 3 times.
        invalid_mask = valid_target_mask & ((z_noisified < 0) | (z_noisified > 6))
        for _ in range(3):
            if not np.any(invalid_mask):
                break
            retry_bias = self._sample_parametric_bias_model(
                data_i=data_i,
                data_z=data_z,
                target_mask=invalid_mask,
            )
            z_bias_samples[invalid_mask] = retry_bias[invalid_mask]
            z_noisified = data_z + z_bias_samples
            invalid_mask = valid_target_mask & ((z_noisified < 0) | (z_noisified > 6))

        # Final clip after retries.
        z_noisified = np.clip(z_noisified, 0, 6)

        data[self.config.col_name] = z_noisified
        return
