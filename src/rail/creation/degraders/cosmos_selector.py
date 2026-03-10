"""Add a bias to redshift using model parameters loaded from a CSV file."""


from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
from ceci.config import StageParameter as Param
from scipy.stats import jf_skew_t

from rail.creation.selector import Selector

class COSMOSSelector(Selector):
    """Add a column of random numbers to a dataframe"""

    name = "COSMOSSelector"
    entrypoint_function = "__call__"  # the user-facing science function for this class
    interactive_function = "COSMOSSelector"
    config_options = Selector.config_options.copy()
    config_options.update(
        col_name=Param(
            str, "photoz_COSMOS", msg="Name of the column to make with mock photometric redshifts"
        ),
        col_name_mag_i=Param(
            str, "mag_i", msg="Name of the i-band magnitude column"
        ),
        col_name_z=Param(
            str, "z", msg="Name of the (true) redshift column"
        ),
        model_params_path=Param(
            str, "cosmos.csv", msg="Path to CSV file with model parameters for Gaussian core and skew-t tail distribution components"
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

    def _addNoise(self) -> None:  # pragma: no cover
        self._addNoiseCOSMOS()

    def _select(self) -> None:  # pragma: no cover
        # for this selector, we currently don't actually select any rows
        data = self.get_data("input")
        selection_mask = np.ones(len(data), dtype=bool)

        # for the COSMOS selector, emulate photo-z's
        self._initNoiseModel()
        self._addNoise()
        return selection_mask
    
    def COSMOSSelector(self, sample: Any, seed: int | None = None, **kwargs: Any):
        return self.__call__(sample, seed=seed, **kwargs)

    def _resolve_model_path(self) -> Path:
        path = Path(self.config.model_params_path).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"photo-z bias model parameter file not found: {self.config.model_params_path}")

    def _load_parametric_model(self) -> None:
        ''' load model parameters from a csv file and convert to a dictionary '''
        path = self._resolve_model_path()
        df = pd.read_csv(path)

        core = df[df["component"] == "gaussian_core"].copy()
        tail = df[df["component"] == "skew_student_t_tail"].copy()

        # Reconstruct bin edges and ordered bin-pairs
        i_edges = np.unique(np.r_[core["i_bin_lo"].to_numpy(), core["i_bin_hi"].to_numpy()])
        z_edges = np.unique(np.r_[core["z_bin_lo"].to_numpy(), core["z_bin_hi"].to_numpy()])
        i_pairs = list(zip(i_edges[:-1], i_edges[1:]))
        z_pairs = list(zip(z_edges[:-1], z_edges[1:]))

        core["i_pair"] = list(zip(core["i_bin_lo"], core["i_bin_hi"]))
        core["z_pair"] = list(zip(core["z_bin_lo"], core["z_bin_hi"]))

        # Core component: make 2D lookup tables for parameters (ordered by i_pairs x z_pairs)
        med_tbl = (core.pivot_table(index="i_pair", columns="z_pair", values="mu", aggfunc="first")
                        .reindex(index=i_pairs, columns=z_pairs).to_numpy())
        std_tbl = (core.pivot_table(index="i_pair", columns="z_pair", values="sigma", aggfunc="first")
                        .reindex(index=i_pairs, columns=z_pairs).to_numpy())

        # Tail component: make parameter arrays per i-bin
        n_i = len(i_pairs)
        f_tail  = np.zeros(n_i)
        t_loc   = np.zeros(n_i)
        t_scale = np.full(n_i, 0.2)
        t_a     = np.full(n_i, 3.0)
        t_b     = np.full(n_i, 3.0)

        tail["i_pair"] = list(zip(tail["i_bin_lo"], tail["i_bin_hi"]))
        i_pair_index = {p: i for i, p in enumerate(i_pairs)}
        for _, r in tail.iterrows():
            j = i_pair_index.get(r["i_pair"])
            if j is None: 
                continue
            f_tail[j]  = float(r["f_tail"])
            t_loc[j]   = float(r["tail_loc"])
            t_scale[j] = float(r["tail_scale"])
            t_a[j]     = float(r["tail_a"])
            t_b[j]     = float(r["tail_b"])

        self._cosmos_model = {
            "mag_i_bin_edges": i_edges,
            "z_bin_edges":     z_edges,
            "bias_median_lookup_table": med_tbl,
            "bias_std_lookup_table":    std_tbl,
            "f_tail_by_mag_i":          f_tail,
            "tail_loc_by_mag_i":        t_loc,
            "tail_scale_by_mag_i":      t_scale,
            "tail_a_by_mag_i":          t_a,
            "tail_b_by_mag_i":          t_b,
        }

    def _sample_parametric_bias_model(
        self,
        data_i: np.ndarray,
        data_z: np.ndarray,
        target_mask: np.ndarray,
    ) -> np.ndarray:
        """Sample bias using a csv-configured Gaussian core + skewed Student-t tail."""
        z_bias_samples = np.zeros_like(data_z)
        if not np.any(target_mask):
            return z_bias_samples

        self._load_parametric_model()
        model = self._cosmos_model

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
        target_mean_bias_component1 = model["bias_median_lookup_table"][target_mag_i_bin, target_z_bin]
        target_std_bias_component1 = model["bias_std_lookup_table"][target_mag_i_bin, target_z_bin]
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

    def _addNoiseCOSMOS(self) -> None:  # pragma: no cover
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