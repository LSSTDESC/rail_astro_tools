"""Stages that curate a cross-matched reference-redshift catalog into a training set and a
representative test set for photometric redshift.

This is the HSC-FY preparation chain, generalised.  Three stages, and a variant of the third:

``PreTrainTestSplitter``
    Shuffle and split the cross-matched reference catalog into a pre-training and a
    pre-testing sample, applying a redshift range cut and, optionally, a quality-flag cut.
    Splitting *first* is what keeps the two branches below disjoint.

``MagRedshiftDownsampler``
    The training branch.  Bin each dominant survey on a 2-D magnitude--redshift grid and cap
    the occupancy of every bin, which flattens the label distribution so a machine-learning
    photo-z does not inherit the bright, low-redshift prior of the big spectroscopic surveys.

``SOMResampler``
    The testing branch.  Resample the pre-testing pool on a Self-Organising Map so its
    colour--magnitude distribution matches a *population* sample drawn from the shear
    catalog the photo-z will actually be applied to: every SOM cell gets a target count from
    the population occupancy, and is filled from the pool objects in it.

``SOMZCellResampler``
    ``SOMResampler`` with each cell's target also split over wide redshift bins, by the
    redshift distribution a *truth* sample (selected like the population, complete in
    redshift -- e.g. a deep field matched to COSMOS2020) has in that cell.  For pools of
    targeted spectroscopy, whose redshifts inside a colour cell are not those of the
    population.

Every survey-specific quantity -- column names, magnitude limits, per-survey caps, SOM
features -- is a configuration parameter.  The defaults are the ones that work for the Rubin
DP2 anacal shear catalog, whose photometry is ``lsst_{r,i,z}_mag_gauss2`` (riz only, already
magnitudes) and whose survey label is ``z_source``.

Reference: the HSC-Y3 photo-z analysis, sections "Test Set Curation" and "Training Set
Curation"; the 2-D downsampling follows Zhou et al. (2021), section 3.3.
"""

import time

import numpy as np
import pandas as pd
from ceci.config import StageParameter as Param

from rail.core.data import Hdf5Handle, PqHandle, TableLike
from rail.core.stage import RailStage
from rail.creation.selector import Selector

# Re-exported so that `from ...pz_prepare import as_dataframe` keeps working; the definitions
# moved to _frame_utils so the crossmatch stages can share them without importing a degrader.
from rail.creation.degraders._frame_utils import (  # noqa: F401
    as_dataframe,
    frame_to_hdf5_dict,
    to_numpy_frame,
)

# --- DP2 anacal defaults ---------------------------------------------------------------
# anacal measures r, i, z only, so the colours available are r-i and i-z.
DEFAULT_MAG_COL = "lsst_i_mag_gauss2"
DEFAULT_COLOR_COLS = ["lsst_r_mag_gauss2", "lsst_i_mag_gauss2", "lsst_z_mag_gauss2"]
DEFAULT_SURVEY_COL = "z_source"

# 40.0 is anacal's non-detect sentinel.  The replacement values are the 99.9th percentiles of
# the measured magnitude in each band, i.e. roughly the depth.
DEFAULT_NONDETECT_VAL = 40.0
DEFAULT_NONCOLOR_NONDET = [26.89]
DEFAULT_COLOR_NONDET = [26.75, 26.89, 26.66]

# The five surveys that dominate the DP2 anacal cross-match.  DESI_DR1 alone is 56% of it.
DEFAULT_CAPS = {
    "DESI_DR1": 100,
    "SDSS_DR17": 100,
    "PRIMUS": 100,
}


def _bin_cap_mask(
    mag: np.ndarray,
    redshift: np.ndarray,
    cap: int,
    d_mag: float,
    d_z: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Boolean mask keeping at most ``cap`` entries per (magnitude, redshift) cell.

    Rows with a non-finite magnitude or redshift cannot be binned; they are kept, since
    dropping them here would be a silent photometric cut rather than a downsampling.
    """
    keep = np.zeros(len(mag), dtype=bool)
    finite = np.isfinite(mag) & np.isfinite(redshift)
    keep[~finite] = True

    idx = np.where(finite)[0]
    if idx.size == 0:  # pragma: no cover
        return keep

    mag_bin = np.floor(mag[idx] / d_mag).astype(np.int64)
    z_bin = np.floor(redshift[idx] / d_z).astype(np.int64)
    order = np.lexsort((z_bin, mag_bin))
    idx, mag_bin, z_bin = idx[order], mag_bin[order], z_bin[order]

    # cell boundaries in the sorted arrays
    new_cell = np.empty(len(idx), dtype=bool)
    new_cell[0] = True
    new_cell[1:] = (mag_bin[1:] != mag_bin[:-1]) | (z_bin[1:] != z_bin[:-1])
    starts = np.where(new_cell)[0]
    ends = np.append(starts[1:], len(idx))

    for start, end in zip(starts, ends):
        members = idx[start:end]
        if len(members) > cap:
            members = rng.choice(members, size=cap, replace=False)
        keep[members] = True
    return keep


class PreTrainTestSplitter(RailStage):
    """Split a reference catalog into pre-training and pre-testing samples.

    Applies a redshift range cut and an optional quality cut, shuffles, then splits by row
    fraction.  Every input column is carried through to both outputs; a ``redshift`` column is
    added as a copy of ``redshift_col`` so downstream RAIL stages find the name they expect.

    Doing the split before any curation is what guarantees the training and testing branches
    stay disjoint.  Curating first and splitting afterwards leaks the test set into training.
    """

    name = "PreTrainTestSplitter"
    config_options = RailStage.config_options.copy()
    config_options.update(
        train_frac=Param(float, 0.8, msg="fraction of rows assigned to the pre-training set"),
        shuffle=Param(bool, True, msg="shuffle before splitting"),
        seed=Param(int, 1337, msg="random seed for the shuffle"),
        redshift_col=Param(
            str, "z", msg="name of the redshift column in the input catalog"
        ),
        output_redshift_col=Param(
            str,
            "redshift",
            msg="a copy of redshift_col is written under this name for downstream RAIL "
            "stages; set empty to skip",
        ),
        redshift_min=Param(float, 0.001, msg="keep redshift > this"),
        redshift_max=Param(float, 7.0, msg="keep redshift < this"),
        apply_quality_cut=Param(
            bool, False, msg="whether to apply the quality-flag cut at all"
        ),
        quality_col=Param(str, "z_flag", msg="name of the quality-flag column"),
        quality_min=Param(float, 3.0, msg="keep quality_col >= this"),
        group_col=Param(
            str,
            "",
            msg="if set, rows sharing a value in this column are kept together on the same "
            "side of the split. Use it when one label can attach to several rows -- e.g. a "
            "reference redshift matched to two blended detections -- which would otherwise "
            "put the same truth value in both training and testing. Empty disables grouping",
        ),
        quality_nan_policy=Param(
            str,
            "keep",
            msg="'keep' or 'drop': what to do with rows whose quality flag is null. "
            "'keep' is the default because a catalog assembled from several compilations "
            "will often have a flag for only some of them",
        ),
        require_detected=Param(
            list,
            [],
            msg="magnitude columns that must hold a detection -- finite and not "
            "nondetect_val; a row failing any is dropped before the split, from both sides",
        ),
        nondetect_val=Param(
            float, np.nan, msg="non-detection sentinel for require_detected (NaN: none)"
        ),
    )

    inputs = [("input", PqHandle)]
    outputs = [("output_pretrain", PqHandle), ("output_pretest", PqHandle)]

    def __call__(self, sample: TableLike, **kwargs) -> tuple[PqHandle, PqHandle]:
        """Split ``sample`` and return handles to the pre-training and pre-testing tables."""
        self.set_data("input", sample)
        self.run()
        self.finalize()
        return self.get_handle("output_pretrain"), self.get_handle("output_pretest")

    def run(self) -> None:
        data = as_dataframe(self.get_data("input"))
        n_input = len(data)

        z_col = self.config.redshift_col
        if z_col not in data.columns:
            raise KeyError(
                f"redshift column '{z_col}' not in input; available: {list(data.columns)[:20]}..."
            )
        redshift = pd.to_numeric(data[z_col], errors="coerce").astype("float64")
        keep = (redshift > self.config.redshift_min) & (redshift < self.config.redshift_max)
        self.log_msg(
            f"redshift cut {self.config.redshift_min} < {z_col} < {self.config.redshift_max}: "
            f"{n_input:,} -> {int(keep.sum()):,}"
        )

        if self.config.apply_quality_cut:
            q_col = self.config.quality_col
            if q_col not in data.columns:
                raise KeyError(f"quality column '{q_col}' not in input")
            quality = pd.to_numeric(data[q_col], errors="coerce").astype("float64")
            good = quality >= self.config.quality_min
            n_null = int(quality.isna().sum())
            if self.config.quality_nan_policy == "keep":
                good = good | quality.isna()
            elif self.config.quality_nan_policy != "drop":
                raise ValueError(
                    f"quality_nan_policy must be 'keep' or 'drop', got "
                    f"'{self.config.quality_nan_policy}'"
                )
            before = int(keep.sum())
            keep = keep & good
            self.log_msg(
                f"quality cut {q_col} >= {self.config.quality_min} "
                f"(nulls: {n_null:,}, policy={self.config.quality_nan_policy}): "
                f"{before:,} -> {int(keep.sum()):,}"
            )

        for col in self.config.require_detected:
            if col not in data.columns:
                raise KeyError(f"require_detected column '{col}' not in input")
            mag = pd.to_numeric(data[col], errors="coerce").astype("float64")
            detected = np.isfinite(mag)
            if np.isfinite(self.config.nondetect_val):
                detected &= mag != self.config.nondetect_val
            before = int(keep.sum())
            keep = keep & detected
            self.log_msg(
                f"detected in {col}: {before:,} -> {int(keep.sum()):,} "
                f"(alone {float(detected.mean()):.2%})"
            )

        data = data[keep.values].reset_index(drop=True)

        out_z = self.config.output_redshift_col
        if out_z and out_z != z_col:
            data[out_z] = redshift[keep.values].values

        pretrain, pretest = self._split(data)
        self.add_data("output_pretrain", pretrain)
        self.add_data("output_pretest", pretest)

    def _split(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(pre-train, pre-test) from the rows that passed the cuts: seeded shuffle, then by
        row fraction (whole ``group_col`` groups with grouping)."""
        if self.config.shuffle:
            data = data.sample(frac=1.0, random_state=self.config.seed).reset_index(drop=True)

        n_train = int(len(data) * self.config.train_frac)
        if self.config.group_col:
            train_mask = self._group_split_mask(data, n_train)
            pretrain = data[train_mask].reset_index(drop=True)
            pretest = data[~train_mask].reset_index(drop=True)
        else:
            pretrain = data.iloc[:n_train].reset_index(drop=True)
            pretest = data.iloc[n_train:].reset_index(drop=True)
        self.log_msg(
            f"split (train_frac={self.config.train_frac}, seed={self.config.seed}"
            + (f", grouped on {self.config.group_col}" if self.config.group_col else "")
            + f"): {len(pretrain):,} pre-train / {len(pretest):,} pre-test"
        )
        return pretrain, pretest

    def _group_split_mask(self, data: pd.DataFrame, n_train: int) -> np.ndarray:
        """Assign whole groups to the training side until it holds ~``n_train`` rows.

        Splitting row by row lets two rows that share a label land on opposite sides, which
        leaks that label across the train/test boundary.  Assigning whole groups makes the two
        sides disjoint in the label, at the cost of the split fraction being approximate.
        """
        group_col = self.config.group_col
        if group_col not in data.columns:
            raise KeyError(f"group column '{group_col}' not in input")

        keys = data[group_col].astype(str).values
        # Nulls are distinct rows, not one giant group.
        null = data[group_col].isna().values
        if null.any():
            keys = keys.copy()
            keys[null] = [f"__null_{i}__" for i in np.where(null)[0]]

        # data is already shuffled, so first appearance order is a random order of groups.
        codes, first_index = pd.factorize(keys)
        counts = np.bincount(codes, minlength=len(first_index))
        order = pd.unique(codes)  # groups in order of first appearance
        cumulative = np.cumsum(counts[order])
        n_groups_train = int(np.searchsorted(cumulative, n_train, side="right"))
        train_groups = set(order[:n_groups_train].tolist())

        n_multi = int((counts > 1).sum())
        self.log_msg(
            f"grouping on {group_col}: {len(first_index):,} groups over {len(data):,} rows "
            f"({n_multi:,} groups hold more than one row)"
        )
        return np.isin(codes, list(train_groups))

    def log_msg(self, msg: str) -> None:
        """Print progress; stage logging goes to stdout under ceci."""
        print(f"  [{self.name}] {msg}", flush=True)


class ReferenceHoldoutSplitter(PreTrainTestSplitter):
    """``PreTrainTestSplitter``'s cuts, but the split is given: a row sharing any of
    ``holdout_cols`` with the ``holdout`` table goes to the pre-test side, all others to
    pre-train.  No shuffle, no fraction.

    For training sets built from another reference compilation than an existing split: hold
    out that split's pre-test pool (its ``output_pretest``) on the shear-object id and the
    reference ids, and every training set can be scored on the one test set without any of
    its galaxies or labels having been trained on.
    """

    name = "ReferenceHoldoutSplitter"
    config_options = PreTrainTestSplitter.config_options.copy()
    config_options.update(
        holdout_cols=Param(
            list,
            ["ref_id"],
            msg="a row whose value in any of these columns appears in the holdout table "
            "is held out",
        ),
    )

    inputs = [("input", PqHandle), ("holdout", PqHandle)]

    def _split(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        holdout = as_dataframe(self.get_data("holdout"))
        held = np.zeros(len(data), dtype=bool)
        for col in self.config.holdout_cols:
            for frame, which in ((data, "input"), (holdout, "holdout")):
                if col not in frame.columns:
                    raise KeyError(f"holdout column '{col}' not in the {which} table")
            match = data[col].isin(holdout[col].dropna().unique()).to_numpy()
            held |= match
            self.log_msg(f"holdout on {col}: {int(match.sum()):,} rows share a value")
        pretrain = data[~held].reset_index(drop=True)
        pretest = data[held].reset_index(drop=True)
        self.log_msg(
            f"held out {len(pretest):,} rows sharing {list(self.config.holdout_cols)} with "
            f"{len(holdout):,} holdout rows: {len(pretrain):,} pre-train"
        )
        return pretrain, pretest


class MagRedshiftDownsampler(Selector):
    """Cap the occupancy of a 2-D magnitude--redshift grid, per survey.

    A reference catalog assembled from real surveys is dominated by a handful of large
    spectroscopic programmes that pile labels up at particular redshifts and bright
    magnitudes.  Training on it directly imprints that distribution on the photo-z posteriors
    as an effective prior.  Binning each dominant survey on a magnitude--redshift grid and
    capping every cell flattens the distribution without discarding the rare regions.

    Surveys absent from ``caps`` are kept whole, so the parameter doubles as the list of
    surveys considered dominant.
    """

    name = "MagRedshiftDownsampler"
    config_options = Selector.config_options.copy()
    config_options.update(
        survey_col=Param(str, DEFAULT_SURVEY_COL, msg="column holding the survey name"),
        mag_col=Param(str, DEFAULT_MAG_COL, msg="magnitude column binned on"),
        redshift_col=Param(str, "redshift", msg="redshift column binned on"),
        caps=Param(
            dict,
            DEFAULT_CAPS,
            msg="survey name -> maximum objects per (magnitude, redshift) cell; "
            "surveys not listed are kept in full",
        ),
        d_mag=Param(float, 0.1, msg="magnitude bin width"),
        d_z=Param(float, 0.1, msg="redshift bin width"),
        random_seed=Param(int, 42, msg="random seed for the within-cell draw"),
    )

    inputs = [("input", PqHandle)]
    outputs = [("output", Hdf5Handle)]

    def _frame(self) -> pd.DataFrame:
        """The input as a DataFrame, converted once and reused by _select() and run()."""
        if getattr(self, "_cached_frame", None) is None:
            self._cached_frame = as_dataframe(self.get_data("input"))
        return self._cached_frame

    def _select(self) -> np.ndarray:
        data = self._frame()
        for col in (self.config.survey_col, self.config.mag_col, self.config.redshift_col):
            if col not in data.columns:
                raise KeyError(f"column '{col}' not in input")

        survey = data[self.config.survey_col].astype(str).values
        mag = pd.to_numeric(data[self.config.mag_col], errors="coerce").astype("float64").values
        redshift = (
            pd.to_numeric(data[self.config.redshift_col], errors="coerce")
            .astype("float64")
            .values
        )

        rng = np.random.default_rng(self.config.random_seed)
        keep = np.ones(len(data), dtype=bool)

        uncapped = int((~np.isin(survey, list(self.config.caps))).sum())
        print(f"  [{self.name}] surveys kept whole: {uncapped:,} rows", flush=True)

        for survey_name, cap in sorted(self.config.caps.items()):
            in_survey = survey == survey_name
            n_in = int(in_survey.sum())
            if n_in == 0:
                print(f"  [{self.name}] {survey_name}: absent, skipping", flush=True)
                continue
            idx = np.where(in_survey)[0]
            sub_keep = _bin_cap_mask(
                mag[idx], redshift[idx], int(cap), self.config.d_mag, self.config.d_z, rng
            )
            keep[idx] = sub_keep
            print(
                f"  [{self.name}] {survey_name}: {n_in:,} -> {int(sub_keep.sum()):,}"
                f"  (cap={cap}/cell)",
                flush=True,
            )

        print(f"  [{self.name}] total: {len(data):,} -> {int(keep.sum()):,}", flush=True)
        return keep

    def run(self) -> None:
        """As ``Selector.run``, but the output is an HDF5 dict-of-arrays rather than parquet."""
        data = self._frame()
        mask = self._select()
        selected = data[mask].reset_index(drop=True)
        self.add_data("output", frame_to_hdf5_dict(selected))


def som_features(
    frame: pd.DataFrame,
    noncolor_cols: list,
    color_cols: list,
    noncolor_nondet: list,
    color_nondet: list,
    nondetect_val: float,
) -> np.ndarray:
    """The SOM feature matrix: ``noncolor_cols`` as they are, then adjacent ``color_cols``
    differences, as float32.

    Non-detections -- the ``nondetect_val`` sentinel, NaN or +/-inf -- are first replaced by
    the per-column values, as ``SOMSpecSelector`` does, so every row can be given a cell.  A
    column listed in both groups is filled once, with its ``noncolor_nondet`` value, which is
    also what ``SOMSpecSelector``'s in-place replacement amounts to.
    """
    filled = {}
    for col, fill in zip([*noncolor_cols, *color_cols], [*noncolor_nondet, *color_nondet]):
        if col in filled:
            continue
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype="float64", copy=True)
        bad = ~np.isfinite(values)
        if np.isfinite(nondetect_val):
            bad |= np.isclose(values, nondetect_val)
        values[bad] = fill
        filled[col] = values
    features = [filled[col] for col in noncolor_cols]
    features += [filled[a] - filled[b] for a, b in zip(color_cols[:-1], color_cols[1:])]
    return np.column_stack(features).astype(np.float32)


def nearest_cell(  # pragma: no cover
    codebook: np.ndarray, data: np.ndarray, chunk: int = 65536
) -> np.ndarray:
    """Index of the closest codebook vector for every row of ``data``.

    Chunked, so 1M objects against a 32x32 map never hold more than a ``chunk x 1024``
    distance block (``rail_som``'s ``get_bmus`` builds the full N x cells matrix, 8 GB here).
    """
    from scipy.spatial.distance import cdist

    out = np.empty(len(data), dtype=np.int64)
    for start in range(0, len(data), chunk):
        block = data[start : start + chunk]
        out[start : start + len(block)] = cdist(block, codebook, "sqeuclidean").argmin(axis=1)
    return out


def train_som(  # pragma: no cover
    train: np.ndarray,
    som_size: list,
    n_epochs: int,
    init: str = "pca",
    seed: int = 42,
    radius_final: float = 1.0,
) -> np.ndarray:
    """Train a rectangular, planar somoclu SOM on ``train``; return its codebook as
    ``(n_x * n_y, n_features)``, flat index ``y * n_x + x``.

    ``init="pca"`` is somoclu's PCA initialisation: the map starts flat on the plane of the
    first two principal components and has to fold to reach the others, which can leave
    stripes in the cell occupancy.  ``init="random"`` starts every node on a training vector
    drawn with ``seed`` (somoclu's own random initialisation lives in its C++ core and cannot
    be seeded), so the map organises itself from the data.

    The neighbourhood radius shrinks linearly from half the map to ``radius_final`` (somoclu's
    ``radiusN``; its default is 1).  The Gaussian width is half the radius, so at 1 the last
    epochs update barely more than the winning node and the map can crinkle; a larger final
    radius keeps neighbouring nodes coupled, a smoother map at some cost in resolution.
    """
    from somoclu import Somoclu

    n_x, n_y = som_size
    data = np.ascontiguousarray(train, dtype=np.float32)
    if init == "pca":
        start = dict(initialization="pca")
    elif init == "random":
        rows = np.random.default_rng(seed).choice(
            len(data), size=n_x * n_y, replace=len(data) < n_x * n_y
        )
        start = dict(initialcodebook=np.ascontiguousarray(data[rows]))
    else:
        raise ValueError(f"init must be 'pca' or 'random', got {init!r}")
    som = Somoclu(
        n_x, n_y, gridtype="rectangular", compactsupport=False, maptype="planar", **start
    )
    som.train(data, epochs=n_epochs, radiusN=radius_final)
    return som.codebook.reshape(-1, som.codebook.shape[-1])


def allocate_cell_targets(
    n_population: np.ndarray, n_pool: np.ndarray, n_target: int
) -> np.ndarray:
    """Integer target per SOM cell: proportional to the population occupancy, summing to
    exactly ``n_target``.

    Only a cell holding at least one pool object can be filled, so the population occupancy is
    renormalised over those cells (the population share that falls elsewhere is what the test
    set cannot represent; the caller reports it).  Largest-remainder rounding, ties broken by
    cell index, makes the total exact and the result deterministic.

    ``n_population`` may be any non-negative weights, not only counts: ``SOMZCellResampler``
    passes the population share of every (cell, redshift bin).
    """
    n_population = np.asarray(n_population, dtype=np.float64)
    covered = np.asarray(n_pool) > 0
    if n_target < 0:
        raise ValueError(f"n_target must be >= 0, got {n_target}")
    weight = np.where(covered, n_population, 0.0)
    if weight.sum() == 0:
        raise ValueError("no population object falls in a SOM cell that holds a pool object")
    raw = n_target * weight / weight.sum()
    targets = np.floor(raw).astype(np.int64)
    short = n_target - int(targets.sum())
    if short:
        order = np.argsort(-(raw - targets), kind="stable")
        targets[order[:short]] += 1
    assert targets.sum() == n_target and not targets[~covered].any()
    return targets


def draw_cell_members(
    cells: np.ndarray, targets: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Pool row indices that give every SOM cell exactly its target.

    A cell with ``m`` pool objects and target ``t`` takes each object ``t // m`` times plus
    ``t % m`` more drawn without replacement: an object repeats only once its cell has been
    used up, and then as evenly as possible.
    """
    cells = np.asarray(cells, dtype=np.int64)
    order = np.argsort(cells, kind="stable")
    bounds = np.searchsorted(cells[order], np.arange(len(targets) + 1))
    picks = []
    for cell in np.flatnonzero(targets):
        members = order[bounds[cell] : bounds[cell + 1]]
        if members.size == 0:
            raise ValueError(f"SOM cell {cell} has target {targets[cell]} but no pool objects")
        full, extra = divmod(int(targets[cell]), members.size)
        if full:
            picks.append(np.tile(members, full))
        if extra:
            picks.append(rng.choice(members, size=extra, replace=False))
    return np.concatenate(picks) if picks else np.empty(0, dtype=np.int64)


class SOMResampler(RailStage):
    """Resample a spectroscopic pool so its colour--magnitude distribution matches a
    photometric population, cell by cell on a Self-Organising Map.

    A reference catalog is not a fair sample of the galaxies a shear catalog contains -- it is
    brighter and bluer, because that is what is spectroscopically reachable.  Photo-z metrics
    measured on it therefore do not generalise to the science sample.

    One SOM is trained (on the population by default, so its cells resolve the colour--
    magnitude space of the sample being represented) and both samples are assigned to its
    cells.  Each cell gets a target ``n_target x`` (population fraction in that cell), and the
    pool objects in the cell are resampled until the target is met -- all of them before any
    repeats, so a cell short of references repeats them evenly.  The test set therefore has
    exactly ``n_target`` rows and the population's cell occupancy, by construction.

    A cell the population occupies but the pool does not cannot be filled; the population's
    occupancy is renormalised over the cells that can, and the share it loses is logged.  That
    share is the part of the shear sample the references cannot represent at this SOM
    resolution.

    The redshift distribution is **not** matched: the shear catalog has no redshifts, so the
    resampling can only correct magnitude and colour, and inside a cell the test set has the
    pool's redshifts.  Where the pool is targeted spectroscopy those are biased;
    ``SOMZCellResampler`` corrects them with a redshift-complete truth sample.

    Outputs: ``output``, the test set (pool columns plus ``som_cell``), and ``cells``, one row
    per SOM cell with its population, pool and target counts -- the occupancy matrix.
    """

    name = "SOMResampler"
    config_options = RailStage.config_options.copy()
    config_options.update(
        som_size=Param(list, [32, 32], msg="SOM dimensions (x, y)"),
        n_epochs=Param(int, 100, msg="SOM training epochs"),
        n_target=Param(int, 100_000, msg="number of rows in the resampled test set"),
        som_train_on=Param(
            str,
            "population",
            msg="'population' or 'pool': the sample the SOM is trained on; both are then "
            "assigned to its cells",
        ),
        som_init=Param(
            str,
            "pca",
            msg="'random' (nodes start on training vectors drawn with som_seed) or 'pca' "
            "(somoclu's PCA plane, which can leave stripes in the occupancy)",
        ),
        noncolor_cols=Param(
            list, [DEFAULT_MAG_COL], msg="columns used directly as SOM features"
        ),
        color_cols=Param(
            list,
            DEFAULT_COLOR_COLS,
            msg="columns differenced in order to make colour features; give them in "
            "increasing wavelength order",
        ),
        nondetect_val=Param(float, DEFAULT_NONDETECT_VAL, msg="non-detect sentinel value"),
        noncolor_nondet=Param(
            list, DEFAULT_NONCOLOR_NONDET, msg="replacement values for noncolor_cols"
        ),
        color_nondet=Param(
            list, DEFAULT_COLOR_NONDET, msg="replacement values for color_cols"
        ),
        mag_col=Param(str, DEFAULT_MAG_COL, msg="magnitude column the range applies to"),
        mag_range=Param(
            list,
            [float("-inf"), 25.0],
            msg="(min, max) on mag_col, applied to both the pool and the population before "
            "the SOM, so the targets and the draw see the same selection",
        ),
        seed=Param(int, 42, msg="random seed of the within-cell draws"),
        init_seed=Param(int, 42, msg="random seed of som_init='random'"),
        som_radius_final=Param(
            float,
            1.0,
            msg="final neighbourhood radius of the SOM training (somoclu radiusN); larger is a "
            "smoother, less folded map at some cost in resolution",
        ),
    )

    inputs = [("input", PqHandle), ("population", PqHandle)]
    outputs = [("output", Hdf5Handle), ("cells", PqHandle)]

    def __call__(self, sample: TableLike, population: TableLike, **kwargs) -> Hdf5Handle:
        """Resample ``sample`` to match ``population`` and return a handle to the result."""
        self.set_data("input", sample)
        self.set_data("population", population)
        self.run()
        self.finalize()
        return self.get_handle("output")

    def _mag_range(self, df: pd.DataFrame, label: str) -> pd.DataFrame:
        lo, hi = (float(v) for v in self.config.mag_range)
        mag = pd.to_numeric(df[self.config.mag_col], errors="coerce").astype("float64").values
        keep = (mag > lo) & (mag < hi)
        print(
            f"  [{self.name}] {label}: {len(df):,} -> {int(keep.sum()):,} "
            f"after {lo} < {self.config.mag_col} < {hi}",
            flush=True,
        )
        return df[keep].reset_index(drop=True)

    def _features(self, df: pd.DataFrame) -> np.ndarray:
        return som_features(
            df,
            list(self.config.noncolor_cols),
            list(self.config.color_cols),
            list(self.config.noncolor_nondet),
            list(self.config.color_nondet),
            self.config.nondetect_val,
        )

    def _train_and_assign(self, train: np.ndarray, samples: list) -> list:  # pragma: no cover
        """Train the SOM on ``train``; return the flat cell index of every row of each sample."""
        n_x, n_y = self.config.som_size
        t0 = time.time()
        codebook = train_som(
            train,
            [n_x, n_y],
            self.config.n_epochs,
            self.config.som_init,
            self.config.init_seed,
            self.config.som_radius_final,
        )
        print(
            f"  [{self.name}] SOM {n_x}x{n_y} ({self.config.som_init} init, final radius "
            f"{self.config.som_radius_final:g}) trained on the "
            f"{self.config.som_train_on} ({len(train):,} objects, {self.config.n_epochs} epochs) "
            f"in {time.time() - t0:.0f}s",
            flush=True,
        )
        return [nearest_cell(codebook, sample) for sample in samples]

    #: what one entry of ``_targets``'s target array is, for the log
    _unit_label = "cells"

    def _extra_samples(self) -> dict:
        """Further samples to place on the SOM, by label, after the magnitude range.  None
        here; ``SOMZCellResampler`` adds its truth sample."""
        return {}

    def _targets(self, frames: dict, cells: dict, n_pop: np.ndarray, n_pool: np.ndarray):
        """What to draw: ``(units, targets, columns, tables)``.

        ``units[k]`` is the draw unit of pool row ``k`` and ``targets[u]`` the number of rows
        to draw from unit ``u`` (see :func:`draw_cell_members`); ``columns`` are extra
        per-pool-row columns for the test set and ``tables`` extra outputs, by tag.  Here a
        unit is a SOM cell and its target the cell's population share.
        """
        targets = allocate_cell_targets(n_pop, n_pool, self.config.n_target)
        return cells["pool"], targets, {}, {}

    def run(self) -> None:
        if self.config.som_init not in ("pca", "random"):
            raise ValueError(f"som_init must be 'pca' or 'random', got {self.config.som_init!r}")
        if self.config.som_train_on not in ("population", "pool"):
            raise ValueError(
                f"som_train_on must be 'population' or 'pool', got {self.config.som_train_on!r}"
            )
        frames = {
            "pool": self._mag_range(as_dataframe(self.get_data("input")), "pool"),
            "population": self._mag_range(
                as_dataframe(self.get_data("population")), "population"
            ),
        }
        frames.update(self._extra_samples())

        needed = list(self.config.noncolor_cols) + list(self.config.color_cols)
        for label, frame in frames.items():
            missing = [c for c in needed if c not in frame.columns]
            if missing:
                raise KeyError(f"SOM feature columns missing from {label}: {missing}")

        features = {label: self._features(frame) for label, frame in frames.items()}
        cells = dict(
            zip(
                features,
                self._train_and_assign(
                    features[self.config.som_train_on], list(features.values())
                ),
            )
        )
        pool, pool_cell, pop_cell = frames["pool"], cells["pool"], cells["population"]

        n_x, n_y = self.config.som_size
        n_cells = n_x * n_y
        n_pop = np.bincount(pop_cell, minlength=n_cells)
        n_pool = np.bincount(pool_cell, minlength=n_cells)
        lost = int(n_pop[n_pool == 0].sum())
        print(
            f"  [{self.name}] cells: {int((n_pop > 0).sum())} hold population, "
            f"{int(((n_pop > 0) & (n_pool > 0)).sum())} of them also pool objects; population "
            f"in cells without any pool object: {lost:,} ({lost / max(len(pop_cell), 1):.2%}), "
            "renormalised away",
            flush=True,
        )
        units, targets, columns, tables = self._targets(frames, cells, n_pop, n_pool)

        rng = np.random.default_rng(self.config.seed)
        picks = draw_cell_members(units, targets, rng)
        picks = picks[rng.permutation(len(picks))]  # not cell-ordered
        result = pool.iloc[picks].reset_index(drop=True)
        result["som_cell"] = pool_cell[picks]
        for name, values in columns.items():
            result[name] = np.asarray(values)[picks]

        uses = np.bincount(picks, minlength=len(pool))
        n_unique = int((uses > 0).sum())
        n_in_unit = np.bincount(units, minlength=len(targets))
        print(
            f"  [{self.name}] {len(result):,} rows (n_target {self.config.n_target:,}) from "
            f"{n_unique:,} unique pool objects ({n_unique / max(len(pool), 1):.1%} of the pool); "
            f"most-used object {int(uses.max()) if len(uses) else 0}x; "
            f"{int((targets > n_in_unit).sum())} {self._unit_label} needed repeats",
            flush=True,
        )

        cell = np.arange(n_cells)
        drawn = np.flatnonzero(uses)
        occupancy = pd.DataFrame(
            dict(
                cell=cell,
                som_x=cell % n_x,
                som_y=cell // n_x,
                n_population=n_pop,
                n_pool=n_pool,
                n_target=np.bincount(pool_cell[picks], minlength=n_cells),
                n_unique=np.bincount(pool_cell[drawn], minlength=n_cells),
            )
        )
        self.add_data("output", frame_to_hdf5_dict(result))
        self.add_data("cells", occupancy)
        for tag, table in tables.items():
            self.add_data(tag, table)


# --- SOMZCellResampler ------------------------------------------------------------------
# Wider than the truth field's large-scale structure (COSMOS: walls at dz ~ 0.05), which should
# not be imprinted on the test set; narrow enough to correct the pool, whose redshifts are kept
# inside a bin.  On DP2, 0.4-wide bins left the pool's pile-up below z = 0.8 as a sharp peak.
DEFAULT_Z_EDGES = [round(0.2 * k, 1) for k in range(11)] + [2.4, 2.8, 3.4, 4.0, 7.0]
DEFAULT_Z_SHRINK = 5.0


def redshift_bins(z: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Index of the ``edges`` bin of every redshift; values outside go to the end bins."""
    z = np.asarray(z, dtype=np.float64)
    return np.clip(np.searchsorted(edges, z, side="right") - 1, 0, len(edges) - 2)


def zcell_fractions(
    truth_counts: np.ndarray, pool_counts: np.ndarray, som_size: list, shrink: float
) -> tuple[np.ndarray, np.ndarray]:
    """p(z bin | cell) on the SOM from a truth sample's ``(n_cells, n_bins)`` counts.

    A cell's own histogram is shrunk toward its 3x3 grid neighbourhood (the cell and up to
    eight neighbours, pooled) with ``shrink`` pseudo-counts:
    ``(n(c, b) + shrink * p_nbr(b | c)) / (n(c) + shrink)`` -- so a well-populated cell speaks
    for itself and a sparse one borrows from its neighbours.  A cell with no truth object in
    the whole neighbourhood gets the pool's own fractions, i.e. ``SOMResampler``'s draw, and
    is flagged in the second return value.
    """
    n_x, n_y = som_size
    n_bins = truth_counts.shape[1]
    grid = np.pad(truth_counts.reshape(n_y, n_x, n_bins), ((1, 1), (1, 1), (0, 0)))
    nbr = sum(
        grid[1 + dy : 1 + dy + n_y, 1 + dx : 1 + dx + n_x]
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    ).reshape(n_x * n_y, n_bins)
    n_nbr = nbr.sum(axis=1)
    p_nbr = nbr / np.maximum(n_nbr, 1)[:, None]
    denom = truth_counts.sum(axis=1) + shrink
    p = np.where(
        denom[:, None] > 0,
        (truth_counts + shrink * p_nbr) / np.where(denom > 0, denom, 1.0)[:, None],
        p_nbr,
    )
    no_truth = n_nbr == 0
    own = pool_counts / np.maximum(pool_counts.sum(axis=1), 1)[:, None]
    p[no_truth] = own[no_truth]
    return p, no_truth


class SOMZCellResampler(SOMResampler):
    """``SOMResampler`` with every cell's target split over wide redshift bins by a *truth*
    sample: the "zcell" test set.

    Matching colour and magnitude is not enough when the pool is mostly targeted
    spectroscopy.  Inside one SOM cell the references are not a fair draw in redshift: DESI
    selects ELGs at 0.6 < z < 1.6 with colour cuts finer than a cell, so cells whose galaxies
    are mostly at z ~ 0.3-0.7 are filled with z ~ 1 references (measured on DP2: the test sets'
    median z 0.78-0.82 against ~0.65 for the population).

    The truth sample fixes that.  It is a sample selected like the population *with complete
    redshifts* -- for DP2, the deep-field shear objects inside COSMOS2020, given wide-field
    noise, put through the population cuts and matched to COSMOS2020.  Placed on the same SOM
    it gives p(z | cell)
    (:func:`zcell_fractions`), and each cell's target ``n_target x f_pop(cell)`` is split into
    ``n_target x f_pop(cell) x p(z bin | cell)``.  Every (cell, z bin) slot is then filled from
    the pool objects in it, all of them before any repeats, exactly as cells are in
    ``SOMResampler``.  A slot with a target but no pool object cannot be filled: the targets
    are renormalised over the fillable slots and the share lost is logged.

    The test set keeps the population's colour--magnitude occupancy up to that lost share and
    gets the truth sample's redshift distribution at the resolution of ``z_edges``; inside a
    bin the redshifts are the pool's, so a bin much wider than the pool's own structure lets
    it through (DP2: 0.4-wide bins left a sharp peak below z = 0.8; the default is 0.2).  It inherits whatever the truth sample gets wrong (its
    photo-z errors, its field's large-scale structure at bin resolution).

    Extra input ``truth`` (``truth_redshift_col`` and the SOM feature columns; the magnitude
    range applies to it too).  Extra output ``zcells``: one row per (cell, z bin) with the
    truth count, p(z bin | cell), the population share, the pool count and the target.  The
    test set gains ``som_zbin``.
    """

    name = "SOMZCellResampler"
    config_options = SOMResampler.config_options.copy()
    config_options.update(
        z_edges=Param(
            list,
            DEFAULT_Z_EDGES,
            msg="redshift bin edges the cell targets are split over: wider than the truth "
            "field's structure, narrow enough to correct the pool's within-bin shape",
        ),
        z_shrink=Param(
            float,
            DEFAULT_Z_SHRINK,
            msg="pseudo-counts of the 3x3 SOM neighbourhood in each cell's p(z bin | cell)",
        ),
        truth_redshift_col=Param(str, "z", msg="redshift column of the truth sample"),
        redshift_col=Param(
            str,
            "redshift",
            msg="redshift column of the pool (PreTrainTestSplitter's output_redshift_col)",
        ),
    )

    inputs = [("input", PqHandle), ("population", PqHandle), ("truth", PqHandle)]
    outputs = [("output", Hdf5Handle), ("cells", PqHandle), ("zcells", PqHandle)]
    _unit_label = "(cell, z bin) slots"

    def __call__(
        self, sample: TableLike, population: TableLike, truth: TableLike, **kwargs
    ) -> Hdf5Handle:
        """Resample ``sample`` to match ``population`` in colour--magnitude and ``truth`` in
        redshift; return a handle to the result."""
        self.set_data("truth", truth)
        return SOMResampler.__call__(self, sample, population, **kwargs)

    def _edges(self) -> np.ndarray:
        edges = np.asarray(self.config.z_edges, dtype=np.float64)
        if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
            raise ValueError(f"z_edges must be at least two increasing values, got {edges}")
        if self.config.z_shrink < 0:  # pragma: no cover
            raise ValueError(f"z_shrink must be >= 0, got {self.config.z_shrink}")
        return edges

    def _extra_samples(self) -> dict:
        self._edges()
        truth = self._mag_range(as_dataframe(self.get_data("truth")), "truth")
        col = self.config.truth_redshift_col
        if col not in truth.columns:
            raise KeyError(f"truth_redshift_col {col!r} is not a column of the truth sample")
        z = pd.to_numeric(truth[col], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(z)
        if not finite.all():  # pragma: no cover
            print(
                f"  [{self.name}] truth: {int((~finite).sum()):,} rows without a finite "
                f"{col} dropped",
                flush=True,
            )
            truth = truth[finite].reset_index(drop=True)
        if truth.empty:  # pragma: no cover
            raise ValueError("the truth sample has no row with a finite redshift")
        return {"truth": truth}

    def _targets(self, frames: dict, cells: dict, n_pop: np.ndarray, n_pool: np.ndarray):
        edges = self._edges()
        n_bins, n_cells = len(edges) - 1, len(n_pop)
        if self.config.redshift_col not in frames["pool"].columns:
            raise KeyError(
                f"redshift_col {self.config.redshift_col!r} is not a column of the pool"
            )
        pool_z = pd.to_numeric(frames["pool"][self.config.redshift_col], errors="coerce")
        if not np.isfinite(pool_z.to_numpy(dtype=np.float64)).all():  # pragma: no cover
            raise ValueError(f"the pool has non-finite {self.config.redshift_col!r} values")
        pool_bin = redshift_bins(pool_z, edges)
        truth_bin = redshift_bins(frames["truth"][self.config.truth_redshift_col], edges)

        truth_counts = np.zeros((n_cells, n_bins))
        np.add.at(truth_counts, (cells["truth"], truth_bin), 1)
        pool_counts = np.zeros((n_cells, n_bins))
        np.add.at(pool_counts, (cells["pool"], pool_bin), 1)
        p_z, no_truth = zcell_fractions(
            truth_counts, pool_counts, self.config.som_size, self.config.z_shrink
        )
        share = (n_pop / n_pop.sum())[:, None] * p_z
        targets = allocate_cell_targets(share.ravel(), pool_counts.ravel(), self.config.n_target)

        fillable = pool_counts > 0
        print(
            f"  [{self.name}] truth: {len(frames['truth']):,} objects in "
            f"{int((truth_counts.sum(axis=1) > 0).sum())} cells; population in cells with no "
            f"truth object in their 3x3 neighbourhood (pool's own p(z) kept): "
            f"{float(share[no_truth].sum()):.2%}; population share in (cell, z bin) slots "
            f"without a pool object (the cells above included): "
            f"{float(share[~fillable].sum()):.2%}, renormalised away",
            flush=True,
        )
        target_by_bin = targets.reshape(n_cells, n_bins).sum(axis=0)
        pool_by_bin = pool_counts.sum(axis=0) / pool_counts.sum()
        print(f"  [{self.name}] {'z bin':>13s} {'population':>11s} {'pool':>8s} {'test':>8s}")
        for b in range(n_bins):
            print(
                f"  [{self.name}] {edges[b]:5.2f} - {edges[b + 1]:5.2f} "
                f"{float(share[:, b].sum()):11.2%} {pool_by_bin[b]:8.2%} "
                f"{target_by_bin[b] / max(self.config.n_target, 1):8.2%}",
                flush=True,
            )

        cell = np.repeat(np.arange(n_cells), n_bins)
        z_bin = np.tile(np.arange(n_bins), n_cells)
        zcells = pd.DataFrame(
            dict(
                cell=cell,
                z_bin=z_bin,
                z_lo=edges[z_bin],
                z_hi=edges[z_bin + 1],
                n_truth=truth_counts.ravel().astype(np.int64),
                p_z=p_z.ravel(),
                no_truth=no_truth[cell],
                population_share=share.ravel(),
                n_pool=pool_counts.ravel().astype(np.int64),
                n_target=targets,
            )
        )
        units = cells["pool"] * n_bins + pool_bin
        return units, targets, {"som_zbin": pool_bin}, {"zcells": zcells}
