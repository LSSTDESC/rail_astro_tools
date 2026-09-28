"""Stages that curate a cross-matched reference-redshift catalog into a training set and a
representative test set for photometric redshift.

This is the HSC-FY preparation chain, generalised.  Three stages:

``PreTrainTestSplitter``
    Shuffle and split the cross-matched reference catalog into a pre-training and a
    pre-testing sample, applying a redshift range cut and, optionally, a quality-flag cut.
    Splitting *first* is what keeps the two branches below disjoint.

``MagRedshiftDownsampler``
    The training branch.  Bin each dominant survey on a 2-D magnitude--redshift grid and cap
    the occupancy of every bin, which flattens the label distribution so a machine-learning
    photo-z does not inherit the bright, low-redshift prior of the big spectroscopic surveys.

``SOMResampler``
    The testing branch.  Resample the pre-testing pool with a Self-Organising Map so its
    colour--magnitude distribution matches a *population* sample drawn from the shear
    catalog the photo-z will actually be applied to.

Every survey-specific quantity -- column names, magnitude limits, per-survey caps, SOM
features -- is a configuration parameter.  The defaults are the ones that work for the Rubin
DP2 anacal shear catalog, whose photometry is ``lsst_{r,i,z}_mag_gauss2`` (riz only, already
magnitudes) and whose survey label is ``z_source``.

Reference: the HSC-Y3 photo-z analysis, sections "Test Set Curation" and "Training Set
Curation"; the 2-D downsampling follows Zhou et al. (2021), section 3.3.
"""

import os

import numpy as np
import pandas as pd
import tables_io
from ceci.config import StageParameter as Param

from rail.core.data import DataStore, Hdf5Handle, PqHandle, TableLike
from rail.core.stage import RailStage
from rail.creation.selector import Selector

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


def as_dataframe(data: TableLike) -> pd.DataFrame:
    """Return ``data`` as a pandas DataFrame, whatever table flavour it arrives as.

    ``PqHandle`` hands back a **pyarrow Table** when a stage runs under ceci, but a DataFrame
    when a stage is called interactively with one.  The difference is silent and vicious:
    ``Table.columns`` is the list of column *data*, not column *names*, so a name lookup
    against it fails in a way that looks nothing like a type error.
    """
    if isinstance(data, pd.DataFrame):
        return data
    return tables_io.convert(data, tables_io.types.PD_DATAFRAME)


def to_numpy_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Cast pandas extension dtypes to plain numpy dtypes so the frame survives HDF5.

    This is not cosmetic.  ``tables_io.convert(df, NUMPY_DICT)`` **silently drops** any column
    it cannot map to a native HDF5 type -- it emits a warning and carries on -- so a catalog
    with pandas ``string`` or nullable-integer columns loses them without raising.  On the DP2
    anacal cross-match that would quietly discard ``z_source``, ``z_type``, ``ref_id``,
    ``ref_cat``, ``region`` and ``desi_spectype``: every column identifying where a redshift
    came from.

    Strings become fixed-width bytes, nullable integers become float64 (so nulls survive as
    NaN), and nullable booleans are filled False.
    """
    out = df.copy()
    for col in out.columns:
        dtype_name = str(out[col].dtype)
        if dtype_name == "string" or out[col].dtype == object:
            out[col] = out[col].astype(str).values.astype("S")
        elif dtype_name.startswith(("Int", "UInt")):
            out[col] = out[col].astype("float64")
        elif dtype_name == "Float32" or dtype_name == "Float64":
            out[col] = out[col].astype("float64")
        elif dtype_name == "boolean":
            out[col] = out[col].fillna(False).astype(bool)
    return out


def frame_to_hdf5_dict(df: pd.DataFrame) -> dict:
    """Convert a DataFrame to the ordered dict-of-arrays that ``Hdf5Handle`` writes."""
    converted = tables_io.convert(to_numpy_frame(df), tables_io.types.NUMPY_DICT)
    n_lost = len(df.columns) - len(converted)
    if n_lost:  # pragma: no cover
        missing = [c for c in df.columns if c not in converted]
        raise RuntimeError(
            f"{n_lost} columns were dropped converting to HDF5: {missing}. "
            "to_numpy_frame() failed to make them HDF5-safe."
        )
    return converted


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

        data = data[keep.values].reset_index(drop=True)

        out_z = self.config.output_redshift_col
        if out_z and out_z != z_col:
            data[out_z] = redshift[keep.values].values

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

        self.add_data("output_pretrain", pretrain)
        self.add_data("output_pretest", pretest)

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


class SOMResampler(RailStage):
    """Resample a spectroscopic pool to match the colour--magnitude distribution of a
    photometric population, using a Self-Organising Map.

    A reference catalog is not a fair sample of the galaxies a shear catalog contains -- it is
    brighter and bluer, because that is what is spectroscopically reachable.  Photo-z metrics
    measured on it therefore do not generalise to the science sample.  Training a SOM on the
    pool, assigning both the pool and a *population* sample drawn from the shear catalog to
    cells, and then drawing from each cell in proportion to its population occupancy produces
    a subset of the pool whose photometry looks like the shear catalog's.

    The redshift distribution is deliberately **not** matched: the shear catalog has no
    redshifts, so the resampling can only correct magnitude and colour.  This is expected, not
    a defect.

    The SOM itself is ``rail.creation.degraders.specz_som.SOMSpecSelector``.  Note the roles
    of its two inputs are the reverse of what its docstring describes: here ``input`` is the
    spectroscopic pool being selected *from* and ``spec_data`` is the photometric population
    whose distribution is the target.

    A single pass yields at most as many objects as the population sample, so the procedure is
    repeated ``n_repeat`` times with an independent population draw and a freshly trained SOM
    each time, and the results are stacked.  Objects therefore repeat across draws by design.
    """

    name = "SOMResampler"
    config_options = RailStage.config_options.copy()
    config_options.update(
        som_size=Param(list, [32, 32], msg="SOM dimensions (x, y)"),
        n_epochs=Param(int, 100, msg="SOM training epochs"),
        n_repeat=Param(int, 6, msg="independent resampling passes to stack"),
        n_population=Param(
            int,
            30000,
            msg="population objects drawn per pass; the whole population sample is used if "
            "it is smaller than this",
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
        mag_col=Param(str, DEFAULT_MAG_COL, msg="magnitude column the limits apply to"),
        mag_cut_pool=Param(
            float,
            25.5,
            msg="keep mag_col < this in both the pool and the population before the SOM; "
            "set to a large number to disable",
        ),
        mag_range_out=Param(
            list,
            [18.5, 25.0],
            msg="[min, max] on mag_col applied to the stacked output",
        ),
        seed=Param(int, 42, msg="base random seed; pass i uses seed + i"),
    )

    inputs = [("input", PqHandle), ("population", PqHandle)]
    outputs = [("output", Hdf5Handle)]

    def __call__(self, sample: TableLike, population: TableLike, **kwargs) -> Hdf5Handle:
        """Resample ``sample`` to match ``population`` and return a handle to the result."""
        self.set_data("input", sample)
        self.set_data("population", population)
        self.run()
        self.finalize()
        return self.get_handle("output")

    def _mag_cut(self, df: pd.DataFrame, label: str) -> pd.DataFrame:
        mag = pd.to_numeric(df[self.config.mag_col], errors="coerce").astype("float64")
        keep = (mag < self.config.mag_cut_pool).values
        print(
            f"  [{self.name}] {label}: {len(df):,} -> {int(keep.sum()):,} "
            f"after {self.config.mag_col} < {self.config.mag_cut_pool}",
            flush=True,
        )
        return df[keep].reset_index(drop=True)

    def run(self) -> None:
        # Imported here so the module still imports where rail_som is unavailable.
        from rail.creation.degraders.specz_som import SOMSpecSelector

        # A fresh sub-stage per pass reuses tags in the DataStore; in rail-1.2 the store is
        # per-instance but `allow_overwrite` is still a class attribute.
        DataStore.allow_overwrite = True

        pool = self._mag_cut(as_dataframe(self.get_data("input")), "pool")
        population = self._mag_cut(
            as_dataframe(self.get_data("population")), "population"
        )

        needed = list(self.config.noncolor_cols) + list(self.config.color_cols)
        for frame, label in ((pool, "pool"), (population, "population")):
            missing = [c for c in needed if c not in frame.columns]
            if missing:
                raise KeyError(f"SOM feature columns missing from {label}: {missing}")

        som_config = dict(
            noncolor_cols=list(self.config.noncolor_cols),
            color_cols=list(self.config.color_cols),
            noncolor_nondet=list(self.config.noncolor_nondet),
            color_nondet=list(self.config.color_nondet),
            nondetect_val=self.config.nondetect_val,
            som_size=list(self.config.som_size),
            n_epochs=self.config.n_epochs,
        )
        out_dir = os.path.dirname(os.path.abspath(self.get_output("output", final_name=True)))

        n_draw = min(self.config.n_population, len(population))
        pieces = []
        for i in range(self.config.n_repeat):
            # SOMSpecSelector picks within a cell with the *global* numpy RNG and exposes no
            # seed of its own, so reproducibility has to be imposed from out here.
            np.random.seed(self.config.seed + i)
            rng = np.random.default_rng(self.config.seed + i)

            draw = population.sample(n=n_draw, random_state=int(rng.integers(1 << 31)))
            som_stage = SOMSpecSelector.make_stage(
                name=f"{self.instance_name}_som_pass{i}",
                output=os.path.join(out_dir, f"{self.instance_name}_som_pass{i}.pq"),
                **som_config,
            )
            # SOMSpecSelector overwrites non-detects in place, so hand it copies and then map
            # the selection back onto the untouched pool by index.
            selected = som_stage(input_data=pool.copy(), spec_data=draw.copy()).data
            piece = pool.loc[selected.index]
            pieces.append(piece)
            print(
                f"  [{self.name}] pass {i + 1}/{self.config.n_repeat}: "
                f"{len(piece):,} selected from {n_draw:,} population objects",
                flush=True,
            )

        stacked = pd.concat(pieces, ignore_index=True)
        lo, hi = self.config.mag_range_out
        mag = pd.to_numeric(stacked[self.config.mag_col], errors="coerce").astype("float64")
        keep = ((mag > lo) & (mag < hi)).values
        result = stacked[keep].reset_index(drop=True)
        n_unique = result.get("ref_id", pd.Series(result.index)).nunique()
        print(
            f"  [{self.name}] stacked {len(stacked):,} -> {len(result):,} after "
            f"{lo} < {self.config.mag_col} < {hi}; {n_unique:,} unique objects",
            flush=True,
        )

        self.add_data("output", frame_to_hdf5_dict(result))
