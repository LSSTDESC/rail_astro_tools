"""Crossmatch a reference catalog against a shear catalog, partitioned by LSST skymap tract.

Two stages.

``AssignTract``
    Add a skymap tract column to a reference catalog, using a pure-numpy port of
    ``lsst.skymap.RingsSkyMap.findTractIdArray``.  There is deliberately **no lsst import**:
    the Rings pixelization depends on nothing but ``num_rings`` and ``ra_start``, so the DM
    stack and its skymap pickle are not needed, and the whole pipeline can run in one
    environment instead of being split across two.

``TractCrossMatch``
    Match each shear object to its nearest reference within a radius, one worker process per
    tract.  The shear catalog is opened directly with h5py and read tract by tract, because
    the catalogs this targets are hundreds of gigabytes and cannot pass through a RAIL
    ``DataHandle``.

Nothing here knows about any particular survey.  The columns, the cuts, the HDF5 group, the
match radius and the skymap geometry are all configuration.

``is_primary`` IS A CORRECTNESS PRECONDITION, NOT A QUALITY CUT
---------------------------------------------------------------
Tracts overlap, so a detection in the overlap region is processed by more than one tract and
appears more than once.  The copies that are *not* primary carry a **neighbouring** tract's
label, so the tract of their sky position is not the tract in their ``tract`` column.  If they
are left in, the reference objects bucketed into tract T by the skymap can never meet them,
and they contribute nothing but missed matches.

Measured on the TXPipe DP2 metadetect catalog, comparing ``find_tract_id_array(ra, dec)``
against the file's own ``tract`` column over 1,384,280 sampled rows:

    is_primary == True      100.000000% agreement  (0 mismatches)
    is_primary == False       0%        agreement

Which is why ``[["is_primary", "==", True]]`` is the default ``shear_cuts`` rather than
something a caller is expected to remember.  The failure is silent.

THE METADETECT STEP IS A GROUP, NOT A COLUMN
--------------------------------------------
TXPipe writes the five metadetect shear steps as sibling HDF5 groups --
``shear/{ns,1p,1m,2p,2m}`` -- with different lengths and no row alignment between them.  There
is no ``metaStep`` column to cut on.  Selecting the unsheared catalog is therefore
``shear_group="shear/ns"``, not a cut.

CUT SPECIFICATIONS
------------------
Both catalogs take a list of ``"column operator value"`` strings::

    shear_cuts=["is_primary == True", "mag_i < 25.5"]
    ref_cuts=["z_source in DESI_DR1,SDSS_DR17", "z_flag >= 3"]

which is what they look like in the pipeline yaml::

    shear_cuts:
    - is_primary == True
    - mag_i < 25.5

Operators are ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``, ``in`` and ``not in``.  ``in`` and
``not in`` take a comma-separated list; the others take one value.  Values are read with the
YAML scalar rules, so ``True``, ``25.5``, ``3`` and ``DESI_DR1`` come out as a bool, a float,
an int and a string respectively.  A column used only in a cut is read and then dropped, so
cutting on ``is_primary`` does not put it in the output.

The equivalent ``[column, operator, value]`` triples are also accepted, which is often easier
to build programmatically::

    shear_cuts=[["is_primary", "==", True], ["z_source", "in", ["DESI_DR1", "SDSS_DR17"]]]

The string form is the default because ceci builds this stage's command-line parser from the
declared defaults and handles only a list of str, int or float -- a list of lists makes every
run of the stage die in ``parse_command_line`` before ``run()`` is reached.
"""

import difflib
import gc
import multiprocessing as mp
import os
import re
import sys
import time
from collections import deque

import h5py
import numpy as np
import pandas as pd
import yaml
from ceci.config import StageParameter as Param

from rail.core.data import PqHandle, TableLike
from rail.core.stage import RailStage
from rail.creation.degraders._frame_utils import as_dataframe, plain_python

# --- skymap defaults -------------------------------------------------------------------
# Rubin's `lsst_cells_v2`: RingsSkyMap, numRings=120, raStart=0, 18,938 tracts.
DEFAULT_NUM_RINGS = 120
DEFAULT_RA_START = 0.0
DEFAULT_NUM_TRACTS = 18938

# Tract id returned for a position that has no tract -- a non-finite ra or dec.
NO_TRACT = -1

# --- TXPipe metadetect defaults --------------------------------------------------------
DEFAULT_SHEAR_GROUP = "shear/ns"
DEFAULT_SHEAR_COLS = [
    "id",
    "mag_g", "mag_r", "mag_i", "mag_z",
    "mag_err_g", "mag_err_r", "mag_err_i", "mag_err_z",
]
DEFAULT_SHEAR_CUTS = ["is_primary == True"]

DEFAULT_REF_COLS = [
    "ref_id", "ra", "dec", "z", "z_err", "z_type", "z_source", "z_flag",
]
DEFAULT_REF_RENAME = {"ra": "ref_ra", "dec": "ref_dec"}

# Rows scanned at a time when indexing the tract column.  16.8M int64 rows is 134 MB, so the
# full ~1 GB column never exists at once -- which matters because everything resident in the
# parent at fork time is inherited copy-on-write by every worker.
TRACT_SCAN_CHUNK = 1 << 24

_OPERATORS = ("==", "!=", "<", "<=", ">", ">=", "in", "not in")
_LIST_OPERATORS = ("in", "not in")


# ======================================================================================= #
# skymap
# ======================================================================================= #

def ring_nums(num_rings: int = DEFAULT_NUM_RINGS) -> np.ndarray:
    """Number of tracts in each declination ring of a RingsSkyMap.

    Transcribed from ``RingsSkyMap.__init__``.  The expression order is load-bearing and must
    not be "simplified": for ``num_rings=120`` the two equatorial rings evaluate to exactly
    242.0 before the truncation, with no margin at all, so a reassociation that perturbs the
    last bit would silently give 242 tracts there instead of 243 and shift every tract id
    north of the equator.
    """
    ring_size = np.pi / (num_rings + 1)
    out = []
    for i in range(num_rings):
        start_dec = ring_size * (i + 0.5) - 0.5 * np.pi
        stop_dec = start_dec + ring_size
        # the edge of the ring nearest the equator, where the ring is widest
        dec = min(abs(start_dec), abs(stop_dec))
        out.append(int(2 * np.pi * np.cos(dec) / ring_size) + 1)
    return np.array(out, dtype=np.int64)


def num_tracts(num_rings: int = DEFAULT_NUM_RINGS) -> int:
    """Total tracts, including the two polar caps."""
    return int(ring_nums(num_rings).sum()) + 2


def find_tract_id_array(
    ra_deg,
    dec_deg,
    num_rings: int = DEFAULT_NUM_RINGS,
    ra_start: float = DEFAULT_RA_START,
) -> np.ndarray:
    """Tract id for each (ra, dec), in degrees.  A numpy port of ``findTractIdArray``.

    Rings run south to north.  Tract 0 is the south polar cap and ``num_tracts - 1`` the
    north; within a ring, tract 0 is *centred* on ``ra_start``.  Positions with a non-finite
    ra or dec get ``NO_TRACT``: casting NaN to int64 yields ``INT64_MIN``, which would
    otherwise sail on as a plausible-looking tract that matches nothing.
    """
    ra = np.deg2rad(np.atleast_1d(np.asarray(ra_deg, dtype="float64")))
    dec = np.deg2rad(np.atleast_1d(np.asarray(dec_deg, dtype="float64")))
    if ra.shape != dec.shape:
        raise ValueError(f"ra and dec differ in shape: {ra.shape} vs {dec.shape}")

    counts = ring_nums(num_rings)
    total = int(counts.sum()) + 2
    ring_size = np.pi / (num_rings + 1)
    first_ring_start = ring_size * 0.5 - 0.5 * np.pi

    out = np.full(ra.size, NO_TRACT, dtype=np.int64)
    good = np.isfinite(ra) & np.isfinite(dec)
    if not good.any():
        return out

    rings = np.zeros(ra.size, dtype=np.int64)
    rings[good & (dec < first_ring_start)] = -1
    rings[good & (dec > -first_ring_start)] = num_rings
    # The order of these three assignments matters: a dec exactly on the north cap boundary
    # satisfies `mid` and lands on num_rings, which the cap assignment below then resolves.
    mid = good & (dec >= first_ring_start) & (dec <= -first_ring_start)
    rings[mid] = ((dec[mid] - first_ring_start) / ring_size).astype(np.int64)

    out[good & (rings == -1)] = 0
    out[good & (rings == num_rings)] = total - 1

    body = np.flatnonzero(good & (out == NO_TRACT))
    if body.size:
        cumulative = np.cumsum(np.insert(counts, 0, 0))
        per_ring = counts[rings[body]]
        delta = (ra[body] - np.deg2rad(ra_start)) % (2.0 * np.pi)
        # + 0.5 then truncate is a round-half-up, because tract 0 straddles ra_start
        tract_num = (delta / (2.0 * np.pi / per_ring) + 0.5).astype(np.int64)
        tract_num[tract_num == per_ring] = 0  # wraparound
        out[body] = cumulative[rings[body]] + tract_num + 1
    return out


# ======================================================================================= #
# cut specifications
# ======================================================================================= #

_CUT_PATTERN = re.compile(
    r"^\s*(?P<column>\S+)\s+(?P<operator>==|!=|<=|>=|<|>|not\s+in|in)\s+(?P<value>.+?)\s*$"
)


def _parse_scalar(text: str):
    """Read one operand with the YAML scalar rules, falling back to the literal string."""
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError:
        return text
    return text if value is None and text.strip().lower() not in ("null", "~", "") else value


def parse_cuts(cuts, label: str) -> list:
    """Validate a cut specification and return it as ``[column, operator, value]`` triples.

    Accepts either the canonical ``"column operator value"`` string form or ready-made
    triples.  ``StageParameter`` does no element validation at all -- ``Param(list, ...)``
    stores whatever it is handed, and ``cast_value(list, "is_primary")`` cheerfully returns a
    ten-element list of single characters -- so every check has to happen here.

    Values are stripped to plain Python types.  That is not cosmetic: ceci writes the stage
    config with ``yaml.dump`` and reads it back with ``yaml.safe_load``, so a ``numpy`` scalar
    or a tuple is written as a ``!!python/...`` tag and then kills the stage subprocess at
    config-load time, before ``run()`` is reached, with a ``ConstructorError`` that mentions
    YAML and says nothing about cuts.  A ``np.int64`` is very easy to introduce by accident.
    """
    out = []
    for i, cut in enumerate(cuts or []):
        if isinstance(cut, bytes):
            cut = cut.decode()
        if isinstance(cut, str):
            match = _CUT_PATTERN.match(cut)
            if match is None:
                raise ValueError(
                    f"{label} cut {i} is not 'column operator value': {cut!r}.  "
                    f"Operators are {list(_OPERATORS)}"
                )
            column = match.group("column")
            operator = " ".join(match.group("operator").split())
            raw = match.group("value")
            if operator in _LIST_OPERATORS:
                value = [_parse_scalar(part) for part in raw.split(",") if part.strip() != ""]
            else:
                value = _parse_scalar(raw)
        elif isinstance(cut, (list, tuple)) and len(cut) == 3:
            column, operator, value = cut
            operator = " ".join(str(operator).split())
        else:
            raise ValueError(
                f"{label} cut {i} is neither a 'column operator value' string nor a "
                f"[column, operator, value] triple: {cut!r}"
            )

        if operator not in _OPERATORS:
            raise ValueError(
                f"{label} cut {i}: operator {operator!r} is not one of {list(_OPERATORS)}"
            )
        value = plain_python(value)
        if operator in _LIST_OPERATORS and not isinstance(value, list):
            # np.isin(arr, "DESI_DR1") does not raise -- it quietly matches nothing.
            raise ValueError(
                f"{label} cut {i}: {operator!r} needs a comma-separated list, got {value!r}"
            )
        if operator not in _LIST_OPERATORS and isinstance(value, list):
            raise ValueError(
                f"{label} cut {i}: {operator!r} needs a single value, got {value!r}"
            )
        out.append([str(column), operator, value])
    return out


def cut_columns(cuts) -> list:
    """The column names a cut specification refers to, in order, without duplicates."""
    return _ordered_unique([c[0] for c in cuts])


def _ordered_unique(names) -> list:
    seen, out = set(), []
    for name in names:
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def _coerce_operand(values: np.ndarray, value):
    """Match the operand's string flavour to the column's, so bytes vs str never bites."""
    if values.dtype.kind == "S":
        if isinstance(value, str):
            return value.encode()
        if isinstance(value, list):
            return [v.encode() if isinstance(v, str) else v for v in value]
    elif values.dtype.kind in "OU":
        if isinstance(value, bytes):
            return value.decode()
        if isinstance(value, list):
            return [v.decode() if isinstance(v, bytes) else v for v in value]
    return value


def _check_operand_type(values: np.ndarray, value, cut, label: str) -> None:
    """Reject an operand whose type cannot meaningfully compare with the column.

    The shape check below catches the old numpy behaviour, where ``bool_array == "True"``
    returned the scalar ``False``.  Numpy 2 instead returns an all-False *array* of the right
    shape, which passes every structural check and silently selects nothing -- so the operand
    type has to be checked directly.
    """
    operands = value if isinstance(value, list) else [value]
    kind = values.dtype.kind
    for operand in operands:
        if kind == "b" and not isinstance(operand, (bool, np.bool_)):
            raise ValueError(
                f"{label} cut {cut!r}: {operand!r} is a {type(operand).__name__}, but "
                f"{cut[0]!r} is boolean -- write True or False, unquoted"
            )
        if kind in "iuf" and isinstance(operand, (str, bytes)):
            raise ValueError(
                f"{label} cut {cut!r}: {operand!r} is a string, but {cut[0]!r} is numeric "
                f"(dtype {values.dtype}) -- remove the quotes"
            )
        if kind in "SU" and not isinstance(operand, (str, bytes)):
            raise ValueError(
                f"{label} cut {cut!r}: {operand!r} is a {type(operand).__name__}, but "
                f"{cut[0]!r} holds strings (dtype {values.dtype}) -- quote it"
            )


def _one_cut(values: np.ndarray, operator: str, value) -> np.ndarray:
    value = _coerce_operand(values, value)
    if operator == "==":
        return values == value
    if operator == "!=":
        return values != value
    if operator == "<":
        return values < value
    if operator == "<=":
        return values <= value
    if operator == ">":
        return values > value
    if operator == ">=":
        return values >= value
    if operator == "in":
        return np.isin(values, value)
    if operator == "not in":
        return ~np.isin(values, value)
    raise ValueError(f"unknown operator {operator!r}")  # pragma: no cover


def apply_cuts(get_column, cuts, n_rows: int, label: str, log=None) -> np.ndarray:
    """Build the boolean keep-mask for ``cuts``.

    ``get_column`` maps a name to a 1-D numpy array, so this works identically over a
    DataFrame and over an open HDF5 group.
    """
    keep = np.ones(n_rows, dtype=bool)
    for cut in cuts:
        column, operator, value = cut
        values = np.asarray(get_column(column))
        _check_operand_type(values, value, cut, label)
        mask = np.asarray(_one_cut(values, operator, value))
        if mask.dtype != bool or mask.shape != values.shape:  # pragma: no cover
            # A backstop.  On numpy 1 `bool_array == "True"` returned the *scalar* False
            # rather than an array, which selected nothing without a word of complaint.
            # _check_operand_type above now rejects those operands before they get here, so
            # this only fires for a dtype combination neither of us anticipated.
            raise ValueError(
                f"{label} cut {cut!r} on a column of dtype {values.dtype} produced a "
                f"{mask.dtype} of shape {mask.shape}, not a boolean mask of shape "
                f"{values.shape} -- check the operand's type"
            )
        before = int(keep.sum())
        keep &= mask
        if log is not None:
            after = int(keep.sum())
            extra = ""
            if values.dtype.kind == "f":
                n_nan = int((~np.isfinite(values) & ~mask).sum())
                if n_nan:
                    # every comparison is False against NaN, so this is a real selection
                    extra = f" ({n_nan:,} of them non-finite)"
            log(
                f"{label} cut {column} {operator} {value!r}: "
                f"{before:,} -> {after:,}{extra}"
            )
    return keep


def _validate_columns(available, wanted, label: str) -> None:
    """Fail once, in the parent, with a usable message -- not 718 times in the workers."""
    available = list(available)
    missing = [c for c in wanted if c not in available]
    if not missing:
        return
    lines = []
    for column in missing:
        near = difflib.get_close_matches(column, available, n=3)
        lines.append(f"  {column!r}" + (f"  (did you mean {near}?)" if near else ""))
    raise KeyError(
        f"{len(missing)} column(s) not in the {label}:\n" + "\n".join(lines)
        + f"\n{len(available)} available, first 20: {sorted(available)[:20]}"
    )


# ======================================================================================= #
# tract index over the shear file
# ======================================================================================= #

def tract_runs(dataset, chunk: int = TRACT_SCAN_CHUNK) -> dict:
    """``{tract: [(start, stop), ...]}`` -- the maximal runs of equal tract, in file order.

    Deliberately **not** ``np.searchsorted``.  A TXPipe catalog written by an MPI stage is the
    per-rank blocks concatenated, so its tract column is monotone only *within* a block: the
    0.1.0 DP2 metadetect file has 64 blocks and 63 inversions, and ``searchsorted`` answers
    confidently and wrongly on it.  Walking the runs is correct whether the column is sorted
    (one run per tract) or block-wise sorted (at most a handful), and costs about a second
    over 122M rows either way.

    Scanned in chunks so the full column is never resident, because the caller is about to
    fork a worker pool and everything resident is inherited copy-on-write.
    """
    n_rows = int(dataset.shape[0])
    runs: dict = {}
    run_start = 0
    previous = None
    for low in range(0, n_rows, chunk):
        block = np.asarray(dataset[low : low + chunk])
        if block.size == 0:  # pragma: no cover
            continue
        if previous is not None and int(block[0]) != previous:
            # the run open across the chunk seam ends exactly here
            runs.setdefault(previous, []).append((run_start, low))
            run_start = low
        edges = np.flatnonzero(block[1:] != block[:-1]) + 1
        for edge in edges.tolist():
            runs.setdefault(int(block[edge - 1]), []).append((run_start, low + edge))
            run_start = low + edge
        previous = int(block[-1])
    if previous is not None:
        runs.setdefault(previous, []).append((run_start, n_rows))

    covered = sum(stop - start for spans in runs.values() for start, stop in spans)
    if covered != n_rows:  # pragma: no cover
        raise RuntimeError(
            f"tract runs cover {covered:,} rows but the column has {n_rows:,} -- "
            "the run-length scan is wrong"
        )
    return runs


def _run_rows(spans) -> int:
    return sum(stop - start for start, stop in spans)


def _assert_closed(path: str) -> None:
    """Raise if ``path`` is still open in this process.

    Checked by name rather than by a global open-file count, because other libraries in the
    process may hold unrelated HDF5 files open and that is none of our business.
    """
    target = os.path.realpath(path)
    still_open = []
    for obj_id in h5py.h5f.get_obj_ids(h5py.h5f.OBJ_ALL, h5py.h5f.OBJ_FILE):
        try:
            name = h5py.h5f.get_name(obj_id).decode()
        except Exception:  # pragma: no cover - a closing id can vanish under us
            continue
        if os.path.realpath(name) == target:
            still_open.append(name)
    if still_open:  # pragma: no cover
        raise RuntimeError(
            f"{path} is still open in this process ({len(still_open)} handle(s)); it must be "
            "closed before forking, or a worker can inherit the parent's handle"
        )


# ======================================================================================= #
# stage 1 -- assign tracts to the reference catalog
# ======================================================================================= #

class AssignTract(RailStage):
    """Add an LSST skymap tract column to a catalog, from ra and dec alone."""

    name = "AssignTract"
    config_options = RailStage.config_options.copy()
    config_options.update(
        ra_col=Param(str, "ra", msg="right ascension column, degrees"),
        dec_col=Param(str, "dec", msg="declination column, degrees"),
        tract_col=Param(str, "tract", msg="name of the tract column to add"),
        num_rings=Param(int, DEFAULT_NUM_RINGS, msg="RingsSkyMap numRings; DP2 uses 120"),
        ra_start=Param(float, DEFAULT_RA_START, msg="RingsSkyMap raStart, degrees"),
        cuts=Param(
            list, [],
            msg="cuts applied before the tract is assigned, as 'column operator value' "
                "strings; see the module docstring",
        ),
        drop_non_finite=Param(
            bool, True,
            msg="drop rows whose ra or dec is non-finite; if False they are kept with "
                f"tract == {NO_TRACT}",
        ),
    )
    inputs = [("input", PqHandle)]
    outputs = [("output", PqHandle)]

    def __call__(self, sample: TableLike, **kwargs) -> PqHandle:
        self.set_data("input", sample)
        self.run()
        self.finalize()
        return self.get_handle("output")

    def run(self) -> None:
        config = self.config
        cuts = parse_cuts(config.cuts, "reference")

        data = as_dataframe(self.get_data("input"))
        _validate_columns(
            data.columns,
            _ordered_unique([config.ra_col, config.dec_col, *cut_columns(cuts)]),
            "reference catalog",
        )
        self.log_msg(f"{len(data):,} rows in")

        if cuts:
            keep = apply_cuts(
                lambda c: data[c].to_numpy(), cuts, len(data), "reference", self.log_msg
            )
            data = data[keep].reset_index(drop=True)

        tract = find_tract_id_array(
            data[config.ra_col].to_numpy(),
            data[config.dec_col].to_numpy(),
            num_rings=config.num_rings,
            ra_start=config.ra_start,
        )
        n_bad = int((tract == NO_TRACT).sum())
        if n_bad:
            self.log_msg(f"{n_bad:,} rows have a non-finite ra or dec")
        data = data.copy()
        data[config.tract_col] = tract

        if config.drop_non_finite and n_bad:
            data = data[data[config.tract_col] != NO_TRACT].reset_index(drop=True)

        if len(data):
            occupancy = data[config.tract_col].value_counts()
            self.log_msg(
                f"{len(data):,} rows over {len(occupancy):,} tracts "
                f"(median {occupancy.median():.0f}, max {occupancy.max():,} "
                f"in tract {occupancy.idxmax()})"
            )
        self.add_data("output", data)

    def log_msg(self, msg: str) -> None:
        print(f"  [{self.name}] {msg}", flush=True)


# ======================================================================================= #
# stage 2 -- the matcher
# ======================================================================================= #

# Set in the parent before the pool is created, so workers inherit them through fork and
# nothing is ever pickled.  Holds only numpy arrays and plain containers on purpose: an
# object-dtype pandas column of a million Python strings would have its refcount pages
# dirtied by a child merely reading it, giving every worker a private copy of data it does
# not need.
_SHARED: dict = {}
_WORKER: dict = {}


def _init_worker(config: dict) -> None:
    _WORKER.clear()
    _WORKER.update(config)
    _WORKER["group"] = None  # opened on the first task, not here


def _worker_group():
    """The shear group, opened once per worker and cached.

    Opened lazily rather than in the initializer because HDF5 keeps an open-file cache keyed
    by inode: a child that opens a file its parent still has open can be handed the parent's
    handle, complete with the parent's metadata cache.  The parent closes the file before
    forking, and nothing here opens it until the first task arrives.
    """
    if _WORKER["group"] is None:
        handle = h5py.File(_WORKER["shear_catalog"], "r", locking=False)
        _WORKER["file"] = handle
        _WORKER["group"] = handle[_WORKER["shear_group"]]
    return _WORKER["group"]


def _empty_result(tract: int, columns) -> dict:
    return dict(
        tract=tract,
        shear={c: np.empty(0) for c in columns},
        ref_idx=np.empty(0, dtype=np.int64),
        sep=np.empty(0, dtype="float32"),
        n_shear=0,
        n_ref=0,
        n_dup_ref=0,
        n_dedup=0,
        seconds=0.0,
    )


def _match_tract(tract: int) -> dict:
    """Match one tract.  Returns numpy arrays and reference *indices*, never reference data."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    started = time.time()
    emit = _WORKER["emit_cols"]
    halo = _WORKER["edge_halo_arcsec"]

    ref_idx = _SHARED["ref_rows"].get(tract)
    if ref_idx is None:
        ref_idx = np.empty(0, dtype=np.int64)
    if ref_idx.size == 0 and halo <= 0:
        # nothing to match against and no halo to widen it with, so read nothing at all
        return _empty_result(tract, emit)

    group = _worker_group()
    spans = _SHARED["runs"][tract]
    columns = {
        name: (
            np.concatenate([group[name][a:b] for a, b in spans])
            if len(spans) > 1
            else np.asarray(group[name][spans[0][0]:spans[0][1]])
        )
        for name in _WORKER["read_cols"]
    }
    n_shear = len(columns[_WORKER["shear_ra_col"]])

    keep = apply_cuts(columns.__getitem__, _WORKER["shear_cuts"], n_shear, "shear")
    columns = {name: values[keep] for name, values in columns.items()}

    n_dedup = 0
    dedup_col = _WORKER["dedup_col"]
    if dedup_col and len(columns[_WORKER["shear_ra_col"]]):
        # return_index makes np.unique use a stable sort, so this keeps the first occurrence
        # and therefore preserves file order.
        _, first = np.unique(columns[dedup_col], return_index=True)
        if first.size != len(columns[dedup_col]):
            n_dedup = len(columns[dedup_col]) - first.size
            first.sort()
            columns = {name: values[first] for name, values in columns.items()}

    shear_ra = columns[_WORKER["shear_ra_col"]]
    if shear_ra.size == 0:
        result = _empty_result(tract, emit)
        result.update(n_shear=n_shear, n_ref=int(ref_idx.size), n_dedup=n_dedup)
        return result

    if halo > 0:
        ref_idx = _halo_references(ref_idx, shear_ra, columns[_WORKER["shear_dec_col"]], halo)
    if ref_idx.size == 0:  # the halo found nothing either
        result = _empty_result(tract, emit)
        result.update(n_shear=n_shear, n_dedup=n_dedup)
        return result

    shear_c = SkyCoord(ra=shear_ra * u.degree,
                       dec=columns[_WORKER["shear_dec_col"]] * u.degree)
    ref_c = SkyCoord(ra=_SHARED["ref_ra"][ref_idx] * u.degree,
                     dec=_SHARED["ref_dec"][ref_idx] * u.degree)

    # Each shear object takes its nearest reference; one output row per matched shear object.
    nearest, d2d, _ = shear_c.match_to_catalog_sky(ref_c)
    separation = d2d.arcsec
    hit = separation <= _WORKER["max_sep_arcsec"]

    chosen = nearest[hit]
    return dict(
        tract=tract,
        shear={name: columns[name][hit] for name in emit},
        ref_idx=ref_idx[chosen],
        sep=separation[hit].astype("float32"),
        n_shear=n_shear,
        n_ref=int(ref_idx.size),
        # a reference claimed by two shear objects is a blend; reported, never removed
        n_dup_ref=int(chosen.size - np.unique(chosen).size),
        n_dedup=n_dedup,
        seconds=time.time() - started,
    )


def _halo_references(ref_idx, shear_ra, shear_dec, halo_arcsec: float) -> np.ndarray:
    """Widen a tract's reference set to anything within ``halo_arcsec`` of its sky span.

    Per-tract matching loses a true pair whose two halves straddle a tract boundary.  Adding a
    halo costs one box test against the globally dec-sorted reference arrays, and it cannot
    duplicate output rows: each shear object still takes exactly one nearest reference, and
    the ``is_primary`` cut means each shear object lives in exactly one tract.
    """
    pad = halo_arcsec / 3600.0
    dec_lo, dec_hi = shear_dec.min() - pad, shear_dec.max() + pad
    order = _SHARED["ref_dec_order"]
    sorted_dec = _SHARED["ref_dec_sorted"]
    lo, hi = np.searchsorted(sorted_dec, [dec_lo, dec_hi])
    candidates = order[lo:hi]
    if candidates.size == 0:
        return ref_idx

    # RA needs a wrap-aware window, so compare angular distance to the tract's ra midpoint.
    ra_lo, ra_hi = shear_ra.min(), shear_ra.max()
    if ra_hi - ra_lo > 180.0:  # the tract straddles ra = 0
        shifted = np.where(shear_ra > 180.0, shear_ra - 360.0, shear_ra)
        ra_lo, ra_hi = shifted.min(), shifted.max()
        ref_ra = _SHARED["ref_ra"][candidates]
        ref_ra = np.where(ref_ra > 180.0, ref_ra - 360.0, ref_ra)
    else:
        ref_ra = _SHARED["ref_ra"][candidates]
    scale = max(np.cos(np.deg2rad(np.clip((dec_lo + dec_hi) / 2.0, -89.9, 89.9))), 1e-6)
    ra_pad = pad / scale
    inside = (ref_ra >= ra_lo - ra_pad) & (ref_ra <= ra_hi + ra_pad)
    return np.union1d(ref_idx, candidates[inside])


class TractCrossMatch(RailStage):
    """Match a shear catalog to a reference catalog, one worker process per skymap tract."""

    name = "TractCrossMatch"
    # This stage forks a worker pool, so it must never be handed an MPI communicator: ceci
    # turns nprocess > 1 into `mpirun ... --mpi`, and forking out of an MPI-initialised
    # process on Slingshot hangs or corrupts rather than failing.  Setting this makes ceci
    # reject --mpi up front instead.
    parallel = False

    config_options = RailStage.config_options.copy()
    config_options.update(
        shear_catalog=Param(
            str, "",
            msg="path to the shear HDF5.  Passed as a path, not a DataHandle, because these "
                "catalogs run to hundreds of GB and must never be loaded whole",
        ),
        shear_group=Param(
            str, DEFAULT_SHEAR_GROUP,
            msg="HDF5 group holding the 1-D column datasets.  For metadetect the shear step "
                "IS the group -- there is no metaStep column",
        ),
        shear_cols=Param(list, DEFAULT_SHEAR_COLS, msg="shear columns carried to the output"),
        shear_ra_col=Param(str, "ra", msg="shear right ascension column, degrees"),
        shear_dec_col=Param(str, "dec", msg="shear declination column, degrees"),
        shear_tract_col=Param(str, "tract", msg="shear tract column; must already exist"),
        shear_cuts=Param(
            list, DEFAULT_SHEAR_CUTS,
            msg="cuts as 'column operator value' strings.  The default is_primary cut is a "
                "correctness precondition, not a quality cut -- see the module docstring",
        ),
        dedup_col=Param(
            str, "",
            msg="drop rows sharing a value in this column, per tract, keeping the first.  "
                "'' disables.  TXPipe catalogs written by an MPI stage need 'id'",
        ),
        ref_cols=Param(list, DEFAULT_REF_COLS, msg="reference columns carried to the output"),
        ref_ra_col=Param(str, "ra", msg="reference right ascension column, degrees"),
        ref_dec_col=Param(str, "dec", msg="reference declination column, degrees"),
        ref_tract_col=Param(str, "tract", msg="reference tract column, from AssignTract"),
        ref_cuts=Param(list, [], msg="cuts as 'column operator value' strings"),
        ref_rename=Param(
            dict, DEFAULT_REF_RENAME,
            msg="renames applied to ref_cols; both catalogs have ra and dec",
        ),
        max_sep_arcsec=Param(float, 0.75, msg="match radius, arcseconds"),
        edge_halo_arcsec=Param(
            float, 0.0,
            msg="also consider references this far outside the tract's sky span, which "
                "recovers pairs straddling a tract boundary.  0 matches tract-only",
        ),
        nproc=Param(
            int, 0,
            msg="worker processes; 0 takes OMP_NUM_THREADS, else the CPU affinity",
        ),
        max_inflight=Param(
            int, 0,
            msg="cap on outstanding tracts; 0 is unbounded, which is right unless the "
                "reference catalog is large enough for the results to pile up",
        ),
        sep_col=Param(str, "match_sep_arcsec", msg="name of the separation column to add"),
        tract_out_col=Param(str, "tract", msg="name of the tract column to add; '' to skip"),
        max_tracts=Param(
            int, 0,
            msg="debug: only the N largest tracts; 0 is all of them.  Note this is a biased "
                "sample -- on an MPI-written catalog a tract is large partly because it was "
                "written twice, so the dedup fraction here runs far above the whole-file one",
        ),
    )
    inputs = [("input", PqHandle)]
    outputs = [("output", PqHandle)]

    def __call__(self, sample: TableLike, **kwargs) -> PqHandle:
        self.set_data("input", sample)
        self.run()
        self.finalize()
        return self.get_handle("output")

    # -- helpers ------------------------------------------------------------------------ #

    def log_msg(self, msg: str) -> None:
        print(f"  [{self.name}] {msg}", flush=True)

    def _resolve_nproc(self, n_tasks: int) -> int:
        affinity = len(os.sched_getaffinity(0))
        requested = self.config.nproc
        if requested <= 0:
            # ceci puts threads_per_process on the subprocess as OMP_NUM_THREADS, so it is
            # the stage's real core budget.  os.cpu_count() would report the whole node.
            requested = int(os.environ.get("OMP_NUM_THREADS") or 0) or affinity
        return max(1, min(requested, affinity, max(n_tasks, 1)))

    # -- run ---------------------------------------------------------------------------- #

    def run(self) -> None:
        config = self.config
        if getattr(self, "comm", None) is not None:
            raise RuntimeError(
                f"{self.name} forks a worker pool and cannot run under MPI; "
                "it is declared parallel = False"
            )
        if not config.shear_catalog:
            raise ValueError("shear_catalog is required -- the path to the shear HDF5")

        shear_cuts = parse_cuts(config.shear_cuts, "shear")
        ref_cuts = parse_cuts(config.ref_cuts, "reference")

        # ---- reference side ---------------------------------------------------------- #
        reference = as_dataframe(self.get_data("input"))
        _validate_columns(
            reference.columns,
            _ordered_unique([
                config.ref_ra_col, config.ref_dec_col, config.ref_tract_col,
                *config.ref_cols, *cut_columns(ref_cuts),
            ]),
            "reference catalog",
        )
        self.log_msg(f"{len(reference):,} reference rows in")
        if ref_cuts:
            keep = apply_cuts(
                lambda c: reference[c].to_numpy(), ref_cuts, len(reference),
                "reference", self.log_msg,
            )
            reference = reference[keep]
        # positional indices into THIS frame are what the workers return
        reference = reference.reset_index(drop=True)

        # ---- shear side: validate and index, then close before forking ---------------- #
        emit_cols = _ordered_unique(
            [config.shear_ra_col, config.shear_dec_col, *config.shear_cols]
        )
        read_cols = _ordered_unique([
            *emit_cols, *cut_columns(shear_cuts),
            *( [config.dedup_col] if config.dedup_col else [] ),
        ])

        with h5py.File(config.shear_catalog, "r", locking=False) as handle:
            if config.shear_group not in handle:
                raise KeyError(
                    f"group {config.shear_group!r} is not in {config.shear_catalog}; "
                    f"top level holds {list(handle.keys())}"
                )
            group = handle[config.shear_group]
            _validate_columns(
                group.keys(),
                _ordered_unique([*read_cols, config.shear_tract_col]),
                f"shear catalog group {config.shear_group!r}",
            )
            n_shear_rows = int(group[config.shear_tract_col].shape[0])
            self.log_msg(
                f"{n_shear_rows:,} shear rows in {config.shear_group}, "
                f"reading {len(read_cols)} columns"
            )
            started = time.time()
            runs = tract_runs(group[config.shear_tract_col])
        max_runs = max((len(v) for v in runs.values()), default=0)
        self.log_msg(
            f"tract index: {len(runs):,} tracts, "
            f"{sum(len(v) for v in runs.values()):,} runs "
            f"(max {max_runs} per tract) in {time.time() - started:.1f} s"
        )

        # HDF5 caches open files by inode, so a child opening this path could otherwise be
        # handed the handle the parent is still holding, metadata cache and all.  Only this
        # file matters -- other libraries in the process may legitimately hold their own.
        gc.collect()
        _assert_closed(config.shear_catalog)

        # ---- tasks -------------------------------------------------------------------- #
        ref_tract = reference[config.ref_tract_col].to_numpy()
        order = np.argsort(ref_tract, kind="stable")
        sorted_tract = ref_tract[order]
        edges = np.flatnonzero(np.diff(sorted_tract)) + 1
        ref_rows = {
            int(sorted_tract[start]): order[start:stop].astype(np.int64)
            for start, stop in zip(
                np.concatenate([[0], edges]),
                np.concatenate([edges, [len(order)]]),
            )
        } if len(order) else {}

        # With a halo, a shear tract whose own references are all just outside it can still
        # match, so every shear tract is a candidate and the worker decides.  Without one,
        # only the tracts the two catalogs share can produce anything.
        candidates = set(runs) if config.edge_halo_arcsec > 0 else set(runs) & set(ref_rows)
        tracts = sorted(candidates, key=lambda t: -_run_rows(runs[t]))
        if config.max_tracts:
            tracts = tracts[: config.max_tracts]
        if not tracts:
            raise RuntimeError(
                "no tract is present in both catalogs -- "
                f"shear has {len(runs):,} tracts, the reference has {len(ref_rows):,}"
            )
        if config.edge_halo_arcsec > 0:
            shared = len(set(runs) & set(ref_rows))
            self.log_msg(
                f"edge halo is on, so all {len(tracts):,} shear tracts are read "
                f"({shared:,} of them hold references of their own)"
            )
        self.log_msg(
            f"{len(tracts):,} tracts in both catalogs; largest has "
            f"{_run_rows(runs[tracts[0]]):,} shear rows (tract {tracts[0]})"
        )

        # ---- shared payload, assigned before the fork --------------------------------- #
        ref_ra = np.ascontiguousarray(reference[config.ref_ra_col].to_numpy(), dtype="float64")
        ref_dec = np.ascontiguousarray(reference[config.ref_dec_col].to_numpy(), dtype="float64")
        _SHARED.clear()
        _SHARED.update(ref_ra=ref_ra, ref_dec=ref_dec, ref_rows=ref_rows, runs=runs)
        if config.edge_halo_arcsec > 0:
            dec_order = np.argsort(ref_dec, kind="stable")
            _SHARED["ref_dec_order"] = dec_order
            _SHARED["ref_dec_sorted"] = ref_dec[dec_order]

        worker_config = dict(
            shear_catalog=config.shear_catalog,
            shear_group=config.shear_group,
            shear_ra_col=config.shear_ra_col,
            shear_dec_col=config.shear_dec_col,
            shear_cuts=shear_cuts,
            dedup_col=config.dedup_col,
            read_cols=read_cols,
            emit_cols=emit_cols,
            max_sep_arcsec=config.max_sep_arcsec,
            edge_halo_arcsec=config.edge_halo_arcsec,
        )

        nproc = self._resolve_nproc(len(tracts))
        self.log_msg(
            f"matching at {config.max_sep_arcsec}\" with {nproc} process(es)"
            + (f", edge halo {config.edge_halo_arcsec}\"" if config.edge_halo_arcsec else "")
        )

        started = time.time()
        pieces = []
        # ceci redirects stdout to a file, so it is block-buffered; without this flush every
        # forked child inherits a copy of whatever is pending and re-emits it at exit.
        sys.stdout.flush()
        if nproc == 1:
            _init_worker(worker_config)
            try:
                for tract in tracts:
                    pieces.append(_match_tract(tract))
                    self._log_progress(pieces, len(tracts), started)
            finally:
                _close_worker()
        else:
            # `fork` explicitly: Python 3.14 makes forkserver the Linux default, which
            # re-imports the module and pickles the initargs, silently destroying the
            # copy-on-write sharing of _SHARED that this design depends on.
            context = mp.get_context("fork")
            with context.Pool(
                nproc, initializer=_init_worker, initargs=(worker_config,)
            ) as pool:
                for result in self._dispatch(pool, tracts):
                    pieces.append(result)
                    self._log_progress(pieces, len(tracts), started)

        self.add_data("output", self._assemble(pieces, reference))

    def _dispatch(self, pool, tracts):
        """Stream results, optionally with a bound on how many tracts are outstanding."""
        limit = self.config.max_inflight
        if limit <= 0:
            yield from pool.imap_unordered(_match_tract, tracts, chunksize=1)
            return
        pending, remaining = deque(), iter(tracts)
        for _ in range(limit):
            nxt = next(remaining, None)
            if nxt is None:
                break
            pending.append(pool.apply_async(_match_tract, (nxt,)))
        while pending:
            result = pending.popleft().get()
            nxt = next(remaining, None)
            if nxt is not None:
                pending.append(pool.apply_async(_match_tract, (nxt,)))
            yield result

    def _log_progress(self, pieces, n_total: int, started: float) -> None:
        done = len(pieces)
        if done % 50 and done != n_total:
            return
        matched = sum(len(p["sep"]) for p in pieces)
        elapsed = time.time() - started
        self.log_msg(
            f"... {done:,}/{n_total:,} tracts  {matched:,} matched  {elapsed:.0f} s"
        )

    def _assemble(self, pieces, reference: pd.DataFrame) -> pd.DataFrame:
        """Build the output frame one column at a time.

        Not ``pd.concat`` of hundreds of frames: that copies roughly twice, once to align and
        once to consolidate blocks, and peaks near double the final size.
        """
        config = self.config
        pieces = [p for p in pieces if len(p["sep"])]
        emit_cols = _ordered_unique(
            [config.shear_ra_col, config.shear_dec_col, *config.shear_cols]
        )

        n_matched = sum(len(p["sep"]) for p in pieces)
        n_shear = sum(p["n_shear"] for p in pieces)
        n_dup = sum(p["n_dup_ref"] for p in pieces)
        n_dedup = sum(p["n_dedup"] for p in pieces)
        if n_dedup:
            self.log_msg(f"{n_dedup:,} duplicate shear rows dropped on {config.dedup_col!r}")
        self.log_msg(f"{n_matched:,} matches from {n_shear:,} shear rows")
        if n_dup:
            self.log_msg(
                f"{n_dup:,} matches share a reference with another shear object "
                "(blends; kept, and separable downstream with group_col)"
            )

        if not pieces:
            self.log_msg("no matches -- writing an empty table")
            columns = {name: np.empty(0) for name in emit_cols}
            frame = pd.DataFrame(columns)
            for name in config.ref_cols:
                frame[config.ref_rename.get(name, name)] = np.empty(0)
            frame[config.sep_col] = np.empty(0, dtype="float32")
            if config.tract_out_col:
                frame[config.tract_out_col] = np.empty(0, dtype=np.int64)
            return frame

        frame = pd.DataFrame(
            {name: np.concatenate([p["shear"][name] for p in pieces]) for name in emit_cols}
        )

        matched_ref = reference.iloc[
            np.concatenate([p["ref_idx"] for p in pieces])
        ].reset_index(drop=True)
        for name in config.ref_cols:
            out_name = config.ref_rename.get(name, name)
            if out_name in frame.columns:
                raise ValueError(
                    f"reference column {name!r} would overwrite shear column {out_name!r}; "
                    "give it a different name through ref_rename"
                )
            frame[out_name] = matched_ref[name].to_numpy()

        frame[config.sep_col] = np.concatenate([p["sep"] for p in pieces])
        if config.tract_out_col:
            frame[config.tract_out_col] = np.concatenate(
                [np.full(len(p["sep"]), p["tract"], dtype=np.int64) for p in pieces]
            )

        separation = frame[config.sep_col].to_numpy()
        self.log_msg(
            f"separation median {np.median(separation):.4f}\", "
            f"90th {np.percentile(separation, 90):.4f}\""
        )
        return frame


def _close_worker() -> None:
    handle = _WORKER.pop("file", None)
    if handle is not None:  # pragma: no cover
        handle.close()
    _WORKER["group"] = None
