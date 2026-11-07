"""Draw a fixed-size random *population* sample from a shear-catalog HDF5, after a selection.

``SOMResampler`` reshapes a reference test pool to look like "the galaxies the photo-z will be
applied to".  That target is this sample: the shear catalog's science selection, drawn
uniformly over the rows that pass it.  Uniform over rows means area-weighted -- a tract with
more selected galaxies contributes proportionally more, which a per-tract quota would not do.

``ShearPopulationSampler``
    Two passes over the HDF5 with a forked worker pool, one task per row chunk:

    1. read only the selection columns, build the keep-mask, return the kept row indices
       (and the dedup ids, if asked for);
    2. after the parent has drawn exactly ``n_sample`` of the kept rows without replacement,
       read the output columns chunk by chunk and keep the drawn rows.

    So no column is ever resident in full, and nothing is pickled to the workers except row
    ranges.  The draw is seeded and independent of ``nproc`` and ``chunk_rows``.

THE SELECTION
-------------
Four kinds of cut, applied in this order and logged one by one:

``cuts``
    ``"column operator value"`` strings, exactly as ``TractCrossMatch.shear_cuts`` -- see
    :mod:`rail.creation.degraders.crossmatch` for the grammar.  Each is also reported
    *individually* (the fraction it alone keeps), which is what a published selection
    table quotes.
``ratio_cuts``
    ``"numerator / denominator operator value"`` strings, e.g.
    ``"gauss_T / gauss_psfReconvolved_T > 0.5"`` -- the metadetect star/galaxy size cut.  The
    operator and value follow the ``cuts`` grammar; a zero or non-finite denominator fails.
``require_finite``
    Columns that must be finite.  Separate from the cuts because ``x == x`` is not expressible
    in the cut grammar, and because a NaN magnitude is often a *non-detection* that a
    downstream stage replaces -- so it is a choice, not a default.
``color_cols`` + ``abs_color_max``
    ``|c[k] - c[k+1]| < abs_color_max`` for every adjacent pair.  This is the one cut in the
    DP2 anacal "diagnostics2" weak-lensing selection that is not a single-column cut.  A NaN
    colour fails it unless ``color_nan_passes``.  That choice is not cosmetic: a metadetect
    magnitude is NaN for a non-positive flux, so a strict colour cut on g-r silently re-imposes
    most of a g-band flux-flag cut (measured: 97 % of the |g-r| < 5 failures are NaN g).

``is_primary`` is **not** a default here, unlike in ``TractCrossMatch``.  There it is a
correctness precondition of per-tract matching; here it is part of the science selection,
and a catalog may not even have the column (the TXPipe anacal file does not).
"""

import gc
import multiprocessing as mp
import os
import re
import sys
import time

import h5py
import numpy as np
import pandas as pd
from ceci.config import StageParameter as Param

from rail.core.data import PqHandle
from rail.core.stage import RailStage
from rail.creation.degraders.crossmatch import (
    DEFAULT_SHEAR_GROUP,
    _assert_closed,
    _ordered_unique,
    _validate_columns,
    apply_cuts,
    cut_columns,
    parse_cuts,
)

DEFAULT_POPULATION_COLS = [
    "id", "ra", "dec", "tract",
    "mag_g", "mag_r", "mag_i", "mag_z",
    "mag_err_g", "mag_err_r", "mag_err_i", "mag_err_z",
]

# 4M rows per task: ~32 MB per float64 column, so a worker holding twenty selection columns
# stays well under a gigabyte, while 146M rows still make ~35 tasks to spread over the pool.
DEFAULT_CHUNK_ROWS = 1 << 22

_WORKER: dict = {}

_RATIO_PATTERN = re.compile(r"^\s*(?P<num>[^\s/]+)\s*/\s*(?P<den>[^\s/]+)\s+(?P<rest>\S.*)$")


def parse_ratio_cuts(cuts) -> list:
    """``"num / den operator value"`` strings -> ``[num, den, operator, value]``.

    The operator and value go through :func:`parse_cuts`, so they obey exactly the grammar
    and type checks of an ordinary cut.
    """
    out = []
    for i, cut in enumerate(cuts or []):
        match = _RATIO_PATTERN.match(cut) if isinstance(cut, str) else None
        if match is None:
            raise ValueError(
                f"ratio cut {i} is not 'numerator / denominator operator value': {cut!r}"
            )
        _, operator, value = parse_cuts([f"ratio {match.group('rest')}"], "ratio")[0]
        out.append([match.group("num"), match.group("den"), operator, value])
    return out


def _init_worker(config: dict) -> None:
    _WORKER.clear()
    _WORKER.update(config)
    _WORKER["group"] = None


def _worker_group():
    """Opened lazily, per worker, for the reason given in ``crossmatch._worker_group``."""
    if _WORKER["group"] is None:
        handle = h5py.File(_WORKER["shear_catalog"], "r", locking=False)
        _WORKER["file"] = handle
        _WORKER["group"] = handle[_WORKER["shear_group"]]
    return _WORKER["group"]


def selection_mask(get_column, n_rows: int, cuts, require_finite, color_cols,
                   abs_color_max: float, ratio_cuts=(), color_nan_passes: bool = False):
    """Keep-mask for one block of rows, plus the per-step counts.

    Returns ``(keep, individual, cumulative)``: ``individual[name]`` counts the rows that step
    alone keeps, ``cumulative[name]`` the rows surviving it and everything before it.  Both
    are ordered dicts in application order, so they print as a selection table.
    """
    keep = np.ones(n_rows, dtype=bool)
    individual, cumulative = {}, {}

    def _step(name, mask):
        nonlocal keep
        individual[name] = int(mask.sum())
        keep &= mask
        cumulative[name] = int(keep.sum())

    for cut in cuts:
        column, operator, value = cut
        _step(f"{column} {operator} {value}", apply_cuts(get_column, [cut], n_rows, "shear"))
    for num, den, operator, value in ratio_cuts:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = (np.asarray(get_column(num), dtype=float)
                     / np.asarray(get_column(den), dtype=float))
        ratio[~np.isfinite(ratio)] = np.nan  # x/0 fails rather than passing as +inf
        _step(f"{num} / {den} {operator} {value}",
              apply_cuts(lambda _: ratio, [["ratio", operator, value]], n_rows, "ratio"))
    for column in require_finite:
        _step(f"isfinite({column})", np.isfinite(np.asarray(get_column(column), dtype=float)))
    if abs_color_max > 0:
        for blue, red in zip(color_cols[:-1], color_cols[1:]):
            with np.errstate(invalid="ignore"):
                colour = (np.asarray(get_column(blue), dtype=float)
                          - np.asarray(get_column(red), dtype=float))
                mask = np.abs(colour) < abs_color_max
                if color_nan_passes:
                    mask |= np.isnan(colour)
                _step(f"|{blue} - {red}| < {abs_color_max:g}"
                      + (" or NaN" if color_nan_passes else ""), mask)
    return keep, individual, cumulative


def _select_chunk(span):
    """Pass 1: kept global row indices for one chunk, and their dedup ids."""
    low, high = span
    group = _worker_group()
    cache = {}

    def get_column(name):
        if name not in cache:
            cache[name] = np.asarray(group[name][low:high])
        return cache[name]

    keep, individual, cumulative = selection_mask(
        get_column, high - low, _WORKER["cuts"], _WORKER["require_finite"],
        _WORKER["color_cols"], _WORKER["abs_color_max"], _WORKER["ratio_cuts"],
        _WORKER["color_nan_passes"],
    )
    kept = np.flatnonzero(keep).astype(np.int64)
    dedup = _WORKER["dedup_col"]
    ids = get_column(dedup)[kept] if dedup else None
    return dict(span=span, kept=kept + low, ids=ids,
                individual=individual, cumulative=cumulative)


def _read_chunk(task):
    """Pass 2: the output columns at the drawn rows of one chunk."""
    low, high, rows = task
    group = _worker_group()
    local = rows - low
    return {name: np.asarray(group[name][low:high])[local] for name in _WORKER["shear_cols"]}


def _close_worker() -> None:
    handle = _WORKER.pop("file", None)
    if handle is not None:
        handle.close()
    _WORKER["group"] = None


class ShearPopulationSampler(RailStage):
    """Selection-cut, fixed-size, uniform random sample of a shear-catalog HDF5."""

    name = "ShearPopulationSampler"
    # Forks a worker pool, so it must never run under MPI; see TractCrossMatch.parallel.
    parallel = False

    config_options = RailStage.config_options.copy()
    config_options.update(
        shear_catalog=Param(
            str, "",
            msg="path to the shear HDF5; a path, not a DataHandle, because the catalogs this "
                "targets are hundreds of GB",
        ),
        shear_group=Param(str, DEFAULT_SHEAR_GROUP, msg="HDF5 group of 1-D column datasets"),
        shear_cols=Param(list, DEFAULT_POPULATION_COLS, msg="columns written to the output"),
        cuts=Param(
            list, [],
            msg="selection as 'column operator value' strings, as in TractCrossMatch",
        ),
        ratio_cuts=Param(
            list, [],
            msg="cuts on a column ratio, as 'numerator / denominator operator value' strings",
        ),
        require_finite=Param(list, [], msg="columns that must be finite"),
        color_cols=Param(
            list, [],
            msg="magnitude columns, blue to red; adjacent pairs form the colours that "
                "abs_color_max applies to",
        ),
        abs_color_max=Param(
            float, 0.0, msg="keep |adjacent colour| < this for every pair; 0 disables",
        ),
        color_nan_passes=Param(
            bool, False,
            msg="let a NaN colour (a non-detection in either band) pass the colour cut",
        ),
        dedup_col=Param(
            str, "",
            msg="drop rows repeating a value in this column, keeping the first in file order; "
                "'' disables",
        ),
        n_sample=Param(int, 1_000_000, msg="rows to draw; all selected rows if fewer pass"),
        seed=Param(int, 42, msg="seed of the draw"),
        chunk_rows=Param(int, DEFAULT_CHUNK_ROWS, msg="rows per worker task"),
        nproc=Param(
            int, 0, msg="worker processes; 0 takes OMP_NUM_THREADS, else the CPU affinity",
        ),
        nonfinite_to_nan=Param(
            bool, True,
            msg="write +/-inf in float output columns as NaN, so one non-detect convention "
                "reaches the SOM",
        ),
        max_rows=Param(int, 0, msg="debug: only the first N rows of the file; 0 is all"),
    )
    inputs = []
    outputs = [("output", PqHandle)]

    def __call__(self, **kwargs) -> PqHandle:
        self.run()
        self.finalize()
        return self.get_handle("output")

    def log_msg(self, msg: str) -> None:
        print(f"  [{self.name}] {msg}", flush=True)

    def _resolve_nproc(self, n_tasks: int) -> int:
        affinity = len(os.sched_getaffinity(0))
        requested = self.config.nproc
        if requested <= 0:  # pragma: no cover
            requested = int(os.environ.get("OMP_NUM_THREADS") or 0) or affinity
        return max(1, min(requested, affinity, max(n_tasks, 1)))

    def _map(self, func, tasks, worker_config):
        """Run ``func`` over ``tasks`` in order, forking a pool unless nproc is 1."""
        nproc = self._resolve_nproc(len(tasks))
        sys.stdout.flush()  # see TractCrossMatch.run: forked children re-emit pending output
        if nproc == 1:
            _init_worker(worker_config)
            try:
                return [func(task) for task in tasks]
            finally:
                _close_worker()
        # fork explicitly -- Python 3.14 defaults to forkserver on Linux
        with mp.get_context("fork").Pool(
            nproc, initializer=_init_worker, initargs=(worker_config,)
        ) as pool:
            return list(pool.imap(func, tasks, chunksize=1))

    def run(self) -> None:
        config = self.config
        if getattr(self, "comm", None) is not None:  # pragma: no cover
            raise RuntimeError(f"{self.name} forks a worker pool and cannot run under MPI")
        if not config.shear_catalog:
            raise ValueError("shear_catalog is required -- the path to the shear HDF5")
        if config.n_sample <= 0:  # pragma: no cover
            raise ValueError(f"n_sample must be positive, got {config.n_sample}")
        if config.chunk_rows <= 0:  # pragma: no cover
            raise ValueError(f"chunk_rows must be positive, got {config.chunk_rows}")
        if config.abs_color_max > 0 and len(config.color_cols) < 2:
            raise ValueError("abs_color_max needs at least two color_cols")

        cuts = parse_cuts(config.cuts, "shear")
        ratio_cuts = parse_ratio_cuts(config.ratio_cuts)
        shear_cols = _ordered_unique(list(config.shear_cols))
        selection_cols = _ordered_unique([
            *cut_columns(cuts), *[c for r in ratio_cuts for c in r[:2]],
            *config.require_finite,
            *(config.color_cols if config.abs_color_max > 0 else []),
            *([config.dedup_col] if config.dedup_col else []),
        ])

        with h5py.File(config.shear_catalog, "r", locking=False) as handle:
            if config.shear_group not in handle:  # pragma: no cover
                raise KeyError(
                    f"group {config.shear_group!r} is not in {config.shear_catalog}; "
                    f"top level holds {list(handle.keys())}"
                )
            group = handle[config.shear_group]
            _validate_columns(
                group.keys(), _ordered_unique([*shear_cols, *selection_cols]),
                f"shear catalog group {config.shear_group!r}",
            )
            lengths = {group[name].shape[0] for name in _ordered_unique([*shear_cols, *selection_cols])}
            if len(lengths) != 1:  # pragma: no cover
                raise ValueError(f"columns in {config.shear_group!r} differ in length: {lengths}")
            n_rows = lengths.pop()
        gc.collect()
        _assert_closed(config.shear_catalog)

        if config.max_rows:
            n_rows = min(n_rows, config.max_rows)
        spans = [(low, min(low + config.chunk_rows, n_rows))
                 for low in range(0, n_rows, config.chunk_rows)]
        self.log_msg(
            f"{n_rows:,} rows in {config.shear_group}; {len(selection_cols)} selection "
            f"columns, {len(shear_cols)} output columns, {len(spans)} chunks, "
            f"{self._resolve_nproc(len(spans))} process(es)"
        )

        worker_config = dict(
            shear_catalog=config.shear_catalog,
            shear_group=config.shear_group,
            cuts=cuts,
            ratio_cuts=ratio_cuts,
            require_finite=list(config.require_finite),
            color_cols=list(config.color_cols),
            abs_color_max=float(config.abs_color_max),
            color_nan_passes=bool(config.color_nan_passes),
            dedup_col=config.dedup_col,
            shear_cols=shear_cols,
        )

        # ---- pass 1: selection ------------------------------------------------------- #
        started = time.time()
        pieces = self._map(_select_chunk, spans, worker_config)
        kept = np.concatenate([p["kept"] for p in pieces]) if pieces else np.empty(0, np.int64)
        self.log_msg(f"selection pass: {time.time() - started:.1f} s")
        self._log_selection(pieces, n_rows)

        if config.dedup_col and kept.size:
            ids = np.concatenate([p["ids"] for p in pieces])
            # return_index uses a stable sort, so the first occurrence in file order wins
            _, first = np.unique(ids, return_index=True)
            n_dup = kept.size - first.size
            if n_dup:
                kept = kept[np.sort(first)]
            self.log_msg(f"dedup on {config.dedup_col!r}: {n_dup:,} repeated rows dropped")
        del pieces

        # ---- the draw ---------------------------------------------------------------- #
        n_selected = int(kept.size)
        if n_selected == 0:  # pragma: no cover
            raise RuntimeError("the selection keeps no rows")
        if n_selected <= config.n_sample:
            self.log_msg(
                f"only {n_selected:,} rows pass, fewer than n_sample={config.n_sample:,}; "
                "taking all of them"
            )
            chosen = kept
        else:
            rng = np.random.default_rng(config.seed)
            chosen = np.sort(kept[rng.choice(n_selected, size=config.n_sample, replace=False)])
        self.log_msg(
            f"drew {chosen.size:,} of {n_selected:,} selected rows "
            f"(rate {chosen.size / n_selected:.4e}, seed {config.seed})"
        )

        # ---- pass 2: read the drawn rows --------------------------------------------- #
        started = time.time()
        bounds = np.searchsorted(chosen, [s[0] for s in spans] + [n_rows])
        tasks = [
            (low, high, chosen[bounds[i]:bounds[i + 1]])
            for i, (low, high) in enumerate(spans)
            if bounds[i + 1] > bounds[i]
        ]
        parts = self._map(_read_chunk, tasks, worker_config)
        frame = pd.DataFrame(
            {name: np.concatenate([p[name] for p in parts]) for name in shear_cols}
        )
        self.log_msg(f"read pass: {time.time() - started:.1f} s over {len(tasks)} chunks")

        if config.nonfinite_to_nan:
            for name in shear_cols:
                values = frame[name].to_numpy()
                if values.dtype.kind == "f":
                    n_inf = int(np.isinf(values).sum())
                    if n_inf:
                        frame[name] = np.where(np.isinf(values), np.nan, values)
                        self.log_msg(f"{name}: {n_inf:,} +/-inf written as NaN")
        self.add_data("output", frame)

    def _log_selection(self, pieces, n_rows: int) -> None:
        if not pieces or not pieces[0]["individual"]:
            self.log_msg("no selection applied")
            return
        names = list(pieces[0]["individual"])
        self.log_msg(f"{'step':52s} {'alone':>8s} {'cumulative':>11s}")
        for name in names:
            alone = sum(p["individual"][name] for p in pieces)
            cumulative = sum(p["cumulative"][name] for p in pieces)
            self.log_msg(
                f"{name:52s} {100.0 * alone / n_rows:7.2f}% {100.0 * cumulative / n_rows:10.2f}%"
            )
        n_kept = sum(p["kept"].size for p in pieces)
        self.log_msg(f"{'ALL':52s} {'':8s} {100.0 * n_kept / n_rows:10.2f}%  ({n_kept:,} rows)")
