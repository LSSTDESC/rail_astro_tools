"""Unit tests for the tract crossmatch stages."""

import os

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

from rail.core.data import DataStore
from rail.creation.degraders.crossmatch import (
    DEFAULT_NUM_TRACTS,
    NO_TRACT,
    AssignTract,
    TractCrossMatch,
    apply_cuts,
    find_tract_id_array,
    parse_cuts,
    ring_nums,
    num_tracts,
    tract_runs,
)


@pytest.fixture(autouse=True)
def _run_in_tmp_dir(tmp_path, monkeypatch):
    """Stages write their outputs relative to the working directory.

    Without this every test drops an ``output_<name>.pq`` wherever pytest happened to be
    started, which for this repo is the project root.
    """
    monkeypatch.chdir(tmp_path)


# --------------------------------------------------------------------------------------- #
# skymap
# --------------------------------------------------------------------------------------- #

def test_ring_geometry():
    counts = ring_nums(120)
    assert len(counts) == 120
    assert num_tracts(120) == DEFAULT_NUM_TRACTS == 18938
    # the two equatorial rings are the widest, and land on exactly 242.0 before truncation
    assert counts[59] == counts[60] == 243
    # a ring and its mirror about the equator hold the same number of tracts
    assert np.array_equal(counts, counts[::-1])


def test_poles_and_wrapping():
    assert find_tract_id_array([0.0], [-90.0])[0] == 0
    assert find_tract_id_array([0.0], [90.0])[0] == DEFAULT_NUM_TRACTS - 1
    # tract 0 of a ring is centred on ra_start, so these three are the same tract
    wrapped = find_tract_id_array([359.999, -0.001, 0.001], [0.0, 0.0, 0.0])
    assert len(set(wrapped.tolist())) == 1


def test_non_finite_positions_are_flagged_not_truncated():
    """NaN cast to int64 is INT64_MIN, which would pass for a tract and match nothing."""
    out = find_tract_id_array([np.nan, 10.0, np.inf], [0.0, np.nan, 5.0])
    assert (out == NO_TRACT).all()


def test_ra_start_shifts_tracts():
    at_zero = find_tract_id_array([0.0], [0.0], ra_start=0.0)[0]
    shifted = find_tract_id_array([0.0], [0.0], ra_start=10.0)[0]
    assert at_zero != shifted


# --------------------------------------------------------------------------------------- #
# the run-length tract index
# --------------------------------------------------------------------------------------- #

class _FakeDataset:
    """Just enough of an h5py dataset for tract_runs."""

    def __init__(self, values):
        self._values = np.asarray(values)
        self.shape = self._values.shape

    def __getitem__(self, item):
        return self._values[item]


@pytest.mark.parametrize("chunk", [1, 3, 4, 7, 8, 16, 32, 1000])
def test_tract_runs_across_chunk_seams(chunk):
    """The value must be allowed to change exactly on a chunk boundary."""
    values = np.array([5] * 10 + [7] * 10 + [9] * 12, dtype=np.int64)
    assert tract_runs(_FakeDataset(values), chunk=chunk) == {
        5: [(0, 10)],
        7: [(10, 20)],
        9: [(20, 32)],
    }


@pytest.mark.parametrize("chunk", [2, 4, 15])
def test_tract_runs_recurring_value(chunk):
    """An unsorted column -- the case np.searchsorted gets silently wrong."""
    values = np.array([1] * 5 + [2] * 5 + [1] * 5, dtype=np.int64)
    assert tract_runs(_FakeDataset(values), chunk=chunk) == {
        1: [(0, 5), (10, 15)],
        2: [(5, 10)],
    }


def test_tract_runs_tile_the_column():
    rng = np.random.default_rng(0)
    values = np.repeat(rng.integers(0, 20, 500), rng.integers(1, 8, 500))
    runs = tract_runs(_FakeDataset(values), chunk=64)
    covered = sum(stop - start for spans in runs.values() for start, stop in spans)
    assert covered == len(values)
    for tract, spans in runs.items():
        for start, stop in spans:
            assert (values[start:stop] == tract).all()


# --------------------------------------------------------------------------------------- #
# cut specifications
# --------------------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "text, expected",
    [
        ("is_primary == True", ["is_primary", "==", True]),
        ("mag_i < 25.5", ["mag_i", "<", 25.5]),
        ("z_flag >= 3", ["z_flag", ">=", 3]),
        ("ref_cat != 252", ["ref_cat", "!=", 252]),
        ('ref_cat != "252"', ["ref_cat", "!=", "252"]),
        ("z_source in A,B", ["z_source", "in", ["A", "B"]]),
        ("z_type not in p,g", ["z_type", "not in", ["p", "g"]]),
    ],
)
def test_parse_cuts_string_form(text, expected):
    assert parse_cuts([text], "t") == [expected]


def test_parse_cuts_accepts_triples_and_strips_numpy():
    assert parse_cuts([["is_primary", "==", True]], "t") == [["is_primary", "==", True]]
    assert parse_cuts([("z_source", "in", ["A"])], "t") == [["z_source", "in", ["A"]]]
    out = parse_cuts([["tract", "==", np.int64(4572)]], "t")
    assert out == [["tract", "==", 4572]]
    assert not isinstance(out[0][2], np.generic)


@pytest.mark.parametrize(
    "bad",
    [
        "is_primary",          # no operator
        "mag_i =< 25",         # not an operator
        "z_source in ",        # no operand
        ["a", "b"],            # not a triple
        [["mag_i", "<", [1, 2]]],   # list operand for a scalar operator
        [["z_source", "in", "A"]],  # scalar operand for a list operator
    ],
)
def test_parse_cuts_rejects(bad):
    with pytest.raises(ValueError):
        parse_cuts([bad] if isinstance(bad, (str, list)) and not _is_cut_list(bad) else bad, "t")


def _is_cut_list(value):
    return isinstance(value, list) and value and isinstance(value[0], list)


def test_parsed_cuts_survive_the_ceci_yaml_round_trip():
    """ceci writes with yaml.dump and reads with yaml.safe_load; a tag there is fatal."""
    cuts = parse_cuts(
        ["is_primary == True", "mag_i < 25.5", "z_source in A,B",
         ["tract", "==", np.int64(4572)]],
        "t",
    )
    assert yaml.safe_load(yaml.dump(cuts)) == cuts


@pytest.mark.parametrize(
    "column, cut",
    [
        # quoting a bool: numpy 2 returns an all-False array, which no structural check sees
        (np.array([True, False, True]), ["is_primary", "==", "True"]),
        (np.array([21.0, 22.0, 23.0]), ["mag_i", "<", "25"]),
        (np.array(["a", "b"], dtype="U1"), ["z_source", "==", 3]),
    ],
)
def test_apply_cuts_rejects_a_mistyped_operand(column, cut):
    """A silently-empty selection is the worst outcome; make it loud instead."""
    columns = {cut[0]: column}
    with pytest.raises(ValueError):
        apply_cuts(columns.__getitem__, [cut], len(column), "t")


def test_apply_cuts_matches_bytes_columns_against_str_operands():
    columns = {"z_source": np.array([b"DESI", b"SDSS", b"DESI"])}
    mask = apply_cuts(columns.__getitem__, [["z_source", "in", ["DESI"]]], 3, "t")
    assert mask.tolist() == [True, False, True]


# --------------------------------------------------------------------------------------- #
# the stages
# --------------------------------------------------------------------------------------- #

@pytest.fixture(name="tiny_catalogs")
def fixture_tiny_catalogs(tmp_path):
    """One shear object and one reference, 0.2 arcsec apart, in a known tract."""
    ra, dec = 45.0, -10.0
    tract = int(find_tract_id_array([ra], [dec])[0])
    offset = 0.2 / 3600.0 / np.cos(np.deg2rad(dec))
    path = tmp_path / "shear.hdf5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("shear/ns")
        group.create_dataset("id", data=np.array([1, 2], dtype="int64"))
        group.create_dataset("ra", data=np.array([ra, ra + 1e-3]))
        group.create_dataset("dec", data=np.array([dec, dec]))
        group.create_dataset("tract", data=np.array([tract, tract], dtype="int64"))
        group.create_dataset("is_primary", data=np.array([True, False]))
    reference = pd.DataFrame(
        dict(ref_id=["R1"], ra=[ra + offset], dec=[dec], z=[0.5])
    )
    return str(path), reference, tract


def test_assign_tract_adds_the_column(tiny_catalogs):
    DataStore.allow_overwrite = True
    _, reference, tract = tiny_catalogs
    stage = AssignTract.make_stage(name="assign_test")
    out = stage(reference).data
    assert out["tract"].tolist() == [tract]


def test_crossmatch_end_to_end(tiny_catalogs):
    DataStore.allow_overwrite = True
    path, reference, _ = tiny_catalogs
    reference = AssignTract.make_stage(name="assign_xm")(reference).data
    stage = TractCrossMatch.make_stage(
        name="xm_test",
        shear_catalog=path,
        shear_cols=["id"],
        ref_cols=["ref_id", "z"],
        max_sep_arcsec=0.75,
        nproc=1,
    )
    out = stage(reference).data
    # the non-primary row is cut, so only object 1 survives and it matches R1
    assert out["id"].tolist() == [1]
    assert out["ref_id"].tolist() == ["R1"]
    assert out["match_sep_arcsec"][0] == pytest.approx(0.2, abs=1e-3)


def test_crossmatch_requires_a_shear_catalog(tiny_catalogs):
    DataStore.allow_overwrite = True
    _, reference, _ = tiny_catalogs
    reference = AssignTract.make_stage(name="assign_req")(reference).data
    stage = TractCrossMatch.make_stage(name="xm_req", shear_catalog="")
    with pytest.raises(ValueError, match="shear_catalog is required"):
        stage(reference)


def test_crossmatch_reports_a_missing_column(tiny_catalogs):
    DataStore.allow_overwrite = True
    path, reference, _ = tiny_catalogs
    reference = AssignTract.make_stage(name="assign_miss")(reference).data
    stage = TractCrossMatch.make_stage(
        name="xm_miss", shear_catalog=path, shear_cols=["mag_z"],
        ref_cols=["ref_id", "z"], nproc=1,
    )
    with pytest.raises(KeyError, match="mag_z"):
        stage(reference)


def test_crossmatch_is_not_parallel():
    """Forking under MPI would hang rather than fail, so ceci must refuse --mpi."""
    assert TractCrossMatch.parallel is False


# --------------------------------------------------------------------------------------- #
# edge cases
# --------------------------------------------------------------------------------------- #

def _write_shear(path, **columns):
    """Write a one-group shear HDF5 from column arrays."""
    with h5py.File(path, "w") as handle:
        group = handle.create_group("shear/ns")
        for name, values in columns.items():
            group.create_dataset(name, data=np.asarray(values))
    return str(path)


@pytest.fixture(name="make_shear")
def fixture_make_shear(tmp_path):
    counter = {"n": 0}

    def _make(**columns):
        counter["n"] += 1
        return _write_shear(tmp_path / f"shear{counter['n']}.hdf5", **columns)

    return _make


# -- skymap ------------------------------------------------------------------------------ #

def test_find_tract_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="differ in shape"):
        find_tract_id_array([1.0, 2.0], [3.0])


# -- the cut grammar --------------------------------------------------------------------- #

def test_parse_cuts_falls_back_to_the_raw_string_on_bad_yaml():
    """An unclosed bracket is not valid YAML; the operand stays the literal text."""
    assert parse_cuts(["x == [1, 2"], "t") == [["x", "==", "[1, 2"]]


def test_parse_cuts_accepts_a_bytes_cut():
    assert parse_cuts([b"is_primary == True"], "t") == [["is_primary", "==", True]]


@pytest.mark.parametrize(
    "operator, value, expected",
    [
        ("==", 2, [False, True, False]),
        ("!=", 2, [True, False, True]),
        ("<", 2, [True, False, False]),
        ("<=", 2, [True, True, False]),
        (">", 2, [False, False, True]),
        (">=", 2, [False, True, True]),
        ("in", [1, 3], [True, False, True]),
        ("not in", [1, 3], [False, True, False]),
    ],
)
def test_every_operator(operator, value, expected):
    columns = {"x": np.array([1, 2, 3])}
    mask = apply_cuts(columns.__getitem__, [["x", operator, value]], 3, "t")
    assert mask.tolist() == expected


def test_str_operands_match_a_bytes_column_for_list_operators():
    columns = {"s": np.array([b"a", b"b", b"c"])}
    mask = apply_cuts(columns.__getitem__, [["s", "not in", ["a", "c"]]], 3, "t")
    assert mask.tolist() == [False, True, False]


def test_bytes_operands_match_a_str_column():
    columns = {"s": np.array(["a", "b"], dtype=object)}
    assert apply_cuts(columns.__getitem__, [["s", "==", b"a"]], 2, "t").tolist() == [True, False]
    mask = apply_cuts(columns.__getitem__, [["s", "in", [b"b"]]], 2, "t")
    assert mask.tolist() == [False, True]


def test_apply_cuts_logs_and_counts_non_finite_drops():
    messages = []
    columns = {"m": np.array([1.0, np.nan, 5.0])}
    mask = apply_cuts(columns.__getitem__, [["m", "<", 3.0]], 3, "t", messages.append)
    assert mask.tolist() == [True, False, False]
    assert any("non-finite" in m for m in messages)


def test_unknown_operator_is_rejected_by_the_low_level_helper():
    with pytest.raises(ValueError, match="not one of"):
        parse_cuts([["x", "~=", 1]], "t")


# -- the fork guard ----------------------------------------------------------------------- #

def test_assert_closed_detects_an_open_file(tmp_path):
    from rail.creation.degraders.crossmatch import _assert_closed

    path = _write_shear(tmp_path / "open.hdf5", ra=np.array([1.0]))
    _assert_closed(path)  # nothing open yet
    handle = h5py.File(path, "r")
    try:
        with pytest.raises(RuntimeError, match="still open"):
            _assert_closed(path)
    finally:
        handle.close()
    _assert_closed(path)


# -- AssignTract ---------------------------------------------------------------------------- #

def test_assign_tract_applies_cuts_first():
    DataStore.allow_overwrite = True
    frame = pd.DataFrame({"ra": [45.0, 46.0], "dec": [-10.0, -10.0], "keep": [True, False]})
    out = AssignTract.make_stage(name="at_cuts", cuts=["keep == True"])(frame).data
    assert len(out) == 1


def test_assign_tract_drops_or_keeps_non_finite():
    DataStore.allow_overwrite = True
    frame = pd.DataFrame({"ra": [45.0, np.nan], "dec": [-10.0, -10.0]})
    dropped = AssignTract.make_stage(name="at_drop")(frame).data
    assert len(dropped) == 1
    kept = AssignTract.make_stage(name="at_keep", drop_non_finite=False)(frame).data
    assert len(kept) == 2
    assert (kept["tract"] == NO_TRACT).sum() == 1


# -- TractCrossMatch: configuration and guards ---------------------------------------------- #

def test_run_refuses_an_mpi_communicator(tiny_catalogs):
    DataStore.allow_overwrite = True
    path, reference, _ = tiny_catalogs
    reference = AssignTract.make_stage(name="a_mpi")(reference).data
    stage = TractCrossMatch.make_stage(name="xm_mpi", shear_catalog=path, ref_cols=["ref_id"])
    stage._comm = object()  # ceci exposes `comm` as a read-only property over this
    with pytest.raises(RuntimeError, match="cannot run under MPI"):
        stage(reference)


def test_missing_shear_group_is_reported(tiny_catalogs):
    DataStore.allow_overwrite = True
    path, reference, _ = tiny_catalogs
    reference = AssignTract.make_stage(name="a_grp")(reference).data
    stage = TractCrossMatch.make_stage(
        name="xm_grp", shear_catalog=path, shear_group="shear/nope", ref_cols=["ref_id"]
    )
    with pytest.raises(KeyError, match="shear/nope"):
        stage(reference)


def test_no_common_tract_is_reported(tiny_catalogs, make_shear):
    DataStore.allow_overwrite = True
    _, reference, tract = tiny_catalogs
    elsewhere = make_shear(
        id=np.array([1], dtype="int64"), ra=np.array([200.0]), dec=np.array([30.0]),
        tract=np.array([tract + 5000], dtype="int64"), is_primary=np.array([True]),
    )
    reference = AssignTract.make_stage(name="a_none")(reference).data
    stage = TractCrossMatch.make_stage(
        name="xm_none", shear_catalog=elsewhere, shear_cols=["id"], ref_cols=["ref_id"], nproc=1
    )
    with pytest.raises(RuntimeError, match="no tract is present in both"):
        stage(reference)


def test_reference_column_collision_is_reported(tiny_catalogs):
    DataStore.allow_overwrite = True
    path, reference, _ = tiny_catalogs
    reference = AssignTract.make_stage(name="a_coll")(reference).data
    stage = TractCrossMatch.make_stage(
        name="xm_coll", shear_catalog=path, shear_cols=["id"],
        ref_cols=["ref_id", "ra"], ref_rename={}, nproc=1,
    )
    with pytest.raises(ValueError, match="would overwrite"):
        stage(reference)


def test_nproc_follows_omp_num_threads(monkeypatch, tiny_catalogs):
    path, _, _ = tiny_catalogs
    stage = TractCrossMatch.make_stage(name="xm_nproc", shear_catalog=path, nproc=0)
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert stage._resolve_nproc(10) == min(3, len(os.sched_getaffinity(0)))
    # never more workers than there are tracts to work on
    assert stage._resolve_nproc(1) == 1


# -- TractCrossMatch: worker paths ------------------------------------------------------------ #

def test_reference_cuts_are_applied(tiny_catalogs):
    DataStore.allow_overwrite = True
    path, reference, _ = tiny_catalogs
    reference = AssignTract.make_stage(name="a_refcut")(reference).data
    stage = TractCrossMatch.make_stage(
        name="xm_refcut", shear_catalog=path, shear_cols=["id"],
        ref_cols=["ref_id"], ref_cuts=["ref_id == NOPE"], nproc=1,
    )
    with pytest.raises(RuntimeError, match="no tract is present in both"):
        stage(reference)


def test_everything_cut_gives_an_empty_table(tiny_catalogs):
    """All shear rows removed by the cuts -- the empty-output branch."""
    DataStore.allow_overwrite = True
    path, reference, _ = tiny_catalogs
    reference = AssignTract.make_stage(name="a_empty")(reference).data
    out = TractCrossMatch.make_stage(
        name="xm_empty", shear_catalog=path, shear_cols=["id"], ref_cols=["ref_id", "z"],
        shear_cuts=["is_primary == False", "id > 1000"], nproc=1,
    )(reference).data
    assert len(out) == 0
    assert "ref_id" in out.columns and "match_sep_arcsec" in out.columns


def test_dedup_keeps_the_first_of_each_duplicate(make_shear):
    DataStore.allow_overwrite = True
    ra, dec = 45.0, -10.0
    tract = int(find_tract_id_array([ra], [dec])[0])
    path = make_shear(
        id=np.array([7, 7, 8], dtype="int64"),
        ra=np.array([ra, ra, ra + 1.0e-3]),
        dec=np.array([dec, dec, dec]),
        tract=np.full(3, tract, dtype="int64"),
        is_primary=np.array([True, True, True]),
    )
    reference = AssignTract.make_stage(name="a_dedup")(
        pd.DataFrame(dict(ref_id=["R1"], ra=[ra], dec=[dec]))
    ).data
    common = dict(shear_catalog=path, shear_cols=["id"], ref_cols=["ref_id"], nproc=1)
    with_dedup = TractCrossMatch.make_stage(
        name="xm_dedup", dedup_col="id", **common
    )(reference).data
    without = TractCrossMatch.make_stage(name="xm_nodedup", **common)(reference).data
    assert sorted(with_dedup["id"].tolist()) == [7]
    assert sorted(without["id"].tolist()) == [7, 7]


def test_max_tracts_limits_the_work(make_shear):
    DataStore.allow_overwrite = True
    positions = [(45.0, -10.0), (60.0, -10.0), (75.0, -10.0)]
    tracts = [int(find_tract_id_array([r], [d])[0]) for r, d in positions]
    path = make_shear(
        id=np.arange(3, dtype="int64"),
        ra=np.array([r for r, _ in positions]),
        dec=np.array([d for _, d in positions]),
        tract=np.array(tracts, dtype="int64"),
        is_primary=np.ones(3, dtype=bool),
    )
    reference = AssignTract.make_stage(name="a_max")(
        pd.DataFrame(dict(ref_id=[f"R{i}" for i in range(3)],
                          ra=[r for r, _ in positions], dec=[d for _, d in positions]))
    ).data
    out = TractCrossMatch.make_stage(
        name="xm_max", shear_catalog=path, shear_cols=["id"], ref_cols=["ref_id"],
        nproc=1, max_tracts=1,
    )(reference).data
    assert len(out) == 1


def test_runs_with_a_worker_pool(make_shear):
    """The multiprocessing path: _SHARED must reach the workers through the fork."""
    DataStore.allow_overwrite = True
    positions = [(45.0, -10.0), (60.0, -10.0)]
    tracts = [int(find_tract_id_array([r], [d])[0]) for r, d in positions]
    path = make_shear(
        id=np.arange(2, dtype="int64"),
        ra=np.array([r for r, _ in positions]),
        dec=np.array([d for _, d in positions]),
        tract=np.array(tracts, dtype="int64"),
        is_primary=np.ones(2, dtype=bool),
    )
    reference = AssignTract.make_stage(name="a_pool")(
        pd.DataFrame(dict(ref_id=["R0", "R1"],
                          ra=[r for r, _ in positions], dec=[d for _, d in positions]))
    ).data
    common = dict(shear_catalog=path, shear_cols=["id"], ref_cols=["ref_id"])
    serial = TractCrossMatch.make_stage(name="xm_ser", nproc=1, **common)(reference).data
    pooled = TractCrossMatch.make_stage(name="xm_pool", nproc=2, **common)(reference).data
    bounded = TractCrossMatch.make_stage(
        name="xm_bound", nproc=2, max_inflight=1, **common
    )(reference).data
    for out in (pooled, bounded):
        assert sorted(out["id"].tolist()) == sorted(serial["id"].tolist())
        assert len(out) == 2


# -- the edge halo ------------------------------------------------------------------------- #

def test_halo_reads_tracts_that_hold_no_references_of_their_own(make_shear):
    DataStore.allow_overwrite = True
    ra, dec = 45.0, -10.0
    tract = int(find_tract_id_array([ra], [dec])[0])
    # a shear tract far from any reference: the halo finds nothing and it returns empty
    path = make_shear(
        id=np.array([1], dtype="int64"), ra=np.array([ra]), dec=np.array([dec]),
        tract=np.array([tract], dtype="int64"), is_primary=np.array([True]),
    )
    reference = AssignTract.make_stage(name="a_halo_far")(
        pd.DataFrame(dict(ref_id=["R1"], ra=[ra + 10.0], dec=[dec]))
    ).data
    out = TractCrossMatch.make_stage(
        name="xm_halo_far", shear_catalog=path, shear_cols=["id"], ref_cols=["ref_id"],
        edge_halo_arcsec=2.0, nproc=1,
    )(reference).data
    assert len(out) == 0


def test_halo_reference_selection_handles_the_ra_wrap():
    """A tract straddling ra = 0 must not be treated as spanning the whole sky."""
    from rail.creation.degraders.crossmatch import _SHARED, _halo_references

    ref_ra = np.array([359.9995, 180.0, 0.0005])
    ref_dec = np.array([-10.0, -10.0, -10.0])
    order = np.argsort(ref_dec, kind="stable")
    _SHARED.clear()
    _SHARED.update(
        ref_ra=ref_ra, ref_dec=ref_dec,
        ref_dec_order=order, ref_dec_sorted=ref_dec[order],
    )
    shear_ra = np.array([359.999, 0.001])   # spans the wrap
    shear_dec = np.array([-10.0, -10.0])
    out = _halo_references(np.empty(0, dtype=np.int64), shear_ra, shear_dec, 10.0)
    assert set(out.tolist()) == {0, 2}      # the far-side reference at ra=180 is excluded
    _SHARED.clear()


def test_scalar_str_operand_matches_a_bytes_column():
    columns = {"s": np.array([b"a", b"b"])}
    mask = apply_cuts(columns.__getitem__, [["s", "==", "a"]], 2, "t")
    assert mask.tolist() == [True, False]


def test_one_cut_rejects_an_operator_parse_cuts_would_have_caught():
    """Defensive: reachable only by bypassing parse_cuts."""
    from rail.creation.degraders.crossmatch import _one_cut

    with pytest.raises(ValueError, match="unknown operator"):
        _one_cut(np.array([1, 2]), "~~", 1)


def test_worker_skips_a_tract_with_no_references(make_shear):
    """Without a halo the parent never dispatches such a tract, but the guard still holds."""
    from rail.creation.degraders.crossmatch import (
        _SHARED, _WORKER, _init_worker, _match_tract,
    )

    path = make_shear(
        id=np.array([1], dtype="int64"), ra=np.array([45.0]), dec=np.array([-10.0]),
        tract=np.array([7809], dtype="int64"), is_primary=np.array([True]),
    )
    _SHARED.clear()
    _SHARED.update(ref_ra=np.array([45.0]), ref_dec=np.array([-10.0]),
                   ref_rows={}, runs={7809: [(0, 1)]})
    _init_worker(dict(
        shear_catalog=path, shear_group="shear/ns", shear_ra_col="ra", shear_dec_col="dec",
        shear_cuts=[], dedup_col="", read_cols=["ra", "dec", "id"],
        emit_cols=["ra", "dec", "id"], max_sep_arcsec=0.75, edge_halo_arcsec=0.0,
    ))
    try:
        result = _match_tract(7809)
        assert result["n_shear"] == 0 and len(result["sep"]) == 0
    finally:
        _SHARED.clear()
        _WORKER.clear()


def test_halo_returns_early_when_no_reference_is_near():
    from rail.creation.degraders.crossmatch import _SHARED, _halo_references

    ref_dec = np.array([80.0])
    order = np.argsort(ref_dec, kind="stable")
    _SHARED.clear()
    _SHARED.update(ref_ra=np.array([10.0]), ref_dec=ref_dec,
                   ref_dec_order=order, ref_dec_sorted=ref_dec[order])
    try:
        given = np.array([3], dtype=np.int64)
        # the only reference is 90 degrees away in dec, so nothing is added
        assert _halo_references(given, np.array([10.0]), np.array([-10.0]), 5.0) is given
    finally:
        _SHARED.clear()


def test_inflight_window_wider_than_the_task_list(make_shear):
    """The priming loop must stop when it runs out of tracts.

    Two tracts, not one: with a single task nproc collapses to 1 and the serial path runs
    instead, so _dispatch is never reached.
    """
    DataStore.allow_overwrite = True
    positions = [(45.0, -10.0), (60.0, -10.0)]
    tracts = [int(find_tract_id_array([r], [d])[0]) for r, d in positions]
    path = make_shear(
        id=np.arange(2, dtype="int64"),
        ra=np.array([r for r, _ in positions]),
        dec=np.array([d for _, d in positions]),
        tract=np.array(tracts, dtype="int64"),
        is_primary=np.ones(2, dtype=bool),
    )
    reference = AssignTract.make_stage(name="a_inflight")(
        pd.DataFrame(dict(ref_id=["R0", "R1"],
                          ra=[r for r, _ in positions], dec=[d for _, d in positions]))
    ).data
    out = TractCrossMatch.make_stage(
        name="xm_inflight", shear_catalog=path, shear_cols=["id"], ref_cols=["ref_id"],
        nproc=2, max_inflight=8,
    )(reference).data
    assert len(out) == 2


def test_pipeline_rejects_an_unknown_catalog_config_key():
    from rail.pipelines.degradation.crossmatch_pipeline import CrossMatchPipeline

    with pytest.raises(KeyError, match="unknown catalog_config keys"):
        CrossMatchPipeline(dict(not_a_real_key=1))
