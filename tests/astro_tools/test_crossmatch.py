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
