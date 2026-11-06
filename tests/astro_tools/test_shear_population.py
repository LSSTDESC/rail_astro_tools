"""Unit tests for the shear population sampler."""

import h5py
import numpy as np
import pytest

from rail.core.data import DataStore
from rail.creation.degraders.shear_population import (
    ShearPopulationSampler,
    parse_ratio_cuts,
    selection_mask,
)
from rail.pipelines.degradation.shear_population_pipeline import ShearPopulationPipeline


@pytest.fixture(autouse=True)
def _run_in_tmp_dir(tmp_path, monkeypatch):
    """Stages write their outputs relative to the working directory."""
    monkeypatch.chdir(tmp_path)
    DataStore.allow_overwrite = True


N_ROWS = 1000


@pytest.fixture(name="shear_file")
def fixture_shear_file(tmp_path):
    """1000 rows; every 10th is non-primary, every 7th has a NaN g, r-i grows with row."""
    rng = np.random.default_rng(0)
    idx = np.arange(N_ROWS)
    mag_r = rng.uniform(20, 25, N_ROWS)
    mag_g = mag_r + 0.5
    mag_g[idx % 7 == 0] = np.nan
    mag_i = mag_r - idx / N_ROWS * 2.0  # r-i runs 0 -> 2, so |r-i| < 1 keeps the first half
    path = tmp_path / "shear.hdf5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("shear/ns")
        group.create_dataset("id", data=idx.astype("int64") + 100)
        group.create_dataset("ra", data=rng.uniform(0, 1, N_ROWS))
        group.create_dataset("dec", data=rng.uniform(0, 1, N_ROWS))
        group.create_dataset("tract", data=(idx // 100).astype("int64"))
        group.create_dataset("is_primary", data=idx % 10 != 0)
        group.create_dataset("mag_g", data=mag_g)
        group.create_dataset("mag_r", data=mag_r)
        group.create_dataset("mag_i", data=mag_i)
        group.create_dataset("flux", data=np.where(idx % 11 == 0, np.inf, 1.0))
    return str(path)


def _sample(path, name, **kwargs):
    config = dict(
        shear_catalog=path,
        shear_cols=["id", "tract", "mag_g", "mag_r", "mag_i"],
        nproc=1,
        chunk_rows=128,
    )
    config.update(kwargs)
    return ShearPopulationSampler.make_stage(name=name, **config)().data


def test_draws_exactly_n_sample_from_the_selection(shear_file):
    out = _sample(shear_file, "pop_exact", cuts=["is_primary == True"], n_sample=300)
    assert len(out) == 300
    assert out["id"].is_unique
    assert not np.any((out["id"].to_numpy() - 100) % 10 == 0)
    # file order is preserved
    assert np.all(np.diff(out["id"].to_numpy()) > 0)


def test_draw_is_seeded_and_independent_of_chunking_and_nproc(shear_file):
    a = _sample(shear_file, "pop_a", cuts=["is_primary == True"], n_sample=200, seed=3)
    b = _sample(shear_file, "pop_b", cuts=["is_primary == True"], n_sample=200, seed=3,
                chunk_rows=37, nproc=2)
    c = _sample(shear_file, "pop_c", cuts=["is_primary == True"], n_sample=200, seed=4)
    assert a["id"].tolist() == b["id"].tolist()
    assert a["id"].tolist() != c["id"].tolist()


def test_takes_everything_when_fewer_rows_pass(shear_file):
    out = _sample(
        shear_file, "pop_all",
        cuts=["is_primary == True"],
        require_finite=["mag_g"],
        color_cols=["mag_r", "mag_i"],
        abs_color_max=0.999,  # r-i = 2*idx/N, so idx < 499.5; clear of float ties
        n_sample=10_000,
    )
    idx = out["id"].to_numpy() - 100
    expected = np.arange(N_ROWS)
    expected = expected[(expected % 10 != 0) & (expected % 7 != 0) & (expected < 500)]
    assert idx.tolist() == expected.tolist()


def test_dedup_keeps_the_first_occurrence(tmp_path):
    path = tmp_path / "dup.hdf5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("shear/ns")
        group.create_dataset("id", data=np.array([1, 2, 2, 3, 1], dtype="int64"))
        group.create_dataset("row", data=np.arange(5))
    out = ShearPopulationSampler.make_stage(
        name="pop_dedup", shear_catalog=str(path), shear_cols=["id", "row"],
        dedup_col="id", n_sample=100, nproc=1,
    )().data
    assert out["row"].tolist() == [0, 1, 3]


def test_infinities_become_nan(shear_file):
    out = _sample(shear_file, "pop_inf", shear_cols=["id", "flux"], n_sample=N_ROWS)
    flux = out["flux"].to_numpy()
    assert not np.isinf(flux).any()
    assert np.isnan(flux).sum() == len(range(0, N_ROWS, 11))
    kept = _sample(shear_file, "pop_inf_kept", shear_cols=["id", "flux"], n_sample=N_ROWS,
                   nonfinite_to_nan=False)
    assert np.isinf(kept["flux"].to_numpy()).sum() == len(range(0, N_ROWS, 11))


def test_max_rows_limits_the_scan(shear_file):
    out = _sample(shear_file, "pop_max", n_sample=N_ROWS, max_rows=250)
    assert out["id"].max() - 100 < 250
    assert len(out) == 250


def test_selection_mask_counts():
    columns = dict(a=np.array([1.0, 2.0, np.nan, 4.0]), b=np.array([0.0, 0.0, 0.0, 9.0]))
    keep, alone, cumulative = selection_mask(
        columns.__getitem__, 4, [["a", "<", 3.0]], ["a"], ["a", "b"], 2.0,
    )
    assert keep.tolist() == [True, False, False, False]
    # |a-b| = 1, 2, nan, 5: only the first is < 2
    assert list(alone.values()) == [2, 3, 1]
    assert list(cumulative.values()) == [2, 2, 1]


def test_nan_colour_policy():
    columns = dict(g=np.array([20.0, np.nan, 30.0]), r=np.array([20.5, 21.0, 20.0]))
    strict, _, _ = selection_mask(columns.__getitem__, 3, [], [], ["g", "r"], 5.0)
    lenient, alone, _ = selection_mask(
        columns.__getitem__, 3, [], [], ["g", "r"], 5.0, color_nan_passes=True,
    )
    assert strict.tolist() == [True, False, False]
    assert lenient.tolist() == [True, True, False]
    assert list(alone) == ["|g - r| < 5 or NaN"]


def test_ratio_cut(shear_file):
    # mag_i / mag_r falls from 1 towards ~0.9 as the row index grows; NaN never reaches it
    out = _sample(shear_file, "pop_ratio", ratio_cuts=["mag_i / mag_r > 0.97"], n_sample=N_ROWS)
    with h5py.File(shear_file) as fh:
        ratio = fh["shear/ns/mag_i"][()] / fh["shear/ns/mag_r"][()]
    assert (out["id"].to_numpy() - 100).tolist() == np.flatnonzero(ratio > 0.97).tolist()


def test_ratio_cut_with_a_zero_denominator_fails_the_row():
    columns = dict(t=np.array([1.0, 1.0, 1.0]), p=np.array([1.0, 0.0, np.nan]))
    keep, alone, _ = selection_mask(
        columns.__getitem__, 3, [], [], [], 0.0, parse_ratio_cuts(["t / p > 0.5"]),
    )
    assert keep.tolist() == [True, False, False]
    assert list(alone) == ["t / p > 0.5"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("gauss_T / gauss_psfReconvolved_T > 0.5",
         [["gauss_T", "gauss_psfReconvolved_T", ">", 0.5]]),
        ("a/b<=2", None),  # operator needs surrounding whitespace, as in ordinary cuts
        ("a / b ~ 2", None),
        ("a > 2", None),
    ],
)
def test_parse_ratio_cuts(text, expected):
    if expected is None:
        with pytest.raises(ValueError):
            parse_ratio_cuts([text])
    else:
        assert parse_ratio_cuts([text]) == expected


def test_requires_a_shear_catalog():
    stage = ShearPopulationSampler.make_stage(name="pop_req", shear_catalog="")
    with pytest.raises(ValueError, match="shear_catalog is required"):
        stage()


def test_reports_a_missing_column(shear_file):
    with pytest.raises(KeyError, match="mag_y"):
        _sample(shear_file, "pop_missing", cuts=["mag_y < 25"])


def test_colour_cut_needs_two_columns(shear_file):
    with pytest.raises(ValueError, match="two color_cols"):
        _sample(shear_file, "pop_colour", color_cols=["mag_r"], abs_color_max=1.0)


def test_is_not_parallel():
    assert ShearPopulationSampler.parallel is False


def test_pipeline_rejects_unknown_keys():
    with pytest.raises(KeyError, match="unknown catalog_config"):
        ShearPopulationPipeline(catalog_config=dict(not_a_key=1))


def test_pipeline_rejects_a_bad_cut():
    with pytest.raises(ValueError, match="operator"):
        ShearPopulationPipeline(catalog_config=dict(cuts=["mfrac ~ 0.1"]))
