"""Unit tests for the pz_prepare curation stages and the table helpers they share."""

import pickle

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import yaml
from astropy.io import fits

from rail.core.data import DataStore
from rail.creation.degraders._frame_utils import (
    as_dataframe,
    frame_to_hdf5_dict,
    plain_python,
    to_numpy_frame,
)
from rail.creation.degraders.pz_prepare import (
    COSMOS2020_REF,
    COSMOS_WEB_REF,
    PAUS_COSMOS_REF,
    MergeFullPZStage,
    ReferenceHoldoutSplitter,
    MagRedshiftDownsampler,
    PreTrainTestSplitter,
    SOMResampler,
    SOMZCellResampler,
    allocate_cell_targets,
    draw_cell_members,
    redshift_bins,
    rebin_pz,
    som_features,
    train_som,
    zcell_fractions,
)
from rail.pipelines.degradation.hscfy_pz_prepare import set_stage_threads


def _somoclu_can_train() -> bool:
    """somoclu ships a compiled core that is missing in some environments."""
    try:
        import somoclu

        somoclu.Somoclu(2, 2).train(
            np.random.default_rng(0).random((10, 2)).astype(np.float32), epochs=1
        )
    except Exception:  # pragma: no cover - environment dependent
        return False
    return True


needs_somoclu = pytest.mark.skipif(
    not _somoclu_can_train(), reason="somoclu cannot train a map in this environment"
)


# --------------------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _run_in_tmp_dir(tmp_path, monkeypatch):
    """Stages write their outputs relative to the working directory.

    Without this every test drops an ``output_<name>.pq`` wherever pytest happened to be
    started, which for this repo is the project root.
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture(name="catalog")
def fixture_catalog():
    """A small reference catalog with the awkward cases the stages have to survive.

    Every ``ref_id`` appears twice, which is the blended-detection case the group-aware split
    exists for.
    """
    rng = np.random.default_rng(7)
    n_groups = 60
    ref_id = np.repeat([f"R{i:03d}" for i in range(n_groups)], 2)
    n = len(ref_id)
    return pd.DataFrame(
        {
            "ref_id": ref_id,
            "z": np.repeat(rng.uniform(0.05, 2.5, n_groups), 2),
            "z_flag": np.repeat(rng.choice([3.0, 4.0], n_groups), 2),
            "z_source": np.repeat(
                rng.choice(["DESI_DR1", "SDSS_DR17", "TINY_SURVEY"], n_groups), 2
            ),
            "lsst_i_mag_gauss2": np.repeat(rng.uniform(19.0, 24.0, n_groups), 2),
            "object_id": np.arange(n),
        }
    )


# --------------------------------------------------------------------------------------- #
# the shared table helpers
# --------------------------------------------------------------------------------------- #

def test_as_dataframe_passes_through_a_dataframe(catalog):
    assert as_dataframe(catalog) is catalog


def test_as_dataframe_converts_a_pyarrow_table(catalog):
    """PqHandle yields a pyarrow Table under ceci but a DataFrame interactively."""
    out = as_dataframe(pa.Table.from_pandas(catalog))
    assert isinstance(out, pd.DataFrame)
    assert set(out.columns) >= set(catalog.columns)


def test_to_numpy_frame_casts_extension_dtypes():
    frame = pd.DataFrame(
        {
            "s": pd.array(["a", "b"], dtype="string"),
            "i": pd.array([1, None], dtype="Int64"),
            "b": pd.array([True, None], dtype="boolean"),
            "f": pd.array([1.5, 2.5], dtype="Float64"),
        }
    )
    out = to_numpy_frame(frame)
    assert out["s"].dtype.kind == "S"
    assert out["i"].dtype == np.float64 and np.isnan(out["i"][1])
    assert out["b"].dtype == bool and out["b"][1] == False  # noqa: E712
    assert out["f"].dtype == np.float64


def test_frame_to_hdf5_dict_keeps_string_columns():
    """tables_io drops columns it cannot map to HDF5 with a warning, not an error."""
    frame = pd.DataFrame(
        {"z_source": pd.array(["DESI_DR1", "SDSS_DR17"], dtype="string"), "z": [0.1, 0.2]}
    )
    out = frame_to_hdf5_dict(frame)
    assert sorted(out) == ["z", "z_source"]
    assert len(out["z_source"]) == 2


@pytest.mark.parametrize(
    "value, expected",
    [
        (np.int64(3), 3),
        (np.float32(1.5), pytest.approx(1.5)),
        ((1, 2), [1, 2]),
        (np.array([1, 2]), [1, 2]),
        ({"a": np.int64(1)}, {"a": 1}),
        ("plain", "plain"),
    ],
)
def test_plain_python_strips_numpy_and_tuples(value, expected):
    out = plain_python(value)
    assert out == expected
    assert yaml.safe_load(yaml.dump(out)) == out


# --------------------------------------------------------------------------------------- #
# PreTrainTestSplitter
# --------------------------------------------------------------------------------------- #

def test_split_fractions_and_disjointness(catalog):
    DataStore.allow_overwrite = True
    stage = PreTrainTestSplitter.make_stage(name="split_basic", train_frac=0.8)
    train, test = stage(catalog)
    train, test = train.data, test.data
    assert len(train) + len(test) == len(catalog)
    assert len(train) == int(len(catalog) * 0.8)
    # every row lands on exactly one side
    assert not set(train["object_id"]) & set(test["object_id"])


def test_split_adds_the_output_redshift_column(catalog):
    DataStore.allow_overwrite = True
    stage = PreTrainTestSplitter.make_stage(name="split_zcol")
    train, _ = stage(catalog)
    assert "redshift" in train.data.columns
    assert train.data["redshift"].dtype == np.float64


def test_split_applies_the_redshift_range(catalog):
    DataStore.allow_overwrite = True
    stage = PreTrainTestSplitter.make_stage(
        name="split_zcut", redshift_min=0.5, redshift_max=1.0, shuffle=False
    )
    train, test = stage(catalog)
    both = pd.concat([train.data, test.data])
    assert len(both) == int(((catalog.z > 0.5) & (catalog.z < 1.0)).sum())
    assert both["z"].between(0.5, 1.0, inclusive="neither").all()


def test_split_is_reproducible(catalog):
    DataStore.allow_overwrite = True
    first = PreTrainTestSplitter.make_stage(name="split_s1", seed=99)(catalog)[0].data
    second = PreTrainTestSplitter.make_stage(name="split_s2", seed=99)(catalog)[0].data
    assert first["object_id"].tolist() == second["object_id"].tolist()


def test_group_split_keeps_a_label_on_one_side(catalog):
    """The leak this guards against is invisible in object_id -- only the label repeats."""
    DataStore.allow_overwrite = True
    grouped = PreTrainTestSplitter.make_stage(name="split_grp", group_col="ref_id")
    train, test = grouped(catalog)
    shared = set(train.data["ref_id"]) & set(test.data["ref_id"])
    assert shared == set()

    ungrouped = PreTrainTestSplitter.make_stage(name="split_ungrp")
    train2, test2 = ungrouped(catalog)
    # object_id is disjoint either way, which is exactly why the leak goes unnoticed
    assert not set(train2.data["object_id"]) & set(test2.data["object_id"])
    assert set(train2.data["ref_id"]) & set(test2.data["ref_id"])


def test_quality_cut_and_nan_policy(catalog):
    DataStore.allow_overwrite = True
    catalog = catalog.copy()
    catalog.loc[:3, "z_flag"] = np.nan

    kept = PreTrainTestSplitter.make_stage(
        name="split_qk", apply_quality_cut=True, quality_min=4.0, quality_nan_policy="keep"
    )(catalog)
    dropped = PreTrainTestSplitter.make_stage(
        name="split_qd", apply_quality_cut=True, quality_min=4.0, quality_nan_policy="drop"
    )(catalog)
    n_kept = len(kept[0].data) + len(kept[1].data)
    n_dropped = len(dropped[0].data) + len(dropped[1].data)
    assert n_kept == n_dropped + 4


def test_quality_nan_policy_is_validated(catalog):
    DataStore.allow_overwrite = True
    stage = PreTrainTestSplitter.make_stage(
        name="split_qbad", apply_quality_cut=True, quality_nan_policy="whatever"
    )
    with pytest.raises(ValueError, match="quality_nan_policy"):
        stage(catalog)


def test_missing_redshift_column_is_reported(catalog):
    DataStore.allow_overwrite = True
    stage = PreTrainTestSplitter.make_stage(name="split_miss", redshift_col="nope")
    with pytest.raises(KeyError, match="nope"):
        stage(catalog)


def test_missing_group_column_is_reported(catalog):
    DataStore.allow_overwrite = True
    stage = PreTrainTestSplitter.make_stage(name="split_gmiss", group_col="nope")
    with pytest.raises(KeyError, match="nope"):
        stage(catalog)


def test_require_detected_drops_nan_and_sentinel(catalog):
    """A row without a detection in a required band leaves both sides; others are untouched."""
    DataStore.allow_overwrite = True
    data = catalog.copy()
    data["mag_r"] = 22.0
    data.loc[[0, 5], "mag_r"] = np.nan
    data.loc[[9], "lsst_i_mag_gauss2"] = 40.0
    stage = PreTrainTestSplitter.make_stage(
        name="split_det", require_detected=["mag_r", "lsst_i_mag_gauss2"], nondetect_val=40.0
    )
    train, test = stage(data)
    kept = set(train.data["object_id"]) | set(test.data["object_id"])
    assert kept == set(data["object_id"]) - {0, 5, 9}
    # without a sentinel only the NaNs go
    stage = PreTrainTestSplitter.make_stage(name="split_det_nan", require_detected=["mag_r"])
    train, test = stage(data)
    assert len(train.data) + len(test.data) == len(data) - 2


def test_require_detected_missing_column(catalog):
    DataStore.allow_overwrite = True
    stage = PreTrainTestSplitter.make_stage(name="split_det_miss", require_detected=["mag_q"])
    with pytest.raises(KeyError, match="mag_q"):
        stage(catalog)


def test_holdout_split_holds_out_every_shared_id(catalog):
    """Rows sharing an object id OR a ref_id with the holdout go to pre-test, nothing else."""
    DataStore.allow_overwrite = True
    holdout = pd.DataFrame({"object_id": [0, 50, 999], "ref_id": ["R010", "XXX", "YYY"]})
    stage = ReferenceHoldoutSplitter.make_stage(
        name="split_hold", holdout_cols=["object_id", "ref_id"], redshift_min=0.0
    )
    stage.set_data("holdout", holdout)
    train, test = stage(catalog)
    held = set(test.data["object_id"])
    # object 0, object 50, and both rows of R010 (objects 20, 21); 999 is not in the input
    assert held == {0, 50, 20, 21}
    assert set(train.data["object_id"]) == set(catalog["object_id"]) - held
    assert not set(train.data["ref_id"]) & set(holdout["ref_id"])


def test_holdout_split_missing_column(catalog):
    DataStore.allow_overwrite = True
    stage = ReferenceHoldoutSplitter.make_stage(name="split_hold_miss", holdout_cols=["mm_object_id"])
    stage.set_data("holdout", pd.DataFrame({"mm_object_id": [1]}))
    with pytest.raises(KeyError, match="mm_object_id"):
        stage(catalog)


# --------------------------------------------------------------------------------------- #
# MagRedshiftDownsampler
# --------------------------------------------------------------------------------------- #

@pytest.fixture(name="dense_catalog")
def fixture_dense_catalog():
    """One survey piled into a single magnitude-redshift cell, one spread out."""
    n = 200
    return pd.DataFrame(
        {
            "z_source": ["DESI_DR1"] * n + ["TINY_SURVEY"] * 20,
            "lsst_i_mag_gauss2": [20.01] * n + list(np.linspace(19, 24, 20)),
            "redshift": [0.51] * n + list(np.linspace(0.1, 2.0, 20)),
        }
    )


def test_downsampler_caps_one_cell(dense_catalog):
    DataStore.allow_overwrite = True
    stage = MagRedshiftDownsampler.make_stage(
        name="down_cap", caps={"DESI_DR1": 10}, d_mag=0.1, d_z=0.1
    )
    out = stage(dense_catalog).data
    source = np.asarray(out["z_source"]).astype(str)
    # the 200 piled-up rows collapse to the cap; the uncapped survey is untouched
    assert (source == "DESI_DR1").sum() == 10
    assert (source == "TINY_SURVEY").sum() == 20


def test_downsampler_writes_an_hdf5_dict(dense_catalog):
    """It overrides Selector.run precisely so the output is a dict of arrays, not parquet."""
    DataStore.allow_overwrite = True
    out = MagRedshiftDownsampler.make_stage(
        name="down_h5", caps={"DESI_DR1": 5}
    )(dense_catalog).data
    assert isinstance(out, dict)
    assert sorted(out) == ["lsst_i_mag_gauss2", "redshift", "z_source"]


def test_downsampler_ignores_an_absent_survey(dense_catalog):
    DataStore.allow_overwrite = True
    out = MagRedshiftDownsampler.make_stage(
        name="down_absent", caps={"NOT_PRESENT": 1}
    )(dense_catalog).data
    assert len(out["z_source"]) == len(dense_catalog)


def test_downsampler_reports_a_missing_column(dense_catalog):
    DataStore.allow_overwrite = True
    stage = MagRedshiftDownsampler.make_stage(name="down_miss", mag_col="nope")
    with pytest.raises(KeyError, match="nope"):
        stage(dense_catalog)


# --------------------------------------------------------------------------------------- #
# SOMResampler
# --------------------------------------------------------------------------------------- #

def test_som_resampler_declares_its_handles():
    """The pool and the population in; the test set and the per-cell occupancy out."""
    assert [tag for tag, _ in SOMResampler.inputs] == ["input", "population"]
    assert [tag for tag, _ in SOMResampler.outputs] == ["output", "cells"]


def test_allocate_cell_targets_is_exact_and_proportional():
    n_pop = np.array([50, 30, 20, 0])
    n_pool = np.array([5, 1, 9, 3])
    targets = allocate_cell_targets(n_pop, n_pool, 1000)
    assert targets.tolist() == [500, 300, 200, 0]
    # largest remainder keeps the total exact when the shares do not divide evenly
    assert allocate_cell_targets(np.array([1, 1, 1]), np.ones(3), 100).sum() == 100


def test_allocate_cell_targets_renormalises_over_cells_with_pool_objects():
    """A cell the pool cannot fill gets nothing; its population share moves to the others."""
    targets = allocate_cell_targets(np.array([25, 25, 50]), np.array([1, 4, 0]), 100)
    assert targets.tolist() == [50, 50, 0]
    with pytest.raises(ValueError, match="no population object"):
        allocate_cell_targets(np.array([0, 5]), np.array([3, 0]), 10)
    with pytest.raises(ValueError, match="n_target"):
        allocate_cell_targets(np.array([1]), np.array([1]), -1)


def test_draw_cell_members_meets_every_target_with_even_repeats():
    cells = np.array([0, 0, 0, 1, 1, 2, 3, 3, 3, 3])
    targets = np.array([7, 1, 0, 2])
    picks = draw_cell_members(cells, targets, np.random.default_rng(0))
    assert np.bincount(cells[picks], minlength=4).tolist() == [7, 1, 0, 2]
    uses = np.bincount(picks, minlength=len(cells))
    # cell 0 (3 objects, target 7): every object used, at most once more than any other
    assert set(uses[:3]) == {2, 3} and uses[:3].sum() == 7
    # cells 1 and 3 have more objects than their targets: no object repeats
    assert uses[3:5].max() <= 1 and uses[6:].max() <= 1
    with pytest.raises(ValueError, match="no pool objects"):
        draw_cell_members(np.array([0]), np.array([0, 1]), np.random.default_rng(0))


def test_som_features_fill_nondetections_and_difference_colours():
    frame = pd.DataFrame({"r": [21.0, 40.0, np.nan], "i": [20.0, 20.5, np.inf]})
    x = som_features(frame, ["i"], ["r", "i"], [26.0], [27.0, 26.0], 40.0)
    assert x.dtype == np.float32
    assert x[:, 0].tolist() == [20.0, 20.5, 26.0]  # i, inf -> 26
    assert x[:, 1].tolist() == [1.0, 6.5, 1.0]  # r - i with 40 and NaN -> 27
    nan_sentinel = som_features(frame, ["r"], ["r", "i"], [27.0], [27.0, 26.0], float("nan"))
    assert nan_sentinel[:, 0].tolist() == [21.0, 40.0, 27.0]  # 40 is a magnitude here


def _som_inputs(catalog):
    """A pool and a population carrying the three bands the default SOM features need."""
    pool = catalog.rename(columns={"z": "redshift"}).copy()
    for band in ("r", "z"):
        pool[f"lsst_{band}_mag_gauss2"] = pool["lsst_i_mag_gauss2"] + 0.2
    population = pool.sample(frac=1.0, random_state=1).reset_index(drop=True)
    return pool, population


def _cell_by_magnitude(self, train, samples):
    """Stands in for the SOM: cell = whole i magnitude above 19, on a 4x4 map."""
    return [np.clip(np.floor(s[:, 0]) - 19, 0, 15).astype(np.int64) for s in samples]


@pytest.fixture(name="stub_som")
def fixture_stub_som(monkeypatch):
    """Exercise SOMResampler.run() without somoclu, which is a heavy optional dependency."""
    monkeypatch.setattr(SOMResampler, "_train_and_assign", _cell_by_magnitude)


def _cells_of(out):
    return np.clip(np.floor(np.asarray(out["lsst_i_mag_gauss2"])) - 19, 0, 15).astype(int)


def test_som_resampler_matches_the_population_occupancy(catalog, stub_som):
    DataStore.allow_overwrite = True
    pool, population = _som_inputs(catalog)
    # a population skewed faint: every object fainter than 23 appears four times
    faint = population[population["lsst_i_mag_gauss2"] > 23.0]
    population = pd.concat([population] + [faint] * 3, ignore_index=True)
    stage = SOMResampler.make_stage(
        name="som_occ", som_size=[4, 4], n_target=2000, mag_range=[0.0, 99.0]
    )
    out = stage(pool, population).data
    assert len(out["lsst_i_mag_gauss2"]) == 2000
    want = allocate_cell_targets(
        np.bincount(_cells_of(population), minlength=16),
        np.bincount(_cells_of(pool), minlength=16),
        2000,
    )
    assert np.bincount(_cells_of(out), minlength=16).tolist() == want.tolist()
    assert np.array_equal(np.asarray(out["som_cell"]), _cells_of(out))
    # drawn from the pool: every output object is a pool object
    assert set(np.asarray(out["object_id"])) <= set(pool["object_id"])
    cells = stage.get_data("cells")
    assert cells["n_target"].sum() == 2000
    assert cells["n_population"].sum() == len(population)
    assert cells["n_pool"].sum() == len(pool)


def test_som_resampler_drops_population_cells_the_pool_cannot_fill(catalog, stub_som):
    DataStore.allow_overwrite = True
    pool, population = _som_inputs(catalog)
    extra = population.iloc[:30].copy()
    extra["lsst_i_mag_gauss2"] = 26.5  # cell 7: no pool object there
    population = pd.concat([population, extra], ignore_index=True)
    stage = SOMResampler.make_stage(
        name="som_cover", som_size=[4, 4], n_target=500, mag_range=[0.0, 99.0]
    )
    out = stage(pool, population).data
    assert len(out["lsst_i_mag_gauss2"]) == 500
    assert 7 not in set(_cells_of(out))
    cells = stage.get_data("cells").set_index("cell")
    assert cells.loc[7, "n_population"] == 30 and cells.loc[7, "n_target"] == 0


def test_som_resampler_applies_the_magnitude_range_to_both_samples(catalog, stub_som):
    DataStore.allow_overwrite = True
    pool, population = _som_inputs(catalog)
    stage = SOMResampler.make_stage(
        name="som_range", som_size=[4, 4], n_target=300, mag_range=[20.0, 22.0]
    )
    out = stage(pool, population).data
    mags = np.asarray(out["lsst_i_mag_gauss2"])
    assert len(mags) == 300 and ((mags > 20.0) & (mags < 22.0)).all()
    cells = stage.get_data("cells")
    in_range = population["lsst_i_mag_gauss2"].between(20.0, 22.0, inclusive="neither")
    assert cells["n_population"].sum() == int(in_range.sum())


def test_som_resampler_is_reproducible(catalog, stub_som):
    DataStore.allow_overwrite = True
    pool, population = _som_inputs(catalog)
    first, second = (
        SOMResampler.make_stage(
            name=f"som_r{k}", som_size=[4, 4], n_target=400, seed=5, mag_range=[0.0, 99.0]
        )(pool, population).data
        for k in (1, 2)
    )
    assert np.array_equal(np.asarray(first["object_id"]), np.asarray(second["object_id"]))


def test_som_resampler_rejects_bad_configuration(catalog, stub_som):
    DataStore.allow_overwrite = True
    pool, population = _som_inputs(catalog)
    stage = SOMResampler.make_stage(name="som_miss", som_size=[4, 4])
    with pytest.raises(KeyError, match="lsst_r_mag_gauss2"):
        stage(pool.drop(columns=["lsst_r_mag_gauss2"]), population)
    stage = SOMResampler.make_stage(name="som_bad", som_size=[4, 4], som_train_on="both")
    with pytest.raises(ValueError, match="som_train_on"):
        stage(pool, population)
    stage = SOMResampler.make_stage(name="som_bad_init", som_size=[4, 4], som_init="kmeans")
    with pytest.raises(ValueError, match="som_init"):
        stage(pool, population)


@needs_somoclu
def test_train_som_random_init_is_seeded_and_radius_is_used():
    data = np.random.default_rng(3).normal(size=(400, 3)).astype(np.float32)
    first = train_som(data, [4, 4], 3, init="random", seed=11)
    again = train_som(data, [4, 4], 3, init="random", seed=11)
    other = train_som(data, [4, 4], 3, init="random", seed=12)
    wide = train_som(data, [4, 4], 3, init="random", seed=11, radius_final=3.0)
    assert first.shape == (16, 3)
    assert np.allclose(first, again)
    assert not np.allclose(first, other)
    assert not np.allclose(first, wide)
    with pytest.raises(ValueError, match="init"):
        train_som(data, [4, 4], 1, init="kmeans")


@needs_somoclu
@pytest.mark.parametrize(
    "train_on, init, radius",
    [("population", "pca", 1.0), ("pool", "pca", 1.0), ("population", "random", 2.0)],
)
def test_som_resampler_trains_a_real_som(catalog, train_on, init, radius):
    """The genuine article: a 4x4 map, two epochs."""
    DataStore.allow_overwrite = True
    pool, population = _som_inputs(catalog)
    stage = SOMResampler.make_stage(
        name=f"som_real_{train_on}_{init}", som_size=[4, 4], n_epochs=2, n_target=300,
        som_train_on=train_on, som_init=init, som_radius_final=radius,
    )
    out = stage(pool, population).data
    assert len(out["lsst_i_mag_gauss2"]) == 300
    assert set(np.asarray(out["lsst_i_mag_gauss2"])) <= set(pool["lsst_i_mag_gauss2"])
    assert stage.get_data("cells")["n_target"].sum() == 300


# --------------------------------------------------------------------------------------- #
# set_stage_threads
# --------------------------------------------------------------------------------------- #

def test_set_stage_threads_patches_the_yaml(tmp_path):
    """ceci will not schedule a stage asking for more threads than the site declares."""
    path = tmp_path / "pipe.yml"
    path.write_text(
        yaml.dump(
            {
                "stages": [{"name": "split"}, {"name": "som_resample_test"}],
                "site": {"max_threads": 2},
            }
        )
    )
    set_stage_threads(str(path), {"som_resample_test": 16})
    out = yaml.safe_load(path.read_text())
    stages = {s["name"]: s for s in out["stages"]}
    assert stages["som_resample_test"]["threads_per_process"] == 16
    assert "threads_per_process" not in stages["split"]
    assert out["site"]["max_threads"] == 16


def test_set_stage_threads_never_lowers_max_threads(tmp_path):
    path = tmp_path / "pipe.yml"
    path.write_text(
        yaml.dump({"stages": [{"name": "a"}], "site": {"max_threads": 64}})
    )
    set_stage_threads(str(path), {"a": 4})
    assert yaml.safe_load(path.read_text())["site"]["max_threads"] == 64


def test_missing_quality_column_is_reported(catalog):
    DataStore.allow_overwrite = True
    stage = PreTrainTestSplitter.make_stage(
        name="split_qmiss", apply_quality_cut=True, quality_col="nope"
    )
    with pytest.raises(KeyError, match="nope"):
        stage(catalog)


def test_group_split_treats_nulls_as_individual_rows(catalog):
    """A null group key must not collapse every such row into one giant group."""
    DataStore.allow_overwrite = True
    catalog = catalog.copy()
    catalog.loc[:9, "ref_id"] = None
    train, test = PreTrainTestSplitter.make_stage(
        name="split_gnull", group_col="ref_id"
    )(catalog)
    both = len(train.data) + len(test.data)
    assert both == len(catalog)
    shared = set(train.data["ref_id"].dropna()) & set(test.data["ref_id"].dropna())
    assert shared == set()


def test_pz_prepare_pipeline_rejects_an_unknown_key():
    from rail.pipelines.degradation.hscfy_pz_prepare import HscfyPzPreparePipeline

    with pytest.raises(KeyError, match="unknown catalog_config keys"):
        HscfyPzPreparePipeline(dict(not_a_real_key=1))


# --------------------------------------------------------------------------------------- #
# SOMZCellResampler
# --------------------------------------------------------------------------------------- #

def test_zcell_resampler_declares_its_handles():
    """The truth sample comes in, the (cell, z bin) table goes out, on top of SOMResampler's."""
    assert [tag for tag, _ in SOMZCellResampler.inputs] == ["input", "population", "truth"]
    assert [tag for tag, _ in SOMZCellResampler.outputs] == ["output", "cells", "zcells"]


def test_allocate_cell_targets_takes_real_weights():
    targets = allocate_cell_targets(np.array([0.25, 0.5, 0.25, 0.1]), np.array([1, 1, 1, 0]), 8)
    assert targets.tolist() == [2, 4, 2, 0]


def test_redshift_bins_clip_to_the_end_bins():
    edges = np.array([0.0, 0.5, 1.0, 2.0])
    z = np.array([-0.1, 0.0, 0.49, 0.5, 1.99, 2.0, 9.0])
    assert redshift_bins(z, edges).tolist() == [0, 0, 0, 1, 2, 2, 2]


def test_zcell_fractions_shrink_toward_the_neighbourhood():
    """A 3x1 map: cell 0 has truth [6, 4], cell 1 has [0, 2], cell 2 none."""
    truth = np.array([[6.0, 4.0], [0.0, 2.0], [0.0, 0.0]])
    pool = np.array([[1.0, 1.0], [1.0, 1.0], [3.0, 1.0]])
    p, no_truth = zcell_fractions(truth, pool, [3, 1], shrink=5.0)
    # cell 0: own [6, 4] + 5 x its neighbourhood (cells 0, 1: [6, 6] -> [.5, .5]), / (10 + 5)
    assert np.allclose(p[0], (np.array([6, 4]) + 5 * np.array([0.5, 0.5])) / 15)
    # cell 1: neighbourhood is all three cells, [6, 6]
    assert np.allclose(p[1], (np.array([0, 2]) + 5 * np.array([0.5, 0.5])) / 7)
    # cell 2 borrows entirely from cells 1-2, which hold [0, 2]
    assert np.allclose(p[2], [0.0, 1.0]) and not no_truth.any()
    # no shrinkage: a cell is its own histogram
    p0, _ = zcell_fractions(truth, pool, [3, 1], shrink=0.0)
    assert np.allclose(p0[0], [0.6, 0.4]) and np.allclose(p0[1], [0.0, 1.0])
    # a cell with no truth anywhere around it keeps the pool's own fractions, and is flagged
    lonely = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [5.0, 5.0]])
    p_far, flagged = zcell_fractions(lonely, np.ones((4, 2)) * [[3, 1]], [4, 1], shrink=5.0)
    assert flagged.tolist() == [True, True, False, False]
    assert np.allclose(p_far[0], [0.75, 0.25])


def _zcell_inputs():
    """Pool, population and truth on the stub SOM (cell = whole i magnitude above 19).

    In cells 1 and 2 (20 < i < 22) half of the pool sits at z = 0.2 and half at z = 1.0;
    the truth puts every galaxy of those cells at z ~ 0.2.
    """
    rng = np.random.default_rng(11)
    n = 400
    mag = rng.uniform(20.0, 22.0, n)
    pool = pd.DataFrame(
        {
            "object_id": np.arange(n),
            "lsst_i_mag_gauss2": mag,
            "lsst_r_mag_gauss2": mag + 0.3,
            "lsst_z_mag_gauss2": mag - 0.2,
            "redshift": np.where(np.arange(n) % 2 == 0, 0.2, 1.0),
        }
    )
    population = pool.drop(columns=["redshift", "object_id"]).sample(
        frac=1.0, random_state=1
    ).reset_index(drop=True)
    truth = population.iloc[:150].copy()
    truth["z"] = rng.uniform(0.1, 0.3, len(truth))
    return pool, population, truth


def test_zcell_resampler_follows_the_truth_inside_each_cell(stub_som):
    DataStore.allow_overwrite = True
    pool, population, truth = _zcell_inputs()
    stage = SOMZCellResampler.make_stage(
        name="zcell_follow", som_size=[4, 4], n_target=1000, mag_range=[0.0, 99.0],
        z_edges=[0.0, 0.5, 2.0], z_shrink=5.0,
    )
    out = stage(pool, population, truth).data
    z = np.asarray(out["redshift"])
    assert len(z) == 1000
    # the plain SOM draw would give half z = 1.0; the truth says none
    assert (z < 0.5).all()
    assert np.asarray(out["som_zbin"]).tolist() == [0] * 1000
    # the colour--magnitude occupancy is untouched: per cell, still the population share
    want = allocate_cell_targets(
        np.bincount(_cells_of(population), minlength=16),
        np.bincount(_cells_of(pool), minlength=16),
        1000,
    )
    assert np.bincount(_cells_of(out), minlength=16).tolist() == want.tolist()
    zcells = stage.get_data("zcells")
    assert len(zcells) == 16 * 2 and zcells["n_target"].sum() == 1000
    assert np.isclose(zcells["population_share"].sum(), 1.0)
    assert zcells["n_truth"].sum() == len(truth)
    cells = stage.get_data("cells")
    assert cells["n_target"].tolist() == want.tolist()


def test_zcell_resampler_renormalises_slots_the_pool_cannot_fill(stub_som):
    DataStore.allow_overwrite = True
    pool, population, truth = _zcell_inputs()
    pool = pool[pool["redshift"] < 0.5].reset_index(drop=True)  # no pool object at z > 0.5
    truth.loc[truth["lsst_i_mag_gauss2"] < 21.0, "z"] = 1.5  # cell 1 is all high-z truth
    stage = SOMZCellResampler.make_stage(
        name="zcell_lost", som_size=[4, 4], n_target=500, mag_range=[0.0, 99.0],
        z_edges=[0.0, 0.5, 2.0], z_shrink=0.0,
    )
    out = stage(pool, population, truth).data
    assert len(out["redshift"]) == 500
    zcells = stage.get_data("zcells").set_index(["cell", "z_bin"])
    assert zcells.loc[(1, 1), "n_pool"] == 0 and zcells.loc[(1, 1), "n_target"] == 0
    # cell 1's whole share was in the unfillable slot, so cell 2 takes the full target
    assert zcells.loc[(2, 0), "n_target"] == 500


def test_zcell_resampler_rejects_bad_configuration(stub_som):
    DataStore.allow_overwrite = True
    pool, population, truth = _zcell_inputs()
    stage = SOMZCellResampler.make_stage(name="zcell_edges", som_size=[4, 4], z_edges=[1.0, 0.5])
    with pytest.raises(ValueError, match="z_edges"):
        stage(pool, population, truth)
    stage = SOMZCellResampler.make_stage(name="zcell_zcol", som_size=[4, 4])
    with pytest.raises(KeyError, match="truth_redshift_col"):
        stage(pool, population, truth.drop(columns=["z"]))
    stage = SOMZCellResampler.make_stage(name="zcell_pcol", som_size=[4, 4])
    with pytest.raises(KeyError, match="redshift_col"):
        stage(pool.drop(columns=["redshift"]), population, truth)


def test_pz_prepare_pipeline_swaps_in_the_zcell_stage():
    from rail.pipelines.degradation.hscfy_pz_prepare import HscfyPzPreparePipeline

    plain = HscfyPzPreparePipeline()
    zcell = HscfyPzPreparePipeline(dict(zcell=True, z_edges=[0.0, 1.0, 7.0]))
    assert type(plain.som_resample_test) is SOMResampler
    assert type(zcell.som_resample_test) is SOMZCellResampler
    assert zcell.som_resample_test.config.z_edges == [0.0, 1.0, 7.0]
    assert zcell.som_resample_test.config.redshift_col == "redshift"


# --------------------------------------------------------------------------------------- #
# MergeFullPZStage
# --------------------------------------------------------------------------------------- #

# The real inputs are a 258 MB parquet, a 28 GB pickle and two multi-GB FITS files, so these
# fixtures rebuild each format in miniature: same layout, same normalization conventions, a
# handful of objects.  Each synthetic p(z) is a single spike at a known redshift, which lets
# a test assert that a row was matched to the *right* object rather than merely to something.

PAUS_IDS = [11, 12, 13]
PAUS_PEAKS = [0.20, 0.50, 0.90]
WEB_IDS = [0, 1, 2, 3]
WEB_PEAKS = [0.30, 1.10, 8.00, 0.70]     # 8.00 is beyond zmax, to exercise the pile-up
C2020_IDS = [1, 2, 3]
C2020_PEAKS = [0.40, 1.60, 0.60]
WEB_POSITIONS = [(150.00, 2.00), (150.01, 2.00), (150.02, 2.00), (150.03, 2.00)]


def _spike(grid, peak, total):
    """A p(z) holding all of ``total`` in the grid bin nearest ``peak``."""
    pz = np.zeros(len(grid))
    pz[np.abs(grid - peak).argmin()] = total
    return pz


@pytest.fixture
def external_pz_files(tmp_path):
    """Miniature stand-ins for the four external p(z) files, in their real on-disk formats."""
    # PAUS+COSMOS: Z<z> columns on a deliberately non-uniform grid, rows summing to 1.
    paus_grid = np.concatenate([np.arange(0, 0.1, 0.001), np.arange(0.1, 1.0, 0.01)])
    paus = pd.DataFrame(
        np.array([_spike(paus_grid, peak, 1.0) for peak in PAUS_PEAKS]),
        columns=[f"Z{z:.3f}" for z in paus_grid],
    )
    paus["ref_id"] = PAUS_IDS
    paus_file = tmp_path / "paus_cosmos.parquet"
    paus.to_parquet(paus_file)

    # COSMOSWeb: {int64 id: (3, nz)} of grid / p(z) summing to 100 / AGN p(z).
    web_grid = np.arange(0, 10.001, 0.01)
    web = {
        np.int64(i): np.vstack([web_grid, _spike(web_grid, peak, 100.0), np.zeros(len(web_grid))])
        for i, peak in zip(WEB_IDS, WEB_PEAKS)
    }
    web_file = tmp_path / "cosmos_web.pkl"
    with open(web_file, "wb") as stream:
        pickle.dump(web, stream, protocol=4)

    # COSMOSWeb photometry: id / ra / dec, for the positional fallback.
    photometry_file = tmp_path / "cosmos_web_photometry.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU.from_columns(
                [
                    fits.Column(name="id", format="K", array=np.array(WEB_IDS)),
                    fits.Column(name="ra", format="D", array=np.array([p[0] for p in WEB_POSITIONS])),
                    fits.Column(name="dec", format="D", array=np.array([p[1] for p in WEB_POSITIONS])),
                ]
            ),
        ]
    ).writeto(photometry_file)

    # COSMOS2020: an image whose row 0 is the grid and column 0 the id == row index.
    c2020_grid = np.arange(0, 3.001, 0.01)
    image = np.zeros((max(C2020_IDS) + 1, len(c2020_grid) + 1), dtype="float32")
    image[0, 0] = np.nan
    image[0, 1:] = c2020_grid
    for i, peak in zip(C2020_IDS, C2020_PEAKS):
        image[i, 0] = i
        image[i, 1:] = _spike(c2020_grid, peak, 100.0)
    c2020_file = tmp_path / "cosmos2020_pz.fits"
    fits.PrimaryHDU(image).writeto(c2020_file)

    return dict(
        paus_cosmos_file=str(paus_file),
        cosmos_web_file=str(web_file),
        cosmos_web_photometry_file=str(photometry_file),
        cosmos2020_file=str(c2020_file),
    )


@pytest.fixture
def merge_catalog():
    """A cross-match-shaped catalog covering every route a row can take to a p(z).

    ``orig_id`` is a string here, as it is in the real cross-match, and row 'noise' carries a
    non-numeric one so the coercion path is exercised.
    """
    return pd.DataFrame(
        {
            "tag": ["paus", "web", "c2020", "position", "nothing", "noise"],
            "orig_id": ["12", "1", "2", "999", "7", "0058-01036"],
            "ref_catalog": [
                PAUS_COSMOS_REF,
                COSMOS_WEB_REF,
                COSMOS2020_REF,
                "314_desi_dr1_lite",
                "314_desi_dr1_lite",
                "183_primus_dr1",
            ],
            # only the 'position' row sits on a COSMOSWeb object
            "ra": [150.5, 150.5, 150.5, 150.02, 151.9, 151.9],
            "dec": [2.5, 2.5, 2.5, 2.00, 2.5, 2.5],
        }
    )


def _peak_redshifts(stage, out):
    """The redshift each output row's p(z) peaks at."""
    pz = np.stack(out["pz"].to_numpy())
    return stage.zgrid[pz.argmax(axis=1)]


def test_merge_full_pz_ids_only(external_pz_files, merge_catalog):
    """With no fallback, only rows whose reference catalog carries a usable id survive."""
    stage = MergeFullPZStage.make_stage(name="ids_only", fallback=[], **external_pz_files)
    out = stage(merge_catalog).data

    assert list(out["tag"]) == ["paus", "web"]
    assert list(out["pz_source"]) == ["paus_cosmos", "cosmos_web"]
    # matched to the right objects: PAUS id 12 peaks at 0.50, COSMOSWeb id 1 at 1.10
    assert _peak_redshifts(stage, out) == pytest.approx([0.50, 1.10], abs=0.011)


def test_merge_full_pz_normalization_and_dtype(external_pz_files, merge_catalog):
    """p(z) come back as normalized float32 on the configured grid."""
    stage = MergeFullPZStage.make_stage(name="norm", **external_pz_files)
    out = stage(merge_catalog).data
    pz = np.stack(out["pz"].to_numpy())

    assert pz.dtype == np.float32
    assert pz.shape[1] == len(stage.zgrid) == 400
    assert pz.sum(axis=1) == pytest.approx(1.0, abs=1e-6)


def test_merge_full_pz_cosmos2020_fallback(external_pz_files, merge_catalog):
    """The COSMOS2020 fallback claims rows by id, and only rows still without a p(z)."""
    stage = MergeFullPZStage.make_stage(
        name="c2020", fallback=["cosmos2020"], **external_pz_files
    )
    out = stage(merge_catalog).data

    assert list(out["tag"]) == ["paus", "web", "c2020"]
    assert out.loc[out.tag == "c2020", "pz_source"].item() == "cosmos2020"
    # COSMOS2020 id 2 peaks at 1.60
    assert _peak_redshifts(stage, out)[2] == pytest.approx(1.60, abs=0.011)


def test_merge_full_pz_position_fallback(external_pz_files, merge_catalog):
    """The positional fallback reaches a row whose reference id belongs to another survey."""
    stage = MergeFullPZStage.make_stage(
        name="pos", fallback=["position"], **external_pz_files
    )
    out = stage(merge_catalog).data

    assert list(out["tag"]) == ["paus", "web", "position"]
    assert out.loc[out.tag == "position", "pz_source"].item() == "cosmos_web_position"
    # it sits on COSMOSWeb id 2, whose p(z) peaks beyond zmax and so piles up in the last bin
    assert _peak_redshifts(stage, out)[2] == pytest.approx(stage.zgrid[-1])


def test_merge_full_pz_position_radius_excludes_distant_rows(external_pz_files, merge_catalog):
    """Shrinking the radius below the offset drops the positional match."""
    far = merge_catalog.copy()
    far.loc[far.tag == "position", "dec"] = 2.0 + 2.0 / 3600.0   # 2 arcsec away
    stage = MergeFullPZStage.make_stage(
        name="radius", fallback=["position"], match_radius_arcsec=0.75, **external_pz_files
    )
    assert "position" not in set(stage(far).data["tag"])


def test_merge_full_pz_chained_fallbacks(external_pz_files, merge_catalog):
    """Chaining reaches the union of both fallbacks, and keeps the same rows either way.

    Only the provenance of rows both fallbacks could supply depends on the order, which is
    also the check that neither step destroys the ids the other matches on.
    """
    both = merge_catalog.copy()
    # this row is reachable either way: a COSMOS2020 id, sitting on a COSMOSWeb object
    both.loc[len(both)] = {
        "tag": "either", "orig_id": "3", "ref_catalog": COSMOS2020_REF,
        "ra": 150.03, "dec": 2.00,
    }

    forward = MergeFullPZStage.make_stage(
        name="fwd", fallback=["position", "cosmos2020"], **external_pz_files
    )(both).data
    reverse = MergeFullPZStage.make_stage(
        name="rev", fallback=["cosmos2020", "position"], **external_pz_files
    )(both).data

    assert set(forward["tag"]) == set(reverse["tag"])
    assert set(forward["tag"]) == {"paus", "web", "c2020", "position", "either"}
    assert forward.loc[forward.tag == "either", "pz_source"].item() == "cosmos_web_position"
    assert reverse.loc[reverse.tag == "either", "pz_source"].item() == "cosmos2020"
    # the rows only one fallback can reach keep their source in both orders
    for table in (forward, reverse):
        assert table.loc[table.tag == "position", "pz_source"].item() == "cosmos_web_position"
        assert table.loc[table.tag == "c2020", "pz_source"].item() == "cosmos2020"


def test_merge_full_pz_unnested_matches_nested(external_pz_files, merge_catalog):
    """The one-column-per-bin format holds the same numbers under the documented names."""
    nested = MergeFullPZStage.make_stage(name="nest", **external_pz_files)(merge_catalog).data
    stage = MergeFullPZStage.make_stage(name="flat", nested=False, **external_pz_files)
    flat = stage(merge_catalog).data

    assert stage.column_names()[:2] == ["Z0_000", "Z0_010"]
    assert stage.column_names()[-1] == "Z3_990"
    assert [c for c in flat.columns if c.startswith("Z")] == stage.column_names()
    assert flat[stage.column_names()].to_numpy() == pytest.approx(
        np.stack(nested["pz"].to_numpy())
    )


def test_merge_full_pz_accepts_arrow_table(external_pz_files, merge_catalog):
    """A PqHandle hands back a pyarrow Table under ceci, which the stage must absorb."""
    stage = MergeFullPZStage.make_stage(name="arrow", fallback=[], **external_pz_files)
    out = stage(pa.Table.from_pandas(merge_catalog, preserve_index=False)).data
    assert isinstance(out, pd.DataFrame)
    assert list(out["pz_source"]) == ["paus_cosmos", "cosmos_web"]


def test_merge_full_pz_custom_grid(external_pz_files, merge_catalog):
    """zmin/zmax/dz drive both the grid and the unnested column names."""
    stage = MergeFullPZStage.make_stage(
        name="grid", zmin=0.0, zmax=1.0, dz=0.1, nested=False, fallback=[], **external_pz_files
    )
    out = stage(merge_catalog).data
    assert stage.column_names() == [f"Z{0.1 * k:.3f}".replace(".", "_") for k in range(11)]
    assert out[stage.column_names()].to_numpy().sum(axis=1) == pytest.approx(1.0, abs=1e-6)


def test_merge_full_pz_requires_id_and_ref_columns(external_pz_files, merge_catalog):
    """A missing id or reference-catalog column is an error, not a silently empty result."""
    stage = MergeFullPZStage.make_stage(name="missing", **external_pz_files)
    with pytest.raises(KeyError, match="ref_catalog"):
        stage(merge_catalog.drop(columns=["ref_catalog"]))


def test_merge_full_pz_rejects_unknown_fallback(external_pz_files, merge_catalog):
    """An unrecognized fallback step is rejected rather than ignored."""
    stage = MergeFullPZStage.make_stage(name="bad", fallback=["nope"], **external_pz_files)
    with pytest.raises(ValueError, match="unknown fallback step"):
        stage(merge_catalog)


def test_merge_full_pz_reports_missing_cosmos_web_id(external_pz_files, merge_catalog):
    """An id the COSMOSWeb pickle does not hold is reported, not silently dropped."""
    absent = merge_catalog.copy()
    absent.loc[absent.tag == "web", "orig_id"] = "4242"
    stage = MergeFullPZStage.make_stage(name="absent", fallback=[], **external_pz_files)
    with pytest.raises(KeyError, match="4242"):
        stage(absent)


def test_rebin_pz_conserves_mass_and_piles_up_tails():
    """Mass is conserved, out-of-range mass lands in the end bins, empty rows stay empty."""
    z_out = np.arange(0.0, 3.995, 0.01)
    z_in = np.arange(0.0, 15.001, 0.01)

    pz = np.zeros((4, len(z_in)))
    pz[0, 50] = 1.0                      # a delta at z = 0.50
    pz[1, 1000] = 1.0                    # a delta at z = 10.0, beyond z_out
    pz[2, 500] = pz[2, 1400] = 0.5       # both beyond z_out
                                         # row 3 is left empty
    rebinned = rebin_pz(z_in, pz, z_out)

    assert rebinned[:3].sum(axis=1) == pytest.approx(1.0)
    assert z_out[rebinned[0].argmax()] == pytest.approx(0.50)
    assert rebinned[1][-1] == pytest.approx(1.0)
    assert rebinned[2][-1] == pytest.approx(1.0)
    assert rebinned[3].sum() == 0.0


def test_rebin_pz_handles_a_non_uniform_input_grid():
    """The PAUS grid changes spacing twice, which a naive reshape would silently mangle."""
    z_out = np.arange(0.0, 3.995, 0.01)
    z_in = np.concatenate([np.arange(0, 1, 0.001), np.arange(1, 2, 0.002), np.arange(2, 3, 0.01)])
    pz = np.random.default_rng(0).random((5, len(z_in)))
    pz /= pz.sum(axis=1, keepdims=True)

    assert rebin_pz(z_in, pz, z_out).sum(axis=1) == pytest.approx(1.0)
