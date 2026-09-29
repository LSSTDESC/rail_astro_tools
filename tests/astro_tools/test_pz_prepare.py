"""Unit tests for the pz_prepare curation stages and the table helpers they share."""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import yaml

from rail.core.data import DataStore
from rail.creation.degraders._frame_utils import (
    as_dataframe,
    frame_to_hdf5_dict,
    plain_python,
    to_numpy_frame,
)
from rail.creation.degraders.pz_prepare import (
    MagRedshiftDownsampler,
    PreTrainTestSplitter,
    SOMResampler,
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

def test_som_resampler_declares_two_inputs():
    """The pool and the population; the population binds to a pipeline-level input."""
    assert [tag for tag, _ in SOMResampler.inputs] == ["input", "population"]
    assert [tag for tag, _ in SOMResampler.outputs] == ["output"]


@needs_somoclu
def test_som_resampler_runs(catalog):  # pragma: no cover - needs a working somoclu
    DataStore.allow_overwrite = True
    pool = catalog.rename(columns={"z": "redshift"}).copy()
    for band in ("r", "z"):
        pool[f"lsst_{band}_mag_gauss2"] = pool["lsst_i_mag_gauss2"] + 0.2
    population = pool.sample(frac=1.0, random_state=1).reset_index(drop=True)
    stage = SOMResampler.make_stage(
        name="som_small", som_size=[4, 4], n_epochs=2, n_repeat=1, n_population=50
    )
    out = stage(pool, population).data
    assert isinstance(out, dict)
    assert len(out["lsst_i_mag_gauss2"]) > 0


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
