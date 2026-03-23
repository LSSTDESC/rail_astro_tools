"""Tests for SpecSelection_DESI_Phy degrader."""

import os

import numpy as np
import pandas as pd
import pytest
from rail.core.data import PqHandle, TableHandle

from rail.creation.degraders.desi_selector_phy import SpecSelection_DESI_Phy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def catalog():
    """Small simulation catalog with a physical parameter and redshift column."""
    df = pd.DataFrame(
        {
            "redshift": np.linspace(0.1, 1.0, 10),
            "log_peak_sub_halo_mass": [11.0, 12.0, 13.0, 14.0, 11.5,
                                        12.5, 13.5, 10.0, 14.5, 11.0],
        }
    )
    return TableHandle("catalog", df, path="dummy_catalog.pq")


@pytest.fixture
def threshold_parquet(tmp_path):
    """Flat threshold of 12.0 across the full redshift range, written to disk."""
    df = pd.DataFrame({"z": [0.0, 0.5, 1.0], "thresh": [12.0, 12.0, 12.0]})
    path = str(tmp_path / "thresh.parquet")
    df.to_parquet(path)
    return path


@pytest.fixture
def varying_threshold_parquet(tmp_path):
    """Threshold rising linearly from 11.0 at z=0 to 13.0 at z=1, written to disk."""
    df = pd.DataFrame({"z": [0.0, 1.0], "thresh": [11.0, 13.0]})
    path = str(tmp_path / "varying_thresh.parquet")
    df.to_parquet(path)
    return path


@pytest.fixture
def bad_threshold_parquet(tmp_path):
    """Threshold table with wrong column names, written to disk."""
    df = pd.DataFrame({"redshift": [0.0, 1.0], "value": [12.0, 12.0]})
    path = str(tmp_path / "bad_thresh.parquet")
    df.to_parquet(path)
    return path


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _cleanup(stage):
    path = stage.get_output(stage.get_aliased_tag("output"), final_name=True)
    if os.path.exists(path):
        os.remove(path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_basic_selection_correct_rows(catalog, threshold_parquet):
    """Objects with log_peak_sub_halo_mass > 12.0 should be selected."""
    sel = SpecSelection_DESI_Phy.make_stage(
        name="test_basic",
        desi_type="lrg",
        threshold_col="log_peak_sub_halo_mass",
        threshold_table=threshold_parquet,
        drop_rows=True,
    )
    result = sel(catalog).data

    # All selected rows must have the physical column above the threshold
    assert (result["log_peak_sub_halo_mass"] > 12.0).all()
    # At least one row was selected and at least one was dropped
    assert len(result) > 0
    assert len(result) < len(catalog.data)

    _cleanup(sel)


def test_output_has_same_columns(catalog, threshold_parquet):
    """Output catalog should have the same columns as the input."""
    sel = SpecSelection_DESI_Phy.make_stage(
        name="test_cols",
        desi_type="lrg",
        threshold_col="log_peak_sub_halo_mass",
        threshold_table=threshold_parquet,
        drop_rows=True,
    )
    result = sel(catalog).data

    assert set(result.columns) == set(catalog.data.columns)
    _cleanup(sel)


def test_drop_rows_false_returns_flag_column(catalog, threshold_parquet):
    """With drop_rows=False all rows are returned with a 'flag' column."""
    sel = SpecSelection_DESI_Phy.make_stage(
        name="test_flag",
        desi_type="lrg",
        threshold_col="log_peak_sub_halo_mass",
        threshold_table=threshold_parquet,
        drop_rows=False,
    )
    result = sel(catalog).data

    assert len(result) == len(catalog.data)
    assert "flag" in result.columns
    # flag=1 for selected objects, flag=0 for rejected
    assert set(result["flag"].unique()).issubset({0, 1})

    _cleanup(sel)


def test_varying_threshold_selection(catalog, varying_threshold_parquet):
    """Verify selection respects a redshift-varying threshold via interpolation."""
    sel = SpecSelection_DESI_Phy.make_stage(
        name="test_varying",
        desi_type="lrg",
        threshold_col="log_peak_sub_halo_mass",
        threshold_table=varying_threshold_parquet,
        drop_rows=False,
    )
    result = sel(catalog).data

    # Recompute expected mask manually
    from scipy.interpolate import interp1d

    thresh_fn = interp1d([0.0, 1.0], [11.0, 13.0], fill_value="extrapolate", bounds_error=False)
    expected_mask = (
        catalog.data["log_peak_sub_halo_mass"].values
        > thresh_fn(catalog.data["redshift"].values)
    ).astype(int)

    np.testing.assert_array_equal(result["flag"].values, expected_mask)
    _cleanup(sel)


@pytest.mark.parametrize("desi_type", ["bgs", "lrg", "elg"])
def test_desi_types_run_without_error(catalog, threshold_parquet, desi_type):
    """All supported desi_type values should produce a valid output."""
    sel = SpecSelection_DESI_Phy.make_stage(
        name=f"test_{desi_type}",
        desi_type=desi_type,
        threshold_col="log_peak_sub_halo_mass",
        threshold_table=threshold_parquet,
        drop_rows=True,
    )
    result = sel(catalog).data
    assert isinstance(result, pd.DataFrame)
    _cleanup(sel)


def test_missing_threshold_col_raises(catalog, threshold_parquet):
    """ValueError raised when the specified threshold_col is absent from catalog."""
    sel = SpecSelection_DESI_Phy.make_stage(
        name="test_missing_col",
        desi_type="lrg",
        threshold_col="nonexistent_column",
        threshold_table=threshold_parquet,
    )
    with pytest.raises(ValueError, match="nonexistent_column"):
        sel(catalog)


def test_missing_redshift_col_raises(catalog, threshold_parquet):
    """ValueError raised when the specified redshift_col is absent from catalog."""
    sel = SpecSelection_DESI_Phy.make_stage(
        name="test_missing_z",
        desi_type="lrg",
        threshold_col="log_peak_sub_halo_mass",
        redshift_col="nonexistent_redshift",
        threshold_table=threshold_parquet,
    )
    with pytest.raises(ValueError, match="nonexistent_redshift"):
        sel(catalog)


def test_missing_thresh_table_columns_raises(catalog, bad_threshold_parquet):
    """ValueError raised when the threshold table lacks 'z' or 'thresh'."""
    sel = SpecSelection_DESI_Phy.make_stage(
        name="test_bad_thresh",
        desi_type="lrg",
        threshold_col="log_peak_sub_halo_mass",
        threshold_table=bad_threshold_parquet,
    )
    with pytest.raises(ValueError, match="z"):
        sel(catalog)


def test_flat_threshold_selects_nothing_when_all_below(threshold_parquet):
    """When all physical values are below the threshold nothing is selected."""
    df = pd.DataFrame(
        {
            "redshift": np.linspace(0.1, 1.0, 5),
            "log_peak_sub_halo_mass": [10.0, 10.5, 11.0, 11.5, 11.9],
        }
    )
    cat = TableHandle("cat_all_below", df, path="dummy_all_below.pq")

    sel = SpecSelection_DESI_Phy.make_stage(
        name="test_none_selected",
        desi_type="lrg",
        threshold_col="log_peak_sub_halo_mass",
        threshold_table=threshold_parquet,
        drop_rows=True,
    )
    result = sel(cat).data
    assert len(result) == 0
    _cleanup(sel)


def test_flat_threshold_selects_all_when_all_above(threshold_parquet):
    """When all physical values are above the threshold everything is selected."""
    df = pd.DataFrame(
        {
            "redshift": np.linspace(0.1, 1.0, 5),
            "log_peak_sub_halo_mass": [13.0, 13.5, 14.0, 14.5, 15.0],
        }
    )
    cat = TableHandle("cat_all_above", df, path="dummy_all_above.pq")

    sel = SpecSelection_DESI_Phy.make_stage(
        name="test_all_selected",
        desi_type="lrg",
        threshold_col="log_peak_sub_halo_mass",
        threshold_table=threshold_parquet,
        drop_rows=True,
    )
    result = sel(cat).data
    assert len(result) == len(df)
    _cleanup(sel)
