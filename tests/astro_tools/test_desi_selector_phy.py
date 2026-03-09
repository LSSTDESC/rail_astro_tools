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
def threshold_table():
    """Flat threshold of 12.0 across the full redshift range."""
    df = pd.DataFrame({"z": [0.0, 0.5, 1.0], "thresh": [12.0, 12.0, 12.0]})
    return TableHandle("threshold_table", df, path="dummy_thresh.pq")


@pytest.fixture
def varying_threshold_table():
    """Threshold that rises linearly from 11.0 at z=0 to 13.0 at z=1."""
    df = pd.DataFrame({"z": [0.0, 1.0], "thresh": [11.0, 13.0]})
    return TableHandle("varying_threshold_table", df, path="dummy_varying_thresh.pq")


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


def test_basic_selection_correct_rows(catalog, threshold_table):
    """Objects with log_peak_sub_halo_mass > 12.0 should be selected."""
    sel = SpecSelection_DESI_Phy.make_stage(
        name="test_basic",
        desi_type="lrg",
        threshold_col="log_peak_sub_halo_mass",
        drop_rows=True,
    )
    result = sel(catalog, threshold_table).data

    # All selected rows must have the physical column above the threshold
    assert (result["log_peak_sub_halo_mass"] > 12.0).all()
    # At least one row was selected and at least one was dropped
    assert len(result) > 0
    assert len(result) < len(catalog.data)

    _cleanup(sel)


def test_output_has_same_columns(catalog, threshold_table):
    """Output catalog should have the same columns as the input."""
    sel = SpecSelection_DESI_Phy.make_stage(
        name="test_cols",
        desi_type="lrg",
        threshold_col="log_peak_sub_halo_mass",
        drop_rows=True,
    )
    result = sel(catalog, threshold_table).data

    assert set(result.columns) == set(catalog.data.columns)
    _cleanup(sel)


def test_drop_rows_false_returns_flag_column(catalog, threshold_table):
    """With drop_rows=False all rows are returned with a 'flag' column."""
    sel = SpecSelection_DESI_Phy.make_stage(
        name="test_flag",
        desi_type="lrg",
        threshold_col="log_peak_sub_halo_mass",
        drop_rows=False,
    )
    result = sel(catalog, threshold_table).data

    assert len(result) == len(catalog.data)
    assert "flag" in result.columns
    # flag=1 for selected objects, flag=0 for rejected
    assert set(result["flag"].unique()).issubset({0, 1})

    _cleanup(sel)


def test_varying_threshold_selection(catalog, varying_threshold_table):
    """Verify selection respects a redshift-varying threshold via interpolation."""
    sel = SpecSelection_DESI_Phy.make_stage(
        name="test_varying",
        desi_type="lrg",
        threshold_col="log_peak_sub_halo_mass",
        drop_rows=False,
    )
    result = sel(catalog, varying_threshold_table).data

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
def test_desi_types_run_without_error(catalog, threshold_table, desi_type):
    """All supported desi_type values should produce a valid output."""
    sel = SpecSelection_DESI_Phy.make_stage(
        name=f"test_{desi_type}",
        desi_type=desi_type,
        threshold_col="log_peak_sub_halo_mass",
        drop_rows=True,
    )
    result = sel(catalog, threshold_table).data
    assert isinstance(result, pd.DataFrame)
    _cleanup(sel)


def test_missing_threshold_col_raises(catalog, threshold_table):
    """ValueError raised when the specified threshold_col is absent from catalog."""
    sel = SpecSelection_DESI_Phy.make_stage(
        name="test_missing_col",
        desi_type="lrg",
        threshold_col="nonexistent_column",
    )
    with pytest.raises(ValueError, match="nonexistent_column"):
        sel(catalog, threshold_table)


def test_missing_redshift_col_raises(catalog, threshold_table):
    """ValueError raised when the specified redshift_col is absent from catalog."""
    sel = SpecSelection_DESI_Phy.make_stage(
        name="test_missing_z",
        desi_type="lrg",
        threshold_col="log_peak_sub_halo_mass",
        redshift_col="nonexistent_redshift",
    )
    with pytest.raises(ValueError, match="nonexistent_redshift"):
        sel(catalog, threshold_table)


def test_missing_thresh_table_columns_raises(catalog):
    """ValueError raised when the threshold table lacks 'z' or 'thresh'."""
    bad_thresh = TableHandle(
        "bad_thresh",
        pd.DataFrame({"redshift": [0.0, 1.0], "value": [12.0, 12.0]}),
        path="dummy_bad.pq",
    )
    sel = SpecSelection_DESI_Phy.make_stage(
        name="test_bad_thresh",
        desi_type="lrg",
        threshold_col="log_peak_sub_halo_mass",
    )
    with pytest.raises(ValueError, match="z"):
        sel(catalog, bad_thresh)


def test_flat_threshold_selects_nothing_when_all_below(threshold_table):
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
        drop_rows=True,
    )
    result = sel(cat, threshold_table).data
    assert len(result) == 0
    _cleanup(sel)


def test_flat_threshold_selects_all_when_all_above(threshold_table):
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
        drop_rows=True,
    )
    result = sel(cat, threshold_table).data
    assert len(result) == len(df)
    _cleanup(sel)
