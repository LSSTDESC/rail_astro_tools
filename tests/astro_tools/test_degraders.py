import os
from typing import Type

import numpy as np
import pandas as pd
import pytest
from photerr import LsstErrorModel as PhoterrErrorModel

from rail.core.data import DATA_STORE, TableHandle
from rail.utils.path_utils import find_rail_file
from rail.tools.table_tools import ColumnMapper
from rail.creation.degraders.spectroscopic_degraders import InvRedshiftIncompleteness, LineConfusion
from rail.creation.degraders.spectroscopic_selections import *
from rail.creation.degraders.observing_condition_degrader import ObsCondition
from rail.creation.degraders.grid_selection import GridSelection
from rail.creation.degraders.photometric_errors import EuclidErrorModel, LSSTErrorModel
from rail.creation.degraders.unrec_bl_model import UnrecBlModel


@pytest.fixture
def data():
    """Some dummy data to use below."""

    DS = DATA_STORE()
    DS.__class__.allow_overwrite = True

    # generate random normal data
    rng = np.random.default_rng(0)
    x = rng.normal(loc=26, scale=1, size=(100, 7))

    # replace redshifts with reasonable values
    x[:, 0] = np.linspace(0, 2, x.shape[0])

    # return data in handle wrapping a pandas DataFrame
    df = pd.DataFrame(x, columns=["redshift", "u", "g", "r", "i", "z", "y"])
    return DS.add_data("data", df, TableHandle, path="dummy.pd")


@pytest.fixture
def data_forspec():
    """Some dummy data to use below."""

    DS = DATA_STORE()
    DS.__class__.allow_overwrite = True

    # generate random normal data
    rng = np.random.default_rng(0)
    x = rng.normal(loc=26, scale=1, size=(200000, 7))

    # replace redshifts with reasonable values
    x[:, 0] = np.linspace(0, 2, x.shape[0])

    # return data in handle wrapping a pandas DataFrame
    df = pd.DataFrame(x, columns=["redshift", "u", "g", "r", "i", "z", "y"])
    return DS.add_data("data_forspec", df, TableHandle, path="dummy_forspec.pd")

@pytest.fixture
def data_with_radec():
    DS = DATA_STORE()
    DS.__class__.allow_overwrite = True

    # generate random normal data
    rng = np.random.default_rng(0)
    x = rng.normal(loc=26, scale=1, size=(100, 7))
    # for simplicity we will not sample uniformly on sphere
    ra = rng.uniform(low=40, high=80, size=(100,1))
    dec = rng.uniform(low=-50, high=-25, size=(100,1))
    # replace redshifts with reasonable values
    x[:, 0] = np.linspace(0, 2, x.shape[0])
    x=np.append(x, ra, axis=1)
    x=np.append(x, dec, axis=1)

    # return data in handle wrapping a pandas DataFrame
    df = pd.DataFrame(x, columns=["redshift", "u", "g", "r", "i", "z", "y", "ra", "dec"])
    return DS.add_data("data_with_radec", df, TableHandle, path="dummy_with_radec.pd")


@pytest.fixture
def data_for_bl():
    """Some dummy data to use below."""

    DS = DATA_STORE()
    DS.__class__.allow_overwrite = True

    # generate random normal data
    columns=['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'y', 'redshift']
    rng = np.random.default_rng(0)
    x = rng.normal(loc=23, scale=3, size=(1000, len(columns)))

    # replace positions with constrained values
    x[:, 0] = np.random.uniform(low=0, high=0.02, size=1000)
    x[:, 1] = np.random.uniform(low=0, high=0.02, size=1000)
    x[:, 8] = np.random.uniform(low=0, high=3.00, size=1000)

    # return data in handle wrapping a pandas DataFrame
    df = pd.DataFrame(x, columns=columns)
    return DS.add_data("data_for_bl", df, TableHandle, path="dummy_for_bl.pd")


@pytest.mark.parametrize("pivot_redshift,errortype", [("fake", TypeError), (-1, ValueError)])
def test_InvRedshiftIncompleteness_bad_params(pivot_redshift, errortype):
    """Test bad parameters that should raise ValueError"""
    with pytest.raises(errortype):
        InvRedshiftIncompleteness.make_stage(pivot_redshift=pivot_redshift)


def test_InvRedshiftIncompleteness_returns_correct_shape(data):
    """Make sure returns same number of columns, fewer rows"""
    degrader = InvRedshiftIncompleteness.make_stage(pivot_redshift=1.0, seed = 12345)
    degraded_data = degrader(data).data
    assert degraded_data.shape[0] < data.data.shape[0]
    assert degraded_data.shape[1] == data.data.shape[1]
    os.remove(degrader.get_output(degrader.get_aliased_tag("output"), final_name=True))

    
def test_LineConfusion_returns_correct_shape(data):
    """Make sure returns same number of columns, fewer rows"""
    OII = 3727
    OIII = 5007

    lc_2p_0II_0III = LineConfusion.make_stage(
        name="lc_2p_0II_0III", true_wavelen=OII, wrong_wavelen=OIII, frac_wrong=0.02
    )
    lc_1p_0III_0II = LineConfusion.make_stage(
        name="lc_1p_0III_0II", true_wavelen=OIII, wrong_wavelen=OII, frac_wrong=0.01
    )
    degraded_data = lc_1p_0III_0II(
        lc_2p_0II_0III(data)
    ).data
    assert degraded_data.shape[0] <= data.data.shape[0]
    assert degraded_data.shape[1] == data.data.shape[1]
    os.remove(lc_2p_0II_0III.get_output(lc_2p_0II_0III.get_aliased_tag("output"), final_name=True))
    os.remove(lc_1p_0III_0II.get_output(lc_1p_0III_0II.get_aliased_tag("output"), final_name=True))
    

@pytest.mark.parametrize(
    "percentile_cut,redshift_cut,errortype",
    [(-1, 1, ValueError), (101, 1, ValueError), (99, -1, ValueError)],
)
def test_GridSelection_bad_params(percentile_cut, redshift_cut, errortype):
    """Test bad parameters that should raise ValueError"""
    with pytest.raises(errortype):
        GridSelection.make_stage(percentile_cut=percentile_cut, redshift_cut=redshift_cut)


def test_GridSelection_returns_correct_shape(data):
    """Make sure returns same number of columns, fewer rows"""
    degrader = GridSelection.make_stage(pessimistic_redshift_cut=1.0)
    degraded_data = degrader(data).data
    assert degraded_data.shape[0] < data.data.shape[0]
    assert degraded_data.shape[1] == data.data.shape[1] 
    os.remove(degrader.get_output(degrader.get_aliased_tag("output"), final_name=True))

    

def test_SpecSelection(data):
    bands = ["u", "g", "r", "i", "z", "y"]
    band_dict = {band: f"mag_{band}_lsst" for band in bands}
    rename_dict = {f"{band}_err": f"mag_err_{band}_lsst" for band in bands}
    rename_dict.update({f"{band}": f"mag_{band}_lsst" for band in bands})
    standard_colnames = [f"mag_{band}_lsst" for band in "ugrizy"]

    col_remapper_test = ColumnMapper.make_stage(
        name="col_remapper_test", hdf5_groupname="", columns=rename_dict
    )
    data = col_remapper_test(data)

    degrader_GAMA = SpecSelection_GAMA.make_stage()
    degrader_GAMA(data)
    degrader_GAMA.__repr__()

    os.remove(degrader_GAMA.get_output(degrader_GAMA.get_aliased_tag("output"), final_name=True))

    degrader_BOSS = SpecSelection_BOSS.make_stage()
    degrader_BOSS(data)
    degrader_BOSS.__repr__()

    os.remove(degrader_BOSS.get_output(degrader_BOSS.get_aliased_tag("output"), final_name=True))

    degrader_DEEP2 = SpecSelection_DEEP2.make_stage()
    degrader_DEEP2(data)
    degrader_DEEP2.__repr__()

    os.remove(degrader_DEEP2.get_output(degrader_DEEP2.get_aliased_tag("output"), final_name=True))

    degrader_VVDSf02 = SpecSelection_VVDSf02.make_stage()
    degrader_VVDSf02(data)
    degrader_VVDSf02.__repr__()

    degrader_zCOSMOS = SpecSelection_zCOSMOS.make_stage(colnames={"i": "mag_i_lsst", "redshift": "redshift"})
    degrader_zCOSMOS(data)
    degrader_zCOSMOS.__repr__()

    os.remove(degrader_zCOSMOS.get_output(degrader_zCOSMOS.get_aliased_tag("output"), final_name=True))

    degrader_HSC = SpecSelection_HSC.make_stage()
    degrader_HSC(data)
    degrader_HSC.__repr__()

    os.remove(degrader_HSC.get_output(degrader_HSC.get_aliased_tag("output"), final_name=True))

    degrader_HSC = SpecSelection_HSC.make_stage(percentile_cut=70)
    degrader_HSC(data)
    degrader_HSC.__repr__()

    os.remove(degrader_HSC.get_output(degrader_HSC.get_aliased_tag("output"), final_name=True))


def test_SpecSelection_low_N_tot(data_forspec):
    bands = ["u", "g", "r", "i", "z", "y"]
    band_dict = {band: f"mag_{band}_lsst" for band in bands}
    rename_dict = {f"{band}_err": f"mag_err_{band}_lsst" for band in bands}
    rename_dict.update({f"{band}": f"mag_{band}_lsst" for band in bands})
    standard_colnames = [f"mag_{band}_lsst" for band in "ugrizy"]

    col_remapper_test = ColumnMapper.make_stage(
        name="col_remapper_test", hdf5_groupname="", columns=rename_dict
    )
    data_forspec = col_remapper_test(data_forspec)

    degrader_zCOSMOS = SpecSelection_zCOSMOS.make_stage(N_tot=1)
    degrader_zCOSMOS(data_forspec)

    os.remove(degrader_zCOSMOS.get_output(degrader_zCOSMOS.get_aliased_tag("output"), final_name=True))


@pytest.mark.parametrize("N_tot, errortype", [(-1, ValueError)])
def test_SpecSelection_bad_params(N_tot, errortype):
    """Test bad parameters that should raise TypeError"""
    with pytest.raises(errortype):
        SpecSelection.make_stage(N_tot=N_tot)


@pytest.mark.parametrize("errortype", [(ValueError)])
def test_SpecSelection_bad_colname(data, errortype):
    """Test bad parameters that should raise TypeError"""
    with pytest.raises(errortype):
        degrader_GAMA = SpecSelection_GAMA.make_stage()
        degrader_GAMA(data)


@pytest.mark.parametrize("success_rate_dir, errortype", [("/this/path/should/not/exist", ValueError)])
def test_SpecSelection_bad_path(success_rate_dir, errortype):
    """Test bad parameters that should raise TypeError"""
    with pytest.raises(errortype):
        SpecSelection.make_stage(success_rate_dir=success_rate_dir)


def test_ObsCondition_returns_correct_shape(data):
    """Test that the ObsCondition returns the correct shape"""

    degrader = ObsCondition.make_stage()

    degraded_data = degrader(data).data
    
    # columns added: pixel, ugrizyerrors, ra, dec
    assert degraded_data.shape == (data.data.shape[0], 2 * data.data.shape[1] + 2)
    os.remove(degrader.get_output(degrader.get_aliased_tag("output"), final_name=True))


def test_ObsCondition_random_seed(data):
    """Test control with random seeds."""
    degrader1 = ObsCondition.make_stage(random_seed=0)
    degrader2 = ObsCondition.make_stage(random_seed=0)

    # make sure setting the same seeds yields the same output
    degraded_data1 = degrader1(data).data
    degraded_data2 = degrader2(data).data
    assert degraded_data1.equals(degraded_data2)

    # make sure setting different seeds yields different output
    degrader3 = ObsCondition.make_stage(random_seed=1)
    degraded_data3 = degrader3(data).data.to_numpy()
    assert not degraded_data1.equals(degraded_data3)

    os.remove(degrader3.get_output(degrader3.get_aliased_tag("output"), final_name=True))


@pytest.mark.parametrize(
    "nside, error",
    [
        (-1, ValueError),
        (123, ValueError),
    ],
)
def test_ObsCondition_bad_nside(nside, error):
    """Test bad nside should raise Value and Type errors."""
    with pytest.raises(error):
        ObsCondition.make_stage(nside=nside)


@pytest.mark.parametrize(
    "mask, error",
    [
        ("xx", ValueError),
        ("", ValueError),
    ],
)
def test_ObsCondition_bad_mask(mask, error):
    """Test bad mask should raise Value and Type errors."""
    with pytest.raises(error):
        ObsCondition.make_stage(mask=mask)


@pytest.mark.parametrize(
    "weight, error",
    [
        ("xx", ValueError),
    ],
)
def test_ObsCondition_bad_weight(weight, error):
    """Test bad weight should raise Value and Type errors."""
    with pytest.raises(error):
        ObsCondition.make_stage(weight=weight)


@pytest.mark.parametrize(
    "map_dict, error",
    [
        # band-dependent
        ({"m5": "xx"}, TypeError),
        ({"m5": {"u": False}}, TypeError),
        ({"m5": {"u": "xx"}}, ValueError),
        ({"m5": {}}, ValueError),
        ({"nVisYr": "xx"}, TypeError),
        ({"gamma": {"u": "xx"}}, ValueError),
        ({"msky": {"u": "xx"}}, ValueError),
        ({"theta": {"u": False}}, TypeError),
        ({"km": {"u": False}}, TypeError),
        # band-independent
        ({"airmass": "xx"}, ValueError),
        ({"airmass": False}, TypeError),
        ({"tvis": False}, TypeError),
        # wrong key name
        ({"m5sigma": {"u": 27}}, ValueError),
        # nYrObs not float
        ({"nYrObs": "xx"}, TypeError),
    ],
)
def test_ObsCondition_bad_map_dict(map_dict, error):
    """Test bad map_dict that should raise Value and Type errors."""
    with pytest.raises(error):
        ObsCondition.make_stage(map_dict=map_dict)


def test_ObsCondition_extended(data):
    # Testing extended parameter values
    weight = ""
    map_dict = {
        "airmass": find_rail_file('examples_data/creation_data/data/survey_conditions/minion_1016_dc2_Median_airmass_i_and_nightlt1825_HEAL.fits'),
        "EBV": 0.0,
        "nVisYr": {"u": 50.0},
        "tvis": 30.0,
    }
    tot_nVis_flag = True
    random_seed = None

    degrader_ext = ObsCondition.make_stage(
        weight=weight,
        tot_nVis_flag=tot_nVis_flag,
        random_seed=random_seed,
        map_dict=map_dict,
    )
    degrader_ext(data)
    degrader_ext.__repr__()

    os.remove(degrader_ext.get_output(degrader_ext.get_aliased_tag("output"), final_name=True))


def test_ObsCondition_empty_map_dict(data):
    """Test control with random seeds."""
    degrader1 = ObsCondition.make_stage(random_seed=0, map_dict={})
    degrader2 = PhoterrErrorModel()

    # make sure setting the same seeds yields the same output
    degraded_data1 = degrader1(data).data
    degraded_data2 = degrader2(data.data, random_state=0)
    assert degraded_data1.equals(degraded_data2)

    os.remove(degrader1.get_output(degrader1.get_aliased_tag("output"), final_name=True))
    

def test_ObsCondition_renameDict(data_with_radec):
    """Test with renameDict included"""
    degrader1 = ObsCondition.make_stage(random_seed=0, map_dict={"EBV": 0.0,"renameDict": {"u": "u", "ra": "ra", "dec":"dec"},})

    # make sure setting the same seeds yields the same output
    degraded_data1 = degrader1(data_with_radec).data

    os.remove(degrader1.get_output(degrader1.get_aliased_tag("output"), final_name=True))
    
    
def test_ObsCondition_data_with_radec(data_with_radec):
    """Test with ra dec in data"""
    degrader1 = ObsCondition.make_stage(random_seed=0, map_dict={"EBV": 0.0})
    degraded_data1 = degrader1(data_with_radec).data

    os.remove(degrader1.get_output(degrader1.get_aliased_tag("output"), final_name=True))


def test_LSSTErrorModel_returns_correct_columns(data):
    # Setup the stage
    degrader = LSSTErrorModel.make_stage()

    # Apply the degrader and get the data out
    degraded_data = degrader(data).data

    # Check that we still have the same number of rows, and added an extra
    # column for each band
    assert degraded_data.shape == (data.data.shape[0], 2 * data.data.shape[1] - 1)
    for band in "ugrizy":
        assert band in degraded_data.columns
        assert f"{band}_err" in degraded_data.columns
    os.remove(degrader.get_output(degrader.get_aliased_tag("output"), final_name=True))


def test_EucliErrorModel(data):
    # Setup the stage
    degrader = EuclidErrorModel.make_stage()


def test_BLModel(data_for_bl):
    # Setup the stage

    degrader = UnrecBlModel.make_stage(
        name='unrec_bl_model',
        ra_label='ra',
        dec_label='dec',
        linking_lengths=1.0,
        bands='ugrizy',
        ref_band='i',
	redshift_col='redshift',
        seed=1234)

    # Apply the degrader and get the data out
    outputs = degrader(data_for_bl)
    degraded_data = outputs['output'].data
    truth_components = outputs['compInd'].data

    # Check output data has less rows than input data
    assert degraded_data.shape[0] < data_for_bl.data.shape[0]

    # Check components has the same rows as input data
    assert truth_components.shape[0] == data_for_bl.data.shape[0]

    os.remove(degrader.get_output(degrader.get_aliased_tag("output"), final_name=True))
    os.remove(degrader.get_output(degrader.get_aliased_tag("compInd"), final_name=True))
