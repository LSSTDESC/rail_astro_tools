import os
import numpy as np
import pytest
from rail.utils.path_utils import RAILDIR
from rail.tools.filter_tools import ccm_alam_over_ebv
from rail.tools.filter_tools import calc_lambda_effective_from_file
from rail.tools.filter_tools import calc_lambda_effective_from_arrays


def test_wl_from_file():
    gfiltpath = os.path.join(RAILDIR, "rail/examples_data/estimation_data/data/FILTER/DC2LSST_g.res")
    leff = calc_lambda_effective_from_file(gfiltpath)
    assert np.isclose(leff, 4826.8517, atol=1.e-2)
    alam = ccm_alam_over_ebv(leff)
    assert np.isclose(alam, 3.64287, atol=1.e-2)

def test_wl_from_dummy_data():
    dummywl = np.linspace(3000, 6000, 301)
    tr = np.exp(-1. * (dummywl - 4500.)**2 / (2.0 * 900.))
    leff = calc_lambda_effective_from_arrays(dummywl, tr)
    assert np.isclose(leff, 4500., atol=1.e-2)
    # alamba for 4500 should be 3.9925
    alam = ccm_alam_over_ebv(leff)
    assert np.isclose(alam, 3.9925, atol=1.e-2)

def test_specific_wls():
    wllist = np.array([10_000, 2_000, 1428, 1000])
    alams = np.array([1.2524, 8.81183, 8.62025, 16.2389]) 
    for wl, true_alam in zip(wllist, alams):
        alam = ccm_alam_over_ebv(wl)
        assert np.isclose(alam, true_alam, atol=1.e-2)

def test_wl_out_of_range():
    badwl = 500.0
    with pytest.raises(ValueError):
        alam = ccm_alam_over_ebv(badwl)
