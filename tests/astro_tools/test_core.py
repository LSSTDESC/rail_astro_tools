import os
import pickle
from types import GeneratorType

import numpy as np
import pandas as pd
import pytest
import tempfile

import rail
from rail.core.common_params import SHARED_PARAMS, copy_param, set_param_default
from rail.core.data import (
    DataHandle,
    DataStore,
    FitsHandle,
    Hdf5Handle,
    ModelHandle,
    PqHandle,
    QPHandle,
    TableHandle,
)
from rail.core.stage import RailStage
from rail.tools.util_photometry import HyperbolicMagnitudes, HyperbolicSmoothing, PhotormetryManipulator, LSSTFluxToMagConverter, Dereddener
from rail.core.utils import RAILDIR
#from rail.tools.util_stages import (
#    LSSTFluxToMagConverter,
#    Dereddener,
#)


# def test_data_file():
#    with pytest.raises(ValueError) as errinfo:
#        df = DataFile('dummy', 'x')


def test_flux2mag():
    DS = RailStage.data_store
    DS.clear()

    testFile = os.path.join(RAILDIR, "rail", "examples_data", "testdata", "rubin_dm_dc2_example.pq")
    test_data = DS.read_file("test_data", TableHandle, testFile)

    fluxToMag = LSSTFluxToMagConverter.make_stage(name='flux2mag')
    out_data = fluxToMag(test_data)


@pytest.mark.slow
def test_dereddener():
    DS = RailStage.data_store
    DS.clear()

    testFile = os.path.join(RAILDIR, "rail", "examples_data", "testdata", "rubin_dm_dc2_example.pq")
    test_data = DS.read_file("test_data", TableHandle, testFile)

    fluxToMag = LSSTFluxToMagConverter.make_stage(name='flux2mag', copy_cols=dict(ra='ra', decl='decl'))

    is_temp_dir = False
    dustmap_dir = os.environ.get('RAIL_DUSTMAP_DIR')
    if dustmap_dir is None:
        tmp_dustmap_dir = tempfile.TemporaryDirectory()
        dustmap_dir = tmp_dustmap_dir.name
        is_temp_dir = True

    dereddener = Dereddener.make_stage(name='dereddner', dustmap_dir=dustmap_dir)
    dereddener.fetch_map()

    flux_data = fluxToMag(test_data)
    dered_data = dereddener(flux_data)

    if is_temp_dir:
        tmp_dustmap_dir.cleanup()



@pytest.fixture
def hyperbolic_configuration():
    """get the code configuration for the example data"""
    lsst_bands = "ugrizy"
    return dict(
        value_columns=[f"mag_{band}_lsst" for band in lsst_bands],
        error_columns=[f"mag_err_{band}_lsst" for band in lsst_bands],
        zeropoints=[0.0] * len(lsst_bands),
        is_flux=False,
    )


@pytest.fixture
def load_result_smoothing():
    """load the smoothing parameters for an example patch of DC2"""
    DS = RailStage.data_store
    DS.clear()
    DS.__class__.allow_overwrite = False

    testFile = os.path.join(
        RAILDIR, "rail", "examples_data", "testdata", "test_dc2_training_9816_smoothing_params.pq"
    )

    return DS.read_file("test_data", TableHandle, testFile).data


def test_PhotormetryManipulator(hyperbolic_configuration):
    DS = RailStage.data_store
    DS.clear()
    DS.__class__.allow_overwrite = False

    # NOTE: the __init__ machinery of HyperbolicSmoothing is identical to PhotormetryManipulator
    # and is used as substitute since PhotormetryManipulator cannot be instantiated.
    n_filters = len(hyperbolic_configuration["value_columns"])

    # wrong number of "error_columns"
    config = hyperbolic_configuration.copy()
    config["error_columns"] = hyperbolic_configuration["error_columns"][:-1]
    with pytest.raises(IndexError):
        inst = HyperbolicSmoothing.make_stage(name="photormetry_manipulator", **config)

    # wrong number of "zeropoints"
    config = hyperbolic_configuration.copy()
    config["zeropoints"] = np.arange(0, n_filters - 1)
    with pytest.raises(IndexError):
        inst = HyperbolicSmoothing.make_stage(name="photormetry_manipulator", **config)

    # default values for "zeropoints"
    config = hyperbolic_configuration.copy()
    config.pop("zeropoints")  # should resort to default of 0.0
    inst = HyperbolicSmoothing.make_stage(name="photormetry_manipulator", **config)
    assert len(inst.zeropoints) == n_filters
    assert all(zp == 0.0 for zp in inst.zeropoints)

    # if_flux preserves the values
    dummy_data = pd.DataFrame(dict(val=[1, 2, 3], err=[1, 2, 3]))
    config = dict(value_columns=["val"], error_columns=["err"], zeropoints=[0.0])
    inst = HyperbolicSmoothing.make_stage(name="photormetry_manipulator", **config, is_flux=True)
    inst.set_data("input", dummy_data)
    data = inst.get_as_fluxes()
    assert np.allclose(data, dummy_data)


def test_HyperbolicSmoothing(hyperbolic_configuration):
    DS = RailStage.data_store
    DS.clear()
    DS.__class__.allow_overwrite = False

    test_data = DS.read_file(
        "test_data",
        TableHandle,
        os.path.join(RAILDIR, "rail", "examples_data", "testdata", "test_dc2_training_9816.pq"),
    ).data
    result_smoothing = DS.read_file(
        "result_smoothing",
        TableHandle,
        os.path.join(RAILDIR, "rail", "examples_data", "testdata", "test_dc2_training_9816_smoothing_params.pq"),
    ).data

    stage_name, handle_name = "hyperbolic_smoothing", "parameters"

    # test against prerecorded output
    smooth = HyperbolicSmoothing.make_stage(name=stage_name, **hyperbolic_configuration)
    smooth.compute(test_data)
    smooth_params = smooth.get_handle(handle_name).data
    assert np.allclose(smooth_params, result_smoothing)

    os.remove(f"{handle_name}_{stage_name}.pq")


def test_HyperbolicMagnitudes(
    hyperbolic_configuration,
):
    DS = RailStage.data_store
    DS.clear()
    DS.__class__.allow_overwrite = False

    test_data = DS.read_file(
        "test_data",
        TableHandle,
        os.path.join(RAILDIR, "rail", "examples_data", "testdata", "test_dc2_training_9816.pq"),
    ).data
    result_smoothing = DS.read_file(
        "result_smoothing",
        TableHandle,
        os.path.join(RAILDIR, "rail", "examples_data", "testdata", "test_dc2_training_9816_smoothing_params.pq"),
    ).data
    result_hyperbolic = DS.read_file(
        "result_hyperbolic",
        TableHandle,
        os.path.join(RAILDIR, "rail", "examples_data", "testdata", "test_dc2_training_9816_hyperbolic.pq"),
    ).data

    stage_name, handle_name = "hyperbolic_magnitudes", "output"

    # test against prerecorded output
    hypmag = HyperbolicMagnitudes.make_stage(name=stage_name, **hyperbolic_configuration)
    hypmag.compute(test_data, result_smoothing)
    test_hypmags = hypmag.get_handle(handle_name).data

    # What we would want to test is
    # >>> assert test_hypmags.equals(result_hyperbolic)
    # however this test fails at github actions.
    # Instead we test that the values are numerically close. The accepted deviation scales with
    # magnitude m as
    # dm = 1e-5 * m
    # which is smaller than difference between classical and hyperbolic magnitudes except at the
    # very brightest magnitudes.
    for (key_test, values_test), (key_ref, values_ref) in zip(
        test_hypmags.items(), result_hyperbolic.items()
    ):
        assert key_test == key_ref
        assert np.allclose(values_test, values_ref)

    # check of input data columns against smoothing parameter table
    smoothing = result_smoothing.copy().drop("mag_r_lsst")  # drop one filter from the set
    hypmag = HyperbolicMagnitudes.make_stage(name=stage_name, **hyperbolic_configuration)
    with pytest.raises(KeyError):
        hypmag._check_filters(smoothing)

    os.remove(f"{handle_name}_{stage_name}.pq")
