"""Dereddener under ceci (pyarrow input), its non-detection and E(B-V) options, and the
dereddening front end of HscfyPzPreparePipeline."""

import os

import numpy as np
import pyarrow as pa
import pytest
import yaml

from rail.core.data import DataStore
from rail.core.stage import RailPipeline
from rail.pipelines.degradation.hscfy_pz_prepare import HscfyPzPreparePipeline  # noqa: F401
from rail.tools.photometry_tools import Dereddener, Reddener

DataStore.allow_overwrite = True

# The SFD maps, if RAIL_DUSTMAP_DIR points at them.  Otherwise -- as in CI, where the download
# is broken (see test_core.test_dereddener) -- the map is faked, so the stage logic is still
# exercised; only the E(B-V) values themselves are not SFD's.
DUSTMAP_DIR = os.environ.get("RAIL_DUSTMAP_DIR", "")
HAVE_MAP = os.path.exists(os.path.join(DUSTMAP_DIR, "sfd", "SFD_dust_4096_ngp.fits"))


class _FakeSFDQuery:
    """Stands in for dustmaps.sfd.SFDQuery without reading any file: E(B-V) > 0 everywhere."""

    def __call__(self, coords):
        return 0.01 + 0.001 * np.abs(coords.dec.deg)


@pytest.fixture(name="dustmap_dir")
def fixture_dustmap_dir(tmp_path, monkeypatch):
    """The real map directory if there is one, else an empty directory and a fake map."""
    if HAVE_MAP:  # pragma: no cover - only where the SFD files are on disk
        return DUSTMAP_DIR
    import dustmaps.sfd

    # an empty directory makes fetch_map() call fetch(); make that a no-op
    monkeypatch.setattr(dustmaps.sfd, "fetch", lambda: None)
    monkeypatch.setattr(dustmaps.sfd, "SFDQuery", _FakeSFDQuery)
    return str(tmp_path)


def _table():
    return pa.table(
        {
            "ra": np.array([10.0, 150.1, 150.1, 60.0]),
            "dec": np.array([-40.0, 2.2, 2.2, -48.0]),
            "mag_i": np.array([22.0, 40.0, np.nan, 23.5]),
            "survey": ["a", "b", "c", "d"],
        }
    )


def test_pyarrow_input_sentinel_and_ebv(dustmap_dir):
    stage = Dereddener.make_stage(
        name="dered_pa",
        dustmap_dir=dustmap_dir,
        band_a_env={"mag_i": 2.06},
        copy_all_cols=True,
        nondetect_val=40.0,
        ebv_name="ebv_sfd",
    )
    out = stage(_table()).data
    assert list(out["survey"]) == ["a", "b", "c", "d"]
    ebv = out["ebv_sfd"].to_numpy()
    assert np.all(ebv > 0)
    np.testing.assert_allclose(out["mag_i"].iloc[[0, 3]], np.array([22.0, 23.5]) - 2.06 * ebv[[0, 3]])
    assert out["mag_i"].iloc[1] == 40.0  # the sentinel is not a magnitude
    assert np.isnan(out["mag_i"].iloc[2])


def test_default_options_unchanged(dustmap_dir):
    """Without the new options a pandas input gives what it always did."""
    stage = Dereddener.make_stage(
        name="dered_pd", dustmap_dir=dustmap_dir, band_a_env={"mag_i": 2.06}, copy_all_cols=True
    )
    out = stage(_table().to_pandas()).data
    assert "ebv_sfd" not in out
    assert out["mag_i"].iloc[1] < 40.0


def test_ebv_name_collision(dustmap_dir):
    stage = Dereddener.make_stage(
        name="dered_clash",
        dustmap_dir=dustmap_dir,
        band_a_env={"mag_i": 2.06},
        copy_all_cols=True,
        ebv_name="survey",
    )
    with pytest.raises(ValueError, match="already a column"):
        stage(_table())


def test_reddener_undoes_the_dereddener(dustmap_dir):
    options = dict(dustmap_dir=dustmap_dir, band_a_env={"mag_i": 2.06}, copy_all_cols=True)
    frame = _table().to_pandas()
    dered = Dereddener.make_stage(name="dered_rt", **options)(frame).data
    back = Reddener.make_stage(name="red_rt", **options)(dered).data
    np.testing.assert_allclose(back["mag_i"].iloc[[0, 3]], frame["mag_i"].iloc[[0, 3]])


def _build(tmp_path, config):
    yml = str(tmp_path / "pz.yml")
    RailPipeline.build_and_write(
        "HscfyPzPreparePipeline",
        yml,
        input_dict=dict(input="xm.pq", population="pop.pq", truth="truth.pq"),
        output_dir=str(tmp_path),
        log_dir=str(tmp_path),
        catalog_config=config,
    )
    with open(yml) as fh:
        pipe = yaml.safe_load(fh)
    with open(yml.replace(".yml", "_config.yml")) as fh:
        stages = yaml.safe_load(fh)
    return {s["name"]: s for s in pipe["stages"]}, stages


def test_pipeline_wiring(tmp_path):
    a_env = {"mag_i": 2.06}
    config = dict(
        zcell=True,
        nondetect_val=40.0,
        dustmap_dir="dustmaps",
        deredden={"input": a_env, "population": a_env, "truth": a_env},
    )
    stages, stage_config = _build(tmp_path, config)
    assert stages["deredden_population"]["aliases"]["input"] == "population"
    assert stages["deredden_truth"]["aliases"]["input"] == "truth"
    assert "input" not in stages["deredden_input"]["aliases"]
    assert stages["split"]["aliases"]["input"] == "output_deredden_input"
    som = stages["som_resample_test"]["aliases"]
    assert som["population"] == "output_deredden_population"
    assert som["truth"] == "output_deredden_truth"
    assert stage_config["deredden_input"]["nondetect_val"] == 40.0
    assert stage_config["deredden_input"]["ebv_name"] == "ebv_sfd"
    assert stage_config["deredden_input"]["band_a_env"] == a_env


def test_pipeline_without_dereddening(tmp_path):
    stages, _ = _build(tmp_path, {})
    assert not [name for name in stages if name.startswith("deredden")]
    assert "population" not in stages["som_resample_test"]["aliases"]


@pytest.mark.parametrize(
    "config, error",
    [
        (dict(deredden={"pool": {"mag_i": 2.06}}, dustmap_dir="d"), KeyError),
        (dict(deredden={"truth": {"mag_i": 2.06}}, dustmap_dir="d"), ValueError),
        (dict(deredden={"input": {"mag_i": 2.06}}), ValueError),
    ],
)
def test_pipeline_deredden_errors(tmp_path, config, error):
    with pytest.raises(error):
        _build(tmp_path, config)
