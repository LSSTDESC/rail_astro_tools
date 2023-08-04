"""Test that the GCR Catalog Reader stage works."""
from rail.core.stage import RailStage
from rail.creation.engines.gcr_engine import GCRCreator
import os

# Allow us to overwrite the data store
DS = RailStage.data_store
DS.__class__.allow_overwrite = True


def test_GCRCreator():
    """Test that GCRCreator returns correct columns and samples, and filters work."""
    # Make the stage
    gc = GCRCreator.make_stage(n_samples=1)

    # Override with the test catalog
    this_dir = os.path.dirname(__file__)
    reader = "dc2_object_run1.1p_tract4850"
    config = {
        "base_dir": os.path.join(this_dir, "gcr_test_data"),
        "filename_pattern": "test_object_tract_4850.hdf5",
    }
    gc._catalog_override = [reader, config]

    # Sample 4 galaxies with LSST bands
    bands = [f"mag_{band}" for band in "ugrizy"]
    cat = gc.sample(n_samples=4, quantities=bands, filters=None)

    # Check the correct columns are returned
    assert cat.data.columns.tolist() == bands

    # Check the correct number of samples were returned
    assert len(cat.data) == 4

    # Check filters work
    assert cat.data["mag_r"].max() > 24
    cat = gc.sample(n_samples=4, quantities=bands, filters=["mag_r < 24"])
    assert cat.data["mag_r"].max() < 24
