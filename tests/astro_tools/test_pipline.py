import os

import pytest
from rail.utils import catalog_utils
from rail.utils.testing_utils import build_and_read_pipeline


@pytest.mark.parametrize(
    "pipeline_class, options",
    [
        ("rail.pipelines.degradation.apply_phot_errors.ApplyPhotErrorsPipeline", {}),
        (
            "rail.pipelines.degradation.apply_phot_errors.ApplyPhotErrorsPipeline",
            {"parallel": True},
        ),
        ("rail.pipelines.degradation.blending.BlendingPipeline", {}),
        (
            "rail.pipelines.degradation.spectroscopic_selection_pipeline.SpectroscopicSelectionPipeline",
            {},
        ),
        ("rail.pipelines.degradation.truth_to_observed.TruthToObservedPipeline", {}),
        (
            "rail.pipelines.degradation.truth_to_observed.TruthToObservedPipeline",
            {"blending": True},
        ),
        (
            "rail.pipelines.degradation.truth_to_observed.TruthToObservedPipeline",
            {"parallel": True},
        ),
        (
            "rail.pipelines.degradation.truth_to_observed.TruthToObservedPipeline",
            {"blending": True, "parallel": True},
        ),
        ("rail.pipelines.degradation.hscfy_pz_prepare.HscfyPzPreparePipeline", {}),
        (
            "rail.pipelines.degradation.hscfy_pz_prepare.HscfyPzPreparePipeline",
            {"catalog_config": {"split_seed": 7, "n_repeat": 2, "group_col": "ref_id"}},
        ),
        ("rail.pipelines.degradation.crossmatch_pipeline.CrossMatchPipeline", {}),
        (
            "rail.pipelines.degradation.crossmatch_pipeline.CrossMatchPipeline",
            {
                "catalog_config": {
                    "shear_catalog": "shear_catalog.hdf5",
                    "shear_cuts": ["is_primary == True", "mag_i < 25.5"],
                    "ref_cuts": ["z_source in DESI_DR1,SDSS_DR17"],
                    "edge_halo_arcsec": 2.0,
                }
            },
        ),
    ],
)
def test_build_and_read_pipeline(pipeline_class, options):
    catalog_utils.apply_defaults("com_cam")
    build_and_read_pipeline(pipeline_class, **options)
