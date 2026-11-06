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
            {"catalog_config": {"split_seed": 7, "n_target": 2000, "group_col": "ref_id"}},
        ),
        (
            "rail.pipelines.degradation.hscfy_pz_prepare.HscfyPzPreparePipeline",
            {"catalog_config": {"zcell": True, "z_edges": [0.0, 0.5, 1.0, 2.0, 7.0]}},
        ),
        (
            "rail.pipelines.degradation.hscfy_pz_prepare.HscfyPzPreparePipeline",
            {
                "catalog_config": {
                    "zcell": True,
                    "nondetect_val": 40.0,
                    "dustmap_dir": "dustmaps",
                    "deredden": {
                        "input": {"mag_r": 2.70, "mag_i": 2.06, "i_gauss2_mag": 2.06},
                        "population": {"mag_r": 2.70, "mag_i": 2.06},
                        "truth": {"mag_r": 2.70, "mag_i": 2.06},
                    },
                }
            },
        ),
        ("rail.pipelines.degradation.hscfy_pz_prepare.HoldoutTrainPipeline", {}),
        (
            "rail.pipelines.degradation.hscfy_pz_prepare.HoldoutTrainPipeline",
            {
                "catalog_config": {
                    "holdout_cols": ["object_id", "ref_id"],
                    "require_detected": ["mag_r", "mag_i", "mag_z"],
                    "nondetect_val": 40.0,
                    "dustmap_dir": "dustmaps",
                    "deredden": {"input": {"mag_r": 2.70, "mag_i": 2.06}},
                }
            },
        ),
        ("rail.pipelines.degradation.deep_field_redshift_pipeline.DeepFieldRedshiftPipeline", {}),
        (
            "rail.pipelines.degradation.deep_field_redshift_pipeline.DeepFieldRedshiftPipeline",
            {
                "catalog_config": {
                    "shear_catalog": "deep_shear_catalog.hdf5",
                    "shear_group": "shear",
                    "cuts": ["mag_i < 24", "s2n > 10", "esq < 0.16", "n_mask_base < 0.035"],
                    "derived_cols": {"esq": "e1_raw**2 + e2_raw**2"},
                    "skip_missing_cuts": True,
                    "noise_bands": ["r", "i", "z"],
                    "sn_col": "s2n",
                    "nondetect_val": 40.0,
                }
            },
        ),
        ("rail.pipelines.degradation.sim_pz_prepare.SimPzPreparePipeline", {}),
        (
            "rail.pipelines.degradation.sim_pz_prepare.SimPzPreparePipeline",
            {"catalog_config": {"n_holdout": 1000, "match_redshift": True, "target_scale": 2}},
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
        ("rail.pipelines.degradation.shear_population_pipeline.ShearPopulationPipeline", {}),
        (
            "rail.pipelines.degradation.shear_population_pipeline.ShearPopulationPipeline",
            {
                "catalog_config": {
                    "shear_catalog": "shear_catalog.hdf5",
                    "shear_group": "shear",
                    "cuts": ["mag_i < 24", "s2n > 10", "psf_g1_i < 0.1"],
                    "require_finite": ["mag_r", "mag_i", "mag_z"],
                    "color_cols": ["mag_r", "mag_i", "mag_z"],
                    "abs_color_max": 1.3,
                    "n_sample": 1000,
                }
            },
        ),
    ],
)
def test_build_and_read_pipeline(pipeline_class, options):
    catalog_utils.apply_defaults("com_cam")
    build_and_read_pipeline(pipeline_class, **options)
