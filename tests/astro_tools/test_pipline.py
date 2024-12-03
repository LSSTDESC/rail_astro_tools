import os
from rail.utils.testing_utils import build_and_read_pipeline

import pytest

@pytest.mark.parametrize(
    "pipeline_class, options",
    [
        ('rail.pipelines.degradation.apply_phot_errors.ApplyPhotErrorsPipeline', {}),
        ('rail.pipelines.degradation.blending.BlendingPipeline', {}),
        ('rail.pipelines.degradation.spectroscopic_selection_pipeline.SpectroscopicSelectionPipeline', {}),
        ('rail.pipelines.degradation.truth_to_observed.TruthToObservedPipeline', {'blending':True}),
    ]
)
def test_build_and_read_pipeline(pipeline_class, options):
    build_and_read_pipeline(pipeline_class, **options)

