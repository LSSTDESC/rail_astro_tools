import os
from rail.utils.testing_utils import build_and_read_pipeline
from rail.utils import catalog_utils

import pytest

@pytest.mark.parametrize(
    "pipeline_class, options",
    [
        ('rail.pipelines.degradation.apply_phot_errors.ApplyPhotErrorsPipeline', {}),
        ('rail.pipelines.degradation.apply_phot_errors.ApplyPhotErrorsPipeline', {'parallel':True}),
        ('rail.pipelines.degradation.blending.BlendingPipeline', {}),
        ('rail.pipelines.degradation.spectroscopic_selection_pipeline.SpectroscopicSelectionPipeline', {}),
        ('rail.pipelines.degradation.truth_to_observed.TruthToObservedPipeline', {}),
        ('rail.pipelines.degradation.truth_to_observed.TruthToObservedPipeline', {'blending':True}),
        ('rail.pipelines.degradation.truth_to_observed.TruthToObservedPipeline', {'parallel':True}),
        ('rail.pipelines.degradation.truth_to_observed.TruthToObservedPipeline', {'blending':True, 'parallel':True}),
    ]
)
def test_build_and_read_pipeline(pipeline_class, options):    
    catalog_utils.apply_defaults('com_cam')
    build_and_read_pipeline(pipeline_class, **options)
