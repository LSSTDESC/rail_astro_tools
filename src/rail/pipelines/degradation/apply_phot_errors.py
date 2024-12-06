#!/usr/bin/env python
# coding: utf-8

# Prerquisites, os, and numpy
import os
import numpy as np

# Various rail modules
from rail.tools.photometry_tools import Dereddener, Reddener

from rail.core.stage import RailStage, RailPipeline

import ceci

from rail.core.utils import RAILDIR
from rail.utils import catalog_utils

if 'PZ_DUSTMAP_DIR' not in os.environ:
    os.environ['PZ_DUSTMAP_DIR'] = '.'

dustmap_dir = os.path.expandvars("${PZ_DUSTMAP_DIR}")


ERROR_MODELS = dict(
    lsst = dict(
        ErrorModel='LSSTErrorModel',
        Module='rail.creation.degraders.photometric_errors',
    ),
    roman = dict(
        ErrorModel='RomanErrorModel',
        Module='rail.creation.degraders.photometric_errors',
    ),
    #euclid = dict(
    #    ErrorModel='EuclidErrorModel',
    #    Module='rail.creation.degraders.photometric_errors',
    #),
)



class ApplyPhotErrorsPipeline(RailPipeline):

    default_input_dict = dict(input='dummy.in')

    def __init__(self, error_models=None):
        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        if error_models is None:
            error_models = ERROR_MODELS

        self.reddener = Reddener.build(
            dustmap_dir=dustmap_dir,
            copy_all_cols=True,
        )

        previous_stage = self.reddener
        for key, val in error_models.items():
            error_model_class = ceci.PipelineStage.get_stage(val['ErrorModel'], val['Module'])
            the_error_model = error_model_class.make_and_connect(
                name=f'error_model_{key}',
                connections=dict(input=previous_stage.io.output),
                hdf5_groupname='',
                renameDict=catalog_utils.CatalogConfigBase.active_class().band_name_dict(),
            )
            self.add_stage(the_error_model)
            previous_stage = the_error_model

        self.dereddener_errors = Dereddener.build(
            dustmap_dir=dustmap_dir,
            connections=dict(input=previous_stage.io.output),
            copy_all_cols=True,
        )
