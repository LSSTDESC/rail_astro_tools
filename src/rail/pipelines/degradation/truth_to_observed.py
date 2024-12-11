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
from rail.creation.degraders.unrec_bl_model import UnrecBlModel

from .spectroscopic_selection_pipeline import SELECTORS, CommonConfigParams
from .apply_phot_errors import ERROR_MODELS


if 'PZ_DUSTMAP_DIR' not in os.environ:  # pragma: no cover
    os.environ['PZ_DUSTMAP_DIR'] = '.'

dustmap_dir = os.path.expandvars("${PZ_DUSTMAP_DIR}")


class TruthToObservedPipeline(RailPipeline):

    default_input_dict = dict(input='dummy.in')

    def __init__(self, error_models=None, selectors=None, blending=False):
        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        active_catalog_config = catalog_utils.CatalogConfigBase.active_class()
        band_name_dict = active_catalog_config.band_name_dict()

        if error_models is None:
            error_models = ERROR_MODELS.copy()

        if selectors is None:
            selectors = SELECTORS.copy()

        config_pars = CommonConfigParams.copy()
        config_pars['colnames'] = band_name_dict.copy()
        config_pars['colnames']['redshift'] = active_catalog_config.redshift_col

        self.reddener = Reddener.build(
            dustmap_dir=dustmap_dir,
            copy_all_cols=True,
        )
        previous_stage = self.reddener

        if blending:
            self.unrec_bl = UnrecBlModel.build()
            previous_stage = self.unrec_bl

        for key, val in error_models.items():
            error_model_class = ceci.PipelineStage.get_stage(val['ErrorModel'], val['Module'])
            the_error_model = error_model_class.make_and_connect(
                name=f'error_model_{key}',
                connections=dict(input=previous_stage.io.output),
                hdf5_groupname='',
                renameDict=band_name_dict,
            )
            self.add_stage(the_error_model)
            previous_stage = the_error_model

            dereddener_errors = Dereddener.make_and_connect(
                name=f"deredden_{key}",
                dustmap_dir=dustmap_dir,
                connections=dict(input=previous_stage.io.output),
                copy_all_cols=True,
            )
            self.add_stage(dereddener_errors)
            previous_stage = dereddener_errors
            
            for key2, val2 in selectors.items():
                the_class = ceci.PipelineStage.get_stage(val2['Select'], val2['Module'])
                the_selector = the_class.make_and_connect(
                    name=f'select_{key}_{key2}',
                    connections=dict(input=previous_stage.io.output),
                    **config_pars,
                )            
                self.add_stage(the_selector)
