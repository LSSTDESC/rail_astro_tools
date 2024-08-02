#!/usr/bin/env python
# coding: utf-8

# Prerquisites, os, and numpy
import os
import numpy as np

# Various rail modules

from rail.core.stage import RailStage, RailPipeline

import ceci

from rail.utils.catalog_utils import CatalogConfigBase
from rail.core.utils import RAILDIR

SELECTORS = dict(
    GAMA = dict(
        Select='SpecSelection_GAMA',
        Module='rail.creation.degraders.spectroscopic_selections',
    ),
    BOSS = dict(
        Select='SpecSelection_BOSS',
        Module='rail.creation.degraders.spectroscopic_selections',
    ),
    VVDSf02 = dict(
        Select='SpecSelection_VVDSf02',
        Module='rail.creation.degraders.spectroscopic_selections',
    ),
    zCOSMOS = dict(
        Select='SpecSelection_zCOSMOS',
        Module='rail.creation.degraders.spectroscopic_selections',
    ),
    HSC = dict(
        Select='SpecSelection_HSC',
        Module='rail.creation.degraders.spectroscopic_selections',
    ),
)


CommonConfigParams = dict(
    N_tot = 100_000,
    nondetect_val = -np.inf,
    downsample= False,
)


class SpectroscopicSelectionPipeline(RailPipeline):

    default_input_dict = dict(input='dummy.in')
    
    def __init__(self, selectors=None):
        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        if selectors is None:
            selectors = SELECTORS.copy()

        config_pars = CommonConfigParams.copy()
        active_catalog = CatalogConfigBase.active_class()        
        if active_catalog:
            colnames = active_catalog.band_name_dict()
            colnames['redshift'] = active_catalog.redshift_col
            config_pars['colnames'] = colnames            

        for key, val in selectors.items():
            the_class = ceci.PipelineStage.get_stage(val['Select'], val['Module'])
            the_selector = the_class.make_and_connect(
                name=f'select_{key}',
                **config_pars,
            )
            
            self.add_stage(the_selector)
