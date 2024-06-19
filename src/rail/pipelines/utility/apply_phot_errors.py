#!/usr/bin/env python
# coding: utf-8

# Prerquisites, os, and numpy
import os
import numpy as np

# Various rail modules
from rail.tools.photometry_tools import Dereddener, Reddener
from rail.creation.degraders.lsst_error_model import LSSTErrorModel

from rail.core.stage import RailStage, RailPipeline

import ceci

from rail.core.utils import RAILDIR

if 'PZ_DUSTMAP_DIR' not in os.environ:
    os.environ['PZ_DUSTMAP_DIR'] = '.'

dustmap_dir = os.path.expandvars("${PZ_DUSTMAP_DIR}")


class ApplyPhotErrorsPipeline(RailPipeline):

    default_input_dict = dict(input='dummy.in')
    
    def __init__(self, namer, selection='default', flavor='baseline'):
        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        local_dir = name.resolve_path_template(
            'degraded_catalog_path',
            selection=selection,
            flavor=flavor,
        )
        
        self.reddener = Reddener.build(
            dustmap_dir=dustmap_dir,
            output=os.path.join(local_dir, "output_reddener.pq"),
        )
        
        self.phot_errors = LSSTErrorModel.build(
            connections=dict(input=self.reddener.io.output),
            output=os.path.join(local_dir, "output_lsst_error_model.pq"),
        )

        self.dereddener_errors = Dereddener.build(
            dustmap_dir=dustmap_dir,
            connections=dict(input=self.phot_errors.io.output),
            output=os.path.join(local_dir, "output_dereddener.pq"),
        )

