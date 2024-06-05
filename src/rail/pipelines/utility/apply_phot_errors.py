#!/usr/bin/env python
# coding: utf-8

# Prerquisites, os, and numpy
import os
import numpy as np

# Various rail modules
from rail.tools.photometry_tools import Dereddener, Reddener
from rail.creation.degraders.lsst_error_model import LSSTErrorModel

from rail.utils.name_utils import NameFactory, DataType, CatalogType, ModelType, PdfType
from rail.core.stage import RailStage, RailPipeline

import ceci

namer = NameFactory()
from rail.core.utils import RAILDIR

if 'PZ_DUSTMAP_DIR' not in os.environ:
    os.environ['PZ_DUSTMAP_DIR'] = '.'

dustmap_dir = os.path.expandvars("${PZ_DUSTMAP_DIR}")


class ApplyPhotErrorsPipeline(RailPipeline):

    default_input_dict = dict(input='dummy.in')
    
    def __init__(self):
        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True
        
        self.reddener = Reddener.build(
            dustmap_dir=dustmap_dir,
            output=os.path.join(
                namer.get_data_dir(DataType.catalogs, CatalogType.degraded), "output_reddener.pq",
            ),
        )
        
        self.phot_errors = LSSTErrorModel.build(
            connections=dict(input=self.reddener.io.output),
            output=os.path.join(
                namer.get_data_dir(DataType.catalogs, CatalogType.degraded), "output_lsst_error_model.pq",
            ),
        )

        self.dereddener_errors = Dereddener.build(
            dustmap_dir=dustmap_dir,
            connections=dict(input=self.phot_errors.io.output),
            output=os.path.join(
                namer.get_data_dir(DataType.catalogs, CatalogType.degraded), "output_dereddener.pq",
            ),
        )

