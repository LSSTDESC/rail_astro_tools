#!/usr/bin/env python
# coding: utf-8

# Prerquisites, os, and numpy
import os
import numpy as np

# Various rail modules
from rail.creation.degraders.unrec_bl_model import UnrecBlModel

from rail.core.stage import RailStage, RailPipeline

import ceci

from rail.core.utils import RAILDIR


class BlendingPipeline(RailPipeline):

    default_input_dict = dict(input='dummy.in')

    def __init__(self):
        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        self.unrec_bl = UnrecBlModel.build()
