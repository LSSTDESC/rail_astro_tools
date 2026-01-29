#!/usr/bin/env python
# coding: utf-8

# Prerquisites, os, and numpy
import os

import ceci
import numpy as np
from rail.core.stage import RailPipeline, RailStage
from rail.core.utils import RAILDIR

# Various rail modules
from rail.creation.degraders.unrec_bl_model import UnrecBlModel


class BlendingPipeline(RailPipeline):

    default_input_dict = dict(input="dummy.in")

    def __init__(self):
        RailPipeline.__init__(self)

        self.unrec_bl = UnrecBlModel.build()
