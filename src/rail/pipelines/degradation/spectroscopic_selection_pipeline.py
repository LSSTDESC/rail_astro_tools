#!/usr/bin/env python
# coding: utf-8

# Prerquisites, os, and numpy
import os

import ceci
import numpy as np
from rail.core.stage import RailPipeline, RailStage
from rail.core.utils import RAILDIR
from rail.utils import catalog_utils
from rail.utils.catalog_utils import CatalogConfigBase

# Various rail modules


SELECTORS = dict(
    GAMA=dict(
        Select="SpecSelection_GAMA",
        Module="rail.creation.degraders.spectroscopic_selections",
    ),
    BOSS=dict(
        Select="SpecSelection_BOSS",
        Module="rail.creation.degraders.spectroscopic_selections",
    ),
    VVDSf02=dict(
        Select="SpecSelection_VVDSf02",
        Module="rail.creation.degraders.spectroscopic_selections",
    ),
    zCOSMOS=dict(
        Select="SpecSelection_zCOSMOS",
        Module="rail.creation.degraders.spectroscopic_selections",
    ),
    HSC=dict(
        Select="SpecSelection_HSC",
        Module="rail.creation.degraders.spectroscopic_selections",
    ),
)


CommonConfigParams = dict(
    N_tot=100_000,
    nondetect_val=-np.inf,
    downsample=False,
)


class SpectroscopicSelectionPipeline(RailPipeline):

    default_input_dict = dict(input="dummy.in")

    def __init__(self, selectors=None):
        RailPipeline.__init__(self)

        if selectors is None:
            selectors = SELECTORS.copy()

        config_pars = CommonConfigParams.copy()

        active_catalog_tag = catalog_utils.get_active_tag()

        colnames = active_catalog_tag.band_name_dict().copy()
        colnames["redshift"] = active_catalog_tag.config.redshift_col
        config_pars["colnames"] = colnames

        for key, val in selectors.items():
            the_class = ceci.PipelineStage.get_stage(val["Select"], val["Module"])
            the_selector = the_class.make_and_connect(
                name=f"select_{key}",
                **config_pars,
            )

            self.add_stage(the_selector)
