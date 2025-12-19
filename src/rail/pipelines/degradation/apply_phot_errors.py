#!/usr/bin/env python
# coding: utf-8

# Prerquisites, os, and numpy
import os

import ceci
import numpy as np
from rail.core.stage import RailPipeline, RailStage
from rail.core.utils import RAILDIR
from rail.utils import catalog_utils

# Various rail modules
from rail.tools.photometry_tools import Dereddener, Reddener

if "PZ_DUSTMAP_DIR" not in os.environ:
    os.environ["PZ_DUSTMAP_DIR"] = "."

dustmap_dir = os.path.expandvars("${PZ_DUSTMAP_DIR}")


ERROR_MODELS = dict(
    lsst=dict(
        ErrorModel="LSSTErrorModel",
        Module="rail.creation.degraders.photometric_errors",
        Bands=["u", "g", "r", "i", "z", "y"],
        Overrides=dict(
            minorCol="minor",
            majorCol="major",
            extendedType="gaap",
            hdf5_groupname="",
        ),
    ),
    # roman = dict(
    #    ErrorModel='RomanErrorModel',
    #    Module='rail.creation.degraders.photometric_errors',
    #    Bands=['Y', 'J', 'H', 'F'],
    # ),
    # euclid = dict(
    #    ErrorModel='EuclidErrorModel',
    #    Module='rail.creation.degraders.photometric_errors',
    # ),
)


class ApplyPhotErrorsPipeline(RailPipeline):

    default_input_dict = dict(input="dummy.in")

    def __init__(
        self,
        error_models: dict | None = None,
        *,
        parallel: bool = False,
    ):
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
        full_rename_dict = catalog_utils.get_active_tag().band_name_dict()
        for key, val in error_models.items():
            error_model_class = ceci.PipelineStage.get_stage(
                val["ErrorModel"], val["Module"]
            )
            if "Bands" in val:
                rename_dict = {band_: full_rename_dict[band_] for band_ in val["Bands"]}
            else:  # pragma: no cover
                rename_dict = full_rename_dict
            overrides = val.get("Overrides", {})
            the_error_model = error_model_class.make_and_connect(
                name=f"error_model_{key}",
                connections=dict(input=previous_stage.io.output),
                renameDict=rename_dict,
                **overrides,
            )
            self.add_stage(the_error_model)
            if parallel:
                the_dereddener = Dereddener.make_and_connect(
                    name=f"deredden_{key}",
                    dustmap_dir=dustmap_dir,
                    connections=dict(input=the_error_model.io.output),
                    copy_all_cols=True,
                )
                self.add_stage(the_dereddener)
                previous_stage = self.reddener
            else:
                previous_stage = the_error_model

        if not parallel:
            self.dereddener_errors = Dereddener.build(
                dustmap_dir=dustmap_dir,
                connections=dict(input=previous_stage.io.output),
                copy_all_cols=True,
            )
