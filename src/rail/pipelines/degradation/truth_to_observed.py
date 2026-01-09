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
from rail.core.common_params import SHARED_PARAMS
from rail.utils import catalog_utils
from rail.creation.degraders.unrec_bl_model import UnrecBlModel

from .spectroscopic_selection_pipeline import SELECTORS, CommonConfigParams
from .apply_phot_errors import ERROR_MODELS


if 'PZ_DUSTMAP_DIR' not in os.environ:  # pragma: no cover
    os.environ['PZ_DUSTMAP_DIR'] = '.'

dustmap_dir = os.path.expandvars("${PZ_DUSTMAP_DIR}")


class TruthToObservedPipeline(RailPipeline):

    default_input_dict = dict(input='dummy.in')

    def __init__(
        self,
        error_models: dict|None=None,
        selectors: dict|None=None,
        models_to_run_select: list[str]|None=None,
        *,
        blending: bool=False,
        parallel: bool=False,
    ):
        RailPipeline.__init__(self)

        DS = RailStage.data_store
        DS.__class__.allow_overwrite = True

        active_catalog_config = catalog_utils.get_active_tag()
        full_rename_dict = active_catalog_config.band_name_dict().copy()
        full_a_env_dict = SHARED_PARAMS.band_a_env.copy()

        if error_models is None:
            error_models = ERROR_MODELS.copy()

        if selectors is None:
            selectors = SELECTORS.copy()

        if models_to_run_select is None:
            models_to_run_select = []

        config_pars = CommonConfigParams.copy()
        config_pars['colnames'] = full_rename_dict.copy()
        config_pars['colnames']['redshift'] = active_catalog_config.config['redshift_col']

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
            if 'Bands' in val:
                rename_dict = {band_: full_rename_dict[band_] for band_ in val['Bands']}
                a_env_dict: dict[str, float] = {}
                for band_ in val['Bands']:
                    if band_ in full_a_env_dict:
                        a_env_dict[band_] = full_a_env_dict[band_]
                    else:
                        renamed_band = rename_dict[band_]
                        a_env_dict[renamed_band] = full_a_env_dict[renamed_band]
            else:  # pragma: no cover
                rename_dict = full_rename_dict
                a_env_dict = full_a_env_dict
            overrides = val.get('Overrides', {})
            the_error_model = error_model_class.make_and_connect(
                name=f'error_model_{key}',
                connections=dict(input=previous_stage.io.output),
                renameDict=rename_dict,
                **overrides,
            )
            self.add_stage(the_error_model)
            spec_selection_dict = rename_dict.copy()
            spec_selection_dict.update(redshift=active_catalog_config.config['redshift_col'])
            if parallel:
                the_dereddener = Dereddener.make_and_connect(
                    name=f'deredden_{key}',
                    dustmap_dir=dustmap_dir,
                    connections=dict(input=the_error_model.io.output),
                    band_a_env=a_env_dict,
                    copy_all_cols=True,
                )
                self.add_stage(the_dereddener)
                if key in models_to_run_select:
                    self._add_selectors(
                        the_dereddener,
                        key,
                        selectors,
                        config_pars,
                    )
                previous_stage = self.reddener
            else:
                previous_stage = the_error_model

        if not parallel:
            self.dereddener_errors = Dereddener.build(
                dustmap_dir=dustmap_dir,
                connections=dict(input=previous_stage.io.output),
                copy_all_cols=True,
            )
            self._add_selectors(
                self.dereddener_errors,
                key,
                selectors,
                config_pars,
            )


    def _add_selectors(
        self,
        previous_stage,
        key: str,
        selectors: dict,
        config_pars: dict,
    ) -> None:


        for keyS, valS in selectors.items():
            the_class = ceci.PipelineStage.get_stage(valS['Select'], valS['Module'])
            the_selector = the_class.make_and_connect(
                name=f'select_{key}_{keyS}',
                connections=dict(input=previous_stage.io.output),
                **config_pars,
            )
            self.add_stage(the_selector)
