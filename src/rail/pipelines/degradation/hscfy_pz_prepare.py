#!/usr/bin/env python
# coding: utf-8
"""Pipeline that turns a cross-matched reference-redshift catalog into a photo-z training set
and a representative test set.

This is the HSC-Y3 ("hscfy") preparation chain:

    reference cross-match
             |
       PreTrainTestSplitter          80/20, seeded, redshift + optional quality cut
          /            \\
    pre-train         pre-test
        |                 |
  MagRedshiftDownsampler  SOMResampler <-- population sample from the shear catalog
        |                 |
    train.hdf5        test.hdf5

The split happens first so the two branches can never share a galaxy.  The training branch
flattens the magnitude--redshift distribution; the testing branch reshapes the pool to look
like the shear catalog it will be used to evaluate.

Build and run:

    from rail.core.stage import RailPipeline
    RailPipeline.build_and_write(
        "HscfyPzPreparePipeline",
        "hscfy_pz_prepare.yml",
        input_dict=dict(input=..., population=...),
        output_dir=...,
        log_dir=...,
    )

then ``ceci hscfy_pz_prepare.yml``.

Call :func:`set_stage_threads` on the written yaml before running it.  Without it ceci gives
every stage a single thread and the SOM stage takes 42 minutes instead of about 90 seconds.

The defaults target the Rubin DP2 anacal shear catalog (riz photometry named
``lsst_{r,i,z}_mag_gauss2``, survey label ``z_source``, redshift ``z``).  Pass ``catalog_config``
to retarget it at another survey without touching this file.
"""

import yaml

from rail.core.data import DataStore
from rail.core.stage import RailPipeline
from rail.creation.degraders.pz_prepare import (
    DEFAULT_CAPS,
    DEFAULT_COLOR_COLS,
    DEFAULT_COLOR_NONDET,
    DEFAULT_MAG_COL,
    DEFAULT_NONCOLOR_NONDET,
    DEFAULT_NONDETECT_VAL,
    DEFAULT_SURVEY_COL,
    MagRedshiftDownsampler,
    PreTrainTestSplitter,
    SOMResampler,
)

#: Stages worth more than one thread, and how many.  ``SOMResampler`` spends essentially all
#: its time inside somoclu's OpenMP training loop: measured on the DP2 anacal pre-test pool
#: (165,775 objects, 32x32 SOM, 100 epochs) training takes 430 s on one thread and 14 s on 64,
#: while BMU assignment is 0.8 s either way.  ceci gives every stage
#: ``threads_per_process = 1`` unless the pipeline yaml says otherwise, which is what made six
#: passes take 42 minutes.
#:
#: This cannot be fixed from inside the stage.  The OpenMP runtime reads OMP_NUM_THREADS when
#: the library loads, so setting os.environ in ``run()`` is too late -- measured, no effect --
#: and somoclu exposes no thread argument.  It has to be in the environment ceci builds for the
#: subprocess, which is exactly what ``threads_per_process`` controls.
DEFAULT_STAGE_THREADS = {"som_resample_test": 64}


def set_stage_threads(
    pipeline_yaml: str,
    stage_threads: dict | None = None,
    site_max_threads: int | None = None,
) -> None:
    """Grant threads to individual stages in an already-written pipeline yaml.

    ``RailPipeline.build_and_write`` offers no way to set ceci's per-stage
    ``threads_per_process``, so it is patched in afterwards.  The site's ``max_threads`` is
    raised to match, because the local runner will not schedule a stage that asks for more
    threads than the node it believes it has.
    """
    stage_threads = DEFAULT_STAGE_THREADS if stage_threads is None else stage_threads
    if site_max_threads is None:
        site_max_threads = max([1, *stage_threads.values()])

    with open(pipeline_yaml) as fh:
        config = yaml.safe_load(fh)

    for stage in config.get("stages", []):
        threads = stage_threads.get(stage["name"])
        if threads:
            stage["threads_per_process"] = int(threads)

    site = config.setdefault("site", {})
    site["max_threads"] = max(int(site.get("max_threads", 1)), int(site_max_threads))

    with open(pipeline_yaml, "w") as fh:
        yaml.dump(config, fh, default_flow_style=False, sort_keys=False)


#: Everything survey-specific, in one place.  Override any subset via ``catalog_config``.
DP2_ANACAL_CONFIG = dict(
    redshift_col="z",
    output_redshift_col="redshift",
    survey_col=DEFAULT_SURVEY_COL,
    mag_col=DEFAULT_MAG_COL,
    color_cols=DEFAULT_COLOR_COLS,
    noncolor_cols=[DEFAULT_MAG_COL],
    nondetect_val=DEFAULT_NONDETECT_VAL,
    noncolor_nondet=DEFAULT_NONCOLOR_NONDET,
    color_nondet=DEFAULT_COLOR_NONDET,
    caps=DEFAULT_CAPS,
    # split
    train_frac=0.8,
    shuffle=True,
    split_seed=1337,
    redshift_min=0.001,
    redshift_max=7.0,
    apply_quality_cut=False,
    quality_col="z_flag",
    quality_min=3.0,
    quality_nan_policy="keep",
    # A reference redshift can be the nearest neighbour of two blended anacal detections, so
    # grouping on ref_id is what keeps the same truth value out of both train and test.
    group_col="ref_id",
    # training downsample
    d_mag=0.1,
    d_z=0.1,
    downsample_seed=42,
    # test SOM resample
    som_size=[32, 32],
    n_epochs=100,
    n_repeat=6,
    n_population=30000,
    mag_cut_pool=25.5,
    mag_range_out=[18.5, 25.0],
    som_seed=42,
)


class HscfyPzPreparePipeline(RailPipeline):
    """Split, downsample and SOM-resample a reference catalog into train/test sets."""

    default_input_dict = dict(
        input="dummy.pq",
        population="dummy.pq",
    )

    def __init__(self, catalog_config: dict | None = None) -> None:
        RailPipeline.__init__(self)

        DataStore.allow_overwrite = True

        config = DP2_ANACAL_CONFIG.copy()
        if catalog_config:
            unknown = set(catalog_config) - set(config)
            if unknown:
                raise KeyError(
                    f"unknown catalog_config keys {sorted(unknown)}; "
                    f"valid keys are {sorted(config)}"
                )
            config.update(catalog_config)

        self.split = PreTrainTestSplitter.build(
            train_frac=config["train_frac"],
            shuffle=config["shuffle"],
            seed=config["split_seed"],
            redshift_col=config["redshift_col"],
            output_redshift_col=config["output_redshift_col"],
            redshift_min=config["redshift_min"],
            redshift_max=config["redshift_max"],
            apply_quality_cut=config["apply_quality_cut"],
            quality_col=config["quality_col"],
            quality_min=config["quality_min"],
            quality_nan_policy=config["quality_nan_policy"],
            group_col=config["group_col"],
        )

        self.downsample_train = MagRedshiftDownsampler.build(
            connections=dict(input=self.split.io.output_pretrain),
            survey_col=config["survey_col"],
            mag_col=config["mag_col"],
            redshift_col=config["output_redshift_col"],
            caps=config["caps"],
            d_mag=config["d_mag"],
            d_z=config["d_z"],
            random_seed=config["downsample_seed"],
        )

        self.som_resample_test = SOMResampler.build(
            connections=dict(input=self.split.io.output_pretest),
            som_size=config["som_size"],
            n_epochs=config["n_epochs"],
            n_repeat=config["n_repeat"],
            n_population=config["n_population"],
            noncolor_cols=config["noncolor_cols"],
            color_cols=config["color_cols"],
            nondetect_val=config["nondetect_val"],
            noncolor_nondet=config["noncolor_nondet"],
            color_nondet=config["color_nondet"],
            mag_col=config["mag_col"],
            mag_cut_pool=config["mag_cut_pool"],
            mag_range_out=config["mag_range_out"],
            seed=config["som_seed"],
        )
