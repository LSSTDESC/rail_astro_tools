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
        |                 |      (SOMZCellResampler with zcell=True <-- also a truth sample)
    train.hdf5        test.hdf5

The split happens first so the two branches can never share a galaxy.  The training branch
flattens the magnitude--redshift distribution; the testing branch reshapes the pool to look
like the shear catalog it will be used to evaluate.

``zcell=True`` swaps the test stage for ``SOMZCellResampler``, which also splits every SOM
cell's target over wide redshift bins by a redshift-complete ``truth`` sample (e.g. deep-field
shear objects matched to a deep photo-z catalog), and adds ``truth`` to the pipeline inputs.

``deredden`` (``{input tag: {magnitude column: A_b / E(B-V)}}``) puts a ``Dereddener`` in
front of each named input -- ``input``, ``population``, ``truth`` -- so the split, the SOM and
the train/test files all see dust-corrected magnitudes (``dustmap_name`` map, ``dustmap_dir``).
``nondetect_val`` is left as it is, and the E(B-V) used is written to ``ebv_col``.

``require_detected`` (magnitude columns) drops, before the split and so from both sides, every
row without a detection in one of them (non-finite or ``nondetect_val``).

``HoldoutTrainPipeline`` makes a training set only: the same dereddening, cuts and caps, but
instead of a random split every row sharing a ``holdout_cols`` value with the ``holdout`` input
-- another run's ``output_pretest_split.pq`` -- is held out.  That is how training sets from
different reference compilations are scored on one test set.

Build and run:

    from rail.core.stage import RailPipeline
    RailPipeline.build_and_write(
        "HscfyPzPreparePipeline",
        "hscfy_pz_prepare.yml",
        input_dict=dict(input=..., population=...),  # + truth=... with zcell
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
    DEFAULT_Z_EDGES,
    DEFAULT_Z_SHRINK,
    ReferenceHoldoutSplitter,
    MagRedshiftDownsampler,
    PreTrainTestSplitter,
    SOMResampler,
    SOMZCellResampler,
)
from rail.tools.photometry_tools import Dereddener

#: Stages worth more than one thread, and how many.  ``SOMResampler`` spends essentially all
#: its time inside somoclu's OpenMP training loop: measured on the DP2 anacal pre-test pool
#: (165,775 objects, 32x32 SOM, 100 epochs) training takes 430 s on one thread and 14 s on 64,
#: while BMU assignment is 0.8 s either way.  ceci gives every stage
#: ``threads_per_process = 1`` unless the pipeline yaml says otherwise, which is what made the
#: (then six-pass) SOM stage take 42 minutes.
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
    # magnitude columns that must hold a detection (finite, not nondetect_val), both sides
    require_detected=[],
    # training downsample
    d_mag=0.1,
    d_z=0.1,
    downsample_seed=42,
    # test SOM resample
    som_size=[32, 32],
    n_epochs=100,
    # one SOM; every cell gets n_target x its population fraction, filled from the pool
    n_target=100_000,
    som_train_on="population",
    som_init="pca",
    # on both the pool and the population, before the SOM (no bright limit)
    mag_range=[float("-inf"), 25.0],
    som_seed=42,
    som_init_seed=42,  # only used with som_init="random"
    som_radius_final=1.0,  # somoclu's radiusN
    # zcell test set: split each cell's target over redshift bins by the `truth` input
    zcell=False,
    z_edges=DEFAULT_Z_EDGES,
    z_shrink=DEFAULT_Z_SHRINK,
    truth_redshift_col="z",
    # dereddening before everything else: {input tag: {magnitude column: A_b / E(B-V)}};
    # empty = the magnitudes as given
    deredden={},
    dustmap_name="sfd",
    dustmap_dir="",
    ebv_col="ebv_sfd",
)

#: The pipeline inputs a ``Dereddener`` can be put in front of.
DEREDDEN_TAGS = ("input", "population", "truth")


def _merged_config(catalog_config: dict | None, extra: dict | None = None) -> dict:
    """DP2_ANACAL_CONFIG (+ ``extra`` defaults) updated by ``catalog_config``; unknown keys
    raise."""
    config = {**DP2_ANACAL_CONFIG, **(extra or {})}
    if catalog_config:
        unknown = set(catalog_config) - set(config)
        if unknown:
            raise KeyError(
                f"unknown catalog_config keys {sorted(unknown)}; valid keys are {sorted(config)}"
            )
        config.update(catalog_config)
    return config


def _add_dereddeners(pipe: RailPipeline, config: dict, tags: tuple) -> dict:
    """A ``Dereddener`` named ``deredden_<tag>`` in front of each input of ``config["deredden"]``
    (only ``tags`` allowed); returns {tag: its output} for the stages that read them."""
    unknown = set(config["deredden"]) - set(tags)
    if unknown:
        raise KeyError(f"deredden: unknown inputs {sorted(unknown)}; valid are {tags}")
    if config["deredden"] and not config["dustmap_dir"]:
        raise ValueError("deredden needs dustmap_dir")
    dereddened = {}
    for tag in tags:
        if tag not in config["deredden"]:
            continue
        # the attribute name below is the stage name, deredden_<tag>
        stage = Dereddener.build(
            **({} if tag == "input" else dict(aliases=dict(input=tag))),
            dustmap_name=config["dustmap_name"],
            dustmap_dir=config["dustmap_dir"],
            band_a_env=dict(config["deredden"][tag]),
            nondetect_val=config["nondetect_val"],
            ebv_name=config["ebv_col"],
            copy_all_cols=True,
        )
        setattr(pipe, f"deredden_{tag}", stage)
        dereddened[tag] = stage.io.output
    return dereddened


def _split_options(config: dict) -> dict:
    """PreTrainTestSplitter's cuts (shared by ReferenceHoldoutSplitter)."""
    return dict(
        redshift_col=config["redshift_col"],
        output_redshift_col=config["output_redshift_col"],
        redshift_min=config["redshift_min"],
        redshift_max=config["redshift_max"],
        apply_quality_cut=config["apply_quality_cut"],
        quality_col=config["quality_col"],
        quality_min=config["quality_min"],
        quality_nan_policy=config["quality_nan_policy"],
        require_detected=list(config["require_detected"]),
        nondetect_val=config["nondetect_val"],
    )


def _downsampler(config: dict, pretrain) -> object:
    return MagRedshiftDownsampler.build(
        connections=dict(input=pretrain),
        survey_col=config["survey_col"],
        mag_col=config["mag_col"],
        redshift_col=config["output_redshift_col"],
        caps=config["caps"],
        d_mag=config["d_mag"],
        d_z=config["d_z"],
        random_seed=config["downsample_seed"],
    )


class HscfyPzPreparePipeline(RailPipeline):
    """Split, downsample and SOM-resample a reference catalog into train/test sets."""

    default_input_dict = dict(
        input="dummy.pq",
        population="dummy.pq",
        # read only with zcell=True; listed so a zcell pipeline builds without it
        truth="dummy.pq",
    )

    def __init__(self, catalog_config: dict | None = None) -> None:
        RailPipeline.__init__(self)

        DataStore.allow_overwrite = True

        config = _merged_config(catalog_config)
        if "truth" in config["deredden"] and not config["zcell"]:
            raise ValueError("deredden: the truth input is only read with zcell=True")
        dereddened = _add_dereddeners(self, config, DEREDDEN_TAGS)

        self.split = PreTrainTestSplitter.build(
            **({"connections": dict(input=dereddened["input"])} if "input" in dereddened else {}),
            train_frac=config["train_frac"],
            shuffle=config["shuffle"],
            seed=config["split_seed"],
            group_col=config["group_col"],
            **_split_options(config),
        )

        self.downsample_train = _downsampler(config, self.split.io.output_pretrain)

        zcell_options = (
            dict(
                z_edges=config["z_edges"],
                z_shrink=config["z_shrink"],
                truth_redshift_col=config["truth_redshift_col"],
                redshift_col=config["output_redshift_col"],
            )
            if config["zcell"]
            else {}
        )
        resampler = SOMZCellResampler if config["zcell"] else SOMResampler
        self.som_resample_test = resampler.build(
            connections=dict(
                input=self.split.io.output_pretest,
                **{tag: dereddened[tag] for tag in ("population", "truth") if tag in dereddened},
            ),
            **zcell_options,
            som_size=config["som_size"],
            n_epochs=config["n_epochs"],
            n_target=config["n_target"],
            som_train_on=config["som_train_on"],
            som_init=config["som_init"],
            init_seed=config["som_init_seed"],
            som_radius_final=config["som_radius_final"],
            noncolor_cols=config["noncolor_cols"],
            color_cols=config["color_cols"],
            nondetect_val=config["nondetect_val"],
            noncolor_nondet=config["noncolor_nondet"],
            color_nondet=config["color_nondet"],
            mag_col=config["mag_col"],
            mag_range=config["mag_range"],
            seed=config["som_seed"],
        )


class HoldoutTrainPipeline(RailPipeline):
    """A training set only: (dereddening,) the split's cuts, then every row sharing a
    ``holdout_cols`` value with the ``holdout`` table held out, then the training caps.

    With ``holdout`` = another HscfyPzPreparePipeline run's ``output_pretest_split.pq`` this
    trains on a different reference compilation without touching that run's test pool, so
    training sets from several compilations are compared on one test set.  The catalog_config
    is HscfyPzPreparePipeline's (its SOM keys are ignored) + ``holdout_cols``; only the
    ``input`` can be dereddened.
    """

    default_input_dict = dict(input="dummy.pq", holdout="dummy.pq")

    def __init__(self, catalog_config: dict | None = None) -> None:
        RailPipeline.__init__(self)

        DataStore.allow_overwrite = True

        config = _merged_config(catalog_config, extra=dict(holdout_cols=["ref_id"]))
        dereddened = _add_dereddeners(self, config, ("input",))

        self.split = ReferenceHoldoutSplitter.build(
            **({"connections": dict(input=dereddened["input"])} if "input" in dereddened else {}),
            holdout_cols=list(config["holdout_cols"]),
            **_split_options(config),
        )
        self.downsample_train = _downsampler(config, self.split.io.output_pretrain)
