"""Assign skymap tracts to a reference catalog, then crossmatch it against a shear catalog.

Two stages, one environment.  The predecessor of this pipeline had to be run as two separate
shell scripts because tract assignment needed ``lsst.skymap`` from the cvmfs DM stack while
everything else needed the RAIL conda environment; ``AssignTract`` reimplements the tract
lookup in numpy, so that split is gone.

    from rail.core.stage import RailPipeline
    from rail.pipelines.degradation.crossmatch_pipeline import (
        CrossMatchPipeline, set_stage_threads,
    )

    RailPipeline.build_and_write(
        "CrossMatchPipeline",
        "crossmatch.yml",
        input_dict=dict(input="reference_catalog.pq"),
        output_dir="./data",
        log_dir="./logs",
    )
    set_stage_threads("crossmatch.yml", {"crossmatch": 32})
    # then: ceci crossmatch.yml

The shear catalog is **not** a pipeline input.  It is a path in the ``crossmatch`` stage's
config, because the catalogs this targets run to hundreds of gigabytes and must never be
loaded through a ``DataHandle``.  Set it through ``catalog_config``.

``set_stage_threads`` is imported from ``hscfy_pz_prepare`` rather than copied: it also raises
the site's ``max_threads``, which is the half that is easy to forget, and without which the
local runner refuses to schedule a stage asking for more threads than it believes it has.
Here the threads are the crossmatch's *worker processes* -- the stage reads the resulting
``OMP_NUM_THREADS`` as its core budget when ``nproc`` is left at 0.
"""

from rail.core.data import DataStore
from rail.core.stage import RailPipeline
from rail.creation.degraders.crossmatch import (
    DEFAULT_NUM_RINGS,
    DEFAULT_RA_START,
    DEFAULT_REF_COLS,
    DEFAULT_REF_RENAME,
    DEFAULT_SHEAR_COLS,
    DEFAULT_SHEAR_CUTS,
    DEFAULT_SHEAR_GROUP,
    AssignTract,
    TractCrossMatch,
    parse_cuts,
)
from rail.pipelines.degradation.hscfy_pz_prepare import set_stage_threads  # noqa: F401

# The crossmatch stage forks this many worker processes.  It is I/O bound on the per-tract
# HDF5 reads, so it scales with cores until the filesystem saturates.
DEFAULT_STAGE_THREADS = {"crossmatch": 32}


# A worked example, not a requirement: the TXPipe-ingested Rubin DP2 metadetect shear catalog
# matched against the DP2 reference-redshift compilation.  Every value is overridable.
DP2_METADETECT_CONFIG = dict(
    # --- tract assignment (reference catalog) --------------------------------------- #
    ref_ra_col="ra",
    ref_dec_col="dec",
    tract_col="tract",
    num_rings=DEFAULT_NUM_RINGS,
    ra_start=DEFAULT_RA_START,
    ref_cuts=[],
    # --- shear catalog ---------------------------------------------------------------- #
    shear_catalog="",
    shear_group=DEFAULT_SHEAR_GROUP,
    shear_cols=DEFAULT_SHEAR_COLS,
    shear_ra_col="ra",
    shear_dec_col="dec",
    shear_tract_col="tract",
    shear_cuts=DEFAULT_SHEAR_CUTS,
    # TXPipe catalogs written by an MPI stage repeat the rows that straddle a rank boundary;
    # the 0.1.0 DP2 metadetect file is 17.2% exact duplicates.  '' for a clean catalog.
    dedup_col="id",
    # --- matching ---------------------------------------------------------------------- #
    ref_cols=DEFAULT_REF_COLS,
    ref_rename=DEFAULT_REF_RENAME,
    max_sep_arcsec=0.75,
    edge_halo_arcsec=0.0,
    nproc=0,
    max_inflight=0,
    max_tracts=0,
)


class CrossMatchPipeline(RailPipeline):
    """Reference catalog in, matched reference-plus-shear catalog out."""

    default_input_dict = dict(
        input="dummy.pq",
    )

    def __init__(self, catalog_config: dict | None = None) -> None:
        RailPipeline.__init__(self)

        DataStore.allow_overwrite = True

        config = DP2_METADETECT_CONFIG.copy()
        if catalog_config:
            unknown = set(catalog_config) - set(config)
            if unknown:
                raise KeyError(
                    f"unknown catalog_config keys {sorted(unknown)}; "
                    f"valid keys are {sorted(config)}"
                )
            config.update(catalog_config)

        # Validate here as well as in the stages, so a bad cut spec is caught while the yaml
        # is being built rather than after ceci has launched the stage subprocess, where it
        # surfaces as an error that says nothing about cuts.  The cuts themselves go into the
        # yaml in whatever form they arrived: the readable string form stays readable.
        parse_cuts(config["shear_cuts"], "shear")
        parse_cuts(config["ref_cuts"], "reference")

        self.assign_tract = AssignTract.build(
            ra_col=config["ref_ra_col"],
            dec_col=config["ref_dec_col"],
            tract_col=config["tract_col"],
            num_rings=config["num_rings"],
            ra_start=config["ra_start"],
            cuts=config["ref_cuts"],
            drop_non_finite=True,
        )

        self.crossmatch = TractCrossMatch.build(
            connections=dict(input=self.assign_tract.io.output),
            shear_catalog=config["shear_catalog"],
            shear_group=config["shear_group"],
            shear_cols=config["shear_cols"],
            shear_ra_col=config["shear_ra_col"],
            shear_dec_col=config["shear_dec_col"],
            shear_tract_col=config["shear_tract_col"],
            shear_cuts=config["shear_cuts"],
            dedup_col=config["dedup_col"],
            ref_cols=config["ref_cols"],
            ref_ra_col=config["ref_ra_col"],
            ref_dec_col=config["ref_dec_col"],
            ref_tract_col=config["tract_col"],
            # already applied by AssignTract; applying them twice would be harmless but
            # would also double the log noise and invite the two lists to drift apart
            ref_cuts=[],
            ref_rename=config["ref_rename"],
            max_sep_arcsec=config["max_sep_arcsec"],
            edge_halo_arcsec=config["edge_halo_arcsec"],
            nproc=config["nproc"],
            max_inflight=config["max_inflight"],
            max_tracts=config["max_tracts"],
        )
