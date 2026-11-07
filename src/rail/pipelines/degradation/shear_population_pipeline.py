"""Draw the population sample that ``HscfyPzPreparePipeline``'s SOM stage matches the test set to.

One stage, no handle inputs: the shear catalog is a *path* in the stage config, for the same
reason as in ``CrossMatchPipeline`` -- hundreds of gigabytes cannot pass through a DataHandle.

    from rail.core.stage import RailPipeline
    from rail.pipelines.degradation.shear_population_pipeline import set_stage_threads

    RailPipeline.build_and_write(
        "ShearPopulationPipeline",
        "population.yml",
        output_dir="./data",
        log_dir="./logs",
        catalog_config=dict(shear_catalog="shear_catalog.hdf5", n_sample=1_000_000),
    )
    set_stage_threads("population.yml", {"population": 32})
    # then: ceci population.yml

The output is ``output_population.pq`` in ``output_dir``, which is what
``HscfyPzPreparePipeline`` takes as its ``population`` input.
"""

from rail.core.data import DataStore
from rail.core.stage import RailPipeline
from rail.creation.degraders.crossmatch import DEFAULT_SHEAR_GROUP, parse_cuts
from rail.creation.degraders.shear_population import (
    DEFAULT_CHUNK_ROWS,
    DEFAULT_POPULATION_COLS,
    ShearPopulationSampler,
    parse_ratio_cuts,
)
from rail.pipelines.degradation.hscfy_pz_prepare import set_stage_threads  # noqa: F401

DEFAULT_STAGE_THREADS = {"population": 32}

# A worked example, not a requirement: the TXPipe DP2 Rubin metadetect catalog with the DESC WL
# fiducial metadetect selection -- every listed flag zero (g_pgaussFlux_flags deliberately not
# among them), is_primary, mfrac < 0.1, the "typical" quality cuts (arXiv:2501.05665 sec. 4.4):
# gauss_snr > 10, gauss_T / gauss_psfReconvolved_T > 0.5, |g-r|, |r-i|, |i-z| < 5, gauss_T < 20.
# metaStep == 'ns' is the shear/ns group.  Every value is overridable.
METADETECT_FIDUCIAL_FLAGS = [
    "bmask_flags",
    "gauss_flags",
    "gauss_object_flags",
    "gauss_psfReconvolved_flags",
    "gauss_shape_flags",
    "i_gaussFlux_flags",
    "i_pgaussFlux_flags",
    "image_flags",
    "pgauss_flags",
    "pgauss_object_flags",
    "pgauss_shape_flags",
    "psfOriginal_flags",
    "r_gaussFlux_flags",
    "r_pgaussFlux_flags",
    "z_gaussFlux_flags",
    "z_pgaussFlux_flags",
]

DP2_METADETECT_POPULATION_CONFIG = dict(
    shear_catalog="",
    shear_group=DEFAULT_SHEAR_GROUP,
    shear_cols=DEFAULT_POPULATION_COLS,
    cuts=(
        [f"{flag} == 0" for flag in METADETECT_FIDUCIAL_FLAGS]
        + ["is_primary == True", "mfrac < 0.1", "gauss_snr > 10", "gauss_T < 20"]
    ),
    ratio_cuts=["gauss_T / gauss_psfReconvolved_T > 0.5"],
    require_finite=[],
    color_cols=["mag_g", "mag_r", "mag_i", "mag_z"],
    abs_color_max=5.0,
    # g_pgaussFlux_flags is left out on purpose, so a g non-detection (NaN g) must not be
    # dropped through the back door of |g-r| < 5
    color_nan_passes=True,
    dedup_col="id",
    n_sample=1_000_000,
    seed=42,
    chunk_rows=DEFAULT_CHUNK_ROWS,
    nproc=0,
    nonfinite_to_nan=True,
    max_rows=0,
)


class ShearPopulationPipeline(RailPipeline):
    """Shear HDF5 in (by path), selection-cut random sample out."""

    default_input_dict: dict = {}

    def __init__(self, catalog_config: dict | None = None) -> None:
        RailPipeline.__init__(self)

        DataStore.allow_overwrite = True

        config = DP2_METADETECT_POPULATION_CONFIG.copy()
        if catalog_config:
            unknown = set(catalog_config) - set(config)
            if unknown:
                raise KeyError(
                    f"unknown catalog_config keys {sorted(unknown)}; "
                    f"valid keys are {sorted(config)}"
                )
            config.update(catalog_config)

        # caught while the yaml is built, not after ceci has launched the subprocess
        parse_cuts(config["cuts"], "shear")
        parse_ratio_cuts(config["ratio_cuts"])

        self.population = ShearPopulationSampler.build(**config)
