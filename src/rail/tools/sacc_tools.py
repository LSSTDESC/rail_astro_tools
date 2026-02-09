"""
Module for interfacing between qp and sacc file formats for n(z) data.

This module implements RailStage classes to convert between qp Ensemble files
and sacc.Sacc files, enabling interoperability between different redshift
distribution storage formats.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import qp
import sacc
from ceci.config import StageParameter as Param

from rail.core.data import Hdf5Handle
from rail.core.stage import RailStage


def normalize_hist(pdfs: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Normalize histogram PDFs to ensure they integrate to 1.

    Parameters
    ----------
    pdfs : np.ndarray
        Array of PDF values. Can be 1D or 2D (multiple PDFs).
    edges : np.ndarray
        Bin edges for the histogram.

    Returns
    -------
    np.ndarray
        Normalized PDF array with same shape as input.
    """
    pdfs_arr = np.array(pdfs, dtype=float, copy=True, ndmin=2)
    widths = np.diff(edges)
    total_width = widths.sum()
    if total_width <= 0:
        raise ValueError("Invalid bin edges provided for normalization.")
    norms = pdfs_arr @ widths
    mask = norms <= 0
    if np.any(mask):
        pdfs_arr[mask] = 1.0 / total_width
        norms = pdfs_arr @ widths
    pdfs_arr /= norms[:, None]
    if np.ndim(pdfs) == 1:
        return pdfs_arr[0]
    return pdfs_arr


def extract_tomographic_bins_from_sacc(
    sacc_catalog: sacc.Sacc,
) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    """Extract tomographic bins data structure from a sacc catalog.

    This utility function extracts a standard tomographic bins format from
    a sacc catalog containing QPNZ tracers. The output format matches the
    structure used in gold_baseline_tutorial and PriorSacc workflows.

    Parameters
    ----------
    sacc_catalog : sacc.Sacc
        SACC catalog containing QPNZ tracers.

    Returns
    -------
    tomographic_bins : List[Dict[str, np.ndarray]]
        List of dictionaries, one per tracer, containing:
        - "label": tracer name
        - "z": redshift centers (1D array)
        - "edges": redshift bin edges (1D array)
        - "truth": truth n(z) distribution (1D array)
        - "estimates": ensemble PDFs (2D array: [n_samples, n_bins])
    edges_lookup : Dict[str, np.ndarray]
        Dictionary mapping tracer names to their bin edges.

    Examples
    --------
    >>> from rail.tools.sacc_tools import extract_tomographic_bins_from_sacc
    >>> tomos, edges = extract_tomographic_bins_from_sacc(sacc_catalog)
    >>> for tomo in tomos:
    ...     print(f"{tomo['label']}: {tomo['estimates'].shape[0]} samples")
    """
    tomos = []
    edges_lookup = {}

    for tracer_name, tracer in sacc_catalog.tracers.items():
        if tracer.tracer_type != "QPNZ":
            continue

        z = np.array(tracer.z, dtype=float)
        # Get edges from ensemble
        if hasattr(tracer, "ensemble") and tracer.ensemble is not None:
            ensemble = tracer.ensemble
            if hasattr(ensemble, "objdata") and "bins" in ensemble.objdata:
                edges = np.array(ensemble.objdata["bins"], dtype=float).flatten()
            elif hasattr(ensemble, "metadata") and "bins" in ensemble.metadata:
                edges = np.array(ensemble.metadata["bins"], dtype=float).flatten()
            else:
                # Estimate edges from z centers
                dz = np.diff(z)
                if len(dz) > 0:
                    edges = np.concatenate(
                        [[z[0] - dz[0] / 2], z[:-1] + dz / 2, [z[-1] + dz[-1] / 2]]
                    )
                else:
                    edges = np.array([z[0] - 0.01, z[0] + 0.01])

            # Get PDFs
            if hasattr(ensemble, "objdata") and "pdfs" in ensemble.objdata:
                pdfs = np.array(ensemble.objdata["pdfs"], dtype=float)
                if pdfs.ndim == 1:
                    pdfs = pdfs.reshape(1, -1)
            else:
                # Fallback: create single PDF from nz
                pdfs = (
                    np.array([tracer.nz], dtype=float)
                    if hasattr(tracer, "nz") and tracer.nz is not None
                    else np.ones((1, len(z)))
                )
        else:
            # Fallback if no ensemble
            edges = np.array([z[0] - 0.01, z[-1] + 0.01])
            pdfs = (
                np.array([tracer.nz], dtype=float)
                if hasattr(tracer, "nz") and tracer.nz is not None
                else np.ones((1, len(z)))
            )

        # Get truth n(z)
        truth = (
            np.array(tracer.nz, dtype=float)
            if hasattr(tracer, "nz") and tracer.nz is not None
            else pdfs.mean(axis=0)
        )

        tomos.append(
            {
                "label": tracer_name,
                "z": z,
                "edges": edges,
                "truth": truth,
                "estimates": pdfs,
            }
        )
        edges_lookup[tracer_name] = edges

    return tomos, edges_lookup


class QPToSACC(RailStage):
    """RailStage to convert n(z) from qp Ensemble format to sacc format.

    This class loads n(z) distributions stored in qp Ensemble files (typically
    one file per tomographic bin) and creates a sacc.Sacc catalog with QPNZ
    tracers. The truth n(z) can optionally be included if provided.

    Attributes
    ----------
    inputs : list of tuples
        List of input data handles. Expects 'qp_input' as Hdf5Handle.
    outputs : list of tuples
        List of output data handles. Produces 'sacc_output' as Hdf5Handle.
    """

    name = "QPToSACC"
    config_options = RailStage.config_options.copy()
    config_options.update(
        tracer_names=Param(
            list,
            default=None,
            required=False,
            msg="List of tracer names corresponding to qp files. "
            "If None, will be inferred from filenames or use bin indices.",
        ),
        truth_files=Param(
            list,
            default=None,
            required=False,
            msg="Optional list of truth qp file paths (same order as input_files).",
        ),
    )

    inputs = [("qp_input", Hdf5Handle)]
    outputs = [("sacc_output", Hdf5Handle)]

    def __init__(self, args, **kwargs):
        """Initialize the QPToSACC stage.

        Parameters
        ----------
        args : dict
            Arguments for RailStage initialization.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(args, **kwargs)

    def run(self):
        """Run the conversion from qp to sacc format.

        Reads qp ensemble files, creates QPNZ tracers in a sacc catalog,
        and saves the result.
        """
        # Get input qp files - can be a single file or list of files
        qp_input_data = self.get_data("qp_input", allow_missing=True)

        # Handle different input formats
        if isinstance(qp_input_data, (list, tuple)):
            qp_files = qp_input_data
        elif isinstance(qp_input_data, str):
            qp_files = [qp_input_data]
        elif isinstance(qp_input_data, dict):
            # If dict, assume it's {tracer_name: file_path}
            if self.config.tracer_names is None:
                self.config.tracer_names = list(qp_input_data.keys())
            qp_files = list(qp_input_data.values())
        else:
            raise ValueError(f"Unsupported input format: {type(qp_input_data)}")

        # Determine tracer names
        if self.config.tracer_names is None:
            # Try to infer from filenames or use indices
            tracer_names = []
            for i, qp_file in enumerate(qp_files):
                if isinstance(qp_file, (str, Path)):
                    # Try to extract tracer name from filename
                    base = Path(qp_file).stem
                    # Look for patterns like "bin_0", "bin0", etc.
                    if "bin" in base.lower():
                        tracer_names.append(base.split("_")[-1] if "_" in base else f"bin_{i}")
                    else:
                        tracer_names.append(f"bin_{i}")
                else:
                    tracer_names.append(f"bin_{i}")
        else:
            tracer_names = self.config.tracer_names

        if len(tracer_names) != len(qp_files):
            raise ValueError(
                f"Number of tracer names ({len(tracer_names)}) does not match "
                f"number of qp files ({len(qp_files)})"
            )

        # Get truth files if provided
        truth_files = self.config.truth_files
        if truth_files is not None and len(truth_files) != len(qp_files):
            raise ValueError(
                f"Number of truth files ({len(truth_files)}) does not match "
                f"number of qp files ({len(qp_files)})"
            )

        # Create sacc catalog
        catalog = sacc.Sacc()

        # Process each qp file
        for i, (tracer_name, qp_file) in enumerate(zip(tracer_names, qp_files)):
            # Read qp ensemble
            if isinstance(qp_file, str):
                qp_path = qp_file
            else:
                qp_path = str(qp_file)

            ensemble = qp.read(qp_path)

            # Extract bins and PDFs
            # Check for histogram format first (bins + pdfs)
            has_bins_objdata = hasattr(ensemble, "objdata") and "bins" in ensemble.objdata
            has_pdfs_objdata = hasattr(ensemble, "objdata") and "pdfs" in ensemble.objdata
            has_bins_metadata = hasattr(ensemble, "metadata") and "bins" in ensemble.metadata
            has_xvals_metadata = hasattr(ensemble, "metadata") and "xvals" in ensemble.metadata
            has_yvals_objdata = hasattr(ensemble, "objdata") and "yvals" in ensemble.objdata

            # Prefer histogram format if available
            if has_pdfs_objdata:
                pdfs = np.array(ensemble.objdata["pdfs"], dtype=float)
                # Use bins from objdata if available, otherwise from metadata
                if has_bins_objdata:
                    edges = np.array(ensemble.objdata["bins"], dtype=float).reshape(-1)
                elif has_bins_metadata:
                    edges = np.array(ensemble.metadata["bins"], dtype=float).reshape(-1)
                else:
                    raise ValueError(f"Could not find bin edges for histogram format in qp file: {qp_path}")
            # Fall back to interpolated format (xvals/yvals)
            elif has_xvals_metadata and has_yvals_objdata:
                # Interpolated format - convert to histogram
                xvals = np.array(ensemble.metadata["xvals"], dtype=float).reshape(-1)
                yvals = np.array(ensemble.objdata["yvals"], dtype=float)
                # Handle NaN/inf values (matches tutorial approach)
                yvals = np.nan_to_num(yvals, nan=0.0, posinf=0.0, neginf=0.0)
                # Ensure yvals is 2D
                if yvals.ndim == 1:
                    yvals = yvals.reshape(1, -1)
                # xvals are grid points (edges), yvals are values at those points
                # Convert to bin values by averaging adjacent grid points
                # This matches the approach in gold_baseline_tutorial.ipynb
                pdfs_centers = 0.5 * (yvals[:, :-1] + yvals[:, 1:])
                # Use xvals as edges (they already represent bin edges)
                edges = xvals
                pdfs = pdfs_centers
            else:
                raise ValueError(
                    f"Could not find PDFs in qp file: {qp_path}. "
                    f"Expected either (pdfs in objdata) or (xvals in metadata and yvals in objdata)."
                )

            # Ensure pdfs is 2D
            if pdfs.ndim == 1:
                pdfs = pdfs.reshape(1, -1)

            # Normalize PDFs
            pdfs = normalize_hist(pdfs, edges)

            # Create z centers
            z_centers = 0.5 * (edges[:-1] + edges[1:])

            # Create qp ensemble for sacc (hist format)
            qp_ensemble = qp.Ensemble(qp.hist, data={"bins": edges, "pdfs": pdfs})

            # Get truth n(z) if provided
            nz_truth = None
            if truth_files is not None:
                truth_path = truth_files[i]
                if truth_path is not None and isinstance(truth_path, str) and os.path.exists(truth_path):
                    truth_ensemble = qp.read(truth_path)
                    if hasattr(truth_ensemble, "objdata") and "pdfs" in truth_ensemble.objdata:
                        truth_pdf = np.array(truth_ensemble.objdata["pdfs"], dtype=float).flatten()
                        truth_pdf = normalize_hist(truth_pdf, edges)
                        nz_truth = truth_pdf
                    elif hasattr(truth_ensemble, "metadata") and "xvals" in truth_ensemble.metadata:
                        truth_xvals = np.array(truth_ensemble.metadata["xvals"], dtype=float).reshape(-1)
                        truth_yvals = np.array(truth_ensemble.objdata["yvals"], dtype=float)
                        if truth_yvals.ndim == 1:
                            truth_yvals = truth_yvals.reshape(1, -1)
                        # Interpolate to match edges
                        truth_pdf = np.mean(truth_yvals, axis=0)
                        # Interpolate to edges
                        truth_pdf = np.interp(z_centers, 0.5 * (truth_xvals[:-1] + truth_xvals[1:]), truth_pdf, left=0.0, right=0.0)
                        nz_truth = normalize_hist(truth_pdf, edges)

            # Add tracer to catalog
            if nz_truth is not None:
                catalog.add_tracer("QPNZ", tracer_name, qp_ensemble, z=z_centers, nz=nz_truth)
            else:
                # Use mean of ensemble as truth
                nz_mean = np.mean(pdfs, axis=0)
                catalog.add_tracer("QPNZ", tracer_name, qp_ensemble, z=z_centers, nz=nz_mean)

        # Save sacc catalog to file if output path is specified
        # The catalog object itself will be stored in the data store
        self.add_data("sacc_output", catalog)

    def __call__(self, qp_input):
        """Callable interface to convert qp to sacc.

        Parameters
        ----------
        qp_input : str, list, or dict
            Input qp file path(s) or dict of {tracer_name: file_path}.

        Returns
        -------
        Hdf5Handle
            Handle to the output sacc file.
        """
        self.set_data("qp_input", qp_input)
        self.run()
        return self.get_handle("sacc_output")


class SACCToQP(RailStage):
    """RailStage to convert n(z) from sacc format to qp Ensemble format.

    This class loads n(z) distributions from a sacc.Sacc file (with QPNZ
    tracers) and saves each tracer as a separate qp Ensemble file on disk.

    Attributes
    ----------
    inputs : list of tuples
        List of input data handles. Expects 'sacc_input' as Hdf5Handle.
    outputs : list of tuples
        List of output data handles. Produces 'qp_output' as Hdf5Handle.
    """

    name = "SACCToQP"
    config_options = RailStage.config_options.copy()
    config_options.update(
        output_dir=Param(
            str,
            default=".",
            required=False,
            msg="Directory to save output qp files. Defaults to current directory.",
        ),
        output_prefix=Param(
            str,
            default="",
            required=False,
            msg="Prefix to add to output filenames. Defaults to empty string.",
        ),
        tracer_names=Param(
            list,
            default=None,
            required=False,
            msg="List of tracer names to extract. If None, extracts all QPNZ tracers.",
        ),
        include_truth=Param(
            bool,
            default=True,
            required=False,
            msg="Whether to include truth n(z) in output qp files.",
        ),
    )

    inputs = [("sacc_input", Hdf5Handle)]
    outputs = [("qp_output", Hdf5Handle)]

    def __init__(self, args, **kwargs):
        """Initialize the SACCToQP stage.

        Parameters
        ----------
        args : dict
            Arguments for RailStage initialization.
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(args, **kwargs)

    def run(self):
        """Run the conversion from sacc to qp format.

        Reads sacc file, extracts QPNZ tracers, and saves each as a qp file.
        """
        # Get input sacc file
        sacc_input_data = self.get_data("sacc_input", allow_missing=True)

        # Load sacc catalog
        if isinstance(sacc_input_data, sacc.Sacc):
            catalog = sacc_input_data
        elif isinstance(sacc_input_data, str):
            # String path - load it directly
            catalog = sacc.Sacc.load_fits(sacc_input_data)
        else:
            raise ValueError(f"Unsupported input format: {type(sacc_input_data)}. Expected sacc.Sacc object or file path string.")

        # Determine which tracers to extract
        if self.config.tracer_names is None:
            # Extract all QPNZ tracers
            tracer_names = [name for name, tracer in catalog.tracers.items() if tracer.tracer_type == "QPNZ"]
        else:
            tracer_names = self.config.tracer_names
            # Verify all requested tracers exist and are QPNZ
            for name in tracer_names:
                if name not in catalog.tracers:
                    raise ValueError(f"Tracer '{name}' not found in sacc catalog.")
                if catalog.tracers[name].tracer_type != "QPNZ":
                    raise ValueError(f"Tracer '{name}' is not a QPNZ tracer (type: {catalog.tracers[name].tracer_type}).")

        if not tracer_names:
            raise ValueError("No QPNZ tracers found in sacc catalog.")

        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each tracer
        output_files = {}
        for tracer_name in tracer_names:
            tracer = catalog.tracers[tracer_name]

            if not hasattr(tracer, "ensemble") or tracer.ensemble is None:
                raise ValueError(f"Tracer '{tracer_name}' does not have an ensemble.")

            ensemble = tracer.ensemble

            # Extract bins and PDFs
            if hasattr(ensemble, "objdata") and "bins" in ensemble.objdata:
                edges = np.array(ensemble.objdata["bins"], dtype=float).reshape(-1)
                if "pdfs" not in ensemble.objdata:
                    raise ValueError(f"Could not find pdfs in objdata for tracer '{tracer_name}'.")
                pdfs = np.array(ensemble.objdata["pdfs"], dtype=float)
            elif hasattr(ensemble, "metadata") and "bins" in ensemble.metadata:
                edges = np.array(ensemble.metadata["bins"], dtype=float).reshape(-1)
                if not (hasattr(ensemble, "objdata") and "pdfs" in ensemble.objdata):
                    raise ValueError(f"Could not find pdfs in objdata for tracer '{tracer_name}' (bins found in metadata).")
                pdfs = np.array(ensemble.objdata["pdfs"], dtype=float)
            else:
                raise ValueError(f"Could not extract bins/PDFs from ensemble for tracer '{tracer_name}'.")

            # Normalize PDFs
            pdfs = normalize_hist(pdfs, edges)

            # Create output ensemble
            output_ensemble = qp.Ensemble(qp.hist, data={"bins": edges, "pdfs": pdfs})

            # Add truth n(z) if available and requested
            if self.config.include_truth and hasattr(tracer, "nz") and tracer.nz is not None:
                # Store truth as metadata
                output_ensemble.metadata["nz_truth"] = np.array(tracer.nz, dtype=float)
                if hasattr(tracer, "z") and tracer.z is not None:
                    output_ensemble.metadata["z_truth"] = np.array(tracer.z, dtype=float)

            # Generate output filename
            prefix = self.config.output_prefix
            if prefix and not prefix.endswith("_"):
                prefix = f"{prefix}_"
            output_filename = f"{prefix}{tracer_name}.hdf5"
            output_path = output_dir / output_filename

            # Save qp ensemble
            output_ensemble.write_to(str(output_path))
            output_files[tracer_name] = str(output_path)

        # Store output file paths
        if len(output_files) == 1:
            self.add_data("qp_output", list(output_files.values())[0])
        else:
            self.add_data("qp_output", output_files)

    def __call__(self, sacc_input):
        """Callable interface to convert sacc to qp.

        Parameters
        ----------
        sacc_input : str or sacc.Sacc
            Input sacc file path or sacc.Sacc object.

        Returns
        -------
        Hdf5Handle or dict
            Handle(s) to the output qp file(s). Returns a single handle if one
            tracer, or a dict of {tracer_name: handle} if multiple.
        """
        # If input is a string path, load it directly to avoid tables_io auto-detection
        if isinstance(sacc_input, str):
            sacc_obj = sacc.Sacc.load_fits(sacc_input)
            self.set_data("sacc_input", sacc_obj)
        else:
            self.set_data("sacc_input", sacc_input)
        self.run()
        return self.get_handle("qp_output")

