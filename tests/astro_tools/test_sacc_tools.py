"""Unit tests for rail.tools.sacc_tools."""
import tempfile
from pathlib import Path

import numpy as np
import pytest
import qp
import sacc

from rail.core.stage import RailStage
from rail.tools.sacc_tools import (
    QPToSACC,
    SACCToQP,
    extract_tomographic_bins_from_sacc,
    normalize_hist,
)

DS = RailStage.data_store
DS.__class__.allow_overwrite = True


# --- Fixtures for synthetic data ---


@pytest.fixture
def edges():
    """Bin edges for synthetic n(z) data."""
    return np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


@pytest.fixture
def pdfs_1d(edges):
    """Single unnormalized PDF (1D)."""
    z_centers = 0.5 * (edges[:-1] + edges[1:])
    return np.exp(-0.5 * ((z_centers - 0.5) ** 2) / 0.1**2)


@pytest.fixture
def pdfs_2d(edges):
    """Multiple unnormalized PDFs (2D)."""
    z_centers = 0.5 * (edges[:-1] + edges[1:])
    pdf1 = np.exp(-0.5 * ((z_centers - 0.3) ** 2) / 0.1**2)
    pdf2 = np.exp(-0.5 * ((z_centers - 0.6) ** 2) / 0.15**2)
    pdf3 = np.ones_like(z_centers)
    return np.array([pdf1, pdf2, pdf3])


def _make_qp_hist_file(tmpdir, edges, pdfs, filename="bin_0.hdf5"):
    """Write a qp histogram ensemble to file."""
    pdfs_norm = normalize_hist(pdfs, edges)
    ens = qp.Ensemble(qp.hist, data={"bins": edges, "pdfs": pdfs_norm})
    path = Path(tmpdir) / filename
    ens.write_to(str(path))
    return str(path)


def _make_sacc_catalog(edges, pdfs_list, tracer_names, nz_truth_list=None):
    """Create a sacc catalog with QPNZ tracers."""
    catalog = sacc.Sacc()
    z_centers = 0.5 * (edges[:-1] + edges[1:])
    for i, (name, pdfs) in enumerate(zip(tracer_names, pdfs_list)):
        pdfs_norm = normalize_hist(pdfs, edges)
        ens = qp.Ensemble(qp.hist, data={"bins": edges, "pdfs": pdfs_norm})
        nz = nz_truth_list[i] if nz_truth_list is not None else np.mean(pdfs_norm, axis=0)
        catalog.add_tracer("QPNZ", name, ens, z=z_centers, nz=nz)
    return catalog


# --- Tests for normalize_hist ---


def test_normalize_hist_single_pdf_1d(pdfs_1d, edges):
    """Single PDF (1D): output integrates to 1, shape preserved."""
    result = normalize_hist(pdfs_1d, edges)
    assert result.ndim == 1
    assert result.shape == pdfs_1d.shape
    widths = np.diff(edges)
    integral = (result * widths).sum()
    assert np.isclose(integral, 1.0)


def test_normalize_hist_multiple_pdfs_2d(pdfs_2d, edges):
    """Multiple PDFs (2D): each row integrates to 1."""
    result = normalize_hist(pdfs_2d, edges)
    assert result.ndim == 2
    assert result.shape == pdfs_2d.shape
    widths = np.diff(edges)
    for row in result:
        integral = (row * widths).sum()
        assert np.isclose(integral, 1.0)


def test_normalize_hist_invalid_edges(pdfs_1d):
    """Invalid edges (non-positive total width) raises ValueError."""
    bad_edges = np.array([1.0, 1.0])  # zero width
    with pytest.raises(ValueError, match="Invalid bin edges"):
        normalize_hist(pdfs_1d, bad_edges)

    bad_edges2 = np.array([2.0, 1.0])  # negative width
    with pytest.raises(ValueError, match="Invalid bin edges"):
        normalize_hist(pdfs_1d, bad_edges2)


def test_normalize_hist_zero_norm(edges):
    """PDFs with zero/negative norm get replaced with uniform."""
    zero_pdfs = np.zeros(5)
    result = normalize_hist(zero_pdfs, edges)
    widths = np.diff(edges)
    integral = (result * widths).sum()
    assert np.isclose(integral, 1.0)
    expected_uniform = 1.0 / widths.sum()
    assert np.allclose(result, expected_uniform)


def test_normalize_hist_preserves_shape(pdfs_1d, pdfs_2d, edges):
    """1D in -> 1D out, 2D in -> 2D out."""
    r1 = normalize_hist(pdfs_1d, edges)
    assert r1.ndim == 1
    r2 = normalize_hist(pdfs_2d, edges)
    assert r2.ndim == 2
    assert r2.shape == pdfs_2d.shape


# --- Tests for extract_tomographic_bins_from_sacc ---


def test_extract_tomographic_bins_single_tracer(edges, pdfs_2d):
    """Single tracer: tomos has 1 element with expected keys."""
    catalog = _make_sacc_catalog(edges, [pdfs_2d], ["bin_0"])
    tomos, edges_lookup = extract_tomographic_bins_from_sacc(catalog)
    assert len(tomos) == 1
    tomo = tomos[0]
    for key in ("label", "z", "edges", "truth", "estimates"):
        assert key in tomo
    assert tomo["label"] == "bin_0"
    assert len(edges_lookup) == 1
    assert "bin_0" in edges_lookup
    np.testing.assert_array_almost_equal(edges_lookup["bin_0"], edges)


def test_extract_tomographic_bins_multiple_tracers(edges, pdfs_2d):
    """Multiple tracers: correct count and labels."""
    pdfs_b = np.ones((2, 5)) * [0.5, 0.3, 0.4, 0.2, 0.6]
    pdfs_c = np.ones((1, 5))
    catalog = _make_sacc_catalog(
        edges,
        [pdfs_2d, pdfs_b, pdfs_c],
        ["bin_0", "bin_1", "bin_2"],
    )
    tomos, edges_lookup = extract_tomographic_bins_from_sacc(catalog)
    assert len(tomos) == 3
    labels = [t["label"] for t in tomos]
    assert labels == ["bin_0", "bin_1", "bin_2"]
    assert len(edges_lookup) == 3


def test_extract_tomographic_bins_skips_non_qpnz(edges, pdfs_2d):
    """Non-QPNZ tracers are omitted from output."""
    catalog = _make_sacc_catalog(edges, [pdfs_2d], ["qpnz_tracer"])
    if hasattr(sacc, "tracers") or True:
        try:
            z = 0.5 * (edges[:-1] + edges[1:])
            nz = np.ones(len(z)) / len(z)
            catalog.add_tracer("NZ", "nz_only_tracer", z=z, nz=nz)
        except (TypeError, ValueError):
            pytest.skip("sacc may not support adding NZ tracer separately")
    tomos, _ = extract_tomographic_bins_from_sacc(catalog)
    labels = [t["label"] for t in tomos]
    assert "qpnz_tracer" in labels
    assert "nz_only_tracer" not in labels


# --- Tests for QPToSACC ---


def test_qp_to_sacc_single_file(edges, pdfs_2d):
    """Single qp file (str) -> sacc with 1 QPNZ tracer."""
    DS.clear()
    with tempfile.TemporaryDirectory() as tmpdir:
        qp_path = _make_qp_hist_file(tmpdir, edges, pdfs_2d, "single.hdf5")
        stage = QPToSACC.make_stage(name="qp_to_sacc", tracer_names=["bin_0"])
        # Use list to avoid DataStore path handling issues with bare string
        handle = stage([qp_path])
        catalog = handle.data
        assert len(catalog.tracers) == 1
        assert "bin_0" in catalog.tracers
        assert catalog.tracers["bin_0"].tracer_type == "QPNZ"


def test_qp_to_sacc_list_of_files(edges, pdfs_2d):
    """List of qp files with tracer_names -> sacc with multiple tracers."""
    DS.clear()
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = [
            _make_qp_hist_file(tmpdir, edges, pdfs_2d, f"bin_{i}.hdf5")
            for i in range(3)
        ]
        stage = QPToSACC.make_stage(
            name="qp_to_sacc",
            tracer_names=["bin_0", "bin_1", "bin_2"],
        )
        handle = stage(paths)
        catalog = handle.data
        assert len(catalog.tracers) == 3
        for name in ["bin_0", "bin_1", "bin_2"]:
            assert name in catalog.tracers


def test_qp_to_sacc_dict_input(edges, pdfs_2d):
    """Dict input {tracer_name: path} uses keys as tracer names."""
    DS.clear()
    with tempfile.TemporaryDirectory() as tmpdir:
        d = {
            "alpha": _make_qp_hist_file(tmpdir, edges, pdfs_2d, "a.hdf5"),
            "beta": _make_qp_hist_file(tmpdir, edges, pdfs_2d, "b.hdf5"),
        }
        stage = QPToSACC.make_stage(name="qp_to_sacc")
        handle = stage(d)
        catalog = handle.data
        assert set(catalog.tracers.keys()) == {"alpha", "beta"}


def test_qp_to_sacc_tracer_name_inference(edges, pdfs_2d):
    """Omit tracer_names; infer from filenames like bin_0.hdf5."""
    DS.clear()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _make_qp_hist_file(tmpdir, edges, pdfs_2d, "bin_0.hdf5")
        stage = QPToSACC.make_stage(name="qp_to_sacc")
        handle = stage([path])
        catalog = handle.data
        assert len(catalog.tracers) == 1
        inferred_name = list(catalog.tracers.keys())[0]
        assert inferred_name in ("0", "bin_0", "bin_0.hdf5") or "bin" in inferred_name.lower()


def test_qp_to_sacc_with_truth_files(edges, pdfs_2d, pdfs_1d):
    """Truth files optional -> tracer nz set from truth."""
    DS.clear()
    with tempfile.TemporaryDirectory() as tmpdir:
        qp_path = _make_qp_hist_file(tmpdir, edges, pdfs_2d, "estimates.hdf5")
        truth_path = _make_qp_hist_file(tmpdir, edges, pdfs_1d, "truth.hdf5")
        stage = QPToSACC.make_stage(
            name="qp_to_sacc",
            tracer_names=["bin_0"],
            truth_files=[truth_path],
        )
        handle = stage([qp_path])
        catalog = handle.data
        tracer = catalog.tracers["bin_0"]
        assert hasattr(tracer, "nz") and tracer.nz is not None
        np.testing.assert_allclose(tracer.nz, normalize_hist(pdfs_1d, edges))


def test_qp_to_sacc_tracer_names_mismatch(edges, pdfs_2d):
    """tracer_names length mismatch raises ValueError."""
    DS.clear()
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = [
            _make_qp_hist_file(tmpdir, edges, pdfs_2d, f"bin_{i}.hdf5")
            for i in range(2)
        ]
        stage = QPToSACC.make_stage(
            name="qp_to_sacc",
            tracer_names=["only_one"],
        )
        with pytest.raises(ValueError, match="Number of tracer names"):
            stage(paths)


def test_qp_to_sacc_truth_files_mismatch(edges, pdfs_2d):
    """truth_files length mismatch raises ValueError."""
    DS.clear()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = _make_qp_hist_file(tmpdir, edges, pdfs_2d, "bin_0.hdf5")
        truth = _make_qp_hist_file(tmpdir, edges, pdfs_2d[0], "truth.hdf5")
        stage = QPToSACC.make_stage(
            name="qp_to_sacc",
            tracer_names=["bin_0"],
            truth_files=[truth, truth],
        )
        with pytest.raises(ValueError, match="Number of truth files"):
            stage([path])


def test_qp_to_sacc_unsupported_input_type():
    """Unsupported input type raises ValueError."""
    DS.clear()
    stage = QPToSACC.make_stage(name="qp_to_sacc")
    with pytest.raises(ValueError, match="Unsupported input format"):
        stage(12345)


# --- Tests for SACCToQP ---


def test_sacc_to_qp_in_memory(edges, pdfs_2d):
    """In-memory sacc object -> qp files."""
    DS.clear()
    catalog = _make_sacc_catalog(edges, [pdfs_2d], ["bin_0"])
    with tempfile.TemporaryDirectory() as tmpdir:
        stage = SACCToQP.make_stage(
            name="sacc_to_qp",
            output_dir=tmpdir,
        )
        handle = stage(catalog)
        out = handle.data
        assert isinstance(out, str)
        assert Path(out).exists()
        loaded = qp.read(out)
        assert hasattr(loaded, "objdata") and "pdfs" in loaded.objdata


def test_sacc_to_qp_file_path(edges, pdfs_2d):
    """File path string -> load and convert."""
    DS.clear()
    catalog = _make_sacc_catalog(edges, [pdfs_2d], ["bin_0"])
    with tempfile.TemporaryDirectory() as tmpdir:
        sacc_path = Path(tmpdir) / "catalog.fits"
        catalog.save_fits(str(sacc_path), overwrite=True)
        stage = SACCToQP.make_stage(name="sacc_to_qp", output_dir=tmpdir)
        handle = stage(str(sacc_path))
        assert Path(handle.data).exists()


def test_sacc_to_qp_multiple_tracers(edges, pdfs_2d):
    """Multiple tracers -> one qp file per tracer."""
    DS.clear()
    pdfs_b = np.ones((2, 5))
    catalog = _make_sacc_catalog(
        edges,
        [pdfs_2d, pdfs_b],
        ["bin_0", "bin_1"],
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        stage = SACCToQP.make_stage(name="sacc_to_qp", output_dir=tmpdir)
        handle = stage(catalog)
        out = handle.data
        assert isinstance(out, dict)
        assert set(out.keys()) == {"bin_0", "bin_1"}
        for p in out.values():
            assert Path(p).exists()


def test_sacc_to_qp_single_tracer_returns_path(edges, pdfs_2d):
    """Single tracer: output is path string, not dict."""
    DS.clear()
    catalog = _make_sacc_catalog(edges, [pdfs_2d], ["bin_0"])
    with tempfile.TemporaryDirectory() as tmpdir:
        stage = SACCToQP.make_stage(name="sacc_to_qp", output_dir=tmpdir)
        handle = stage(catalog)
        assert isinstance(handle.data, str)


def test_sacc_to_qp_output_dir_and_prefix(edges, pdfs_2d):
    """output_dir and output_prefix applied correctly."""
    DS.clear()
    catalog = _make_sacc_catalog(edges, [pdfs_2d], ["bin_0"])
    with tempfile.TemporaryDirectory() as tmpdir:
        out_subdir = Path(tmpdir) / "subdir"
        stage = SACCToQP.make_stage(
            name="sacc_to_qp",
            output_dir=str(out_subdir),
            output_prefix="nz_",
        )
        handle = stage(catalog)
        path = Path(handle.data)
        assert path.parent == out_subdir
        assert path.name == "nz_bin_0.hdf5"


def test_sacc_to_qp_include_truth_false(edges, pdfs_2d):
    """include_truth=False -> no nz_truth in output metadata."""
    DS.clear()
    catalog = _make_sacc_catalog(edges, [pdfs_2d], ["bin_0"])
    with tempfile.TemporaryDirectory() as tmpdir:
        stage = SACCToQP.make_stage(
            name="sacc_to_qp",
            output_dir=tmpdir,
            include_truth=False,
        )
        handle = stage(catalog)
        loaded = qp.read(handle.data)
        assert "nz_truth" not in loaded.metadata


def test_sacc_to_qp_include_truth_true(edges, pdfs_2d):
    """include_truth=True runs successfully; qp does not persist custom metadata."""
    DS.clear()
    catalog = _make_sacc_catalog(edges, [pdfs_2d], ["bin_0"])
    with tempfile.TemporaryDirectory() as tmpdir:
        stage = SACCToQP.make_stage(
            name="sacc_to_qp",
            output_dir=tmpdir,
            include_truth=True,
        )
        handle = stage(catalog)
        loaded = qp.read(handle.data)
        # qp.Ensemble.write_to does not persist custom metadata; verify valid output
        assert "pdfs" in loaded.objdata
        assert "bins" in loaded.objdata or "bins" in loaded.metadata


def test_sacc_to_qp_tracer_not_found(edges, pdfs_2d):
    """Requested tracer not in catalog -> ValueError."""
    DS.clear()
    catalog = _make_sacc_catalog(edges, [pdfs_2d], ["bin_0"])
    with tempfile.TemporaryDirectory() as tmpdir:
        stage = SACCToQP.make_stage(
            name="sacc_to_qp",
            output_dir=tmpdir,
            tracer_names=["nonexistent"],
        )
        with pytest.raises(ValueError, match="not found in sacc catalog"):
            stage(catalog)


def test_sacc_to_qp_unsupported_input_type():
    """Unsupported input type -> ValueError."""
    DS.clear()
    with tempfile.TemporaryDirectory() as tmpdir:
        stage = SACCToQP.make_stage(name="sacc_to_qp", output_dir=tmpdir)
        with pytest.raises(ValueError, match="Unsupported input format"):
            stage(12345)


# --- Round-trip test ---


def test_round_trip_qp_sacc_qp(edges, pdfs_2d):
    """QP -> SACC -> QP round-trip preserves PDFs."""
    DS.clear()
    with tempfile.TemporaryDirectory() as tmpdir:
        orig_path = _make_qp_hist_file(tmpdir, edges, pdfs_2d, "orig.hdf5")
        qp_to_sacc = QPToSACC.make_stage(
            name="qp_to_sacc",
            tracer_names=["bin_0"],
        )
        sacc_handle = qp_to_sacc([orig_path])
        catalog = sacc_handle.data
        sacc_path = Path(tmpdir) / "roundtrip.fits"
        catalog.save_fits(str(sacc_path), overwrite=True)

        sacc_to_qp = SACCToQP.make_stage(
            name="sacc_to_qp",
            output_dir=tmpdir,
        )
        qp_handle = sacc_to_qp(str(sacc_path))
        roundtrip_path = qp_handle.data

        orig_ens = qp.read(orig_path)
        rt_ens = qp.read(roundtrip_path)
        orig_pdfs = np.array(orig_ens.objdata["pdfs"])
        rt_pdfs = np.array(rt_ens.objdata["pdfs"])
        np.testing.assert_allclose(orig_pdfs, rt_pdfs, rtol=1e-9, atol=1e-9)
