"""Table-flavour helpers shared by the degrader stages.

These exist because two conversions in the RAIL stack fail *quietly* rather than raising, and
both are easy to hit from any stage that takes a ``PqHandle`` in or writes an ``Hdf5Handle``
out.  They live here rather than in one stage module so that unrelated stages can share them
without importing each other.
"""

import numpy as np
import pandas as pd
import tables_io

from rail.core.data import TableLike


def as_dataframe(data: TableLike) -> pd.DataFrame:
    """Return ``data`` as a pandas DataFrame, whatever table flavour it arrives as.

    ``PqHandle`` hands back a **pyarrow Table** when a stage runs under ceci, but a DataFrame
    when a stage is called interactively with one.  The difference is silent and vicious:
    ``Table.columns`` is the list of column *data*, not column *names*, so a name lookup
    against it fails in a way that looks nothing like a type error.
    """
    if isinstance(data, pd.DataFrame):
        return data
    return tables_io.convert(data, tables_io.types.PD_DATAFRAME)


def to_numpy_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Cast pandas extension dtypes to plain numpy dtypes so the frame survives HDF5.

    This is not cosmetic.  ``tables_io.convert(df, NUMPY_DICT)`` **silently drops** any column
    it cannot map to a native HDF5 type -- it emits a warning and carries on -- so a catalog
    with pandas ``string`` or nullable-integer columns loses them without raising.  On the DP2
    anacal cross-match that would quietly discard ``z_source``, ``z_type``, ``ref_id``,
    ``ref_cat``, ``region`` and ``desi_spectype``: every column identifying where a redshift
    came from.

    Strings become fixed-width bytes, nullable integers become float64 (so nulls survive as
    NaN), and nullable booleans are filled False.

    ``"str"`` is matched alongside ``"string"`` because pandas 3.0 renames the dtype it gives
    string columns by default.  Without it this helper silently stops converting them there,
    which is the very failure it exists to prevent.
    """
    out = df.copy()
    for col in out.columns:
        dtype_name = str(out[col].dtype)
        if dtype_name in ("string", "str") or out[col].dtype == object:
            out[col] = out[col].astype(str).values.astype("S")
        elif dtype_name.startswith(("Int", "UInt")):
            out[col] = out[col].astype("float64")
        elif dtype_name == "Float32" or dtype_name == "Float64":
            out[col] = out[col].astype("float64")
        elif dtype_name == "boolean":
            out[col] = out[col].fillna(False).astype(bool)
    return out


def frame_to_hdf5_dict(df: pd.DataFrame) -> dict:
    """Convert a DataFrame to the ordered dict-of-arrays that ``Hdf5Handle`` writes."""
    converted = tables_io.convert(to_numpy_frame(df), tables_io.types.NUMPY_DICT)
    n_lost = len(df.columns) - len(converted)
    if n_lost:  # pragma: no cover
        missing = [c for c in df.columns if c not in converted]
        raise RuntimeError(
            f"{n_lost} columns were dropped converting to HDF5: {missing}. "
            "to_numpy_frame() failed to make them HDF5-safe."
        )
    return converted


def plain_python(value):
    """Strip numpy and tuple types out of a config value, recursively.

    ceci writes a stage's config with ``yaml.dump`` (the unsafe Dumper) and reads it back with
    ``yaml.safe_load``.  A ``numpy`` scalar or a ``tuple`` survives the write as a
    ``!!python/...`` tag and then kills the stage subprocess at config-load time -- before
    ``run()`` is ever called -- with a ``ConstructorError`` that talks about YAML and says
    nothing about where the value came from.  Both directions measured.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [plain_python(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [plain_python(v) for v in value]
    if isinstance(value, dict):
        return {plain_python(k): plain_python(v) for k, v in value.items()}
    return value
