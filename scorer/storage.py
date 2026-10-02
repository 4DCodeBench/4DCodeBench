"""Frame-chunked HDF5 storage for dense flow and depth arrays."""

from pathlib import Path

import h5py
import numpy as np


def open_dense(path: str | Path) -> h5py.File:
    """Open a dense archive read-only."""

    return h5py.File(path, "r")


def write_dense(path: str | Path, **arrays) -> None:
    """Write each array as one dataset.

    Unicode arrays become UTF-8 strings; arrays of rank 3 or more are stored one
    frame per chunk with LZF compression.
    """

    with h5py.File(path, "w") as archive:
        for name, values in arrays.items():
            values = np.asarray(values)
            if values.dtype.kind == "U":
                archive.create_dataset(name, data=values.astype(object),
                                       dtype=h5py.string_dtype("utf-8"))
            elif values.ndim >= 3:
                archive.create_dataset(name, data=values, chunks=(1, *values.shape[1:]),
                                       compression="lzf", shuffle=True)
            else:
                archive.create_dataset(name, data=values)
