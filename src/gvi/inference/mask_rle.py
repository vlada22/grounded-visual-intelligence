from __future__ import annotations

import numpy as np
from pydantic import Field

from gvi.models import StrictModel


class BinaryMaskRle(StrictModel):
    """Simple row-major run-length encoding for a two-dimensional binary mask."""

    schema_version: str = "1.0"
    height: int = Field(gt=0)
    width: int = Field(gt=0)
    order: str = "row-major"
    counts: tuple[int, ...] = Field(min_length=1)


def encode_binary_mask(mask: np.ndarray) -> BinaryMaskRle:
    array = np.asarray(mask, dtype=np.bool_)
    if array.ndim != 2:
        raise ValueError("binary mask must have exactly two dimensions")
    if array.size == 0:
        raise ValueError("binary mask dimensions must be non-zero")

    flattened = array.reshape(-1, order="C").astype(np.int8, copy=False)
    boundaries = np.flatnonzero(np.diff(flattened)) + 1
    starts = np.concatenate((np.array([0]), boundaries))
    ends = np.concatenate((boundaries, np.array([flattened.size])))
    counts = (ends - starts).tolist()
    if bool(flattened[0]):
        counts.insert(0, 0)

    return BinaryMaskRle(
        height=array.shape[0],
        width=array.shape[1],
        counts=tuple(counts),
    )


def decode_binary_mask(encoded: BinaryMaskRle) -> np.ndarray:
    counts = np.asarray(encoded.counts, dtype=np.int64)
    if np.any(counts < 0):
        raise ValueError("RLE counts cannot be negative")
    expected_size = encoded.height * encoded.width
    expanded_size = int(counts.sum())
    if expanded_size != expected_size:
        raise ValueError(f"RLE expands to {expanded_size} pixels; expected {expected_size}")

    run_values = np.arange(len(counts)) % 2 == 1
    return np.repeat(run_values, counts).reshape((encoded.height, encoded.width))
