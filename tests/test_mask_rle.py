import numpy as np
import pytest

from gvi.inference.mask_rle import BinaryMaskRle, decode_binary_mask, encode_binary_mask


def test_binary_mask_rle_round_trip() -> None:
    mask = np.asarray(
        [
            [False, False, True, True],
            [False, True, True, False],
        ],
        dtype=np.bool_,
    )

    encoded = encode_binary_mask(mask)
    decoded = decode_binary_mask(encoded)

    assert encoded.counts == (2, 2, 1, 2, 1)
    assert np.array_equal(decoded, mask)


def test_binary_mask_rle_rejects_invalid_expanded_size() -> None:
    encoded = BinaryMaskRle(height=2, width=2, counts=(2, 1))

    with pytest.raises(ValueError, match="expected 4"):
        decode_binary_mask(encoded)


def test_binary_mask_rle_rejects_empty_mask() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        encode_binary_mask(np.zeros((0, 3), dtype=np.bool_))
