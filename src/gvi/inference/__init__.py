"""Optional model-runtime components kept outside the evidence core."""

from gvi.inference.mask_rle import BinaryMaskRle, decode_binary_mask, encode_binary_mask
from gvi.inference.sam3_worker import Sam3InferenceWorker, VideoDescriptor

__all__ = [
    "BinaryMaskRle",
    "Sam3InferenceWorker",
    "VideoDescriptor",
    "decode_binary_mask",
    "encode_binary_mask",
]
