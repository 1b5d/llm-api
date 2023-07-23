"""
utils for the Huggingface model
"""

from typing import Union

import torch


def get_dtype_from_str(  # pylint: disable=too-many-return-statements
    dtype: str,
) -> Union[str, torch.dtype]:
    """
    convert torch dtype from string to enum value
    """
    if dtype in ("torch.float32", "torch.float"):
        return torch.float32
    if dtype in ("torch.float64", "torch.double"):
        return torch.float64
    if dtype in ("torch.complex64", "torch.cfloat"):
        return torch.complex64
    if dtype in ("torch.complex128", "torch.cdouble"):
        return torch.complex128
    if dtype in ("torch.float16", "torch.half"):
        return torch.float16
    if dtype == "torch.bfloat16":
        return torch.bfloat16
    if dtype == "torch.uint8":
        return torch.uint8
    if dtype == "torch.int8":
        return torch.int8
    if dtype in ("torch.int16", "torch.short"):
        return torch.int16
    if dtype in ("torch.int32", "torch.int"):
        return torch.int32
    if dtype in ("torch.int64", "torch.long"):
        return torch.int64
    if dtype == "torch.bool":
        return torch.bool
    return "auto"
