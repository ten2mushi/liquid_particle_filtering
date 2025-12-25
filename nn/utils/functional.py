"""Functional utilities for particle filter operations."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def batched_linear(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """Linear operation supporting both regular and batched weights.

    This function handles the case where weights come from parameter particles,
    resulting in 3D weight tensors [batch, out_features, in_features] instead
    of the standard 2D [out_features, in_features].

    Args:
        x: Input tensor [batch, in_features]
        weight: Weight tensor [out, in] (2D) or [batch, out, in] (3D)
        bias: Bias tensor [out] or [batch, out] or None

    Returns:
        Output tensor [batch, out_features]
    """
    if weight.dim() == 2:
        return F.linear(x, weight, bias)
    # Batched: weight [batch, out, in], x [batch, in]
    # y = bmm(x.unsqueeze(1), W.transpose(1,2)).squeeze(1)
    out = torch.bmm(x.unsqueeze(1), weight.transpose(1, 2)).squeeze(1)
    if bias is not None:
        out = out + bias
    return out
