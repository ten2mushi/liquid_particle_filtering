"""Spatial heatmap generation from particle filter states.

This module provides components for transforming latent particle filter states
into interpretable spatial probability distributions (heatmaps).

Components:
    - SpatialProjectionHead: Projects latent particles to polar coordinates
    - SoftSpatialRenderer: Renders particles as differentiable Gaussian mixture
    - SpatialPFNCP: Complete wrapper combining PFNCP with spatial output

Example:
    >>> from pfncps.nn.spatial import SpatialPFNCP
    >>> from ncps.wirings import AutoNCP
    >>>
    >>> wiring = AutoNCP(units=32, output_size=2)
    >>> model = SpatialPFNCP(
    ...     wiring=wiring,
    ...     input_size=5,
    ...     n_particles=8,
    ...     map_size=64,
    ...     r_max=150.0,
    ...     env_size=200.0,
    ... )
    >>> x = torch.randn(4, 50, 5)  # [batch, seq, features]
    >>> sensor_pos = torch.rand(4, 50, 3)  # [batch, seq, (x, y, heading)]
    >>> outputs, heatmaps, state = model(x, sensor_positions=sensor_pos)
    >>> # outputs: [4, 50, 2] point estimates
    >>> # heatmaps: [4, 50, 1, 64, 64] spatial probability
"""

from .projection_head import SpatialProjectionHead
from .soft_renderer import SoftSpatialRenderer
from .spatial_pf_ncp import SpatialPFNCP

__all__ = [
    "SpatialProjectionHead",
    "SoftSpatialRenderer",
    "SpatialPFNCP",
]
