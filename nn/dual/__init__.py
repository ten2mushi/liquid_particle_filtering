"""Dual particle filter cells (Approach C).

Dual particle filtering maintains K joint (state, parameter) particles,
combining the benefits of both state-level (A) and parameter-level (B)
approaches.

Advantages:
- Captures state-parameter correlations
- Complete uncertainty quantification
- Handles non-stationary systems
- Rao-Blackwellization for variance reduction

Provides:
- DualPFCfCCell: Dual PF over CfC
- DualPFLTCCell: Dual PF over LTC
- DualPFWiredCfCCell: Dual PF over wired CfC
- RaoBlackwellEstimator: Variance-reduced estimation utilities
"""

from .base import DualPFCell
from .dual_pf_cfc_cell import DualPFCfCCell
from .dual_pf_ltc_cell import DualPFLTCCell
from .dual_pf_wired_cell import DualPFWiredCfCCell
from .rao_blackwell import (
    RaoBlackwellEstimator,
    rao_blackwell_state_estimate,
    rao_blackwell_param_estimate,
    stratified_joint_resample,
)

__all__ = [
    # Base
    "DualPFCell",
    # Cells
    "DualPFCfCCell",
    "DualPFLTCCell",
    "DualPFWiredCfCCell",
    # Rao-Blackwell
    "RaoBlackwellEstimator",
    "rao_blackwell_state_estimate",
    "rao_blackwell_param_estimate",
    "stratified_joint_resample",
]
