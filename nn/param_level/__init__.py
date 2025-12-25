"""Parameter-level particle filter cells (Approach B).

Parameter-level particle filtering maintains K particles over model
parameters, enabling uncertainty estimation over learned representations.
This is useful when parameters should adapt during inference.

Key differences from state-level (Approach A):
- Particles are parameter vectors, not hidden states
- Single hidden state, multiple parameter hypotheses
- Parameters evolve with noise during inference
- Typically use fewer particles (K=8) due to higher parameter dimensions

Provides:
- ParamPFCfCCell: Parameter PF over CfC
- ParamPFLTCCell: Parameter PF over LTC
- ParamPFWiredCfCCell: Parameter PF over wired CfC
- ParameterRegistry: Configurable parameter group tracking
"""

from .base import ParamLevelPFCell
from .param_registry import ParameterRegistry, ParameterGroup
from .param_pf_cfc_cell import ParamPFCfCCell
from .param_pf_ltc_cell import ParamPFLTCCell
from .param_pf_wired_cell import ParamPFWiredCfCCell

__all__ = [
    # Base
    "ParamLevelPFCell",
    # Registry
    "ParameterRegistry",
    "ParameterGroup",
    # Cells
    "ParamPFCfCCell",
    "ParamPFLTCCell",
    "ParamPFWiredCfCCell",
]
