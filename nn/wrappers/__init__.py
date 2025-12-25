"""High-level sequence wrappers for particle filter NCPs.

Provides user-friendly interfaces similar to ncps but with
particle filter capabilities. Wrappers handle sequence processing
and support all four approaches (A, B, C, D).

Classes:
- PFCfC: Sequence wrapper for CfC-based particle filters
- PFLTC: Sequence wrapper for LTC-based particle filters
- PFNCP: Sequence wrapper for wired NCP-based particle filters
"""

from .pf_cfc import PFCfC
from .pf_ltc import PFLTC
from .pf_ncp import PFNCP

__all__ = [
    "PFCfC",
    "PFLTC",
    "PFNCP",
]
