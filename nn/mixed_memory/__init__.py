"""Mixed memory architectures combining LSTM with particle filtering.

Provides wrappers that combine deterministic LSTM memory with
probabilistic particle filter inference. Works with any PF cell
from approaches A, B, C, or D.

Two architectures:
- LSTMAugmentedPFCell: Sequential (LSTM -> PF)
- MixedMemoryPFCell: Parallel (LSTM || PF -> combine)
"""

from .lstm_augmentation import LSTMAugmentedPFCell, MixedMemoryPFCell

__all__ = [
    "LSTMAugmentedPFCell",
    "MixedMemoryPFCell",
]
