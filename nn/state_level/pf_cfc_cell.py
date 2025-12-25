"""Particle filter CfC cell - state-level particle filtering over CfC dynamics."""

from typing import Tuple, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from .base import StateLevelPFCell
from ..utils import AlphaMode, NoiseType
from ..observation import ObservationModel


class LeCun(nn.Module):
    """LeCun tanh activation: 1.7159 * tanh(0.666 * x)"""
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 1.7159 * self.tanh(0.666 * x)


class PFCfCCell(StateLevelPFCell):
    """Particle filter CfC (Closed-form Continuous-time) cell.

    Combines CfC dynamics with state-level particle filtering.
    Maintains K particles over the hidden state, each propagated
    through the CfC dynamics with noise injection.

    Example:
        >>> cell = PFCfCCell(input_size=20, hidden_size=64, n_particles=32)
        >>> x = torch.randn(8, 20)  # [batch, input_size]
        >>> output, (particles, log_weights) = cell(x)
        >>> # output: [8, 64], particles: [8, 32, 64], log_weights: [8, 32]
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_particles: int = 32,
        # CfC specific parameters
        mode: str = "default",
        backbone_activation: str = "lecun_tanh",
        backbone_units: int = 128,
        backbone_layers: int = 1,
        backbone_dropout: float = 0.0,
        # Noise configuration
        noise_type: Union[str, NoiseType] = "time_scaled",
        noise_init: float = 0.1,
        noise_learnable: bool = True,
        # Resampling configuration
        alpha_mode: Union[str, AlphaMode] = "adaptive",
        alpha_init: float = 0.5,
        resample_threshold: float = 0.5,
        # Observation model
        observation_model: Optional[ObservationModel] = None,
    ):
        """Initialize particle filter CfC cell.

        Args:
            input_size: Dimension of input
            hidden_size: Dimension of hidden state
            n_particles: Number of particles K
            mode: CfC mode ('default', 'pure', 'no_gate')
            backbone_activation: Activation for backbone network
            backbone_units: Hidden units in backbone
            backbone_layers: Number of backbone layers
            backbone_dropout: Dropout in backbone
            noise_type: Type of noise injection
            noise_init: Initial noise scale
            noise_learnable: Whether noise is learnable
            alpha_mode: Soft resampling alpha mode
            alpha_init: Initial alpha value
            resample_threshold: ESS threshold for resampling
            observation_model: Model for p(y|h)
        """
        # Store CfC-specific parameters before calling super().__init__
        self._mode = mode
        self._backbone_activation = backbone_activation
        self._backbone_units = backbone_units
        self._backbone_layers = backbone_layers
        self._backbone_dropout = backbone_dropout

        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            n_particles=n_particles,
            noise_type=noise_type,
            noise_init=noise_init,
            noise_learnable=noise_learnable,
            alpha_mode=alpha_mode,
            alpha_init=alpha_init,
            resample_threshold=resample_threshold,
            observation_model=observation_model,
        )

    def _create_base_cell(self):
        """Create the underlying CfC cell."""
        # Activation mapping
        if self._backbone_activation == "silu":
            backbone_act = nn.SiLU
        elif self._backbone_activation == "relu":
            backbone_act = nn.ReLU
        elif self._backbone_activation == "tanh":
            backbone_act = nn.Tanh
        elif self._backbone_activation == "gelu":
            backbone_act = nn.GELU
        elif self._backbone_activation == "lecun_tanh":
            backbone_act = LeCun
        else:
            raise ValueError(f"Unknown activation: {self._backbone_activation}")

        # Build backbone
        self.backbone = None
        if self._backbone_layers > 0:
            layer_list = [
                nn.Linear(self.input_size + self.hidden_size, self._backbone_units),
                backbone_act(),
            ]
            for i in range(1, self._backbone_layers):
                layer_list.append(nn.Linear(self._backbone_units, self._backbone_units))
                layer_list.append(backbone_act())
                if self._backbone_dropout > 0.0:
                    layer_list.append(nn.Dropout(self._backbone_dropout))
            self.backbone = nn.Sequential(*layer_list)

        # CfC layers
        cat_shape = (
            self.hidden_size + self.input_size
            if self._backbone_layers == 0
            else self._backbone_units
        )

        self.ff1 = nn.Linear(cat_shape, self.hidden_size)

        if self._mode == "pure":
            self.w_tau = nn.Parameter(torch.zeros(1, self.hidden_size))
            self.A = nn.Parameter(torch.ones(1, self.hidden_size))
        else:
            self.ff2 = nn.Linear(cat_shape, self.hidden_size)
            self.time_a = nn.Linear(cat_shape, self.hidden_size)
            self.time_b = nn.Linear(cat_shape, self.hidden_size)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for w in self.parameters():
            if w.dim() == 2 and w.requires_grad:
                nn.init.xavier_uniform_(w)

    def _propagate_single(
        self,
        input: Tensor,
        state: Tensor,
        timespans: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Propagate single state through CfC dynamics.

        Args:
            input: Input tensor [batch, input_size]
            state: Hidden state [batch, hidden_size]
            timespans: Time deltas [batch, 1] or scalar

        Returns:
            output: Cell output [batch, hidden_size]
            new_state: New hidden state [batch, hidden_size]
        """
        # Default timespans
        if timespans is None:
            ts = torch.ones(input.shape[0], 1, device=input.device, dtype=input.dtype)
        elif timespans.dim() == 0:
            ts = timespans.expand(input.shape[0], 1)
        elif timespans.dim() == 1:
            ts = timespans.unsqueeze(-1)
        else:
            ts = timespans

        # Concatenate input and state
        x = torch.cat([input, state], dim=1)

        # Process through backbone
        if self.backbone is not None:
            x = self.backbone(x)

        # CfC computation
        ff1 = self.ff1(x)

        if self._mode == "pure":
            # Pure solution mode
            new_state = (
                -self.A
                * torch.exp(-ts * (torch.abs(self.w_tau) + torch.abs(ff1)))
                * ff1
                + self.A
            )
        else:
            # Standard CfC
            ff2 = self.ff2(x)
            ff1 = self.tanh(ff1)
            ff2 = self.tanh(ff2)

            t_a = self.time_a(x)
            t_b = self.time_b(x)
            t_interp = self.sigmoid(t_a * ts + t_b)

            if self._mode == "no_gate":
                new_state = ff1 + t_interp * ff2
            else:
                new_state = ff1 * (1.0 - t_interp) + t_interp * ff2

        return new_state, new_state

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"n_particles={self.n_particles}, mode={self._mode}"
        )
