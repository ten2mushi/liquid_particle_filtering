"""Parameter-level particle filter CfC cell."""

from typing import Tuple, Optional, Union, Dict, List

import torch
import torch.nn as nn
from torch import Tensor

from .base import ParamLevelPFCell
from ..utils import AlphaMode
from ..observation import ObservationModel


class LeCun(nn.Module):
    """LeCun tanh activation."""
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 1.7159 * self.tanh(0.666 * x)


class ParamPFCfCCell(ParamLevelPFCell):
    """Parameter-level particle filter CfC cell.

    Maintains K particles over CfC parameters, enabling uncertainty
    estimation over the model's learned dynamics.

    Trackable parameter groups:
    - backbone: Backbone network weights
    - ff1, ff2: Feature projection weights
    - time_a, time_b: Time interpolation weights

    Example:
        >>> cell = ParamPFCfCCell(
        ...     input_size=20,
        ...     hidden_size=64,
        ...     n_particles=8,
        ...     tracked_params=['ff1', 'ff2'],  # Track output layers only
        ... )
        >>> x = torch.randn(8, 20)
        >>> output, (state, params, weights) = cell(x)
    """

    # Default trackable parameter groups
    TRACKABLE_PARAMS = ['ff1_weight', 'ff1_bias', 'ff2_weight', 'ff2_bias',
                        'time_a_weight', 'time_a_bias', 'time_b_weight', 'time_b_bias']

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_particles: int = 8,
        # CfC specific
        mode: str = "default",
        backbone_activation: str = "lecun_tanh",
        backbone_units: int = 128,
        backbone_layers: int = 1,
        backbone_dropout: float = 0.0,
        # Parameter tracking
        tracked_params: Optional[List[str]] = None,
        param_evolution_noise: float = 0.01,
        # Resampling
        alpha_mode: Union[str, AlphaMode] = "adaptive",
        alpha_init: float = 0.5,
        resample_threshold: float = 0.5,
        # Observation model
        observation_model: Optional[ObservationModel] = None,
    ):
        """Initialize parameter-level PF CfC cell.

        Args:
            input_size: Dimension of input
            hidden_size: Dimension of hidden state
            n_particles: Number of particles K
            mode: CfC mode ('default', 'pure', 'no_gate')
            backbone_activation: Backbone activation function
            backbone_units: Backbone hidden units
            backbone_layers: Number of backbone layers
            backbone_dropout: Backbone dropout
            tracked_params: Parameter names to track (None = defaults)
            param_evolution_noise: Evolution noise std
            alpha_mode: Soft resampling alpha mode
            alpha_init: Initial alpha value
            resample_threshold: ESS threshold for resampling
            observation_model: Model for p(y|h)
        """
        self._mode = mode
        self._backbone_activation = backbone_activation
        self._backbone_units = backbone_units
        self._backbone_layers = backbone_layers
        self._backbone_dropout = backbone_dropout

        # Set default tracked params if not specified
        if tracked_params is None:
            if mode == "pure":
                tracked_params = ['ff1_weight', 'ff1_bias']
            else:
                tracked_params = ['ff1_weight', 'ff1_bias', 'ff2_weight', 'ff2_bias']

        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            n_particles=n_particles,
            tracked_params=tracked_params,
            param_evolution_noise=param_evolution_noise,
            alpha_mode=alpha_mode,
            alpha_init=alpha_init,
            resample_threshold=resample_threshold,
            observation_model=observation_model,
        )

    def _create_base_cell(self):
        """Create CfC cell components."""
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

        # Initialize
        self._init_weights()

    def _init_weights(self):
        for w in self.parameters():
            if w.dim() == 2 and w.requires_grad:
                nn.init.xavier_uniform_(w)

    def _get_trackable_params(self) -> Dict[str, Tensor]:
        """Return dict of trackable parameters."""
        params = {
            'ff1_weight': self.ff1.weight,
            'ff1_bias': self.ff1.bias,
        }

        if self._mode == "pure":
            params['w_tau'] = self.w_tau
            params['A'] = self.A
        else:
            params.update({
                'ff2_weight': self.ff2.weight,
                'ff2_bias': self.ff2.bias,
                'time_a_weight': self.time_a.weight,
                'time_a_bias': self.time_a.bias,
                'time_b_weight': self.time_b.weight,
                'time_b_bias': self.time_b.bias,
            })

        return params

    def _forward_with_params(
        self,
        input: Tensor,
        state: Tensor,
        params: Dict[str, Tensor],
        timespans: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward with specific parameters."""
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

        # Process through backbone (shared, not tracked)
        if self.backbone is not None:
            x = self.backbone(x)

        # Use particle-specific parameters for ff1
        ff1_weight = params.get('ff1_weight', self.ff1.weight)
        ff1_bias = params.get('ff1_bias', self.ff1.bias)

        # Batched linear: need to handle per-sample weights
        # x: [batch * K, cat_shape], ff1_weight: [batch * K, hidden, cat_shape]
        if ff1_weight.dim() == 3:
            # Per-particle weights
            ff1 = torch.bmm(ff1_weight, x.unsqueeze(-1)).squeeze(-1)
            if ff1_bias.dim() == 2:
                ff1 = ff1 + ff1_bias
            else:
                ff1 = ff1 + ff1_bias.unsqueeze(0)
        else:
            ff1 = torch.nn.functional.linear(x, ff1_weight, ff1_bias)

        if self._mode == "pure":
            w_tau = params.get('w_tau', self.w_tau)
            A = params.get('A', self.A)
            new_state = (
                -A * torch.exp(-ts * (torch.abs(w_tau) + torch.abs(ff1))) * ff1 + A
            )
        else:
            # ff2
            ff2_weight = params.get('ff2_weight', self.ff2.weight)
            ff2_bias = params.get('ff2_bias', self.ff2.bias)

            if ff2_weight.dim() == 3:
                ff2 = torch.bmm(ff2_weight, x.unsqueeze(-1)).squeeze(-1)
                if ff2_bias.dim() == 2:
                    ff2 = ff2 + ff2_bias
                else:
                    ff2 = ff2 + ff2_bias.unsqueeze(0)
            else:
                ff2 = torch.nn.functional.linear(x, ff2_weight, ff2_bias)

            ff1 = self.tanh(ff1)
            ff2 = self.tanh(ff2)

            # time_a, time_b
            time_a_weight = params.get('time_a_weight', self.time_a.weight)
            time_a_bias = params.get('time_a_bias', self.time_a.bias)
            time_b_weight = params.get('time_b_weight', self.time_b.weight)
            time_b_bias = params.get('time_b_bias', self.time_b.bias)

            if time_a_weight.dim() == 3:
                t_a = torch.bmm(time_a_weight, x.unsqueeze(-1)).squeeze(-1)
                t_b = torch.bmm(time_b_weight, x.unsqueeze(-1)).squeeze(-1)
                if time_a_bias.dim() == 2:
                    t_a = t_a + time_a_bias
                    t_b = t_b + time_b_bias
                else:
                    t_a = t_a + time_a_bias.unsqueeze(0)
                    t_b = t_b + time_b_bias.unsqueeze(0)
            else:
                t_a = torch.nn.functional.linear(x, time_a_weight, time_a_bias)
                t_b = torch.nn.functional.linear(x, time_b_weight, time_b_bias)

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
