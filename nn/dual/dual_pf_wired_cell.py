"""Dual particle filter Wired CfC cell."""

from typing import Tuple, Optional, Union, Dict, List

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

from .base import DualPFCell
from ..utils import AlphaMode, NoiseType, batched_linear
from ..observation import ObservationModel


class DualPFWiredCfCCell(DualPFCell):
    """Dual particle filter Wired CfC cell.

    Maintains K joint (state, parameter) particles with NCP wiring.

    Example:
        >>> from ncps.wirings import AutoNCP
        >>> wiring = AutoNCP(units=64, output_size=10)
        >>> cell = DualPFWiredCfCCell(
        ...     wiring=wiring,
        ...     input_size=20,
        ...     n_particles=16,
        ... )
    """

    def __init__(
        self,
        wiring,
        input_size: Optional[int] = None,
        n_particles: int = 16,
        mode: str = "default",
        # Parameter tracking
        tracked_params: Optional[List[str]] = None,
        param_evolution_noise: float = 0.01,
        # State noise
        state_noise_type: Union[str, NoiseType] = "time_scaled",
        state_noise_init: float = 0.1,
        # Resampling
        alpha_mode: Union[str, AlphaMode] = "adaptive",
        alpha_init: float = 0.5,
        resample_threshold: float = 0.5,
        use_rao_blackwell: bool = True,
        # Observation model
        observation_model: Optional[ObservationModel] = None,
    ):
        if input_size is not None:
            wiring.build(input_size)
        if not wiring.is_built():
            raise ValueError("Wiring not built.")

        self._wiring = wiring
        self._mode = mode

        # Default: track last layer
        if tracked_params is None:
            last_layer = wiring.num_layers - 1
            tracked_params = [
                f'layer_{last_layer}_ff1_weight',
                f'layer_{last_layer}_ff1_bias',
            ]

        super().__init__(
            input_size=wiring.input_dim,
            hidden_size=wiring.units,
            n_particles=n_particles,
            tracked_params=tracked_params,
            state_noise_type=state_noise_type,
            state_noise_init=state_noise_init,
            param_evolution_noise=param_evolution_noise,
            alpha_mode=alpha_mode,
            alpha_init=alpha_init,
            resample_threshold=resample_threshold,
            use_rao_blackwell=use_rao_blackwell,
            observation_model=observation_model,
        )

    @property
    def layer_sizes(self) -> List[int]:
        return [
            len(self._wiring.get_neurons_of_layer(i))
            for i in range(self._wiring.num_layers)
        ]

    @property
    def num_layers(self):
        return self._wiring.num_layers

    def _create_base_cell(self):
        """Create wired CfC layers."""
        in_features = self._wiring.input_dim

        for l in range(self._wiring.num_layers):
            hidden_units = self._wiring.get_neurons_of_layer(l)
            layer_size = len(hidden_units)

            # Sparsity mask
            if l == 0:
                input_sparsity = self._wiring.sensory_adjacency_matrix[:, hidden_units]
            else:
                prev_neurons = self._wiring.get_neurons_of_layer(l - 1)
                input_sparsity = self._wiring.adjacency_matrix[:, hidden_units]
                input_sparsity = input_sparsity[prev_neurons, :]

            input_sparsity = np.concatenate([
                input_sparsity,
                np.ones((layer_size, layer_size)),
            ], axis=0)

            self.register_buffer(
                f"sparsity_mask_{l}",
                torch.from_numpy(np.abs(input_sparsity.T).astype(np.float32))
            )

            cat_shape = in_features + layer_size

            ff1 = nn.Linear(cat_shape, layer_size)
            setattr(self, f"layer_{l}_ff1", ff1)

            if self._mode == "pure":
                setattr(self, f"layer_{l}_w_tau", nn.Parameter(torch.zeros(1, layer_size)))
                setattr(self, f"layer_{l}_A", nn.Parameter(torch.ones(1, layer_size)))
            else:
                setattr(self, f"layer_{l}_ff2", nn.Linear(cat_shape, layer_size))
                setattr(self, f"layer_{l}_time_a", nn.Linear(cat_shape, layer_size))
                setattr(self, f"layer_{l}_time_b", nn.Linear(cat_shape, layer_size))

            in_features = layer_size

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if param.dim() == 2 and param.requires_grad:
                nn.init.xavier_uniform_(param)

    def _get_trackable_params(self) -> Dict[str, Tensor]:
        params = {}
        for l in range(self.num_layers):
            ff1 = getattr(self, f"layer_{l}_ff1")
            params[f"layer_{l}_ff1_weight"] = ff1.weight
            params[f"layer_{l}_ff1_bias"] = ff1.bias

            if self._mode == "pure":
                params[f"layer_{l}_w_tau"] = getattr(self, f"layer_{l}_w_tau")
                params[f"layer_{l}_A"] = getattr(self, f"layer_{l}_A")
            else:
                ff2 = getattr(self, f"layer_{l}_ff2")
                time_a = getattr(self, f"layer_{l}_time_a")
                time_b = getattr(self, f"layer_{l}_time_b")
                params[f"layer_{l}_ff2_weight"] = ff2.weight
                params[f"layer_{l}_ff2_bias"] = ff2.bias
                params[f"layer_{l}_time_a_weight"] = time_a.weight
                params[f"layer_{l}_time_a_bias"] = time_a.bias
                params[f"layer_{l}_time_b_weight"] = time_b.weight
                params[f"layer_{l}_time_b_bias"] = time_b.bias
        return params

    def _forward_with_params(
        self,
        input: Tensor,
        state: Tensor,
        params: Dict[str, Tensor],
        timespans: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward with specific parameters."""
        if timespans is None:
            ts = torch.ones(input.shape[0], 1, device=input.device, dtype=input.dtype)
        elif timespans.dim() == 0:
            ts = timespans.expand(input.shape[0], 1)
        elif timespans.dim() == 1:
            ts = timespans.unsqueeze(-1)
        else:
            ts = timespans

        h_state = torch.split(state, self.layer_sizes, dim=1)
        new_h_state = []
        layer_input = input

        for l in range(self.num_layers):
            hx = h_state[l]
            x = torch.cat([layer_input, hx], dim=1)

            sparsity_mask = getattr(self, f"sparsity_mask_{l}")

            ff1_weight = params.get(f"layer_{l}_ff1_weight", getattr(self, f"layer_{l}_ff1").weight)
            ff1_bias = params.get(f"layer_{l}_ff1_bias", getattr(self, f"layer_{l}_ff1").bias)
            masked_weight = ff1_weight * sparsity_mask
            ff1 = batched_linear(x, masked_weight, ff1_bias)

            if self._mode == "pure":
                w_tau = params.get(f"layer_{l}_w_tau", getattr(self, f"layer_{l}_w_tau"))
                A = params.get(f"layer_{l}_A", getattr(self, f"layer_{l}_A"))
                new_h = -A * torch.exp(-ts * (torch.abs(w_tau) + torch.abs(ff1))) * ff1 + A
            else:
                ff2_weight = params.get(f"layer_{l}_ff2_weight", getattr(self, f"layer_{l}_ff2").weight)
                ff2_bias = params.get(f"layer_{l}_ff2_bias", getattr(self, f"layer_{l}_ff2").bias)
                masked_ff2 = ff2_weight * sparsity_mask
                ff2 = batched_linear(x, masked_ff2, ff2_bias)

                ff1 = self.tanh(ff1)
                ff2 = self.tanh(ff2)

                time_a_weight = params.get(f"layer_{l}_time_a_weight", getattr(self, f"layer_{l}_time_a").weight)
                time_a_bias = params.get(f"layer_{l}_time_a_bias", getattr(self, f"layer_{l}_time_a").bias)
                time_b_weight = params.get(f"layer_{l}_time_b_weight", getattr(self, f"layer_{l}_time_b").weight)
                time_b_bias = params.get(f"layer_{l}_time_b_bias", getattr(self, f"layer_{l}_time_b").bias)

                t_a = batched_linear(x, time_a_weight, time_a_bias)
                t_b = batched_linear(x, time_b_weight, time_b_bias)
                t_interp = self.sigmoid(t_a * ts + t_b)

                if self._mode == "no_gate":
                    new_h = ff1 + t_interp * ff2
                else:
                    new_h = ff1 * (1.0 - t_interp) + t_interp * ff2

            new_h_state.append(new_h)
            layer_input = new_h

        new_state = torch.cat(new_h_state, dim=1)
        return new_h, new_state
