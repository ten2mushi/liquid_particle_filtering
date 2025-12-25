"""SDE Wired LTC cell - Stochastic ODE with NCP wiring structure.

Extends SDELTCCell to use NCP-style wiring with multiple layers.
"""

from typing import Tuple, Optional, Union, List

import torch
import torch.nn as nn
from torch import Tensor

from .sde_ltc_cell import SDELTCCell
from .diffusion import DiffusionType, create_diffusion
from ..utils import AlphaMode
from ..observation import ObservationModel


class SDEWiredLTCCell(nn.Module):
    """SDE Wired LTC cell with NCP architecture.

    Combines SDELTCCell with NCP wiring structure, where multiple
    LTC layers are connected according to a sparse wiring pattern.
    Each layer uses SDE dynamics with per-unfold noise injection.

    This provides:
    1. NCP-style sparse connectivity
    2. SDE-based uncertainty propagation
    3. Multi-layer hierarchical dynamics

    Example:
        >>> from ncps.wirings import AutoNCP
        >>> wiring = AutoNCP(units=64, output_size=10)
        >>> cell = SDEWiredLTCCell(wiring=wiring, in_features=20, n_particles=32)
        >>> x = torch.randn(8, 20)
        >>> output, (particles, log_weights) = cell(x)
    """

    def __init__(
        self,
        wiring,
        in_features: Optional[int] = None,
        n_particles: int = 32,
        # LTC parameters
        ode_unfolds: int = 6,
        epsilon: float = 1e-8,
        # SDE configuration
        diffusion_type: Union[str, DiffusionType] = "learned",
        sigma_init: float = 0.1,
        solver: str = "euler_maruyama",
        # Resampling configuration
        alpha_mode: Union[str, AlphaMode] = "adaptive",
        alpha_init: float = 0.5,
        resample_threshold: float = 0.5,
        # Observation model
        observation_model: Optional[ObservationModel] = None,
    ):
        """Initialize SDE Wired LTC cell.

        Args:
            wiring: NCP wiring configuration
            in_features: Input dimension
            n_particles: Number of particles K
            ode_unfolds: Number of ODE/SDE unfolds per step
            epsilon: Numerical stability constant
            diffusion_type: Type of diffusion coefficient
            sigma_init: Initial diffusion magnitude
            solver: SDE solver type
            alpha_mode: Soft resampling alpha mode
            alpha_init: Initial alpha value
            resample_threshold: ESS threshold for resampling
            observation_model: Model for p(y|h)
        """
        super().__init__()

        if in_features is not None:
            wiring.build(in_features)
        if not wiring.is_built():
            raise ValueError("Wiring not built.")

        self._wiring = wiring
        self.n_particles = n_particles
        self.n_unfolds = ode_unfolds
        self._epsilon = epsilon

        # Create per-layer diffusion coefficients
        self._layers = nn.ModuleList()
        self._diffusions = nn.ModuleList()

        in_size = wiring.input_dim
        for l in range(wiring.num_layers):
            layer_neurons = wiring.get_neurons_of_layer(l)
            layer_size = len(layer_neurons)

            # Diffusion for this layer
            diffusion = create_diffusion(
                diffusion_type=diffusion_type,
                hidden_size=layer_size,
                sigma_init=sigma_init,
            )
            self._diffusions.append(diffusion)

            # Layer parameters
            layer = _SDEWiredLTCLayer(
                in_features=in_size,
                hidden_size=layer_size,
                wiring=wiring,
                layer_idx=l,
                epsilon=epsilon,
            )
            self._layers.append(layer)
            in_size = layer_size

        # Resampler (shared across layers)
        from ..utils import SoftResampler
        self.resampler = SoftResampler(
            n_particles=n_particles,
            alpha_mode=alpha_mode,
            alpha_init=alpha_init,
            resample_threshold=resample_threshold,
        )

        self.observation_model = observation_model

    @property
    def state_size(self):
        return self._wiring.units

    @property
    def layer_sizes(self) -> List[int]:
        return [
            len(self._wiring.get_neurons_of_layer(i))
            for i in range(self._wiring.num_layers)
        ]

    @property
    def num_layers(self):
        return self._wiring.num_layers

    @property
    def sensory_size(self):
        return self._wiring.input_dim

    @property
    def motor_size(self):
        return self._wiring.output_dim

    @property
    def output_size(self):
        return self.motor_size

    def init_hidden(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Initialize particles and log weights."""
        import math

        particles = torch.randn(
            batch_size, self.n_particles, self.state_size,
            device=device, dtype=dtype,
        ) * 0.1

        log_weights = torch.full(
            (batch_size, self.n_particles),
            -math.log(self.n_particles),
            device=device, dtype=dtype,
        )

        return particles, log_weights

    def forward(
        self,
        input: Tensor,
        hx: Optional[Tuple[Tensor, Tensor]] = None,
        timespans: Optional[Tensor] = None,
        observation: Optional[Tensor] = None,
        return_all_particles: bool = False,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass through wired SDE LTC.

        Args:
            input: Input tensor [batch, input_size]
            hx: (particles, log_weights)
            timespans: Time deltas
            observation: Optional observation
            return_all_particles: Return all particle outputs

        Returns:
            output: Cell output
            (particles, log_weights): Updated state
        """
        batch = input.shape[0]
        device = input.device
        dtype = input.dtype

        # Initialize if needed
        if hx is None:
            particles, log_weights = self.init_hidden(batch, device, dtype)
        else:
            particles, log_weights = hx

        K = self.n_particles

        # Default elapsed time
        if timespans is None:
            elapsed = torch.ones(batch, 1, device=device, dtype=dtype)
        elif timespans.dim() == 0:
            elapsed = timespans.expand(batch, 1)
        elif timespans.dim() == 1:
            elapsed = timespans.unsqueeze(-1)
        else:
            elapsed = timespans

        # Split state by layer
        state_splits = torch.split(particles, self.layer_sizes, dim=2)

        new_states = []
        layer_input = input.unsqueeze(1).expand(-1, K, -1)  # [batch, K, in]

        for l in range(self.num_layers):
            layer_state = state_splits[l]  # [batch, K, layer_size]

            # Process each particle through this layer with SDE
            new_layer_state = self._propagate_layer(
                layer_idx=l,
                input=layer_input,
                state=layer_state,
                elapsed_time=elapsed,
            )

            new_states.append(new_layer_state)
            layer_input = new_layer_state  # Output becomes input to next layer

        # Concatenate new states
        new_particles = torch.cat(new_states, dim=2)  # [batch, K, state_size]

        # Update weights if observation provided
        weights_normalized = False
        if observation is not None and self.observation_model is not None:
            from ..utils import log_weight_update
            log_likelihoods = self.observation_model.log_likelihood(
                new_particles, observation
            )
            log_weights = log_weight_update(log_weights, log_likelihoods)
            weights_normalized = True  # log_weight_update normalizes by default

        # Resample (skip ESS normalization if already normalized)
        new_particles, log_weights = self.resampler(
            new_particles, log_weights, already_normalized=weights_normalized
        )

        # Compute output
        if return_all_particles:
            output = layer_input  # Last layer output [batch, K, motor_size]
        else:
            from ..utils import normalize_log_weights
            weights = torch.exp(normalize_log_weights(log_weights))
            weights = weights.unsqueeze(-1)
            output = (weights * layer_input).sum(dim=1)

        return output, (new_particles, log_weights)

    def _propagate_layer(
        self,
        layer_idx: int,
        input: Tensor,
        state: Tensor,
        elapsed_time: Tensor,
    ) -> Tensor:
        """Propagate particles through one layer with SDE dynamics.

        Args:
            layer_idx: Index of layer
            input: Layer input [batch, K, in_size]
            state: Layer state [batch, K, layer_size]
            elapsed_time: Time step [batch, 1]

        Returns:
            new_state: Updated layer state [batch, K, layer_size]
        """
        batch, K, layer_size = state.shape

        # Flatten for processing
        input_flat = input.reshape(batch * K, -1)
        state_flat = state.reshape(batch * K, layer_size)
        elapsed_flat = elapsed_time.unsqueeze(1).expand(-1, K, -1).reshape(batch * K, -1)

        layer = self._layers[layer_idx]
        diffusion = self._diffusions[layer_idx]

        # Compute dt per unfold
        dt = elapsed_flat / self.n_unfolds

        # SDE integration loop
        v_pre = state_flat
        for step in range(self.n_unfolds):
            # Compute drift
            drift = layer.compute_drift(v_pre, input_flat)

            # Compute diffusion
            g = diffusion(v_pre)

            # Generate noise
            sqrt_dt = torch.sqrt(dt.clamp(min=1e-8))
            dW = sqrt_dt * torch.randn_like(v_pre)

            # Euler-Maruyama step
            v_pre = v_pre + drift * dt + g * dW

        return v_pre.reshape(batch, K, layer_size)


class _SDEWiredLTCLayer(nn.Module):
    """Single layer of wired SDE LTC."""

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        wiring,
        layer_idx: int,
        epsilon: float = 1e-8,
    ):
        super().__init__()

        self.in_features = in_features
        self.hidden_size = hidden_size
        self._epsilon = epsilon

        # Get neurons for this layer
        layer_neurons = wiring.get_neurons_of_layer(layer_idx)

        # Simplified LTC parameters for single layer
        self.gleak = nn.Parameter(torch.rand(hidden_size) * 0.999 + 0.001)
        self.vleak = nn.Parameter(torch.rand(hidden_size) * 0.4 - 0.2)
        self.cm = nn.Parameter(torch.rand(hidden_size) * 0.2 + 0.4)

        # Synapse from input
        self.input_w = nn.Parameter(torch.rand(in_features, hidden_size) * 0.999 + 0.001)
        self.input_sigma = nn.Parameter(torch.rand(in_features, hidden_size) * 5 + 3)
        self.input_mu = nn.Parameter(torch.rand(in_features, hidden_size) * 0.5 + 0.3)
        self.input_erev = nn.Parameter(torch.randn(in_features, hidden_size))

        # Recurrent synapses
        self.w = nn.Parameter(torch.rand(hidden_size, hidden_size) * 0.999 + 0.001)
        self.sigma = nn.Parameter(torch.rand(hidden_size, hidden_size) * 5 + 3)
        self.mu = nn.Parameter(torch.rand(hidden_size, hidden_size) * 0.5 + 0.3)
        self.erev = nn.Parameter(torch.randn(hidden_size, hidden_size))

    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = v_pre.unsqueeze(-1)
        mues = v_pre - mu
        return torch.sigmoid(sigma * mues)

    def compute_drift(self, state: Tensor, input: Tensor) -> Tensor:
        """Compute LTC drift for this layer."""
        v_pre = state

        # Input contributions
        input_activation = torch.relu(self.input_w) * self._sigmoid(
            input, self.input_mu, self.input_sigma
        )
        input_rev = input_activation * self.input_erev
        w_num_input = input_rev.sum(dim=1)
        w_den_input = input_activation.sum(dim=1)

        # Recurrent contributions
        w_activation = torch.relu(self.w) * self._sigmoid(v_pre, self.mu, self.sigma)
        rev_activation = w_activation * self.erev
        w_num = rev_activation.sum(dim=1) + w_num_input
        w_den = w_activation.sum(dim=1) + w_den_input

        # Drift
        gleak = torch.relu(self.gleak)
        cm = torch.relu(self.cm) + self._epsilon

        numerator = gleak * self.vleak + w_num - (gleak + w_den) * v_pre
        drift = numerator / cm

        return drift
