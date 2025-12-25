"""Sequence wrapper for particle filter CfC."""

from typing import Tuple, Optional, Union, List, Literal

import torch
import torch.nn as nn
from torch import Tensor

from ..state_level import PFCfCCell
from ..param_level import ParamPFCfCCell
from ..dual import DualPFCfCCell
from ..utils import AlphaMode, NoiseType
from ..observation import ObservationModel


class PFCfC(nn.Module):
    """Particle filter CfC sequence model.

    High-level wrapper that processes sequences through particle filter
    CfC cells. Supports all approaches (state-level, param-level, dual).

    Similar API to ncps.torch.CfC but with particle filter capabilities.

    Example:
        >>> model = PFCfC(
        ...     input_size=20,
        ...     hidden_size=64,
        ...     n_particles=32,
        ...     approach='state',
        ... )
        >>> x = torch.randn(8, 100, 20)  # [batch, seq, input]
        >>> outputs, (particles, log_weights) = model(x)
        >>> # outputs: [8, 100, 64]
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_particles: int = 32,
        approach: Literal["state", "param", "dual"] = "state",
        # CfC parameters
        mode: str = "default",
        backbone_activation: str = "lecun_tanh",
        backbone_units: int = 128,
        backbone_layers: int = 1,
        backbone_dropout: float = 0.0,
        # Particle filter parameters
        noise_type: Union[str, NoiseType] = "time_scaled",
        noise_init: float = 0.1,
        noise_learnable: bool = True,
        alpha_mode: Union[str, AlphaMode] = "adaptive",
        alpha_init: float = 0.5,
        resample_threshold: float = 0.5,
        # Parameter-level specific
        tracked_params: Optional[List[str]] = None,
        param_evolution_noise: float = 0.01,
        # Output options
        return_sequences: bool = True,
        return_state: bool = True,
        # Observation model
        observation_model: Optional[ObservationModel] = None,
        # Performance options
        torch_compile: bool = False,
        compile_mode: str = "reduce-overhead",
    ):
        """Initialize PFCfC model.

        Args:
            input_size: Dimension of input
            hidden_size: Dimension of hidden state
            n_particles: Number of particles K
            approach: Which PF approach ('state', 'param', 'dual')
            mode: CfC mode ('default', 'pure', 'no_gate')
            backbone_activation: Backbone activation function
            backbone_units: Backbone hidden units
            backbone_layers: Number of backbone layers
            backbone_dropout: Backbone dropout
            noise_type: Type of noise injection (state/dual approaches)
            noise_init: Initial noise scale
            noise_learnable: Whether noise parameters are learnable
            alpha_mode: Soft resampling alpha mode
            alpha_init: Initial alpha value
            resample_threshold: ESS threshold for resampling
            tracked_params: Parameters to track (param/dual approaches)
            param_evolution_noise: Parameter evolution noise std
            return_sequences: Return output at each timestep
            return_state: Return final hidden state
            observation_model: Model for p(y|h)
            torch_compile: If True, compile the cell for faster execution (PyTorch 2.0+)
            compile_mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_particles = n_particles
        self.approach = approach
        self.return_sequences = return_sequences
        self.return_state = return_state

        # Create appropriate cell
        if approach == "state":
            self.cell = PFCfCCell(
                input_size=input_size,
                hidden_size=hidden_size,
                n_particles=n_particles,
                mode=mode,
                backbone_activation=backbone_activation,
                backbone_units=backbone_units,
                backbone_layers=backbone_layers,
                backbone_dropout=backbone_dropout,
                noise_type=noise_type,
                noise_init=noise_init,
                noise_learnable=noise_learnable,
                alpha_mode=alpha_mode,
                alpha_init=alpha_init,
                resample_threshold=resample_threshold,
                observation_model=observation_model,
            )
        elif approach == "param":
            self.cell = ParamPFCfCCell(
                input_size=input_size,
                hidden_size=hidden_size,
                n_particles=n_particles,
                mode=mode,
                backbone_activation=backbone_activation,
                backbone_units=backbone_units,
                backbone_layers=backbone_layers,
                backbone_dropout=backbone_dropout,
                tracked_params=tracked_params,
                param_evolution_noise=param_evolution_noise,
                alpha_mode=alpha_mode,
                alpha_init=alpha_init,
                resample_threshold=resample_threshold,
                observation_model=observation_model,
            )
        elif approach == "dual":
            self.cell = DualPFCfCCell(
                input_size=input_size,
                hidden_size=hidden_size,
                n_particles=n_particles,
                mode=mode,
                backbone_activation=backbone_activation,
                backbone_units=backbone_units,
                backbone_layers=backbone_layers,
                backbone_dropout=backbone_dropout,
                tracked_params=tracked_params,
                param_evolution_noise=param_evolution_noise,
                state_noise_type=noise_type,
                state_noise_init=noise_init,
                alpha_mode=alpha_mode,
                alpha_init=alpha_init,
                resample_threshold=resample_threshold,
                observation_model=observation_model,
            )
        else:
            raise ValueError(f"Unknown approach: {approach}")

        # Apply torch.compile if requested (PyTorch 2.0+)
        if torch_compile:
            try:
                self.cell = torch.compile(self.cell, mode=compile_mode)
            except Exception as e:
                import warnings
                warnings.warn(f"torch.compile failed, falling back to eager mode: {e}")

    def forward(
        self,
        input: Tensor,
        hx: Optional[Tuple] = None,
        timespans: Optional[Tensor] = None,
        observations: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tuple]:
        """Forward pass through sequence.

        Args:
            input: Input tensor [batch, seq_len, input_size]
            hx: Initial hidden state (format depends on approach)
            timespans: Optional time deltas [batch, seq_len, 1] or [batch, seq_len]
            observations: Optional observations [batch, seq_len, obs_size]

        Returns:
            outputs: Output tensor [batch, seq_len, hidden_size] if return_sequences
                    else [batch, hidden_size]
            hx: Final hidden state
        """
        batch_size, seq_len, _ = input.shape
        device = input.device
        dtype = input.dtype

        # Pre-allocate output tensor to avoid Python list append overhead
        if self.return_sequences:
            outputs = torch.empty(
                batch_size, seq_len, self.hidden_size,
                device=device, dtype=dtype
            )
        else:
            outputs = None

        state = hx

        # Pre-slice tensors to avoid per-timestep slicing overhead
        # unbind returns a tuple of views, avoiding repeated tensor creation
        input_slices = input.unbind(dim=1)

        if timespans is not None:
            if timespans.dim() == 3:
                ts_slices = timespans.unbind(dim=1)
            else:
                # For 2D timespans [batch, seq], expand to [batch, seq, 1] then unbind
                ts_slices = timespans.unsqueeze(-1).unbind(dim=1)
        else:
            ts_slices = None

        if observations is not None:
            obs_slices = observations.unbind(dim=1)
        else:
            obs_slices = None

        for t in range(seq_len):
            x_t = input_slices[t]
            ts_t = ts_slices[t] if ts_slices is not None else None
            obs_t = obs_slices[t] if obs_slices is not None else None

            # Forward through cell
            output, state = self.cell(
                x_t, state,
                timespans=ts_t,
                observation=obs_t,
            )

            # In-place assignment instead of append
            if self.return_sequences:
                outputs[:, t, :] = output
            elif t == seq_len - 1:
                outputs = output

        if self.return_state:
            return outputs, state
        return outputs

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"n_particles={self.n_particles}, approach={self.approach}"
        )
