"""Sequence wrapper for particle filter LTC."""

from typing import Tuple, Optional, Union, List, Literal

import torch
import torch.nn as nn
from torch import Tensor

from ..state_level import PFLTCCell
from ..param_level import ParamPFLTCCell
from ..dual import DualPFLTCCell
from ..sde import SDELTCCell
from ..utils import AlphaMode, NoiseType
from ..observation import ObservationModel


class PFLTC(nn.Module):
    """Particle filter LTC sequence model.

    High-level wrapper that processes sequences through particle filter
    LTC cells. Supports all four approaches (state, param, dual, SDE).

    Similar API to ncps.torch.LTC but with particle filter capabilities.

    Example:
        >>> from ncps.wirings import AutoNCP
        >>> wiring = AutoNCP(units=64, output_size=10)
        >>> model = PFLTC(
        ...     wiring=wiring,
        ...     in_features=20,
        ...     n_particles=32,
        ...     approach='state',
        ... )
        >>> x = torch.randn(8, 100, 20)
        >>> outputs, (particles, log_weights) = model(x)
    """

    def __init__(
        self,
        wiring,
        in_features: Optional[int] = None,
        n_particles: int = 32,
        approach: Literal["state", "param", "dual", "sde"] = "state",
        # LTC parameters
        input_mapping: str = "affine",
        output_mapping: str = "affine",
        ode_unfolds: int = 6,
        epsilon: float = 1e-8,
        implicit_param_constraints: bool = False,
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
        # SDE specific
        diffusion_type: str = "learned",
        solver: str = "euler_maruyama",
        state_bounds: Optional[float] = 5.0,
        clamp_mode: str = "hard",
        diffusion_scale_by_dim: bool = False,
        # Output options
        return_sequences: bool = True,
        return_state: bool = True,
        # Observation model
        observation_model: Optional[ObservationModel] = None,
        # Performance options
        torch_compile: bool = False,
        compile_mode: str = "reduce-overhead",
    ):
        """Initialize PFLTC model.

        Args:
            wiring: NCP wiring configuration
            in_features: Input dimension
            n_particles: Number of particles K
            approach: Which PF approach ('state', 'param', 'dual', 'sde')
            input_mapping: Input mapping type
            output_mapping: Output mapping type
            ode_unfolds: Number of ODE unfolds
            epsilon: Numerical stability constant
            implicit_param_constraints: Use softplus for positive params
            noise_type: Type of noise injection
            noise_init: Initial noise scale
            noise_learnable: Whether noise parameters are learnable
            alpha_mode: Soft resampling alpha mode
            alpha_init: Initial alpha value
            resample_threshold: ESS threshold for resampling
            tracked_params: Parameters to track (param/dual approaches)
            param_evolution_noise: Parameter evolution noise std
            diffusion_type: Diffusion type for SDE approach
            solver: SDE solver type
            state_bounds: Maximum absolute value for state clamping (SDE)
            clamp_mode: Clamping mode for SDE ('hard' or 'soft')
            diffusion_scale_by_dim: Scale diffusion by 1/sqrt(hidden_size)
            return_sequences: Return output at each timestep
            return_state: Return final hidden state
            observation_model: Model for p(y|h)
            torch_compile: If True, compile the cell for faster execution (PyTorch 2.0+)
            compile_mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
        """
        super().__init__()

        self.approach = approach
        self.return_sequences = return_sequences
        self.return_state = return_state

        # Build wiring
        if in_features is not None:
            wiring.build(in_features)

        self.wiring = wiring
        self.input_size = wiring.input_dim
        self.hidden_size = wiring.units
        self.output_size = wiring.output_dim
        self.n_particles = n_particles

        # Create appropriate cell
        if approach == "state":
            self.cell = PFLTCCell(
                wiring=wiring,
                n_particles=n_particles,
                input_mapping=input_mapping,
                output_mapping=output_mapping,
                ode_unfolds=ode_unfolds,
                epsilon=epsilon,
                implicit_param_constraints=implicit_param_constraints,
                noise_type=noise_type,
                noise_init=noise_init,
                noise_learnable=noise_learnable,
                alpha_mode=alpha_mode,
                alpha_init=alpha_init,
                resample_threshold=resample_threshold,
                observation_model=observation_model,
            )
        elif approach == "param":
            self.cell = ParamPFLTCCell(
                wiring=wiring,
                n_particles=n_particles,
                input_mapping=input_mapping,
                output_mapping=output_mapping,
                ode_unfolds=ode_unfolds,
                epsilon=epsilon,
                tracked_params=tracked_params,
                param_evolution_noise=param_evolution_noise,
                alpha_mode=alpha_mode,
                alpha_init=alpha_init,
                resample_threshold=resample_threshold,
                observation_model=observation_model,
            )
        elif approach == "dual":
            self.cell = DualPFLTCCell(
                wiring=wiring,
                n_particles=n_particles,
                input_mapping=input_mapping,
                output_mapping=output_mapping,
                ode_unfolds=ode_unfolds,
                epsilon=epsilon,
                tracked_params=tracked_params,
                param_evolution_noise=param_evolution_noise,
                state_noise_type=noise_type,
                state_noise_init=noise_init,
                alpha_mode=alpha_mode,
                alpha_init=alpha_init,
                resample_threshold=resample_threshold,
                observation_model=observation_model,
            )
        elif approach == "sde":
            self.cell = SDELTCCell(
                wiring=wiring,
                n_particles=n_particles,
                input_mapping=input_mapping,
                output_mapping=output_mapping,
                ode_unfolds=ode_unfolds,
                epsilon=epsilon,
                implicit_param_constraints=implicit_param_constraints,
                diffusion_type=diffusion_type,
                sigma_init=noise_init,
                solver=solver,
                alpha_mode=alpha_mode,
                alpha_init=alpha_init,
                resample_threshold=resample_threshold,
                observation_model=observation_model,
                state_bounds=state_bounds,
                clamp_mode=clamp_mode,
                diffusion_scale_by_dim=diffusion_scale_by_dim,
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
            hx: Initial hidden state
            timespans: Optional time deltas [batch, seq_len, 1]
            observations: Optional observations [batch, seq_len, obs_size]

        Returns:
            outputs: Output tensor
            hx: Final hidden state
        """
        batch_size, seq_len, _ = input.shape
        device = input.device
        dtype = input.dtype

        # Pre-allocate output tensor to avoid Python list append overhead
        if self.return_sequences:
            outputs = torch.empty(
                batch_size, seq_len, self.output_size,
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
            f"output_size={self.output_size}, n_particles={self.n_particles}, "
            f"approach={self.approach}"
        )
