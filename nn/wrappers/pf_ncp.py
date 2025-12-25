"""Sequence wrapper for particle filter NCP (wired variants)."""

from typing import Tuple, Optional, Union, List, Literal

import torch
import torch.nn as nn
from torch import Tensor

from ..state_level import PFWiredCfCCell
from ..param_level import ParamPFWiredCfCCell
from ..dual import DualPFWiredCfCCell
from ..sde import SDEWiredLTCCell
from ..utils import AlphaMode, NoiseType
from ..observation import ObservationModel


class PFNCP(nn.Module):
    """Particle filter NCP (Neural Circuit Policy) sequence model.

    High-level wrapper for wired particle filter cells with NCP architecture.
    Supports all four approaches (state, param, dual, SDE).

    Similar API to ncps but with particle filter capabilities.

    Example:
        >>> from ncps.wirings import AutoNCP
        >>> wiring = AutoNCP(units=64, output_size=10)
        >>> model = PFNCP(
        ...     wiring=wiring,
        ...     input_size=20,
        ...     n_particles=32,
        ...     approach='state',
        ...     cell_type='cfc',
        ... )
        >>> x = torch.randn(8, 100, 20)
        >>> outputs, (particles, log_weights) = model(x)
    """

    def __init__(
        self,
        wiring,
        input_size: Optional[int] = None,
        n_particles: int = 32,
        approach: Literal["state", "param", "dual", "sde"] = "state",
        cell_type: Literal["cfc", "ltc"] = "cfc",
        # CfC parameters
        mode: str = "default",
        # LTC parameters (for SDE)
        ode_unfolds: int = 6,
        # Particle filter parameters
        noise_type: Union[str, NoiseType] = "time_scaled",
        noise_init: float = 0.1,
        alpha_mode: Union[str, AlphaMode] = "adaptive",
        alpha_init: float = 0.5,
        resample_threshold: float = 0.5,
        # Parameter-level specific
        tracked_params: Optional[List[str]] = None,
        param_evolution_noise: float = 0.01,
        # SDE specific
        diffusion_type: str = "learned",
        solver: str = "euler_maruyama",
        # Output options
        return_sequences: bool = True,
        return_state: bool = True,
        # Observation model
        observation_model: Optional[ObservationModel] = None,
    ):
        """Initialize PFNCP model.

        Args:
            wiring: NCP wiring configuration
            input_size: Input dimension
            n_particles: Number of particles K
            approach: Which PF approach ('state', 'param', 'dual', 'sde')
            cell_type: Cell type ('cfc' or 'ltc' for SDE)
            mode: CfC mode ('default', 'pure', 'no_gate')
            ode_unfolds: Number of ODE unfolds (for SDE)
            noise_type: Type of noise injection
            noise_init: Initial noise scale
            alpha_mode: Soft resampling alpha mode
            alpha_init: Initial alpha value
            resample_threshold: ESS threshold for resampling
            tracked_params: Parameters to track (param/dual approaches)
            param_evolution_noise: Parameter evolution noise std
            diffusion_type: Diffusion type for SDE approach
            solver: SDE solver type
            return_sequences: Return output at each timestep
            return_state: Return final hidden state
            observation_model: Model for p(y|h)
        """
        super().__init__()

        # Build wiring
        if input_size is not None:
            wiring.build(input_size)
        if not wiring.is_built():
            raise ValueError("Wiring not built.")

        self.wiring = wiring
        self.input_size = wiring.input_dim
        self.hidden_size = wiring.units
        self.output_size = wiring.output_dim
        self.n_particles = n_particles
        self.approach = approach
        self.cell_type = cell_type
        self.return_sequences = return_sequences
        self.return_state = return_state

        # Validate approach/cell_type combination
        if approach == "sde" and cell_type != "ltc":
            raise ValueError("SDE approach only supports LTC cell type")

        # Create appropriate cell
        if approach == "state":
            self.cell = PFWiredCfCCell(
                wiring=wiring,
                n_particles=n_particles,
                mode=mode,
                noise_type=noise_type,
                noise_init=noise_init,
                alpha_mode=alpha_mode,
                alpha_init=alpha_init,
                resample_threshold=resample_threshold,
                observation_model=observation_model,
            )
        elif approach == "param":
            self.cell = ParamPFWiredCfCCell(
                wiring=wiring,
                n_particles=n_particles,
                mode=mode,
                tracked_params=tracked_params,
                param_evolution_noise=param_evolution_noise,
                alpha_mode=alpha_mode,
                alpha_init=alpha_init,
                resample_threshold=resample_threshold,
                observation_model=observation_model,
            )
        elif approach == "dual":
            self.cell = DualPFWiredCfCCell(
                wiring=wiring,
                n_particles=n_particles,
                mode=mode,
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
            self.cell = SDEWiredLTCCell(
                wiring=wiring,
                n_particles=n_particles,
                ode_unfolds=ode_unfolds,
                diffusion_type=diffusion_type,
                sigma_init=noise_init,
                solver=solver,
                alpha_mode=alpha_mode,
                alpha_init=alpha_init,
                resample_threshold=resample_threshold,
                observation_model=observation_model,
            )
        else:
            raise ValueError(f"Unknown approach: {approach}")

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

        outputs = []
        state = hx

        for t in range(seq_len):
            x_t = input[:, t, :]

            ts_t = None
            if timespans is not None:
                if timespans.dim() == 3:
                    ts_t = timespans[:, t, :]
                elif timespans.dim() == 2:
                    ts_t = timespans[:, t:t+1]

            obs_t = None
            if observations is not None:
                obs_t = observations[:, t, :]

            output, state = self.cell(
                x_t, state,
                timespans=ts_t,
                observation=obs_t,
            )

            outputs.append(output)

        if self.return_sequences:
            outputs = torch.stack(outputs, dim=1)
        else:
            outputs = outputs[-1]

        if self.return_state:
            return outputs, state
        return outputs

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"output_size={self.output_size}, n_particles={self.n_particles}, "
            f"approach={self.approach}, cell_type={self.cell_type}"
        )
