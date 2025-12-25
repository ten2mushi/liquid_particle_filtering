"""Parameter registry for parameter-level particle filtering.

Manages which parameters are tracked with particles and provides
utilities for parameter evolution and particle management.
"""

from typing import Dict, List, Optional, Tuple, Set, Union
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class ParameterGroup:
    """Configuration for a group of trackable parameters."""
    name: str
    shape: Tuple[int, ...]
    evolution_noise: float = 0.01
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class ParameterRegistry:
    """Registry for managing trackable parameters in parameter-level PF.

    Allows users to specify which parameter groups to track with particles.
    Each registered group will have K copies (one per particle) that evolve
    with noise during inference.

    Example:
        >>> registry = ParameterRegistry()
        >>> registry.register_group("output_weights", (64, 10), evolution_noise=0.01)
        >>> registry.register_group("time_constants", (64,), evolution_noise=0.001)
        >>> # Create particles
        >>> particles = registry.init_particles(batch_size=8, n_particles=32)
    """

    def __init__(self):
        """Initialize parameter registry."""
        self._groups: Dict[str, ParameterGroup] = {}
        self._param_indices: Dict[str, Tuple[int, int]] = {}
        self._total_params = 0
        self._is_frozen = False

    def register_group(
        self,
        name: str,
        shape: Tuple[int, ...],
        evolution_noise: float = 0.01,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        """Register a parameter group for tracking.

        Args:
            name: Unique name for this group
            shape: Shape of parameters in this group
            evolution_noise: Std of noise added during evolution
            min_value: Optional minimum value constraint
            max_value: Optional maximum value constraint
        """
        if self._is_frozen:
            raise RuntimeError("Cannot register groups after registry is frozen")
        if name in self._groups:
            raise ValueError(f"Group '{name}' already registered")

        group = ParameterGroup(
            name=name,
            shape=shape,
            evolution_noise=evolution_noise,
            min_value=min_value,
            max_value=max_value,
        )
        self._groups[name] = group

        # Compute flat indices
        n_params = math.prod(shape)
        start_idx = self._total_params
        end_idx = start_idx + n_params
        self._param_indices[name] = (start_idx, end_idx)
        self._total_params = end_idx

    def register_from_module(
        self,
        module: nn.Module,
        param_names: List[str],
        evolution_noise: float = 0.01,
        prefix: str = "",
    ):
        """Register parameters from a PyTorch module.

        Args:
            module: Module containing parameters
            param_names: Names of parameters to register
            evolution_noise: Default evolution noise
            prefix: Prefix for group names
        """
        for name in param_names:
            if not hasattr(module, name):
                raise ValueError(f"Module has no parameter '{name}'")
            param = getattr(module, name)
            if not isinstance(param, (nn.Parameter, Tensor)):
                raise ValueError(f"'{name}' is not a parameter")

            group_name = f"{prefix}{name}" if prefix else name
            self.register_group(
                name=group_name,
                shape=tuple(param.shape),
                evolution_noise=evolution_noise,
            )

    def freeze(self):
        """Freeze the registry (no more registrations allowed)."""
        self._is_frozen = True

    @property
    def is_frozen(self) -> bool:
        return self._is_frozen

    @property
    def total_params(self) -> int:
        """Total number of tracked parameters (flattened)."""
        return self._total_params

    @property
    def group_names(self) -> List[str]:
        """Names of all registered groups."""
        return list(self._groups.keys())

    @property
    def n_groups(self) -> int:
        """Number of registered groups."""
        return len(self._groups)

    def get_group(self, name: str) -> ParameterGroup:
        """Get a parameter group by name."""
        if name not in self._groups:
            raise KeyError(f"Unknown parameter group: {name}")
        return self._groups[name]

    def get_indices(self, name: str) -> Tuple[int, int]:
        """Get flat indices for a parameter group."""
        if name not in self._param_indices:
            raise KeyError(f"Unknown parameter group: {name}")
        return self._param_indices[name]

    def init_particles(
        self,
        batch_size: int,
        n_particles: int,
        base_params: Optional[Dict[str, Tensor]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """Initialize parameter particles.

        Args:
            batch_size: Batch dimension
            n_particles: Number of particles K
            base_params: Optional base parameters to initialize from
            device: Target device
            dtype: Target dtype

        Returns:
            param_particles: Flattened parameter particles [batch, K, total_params]
        """
        self.freeze()  # Freeze on first use

        particles = torch.zeros(
            batch_size, n_particles, self._total_params,
            device=device, dtype=dtype,
        )

        for name, group in self._groups.items():
            start, end = self._param_indices[name]

            if base_params is not None and name in base_params:
                # Initialize from base parameters with small noise
                base = base_params[name].flatten()
                base = base.unsqueeze(0).unsqueeze(0)  # [1, 1, n_params]
                noise = torch.randn(
                    batch_size, n_particles, end - start,
                    device=device, dtype=dtype,
                ) * group.evolution_noise
                particles[:, :, start:end] = base + noise
            else:
                # Random initialization
                particles[:, :, start:end] = torch.randn(
                    batch_size, n_particles, end - start,
                    device=device, dtype=dtype,
                ) * 0.1

        return particles

    def extract_group(
        self,
        particles: Tensor,
        name: str,
    ) -> Tensor:
        """Extract a parameter group from flattened particles.

        Args:
            particles: Flattened particles [batch, K, total_params]
            name: Group name to extract

        Returns:
            group_params: Parameters reshaped to original [batch, K, *shape]
        """
        group = self._groups[name]
        start, end = self._param_indices[name]

        batch, K, _ = particles.shape
        flat_params = particles[:, :, start:end]

        # Reshape to original shape
        return flat_params.reshape(batch, K, *group.shape)

    def inject_group(
        self,
        particles: Tensor,
        name: str,
        values: Tensor,
    ) -> Tensor:
        """Inject values into a parameter group.

        Args:
            particles: Flattened particles [batch, K, total_params]
            name: Group name to inject
            values: Values to inject [batch, K, *shape]

        Returns:
            Updated particles tensor
        """
        start, end = self._param_indices[name]
        particles[:, :, start:end] = values.flatten(start_dim=2)
        return particles

    def evolve_particles(
        self,
        particles: Tensor,
        timespans: Optional[Tensor] = None,
    ) -> Tensor:
        """Evolve parameter particles with noise.

        Args:
            particles: Parameter particles [batch, K, total_params]
            timespans: Optional time step for scaling

        Returns:
            Evolved particles [batch, K, total_params]
        """
        batch, K, _ = particles.shape
        device = particles.device
        dtype = particles.dtype

        new_particles = particles.clone()

        for name, group in self._groups.items():
            start, end = self._param_indices[name]

            # Scale noise by time if provided
            noise_scale = group.evolution_noise
            if timespans is not None:
                if timespans.dim() == 0:
                    noise_scale *= math.sqrt(timespans.item())
                else:
                    sqrt_dt = torch.sqrt(timespans.mean())
                    noise_scale *= sqrt_dt.item()

            # Add noise
            noise = torch.randn(
                batch, K, end - start,
                device=device, dtype=dtype,
            ) * noise_scale
            new_particles[:, :, start:end] += noise

            # Apply constraints
            if group.min_value is not None or group.max_value is not None:
                new_particles[:, :, start:end] = torch.clamp(
                    new_particles[:, :, start:end],
                    min=group.min_value,
                    max=group.max_value,
                )

        return new_particles

    def to_dict(self, particles: Tensor) -> Dict[str, Tensor]:
        """Convert flattened particles to dictionary of shaped tensors.

        Args:
            particles: Flattened particles [batch, K, total_params]

        Returns:
            Dict mapping group names to shaped tensors
        """
        return {
            name: self.extract_group(particles, name)
            for name in self._groups
        }

    def from_dict(
        self,
        params_dict: Dict[str, Tensor],
        batch_size: int,
        n_particles: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        """Convert dictionary of shaped tensors to flattened particles.

        Args:
            params_dict: Dict mapping group names to shaped tensors
            batch_size: Batch dimension
            n_particles: Number of particles
            device: Target device
            dtype: Target dtype

        Returns:
            Flattened particles [batch, K, total_params]
        """
        self.freeze()

        particles = torch.zeros(
            batch_size, n_particles, self._total_params,
            device=device, dtype=dtype,
        )

        for name, values in params_dict.items():
            if name in self._groups:
                self.inject_group(particles, name, values)

        return particles

    def __repr__(self) -> str:
        groups_str = ", ".join(
            f"{name}: {group.shape}" for name, group in self._groups.items()
        )
        return f"ParameterRegistry(groups=[{groups_str}], total={self._total_params})"
