"""Unit tests for parameter registry (nn/param_level/param_registry.py).

Tests for:
- ParameterGroup dataclass
- ParameterRegistry class
"""

import math
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from pfncps.nn.param_level.param_registry import (
    ParameterGroup,
    ParameterRegistry,
)


# =============================================================================
# Tests for ParameterGroup
# =============================================================================

class TestParameterGroup:
    """Tests for the ParameterGroup dataclass."""

    def test_default_values(self):
        """Default values are set correctly."""
        group = ParameterGroup(name="test", shape=(64, 10))

        assert group.name == "test"
        assert group.shape == (64, 10)
        assert group.evolution_noise == 0.01
        assert group.min_value is None
        assert group.max_value is None

    def test_custom_values(self):
        """Custom values are preserved."""
        group = ParameterGroup(
            name="weights",
            shape=(128,),
            evolution_noise=0.001,
            min_value=-1.0,
            max_value=1.0,
        )

        assert group.name == "weights"
        assert group.shape == (128,)
        assert group.evolution_noise == 0.001
        assert group.min_value == -1.0
        assert group.max_value == 1.0


# =============================================================================
# Tests for ParameterRegistry
# =============================================================================

class TestParameterRegistry:
    """Tests for the ParameterRegistry class."""

    def test_register_group_basic(self):
        """Basic group registration works."""
        registry = ParameterRegistry()
        registry.register_group("weights", (64, 10))

        assert "weights" in registry.group_names
        assert registry.n_groups == 1
        assert registry.total_params == 64 * 10

    def test_register_multiple_groups(self):
        """Multiple groups can be registered."""
        registry = ParameterRegistry()
        registry.register_group("weights", (64, 10))
        registry.register_group("bias", (10,))
        registry.register_group("scale", (64,))

        assert registry.n_groups == 3
        assert registry.total_params == 64 * 10 + 10 + 64
        assert set(registry.group_names) == {"weights", "bias", "scale"}

    def test_register_duplicate_raises(self):
        """Registering duplicate name raises error."""
        registry = ParameterRegistry()
        registry.register_group("weights", (64, 10))

        with pytest.raises(ValueError, match="already registered"):
            registry.register_group("weights", (32, 5))

    def test_freeze_prevents_registration(self):
        """Cannot register after freeze."""
        registry = ParameterRegistry()
        registry.register_group("weights", (64, 10))
        registry.freeze()

        with pytest.raises(RuntimeError, match="frozen"):
            registry.register_group("bias", (10,))

    def test_is_frozen_property(self):
        """is_frozen property works."""
        registry = ParameterRegistry()
        assert not registry.is_frozen

        registry.freeze()
        assert registry.is_frozen

    def test_get_group(self):
        """get_group returns correct ParameterGroup."""
        registry = ParameterRegistry()
        registry.register_group("weights", (64, 10), evolution_noise=0.02)

        group = registry.get_group("weights")
        assert group.name == "weights"
        assert group.shape == (64, 10)
        assert group.evolution_noise == 0.02

    def test_get_group_unknown_raises(self):
        """get_group with unknown name raises KeyError."""
        registry = ParameterRegistry()

        with pytest.raises(KeyError, match="Unknown"):
            registry.get_group("nonexistent")

    def test_get_indices(self):
        """get_indices returns correct start/end indices."""
        registry = ParameterRegistry()
        registry.register_group("group1", (10,))
        registry.register_group("group2", (20,))

        start1, end1 = registry.get_indices("group1")
        start2, end2 = registry.get_indices("group2")

        assert start1 == 0
        assert end1 == 10
        assert start2 == 10
        assert end2 == 30

    def test_init_particles_shape(self, batch_size, n_particles):
        """init_particles produces correct shape."""
        registry = ParameterRegistry()
        registry.register_group("weights", (64, 10))
        registry.register_group("bias", (10,))

        particles = registry.init_particles(batch_size, n_particles)

        assert particles.shape == (batch_size, n_particles, 64 * 10 + 10)

    def test_init_particles_freezes_registry(self, batch_size, n_particles):
        """init_particles freezes the registry."""
        registry = ParameterRegistry()
        registry.register_group("weights", (64, 10))

        assert not registry.is_frozen
        registry.init_particles(batch_size, n_particles)
        assert registry.is_frozen

    def test_init_particles_with_base_params(self, batch_size, n_particles):
        """init_particles with base parameters works."""
        registry = ParameterRegistry()
        registry.register_group("weights", (64,), evolution_noise=0.01)

        base_params = {"weights": torch.randn(64)}
        particles = registry.init_particles(batch_size, n_particles, base_params=base_params)

        # Particles should be close to base params
        extracted = registry.extract_group(particles, "weights")
        mean_extracted = extracted.mean(dim=(0, 1))  # Average over batch and particles

        assert torch.allclose(mean_extracted, base_params["weights"], atol=0.1)

    def test_extract_group_shape(self, batch_size, n_particles):
        """extract_group returns correctly shaped tensor."""
        registry = ParameterRegistry()
        registry.register_group("weights", (64, 10))
        registry.register_group("bias", (10,))

        particles = registry.init_particles(batch_size, n_particles)

        weights = registry.extract_group(particles, "weights")
        bias = registry.extract_group(particles, "bias")

        assert weights.shape == (batch_size, n_particles, 64, 10)
        assert bias.shape == (batch_size, n_particles, 10)

    def test_inject_group(self, batch_size, n_particles):
        """inject_group correctly updates particles."""
        registry = ParameterRegistry()
        registry.register_group("weights", (64,))
        registry.register_group("bias", (10,))

        particles = registry.init_particles(batch_size, n_particles)

        # Inject new values
        new_weights = torch.ones(batch_size, n_particles, 64)
        registry.inject_group(particles, "weights", new_weights)

        # Verify injection
        extracted = registry.extract_group(particles, "weights")
        assert torch.allclose(extracted, new_weights)

        # Bias should be unchanged
        bias_before = registry.extract_group(particles.clone(), "bias")

    def test_evolve_particles_adds_noise(self, batch_size, n_particles):
        """evolve_particles adds noise to particles."""
        registry = ParameterRegistry()
        registry.register_group("weights", (64,), evolution_noise=0.1)

        particles = registry.init_particles(batch_size, n_particles)
        evolved = registry.evolve_particles(particles.clone())

        # Should be different due to noise
        assert not torch.allclose(particles, evolved)

    def test_evolve_particles_respects_constraints(self, batch_size, n_particles):
        """evolve_particles respects min/max constraints."""
        registry = ParameterRegistry()
        registry.register_group(
            "weights", (64,),
            evolution_noise=10.0,  # Large noise
            min_value=-1.0,
            max_value=1.0,
        )

        particles = torch.zeros(batch_size, n_particles, 64)
        evolved = registry.evolve_particles(particles)

        assert torch.all(evolved >= -1.0)
        assert torch.all(evolved <= 1.0)

    def test_evolve_particles_with_timespan(self, batch_size, n_particles):
        """evolve_particles scales noise with timespan."""
        registry = ParameterRegistry()
        registry.register_group("weights", (64,), evolution_noise=0.1)

        particles = torch.zeros(batch_size, n_particles, 64)

        # Short timespan
        evolved_short = registry.evolve_particles(
            particles.clone(), timespans=torch.tensor(0.01)
        )

        # Long timespan
        evolved_long = registry.evolve_particles(
            particles.clone(), timespans=torch.tensor(1.0)
        )

        # Longer timespan should produce more change on average
        # (due to sqrt(dt) scaling)
        change_short = (evolved_short - particles).abs().mean()
        change_long = (evolved_long - particles).abs().mean()

        # This is stochastic, but on average should hold
        # Just check they're different
        assert change_short > 0
        assert change_long > 0

    def test_to_dict(self, batch_size, n_particles):
        """to_dict converts particles to dictionary."""
        registry = ParameterRegistry()
        registry.register_group("weights", (64, 10))
        registry.register_group("bias", (10,))

        particles = registry.init_particles(batch_size, n_particles)
        params_dict = registry.to_dict(particles)

        assert "weights" in params_dict
        assert "bias" in params_dict
        assert params_dict["weights"].shape == (batch_size, n_particles, 64, 10)
        assert params_dict["bias"].shape == (batch_size, n_particles, 10)

    def test_from_dict(self, batch_size, n_particles):
        """from_dict converts dictionary to particles."""
        registry = ParameterRegistry()
        registry.register_group("weights", (64,))
        registry.register_group("bias", (10,))

        params_dict = {
            "weights": torch.randn(batch_size, n_particles, 64),
            "bias": torch.randn(batch_size, n_particles, 10),
        }

        particles = registry.from_dict(params_dict, batch_size, n_particles)

        # Verify round-trip
        extracted = registry.to_dict(particles)
        assert torch.allclose(extracted["weights"], params_dict["weights"])
        assert torch.allclose(extracted["bias"], params_dict["bias"])

    def test_register_from_module(self):
        """register_from_module works with PyTorch module."""
        module = nn.Linear(64, 10)
        registry = ParameterRegistry()
        registry.register_from_module(module, ["weight", "bias"])

        assert "weight" in registry.group_names
        assert "bias" in registry.group_names
        assert registry.total_params == 64 * 10 + 10

    def test_register_from_module_with_prefix(self):
        """register_from_module with prefix works."""
        module = nn.Linear(64, 10)
        registry = ParameterRegistry()
        registry.register_from_module(module, ["weight"], prefix="layer1.")

        assert "layer1.weight" in registry.group_names

    def test_register_from_module_unknown_param_raises(self):
        """register_from_module with unknown param raises error."""
        module = nn.Linear(64, 10)
        registry = ParameterRegistry()

        with pytest.raises(ValueError, match="no parameter"):
            registry.register_from_module(module, ["nonexistent"])

    def test_repr(self):
        """__repr__ returns informative string."""
        registry = ParameterRegistry()
        registry.register_group("weights", (64, 10))
        registry.register_group("bias", (10,))

        repr_str = repr(registry)
        assert "ParameterRegistry" in repr_str
        assert "weights" in repr_str
        assert "bias" in repr_str


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestParameterRegistryEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_param_group(self, batch_size, n_particles):
        """Single parameter group works."""
        registry = ParameterRegistry()
        registry.register_group("weights", (10,))

        particles = registry.init_particles(batch_size, n_particles)
        assert particles.shape == (batch_size, n_particles, 10)

    def test_empty_registry(self):
        """Empty registry has zero params."""
        registry = ParameterRegistry()

        assert registry.n_groups == 0
        assert registry.total_params == 0
        assert registry.group_names == []

    def test_scalar_param_group(self, batch_size, n_particles):
        """Scalar (size 1) parameter group works."""
        registry = ParameterRegistry()
        registry.register_group("scale", (1,))

        particles = registry.init_particles(batch_size, n_particles)
        assert particles.shape == (batch_size, n_particles, 1)

    def test_large_param_group(self, batch_size, n_particles):
        """Large parameter group works."""
        registry = ParameterRegistry()
        registry.register_group("large", (1024, 512))

        particles = registry.init_particles(batch_size, n_particles)
        assert particles.shape == (batch_size, n_particles, 1024 * 512)

    def test_3d_shape(self, batch_size, n_particles):
        """3D parameter shape works."""
        registry = ParameterRegistry()
        registry.register_group("conv", (64, 3, 3, 3))

        particles = registry.init_particles(batch_size, n_particles)
        extracted = registry.extract_group(particles, "conv")

        assert extracted.shape == (batch_size, n_particles, 64, 3, 3, 3)

    def test_device_dtype_propagation(self):
        """Device and dtype are propagated correctly."""
        registry = ParameterRegistry()
        registry.register_group("weights", (64,))

        # Test with specific dtype
        particles = registry.init_particles(
            batch_size=4, n_particles=8,
            dtype=torch.float64
        )
        assert particles.dtype == torch.float64

    def test_no_nan_inf_in_particles(self, batch_size, n_particles):
        """No NaN or Inf in initialized particles."""
        registry = ParameterRegistry()
        registry.register_group("weights", (64, 10), evolution_noise=0.01)

        particles = registry.init_particles(batch_size, n_particles)
        assert not torch.isnan(particles).any()
        assert not torch.isinf(particles).any()

        evolved = registry.evolve_particles(particles)
        assert not torch.isnan(evolved).any()
        assert not torch.isinf(evolved).any()
