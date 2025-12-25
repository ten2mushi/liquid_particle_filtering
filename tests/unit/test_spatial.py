"""The spatial module consists of:
- SpatialProjectionHead: R^H → (r_norm, θ_norm, σ_r, σ_θ)
- SoftSpatialRenderer: (positions, sigmas, weights, sensor_pos) → heatmap
- SpatialPFNCP: Complete wrapper combining PFNCP with spatial output

Mathematical Invariants Tested:
1. Shape Preservation: ∀ inputs, outputs have specified shapes
2. Bounded Outputs: positions ∈ [0,1]², sigmas ∈ [σ_min, σ_max]²
3. Probability Axiom: ∑ weights = 1, ∑ heatmap = 1
4. Differentiability: ∂L/∂θ exists for all learnable parameters
5. Composition: SpatialPFNCP = PFNCP ∘ SpatialProjectionHead ∘ SoftSpatialRenderer
"""

import math
from typing import Tuple

import pytest
import torch
import torch.nn as nn
from torch import Tensor
from hypothesis import given, strategies as st, settings, assume

from pfncps.nn.spatial import (
    SpatialProjectionHead,
    SoftSpatialRenderer,
    SpatialPFNCP,
)
from pfncps.wirings import AutoNCP


# =============================================================================
# Helper Functions
# =============================================================================

def assert_no_nan_inf(tensor: Tensor, name: str = "tensor") -> None:
    """Assert that a tensor contains no NaN or Inf values."""
    assert not torch.isnan(tensor).any(), f"NaN detected in {name}"
    assert not torch.isinf(tensor).any(), f"Inf detected in {name}"


def assert_bounded(
    tensor: Tensor,
    low: float,
    high: float,
    name: str = "tensor",
    atol: float = 1e-6
) -> None:
    """Assert tensor values are within [low, high]."""
    assert tensor.min() >= low - atol, \
        f"{name} has values below {low}: min={tensor.min()}"
    assert tensor.max() <= high + atol, \
        f"{name} has values above {high}: max={tensor.max()}"


def assert_sums_to_one(tensor: Tensor, dim: int = -1, atol: float = 1e-5) -> None:
    """Assert tensor sums to 1 along specified dimension."""
    sums = tensor.sum(dim=dim)
    ones = torch.ones_like(sums)
    assert torch.allclose(sums, ones, atol=atol), \
        f"Expected sum=1, got {sums}"


# =============================================================================
# SECTION 1: SpatialProjectionHead Tests
# =============================================================================

class TestSpatialProjectionHeadShapeInvariants:
    """Shape invariants define the type signature of the projection head.

    These are the fundamental morphisms in the category of tensor shapes.
    For any input (particles, log_weights), the output shapes are fixed.
    """

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    @pytest.mark.parametrize("n_particles", [1, 8, 32])
    @pytest.mark.parametrize("hidden_size", [16, 32, 64])
    def test_forward_output_shapes(self, batch_size, n_particles, hidden_size):
        """Shape Axiom S1: forward(particles[B,K,H], log_weights[B,K])
        returns (positions[B,K,2], sigmas[B,K,2], weights[B,K])"""
        head = SpatialProjectionHead(hidden_size=hidden_size)

        particles = torch.randn(batch_size, n_particles, hidden_size)
        log_weights = torch.randn(batch_size, n_particles)

        positions, sigmas, weights = head(particles, log_weights)

        assert positions.shape == (batch_size, n_particles, 2), \
            f"positions shape mismatch: {positions.shape}"
        assert sigmas.shape == (batch_size, n_particles, 2), \
            f"sigmas shape mismatch: {sigmas.shape}"
        assert weights.shape == (batch_size, n_particles), \
            f"weights shape mismatch: {weights.shape}"

    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    @pytest.mark.parametrize("n_particles", [1, 8, 32])
    def test_weighted_position_shape(self, batch_size, n_particles):
        """Shape Axiom S2: get_weighted_position returns [B, 2]"""
        hidden_size = 32
        head = SpatialProjectionHead(hidden_size=hidden_size)

        particles = torch.randn(batch_size, n_particles, hidden_size)
        log_weights = torch.randn(batch_size, n_particles)

        mean_pos = head.get_weighted_position(particles, log_weights)

        assert mean_pos.shape == (batch_size, 2)

    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("n_particles", [4, 16])
    def test_position_uncertainty_shapes(self, batch_size, n_particles):
        """Shape Axiom S3: get_position_uncertainty returns ([B,2], [B,2])"""
        hidden_size = 32
        head = SpatialProjectionHead(hidden_size=hidden_size)

        particles = torch.randn(batch_size, n_particles, hidden_size)
        log_weights = torch.randn(batch_size, n_particles)

        mean_pos, uncertainty = head.get_position_uncertainty(particles, log_weights)

        assert mean_pos.shape == (batch_size, 2)
        assert uncertainty.shape == (batch_size, 2)


class TestSpatialProjectionHeadBoundInvariants:
    """Output bound invariants ensure outputs are within valid ranges.

    These define the codomain of the projection morphism.
    """

    @pytest.mark.parametrize("sigma_min,sigma_max", [
        (0.01, 0.5),
        (0.001, 0.1),
        (0.1, 1.0),
    ])
    def test_position_bounds(self, sigma_min, sigma_max):
        """Bound Axiom B1: positions ∈ [0, 1]² for all inputs"""
        head = SpatialProjectionHead(
            hidden_size=32,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

        # Test with various input scales
        for scale in [0.1, 1.0, 10.0, 100.0]:
            particles = torch.randn(4, 8, 32) * scale
            log_weights = torch.randn(4, 8)

            positions, _, _ = head(particles, log_weights)

            assert_bounded(positions, 0.0, 1.0, "positions")

    @pytest.mark.parametrize("sigma_min,sigma_max", [
        (0.01, 0.5),
        (0.001, 0.1),
        (0.1, 1.0),
        (0.0001, 0.9999),
    ])
    def test_sigma_bounds(self, sigma_min, sigma_max):
        """Bound Axiom B2: sigmas ∈ [sigma_min, sigma_max]² for all inputs"""
        head = SpatialProjectionHead(
            hidden_size=32,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

        for scale in [0.1, 1.0, 10.0, 100.0]:
            particles = torch.randn(4, 8, 32) * scale
            log_weights = torch.randn(4, 8)

            _, sigmas, _ = head(particles, log_weights)

            assert_bounded(sigmas, sigma_min, sigma_max, "sigmas")

    def test_weights_probability_distribution(self):
        """Bound Axiom B3: weights form a valid probability distribution"""
        head = SpatialProjectionHead(hidden_size=32)

        particles = torch.randn(4, 8, 32)
        log_weights = torch.randn(4, 8)

        _, _, weights = head(particles, log_weights)

        # Weights are non-negative
        assert torch.all(weights >= 0), "Weights must be non-negative"

        # Weights sum to 1
        assert_sums_to_one(weights, dim=-1)


class TestSpatialProjectionHeadNumericalStability:
    """Numerical stability invariants ensure robustness to extreme inputs.

    These test the projection's behavior at the boundary of its domain.
    """

    def test_extreme_particle_values(self):
        """Stability S1: No NaN/Inf for extreme particle values"""
        head = SpatialProjectionHead(hidden_size=32)

        extreme_cases = [
            torch.randn(4, 8, 32) * 1e6,   # Very large
            torch.randn(4, 8, 32) * 1e-6,  # Very small
            torch.zeros(4, 8, 32),          # All zeros
            torch.ones(4, 8, 32) * 1e3,    # Large uniform
        ]

        for particles in extreme_cases:
            log_weights = torch.randn(4, 8)
            positions, sigmas, weights = head(particles, log_weights)

            assert_no_nan_inf(positions, "positions")
            assert_no_nan_inf(sigmas, "sigmas")
            assert_no_nan_inf(weights, "weights")

    def test_extreme_log_weights(self):
        """Stability S2: No NaN/Inf for extreme log weight values"""
        head = SpatialProjectionHead(hidden_size=32)
        particles = torch.randn(4, 8, 32)

        extreme_log_weights = [
            torch.zeros(4, 8),              # Uniform
            torch.randn(4, 8) * 100,        # Large variance
            torch.full((4, 8), -100.0),     # All very negative
        ]

        # Peaked distribution
        peaked = torch.full((4, 8), -1e10)
        peaked[:, 0] = 0.0
        extreme_log_weights.append(peaked)

        for log_weights in extreme_log_weights:
            positions, sigmas, weights = head(particles, log_weights)

            assert_no_nan_inf(positions, "positions")
            assert_no_nan_inf(sigmas, "sigmas")
            assert_no_nan_inf(weights, "weights")

    def test_single_particle(self):
        """Stability S3: Works with K=1 particle (degenerate case)"""
        head = SpatialProjectionHead(hidden_size=32)

        particles = torch.randn(4, 1, 32)
        log_weights = torch.zeros(4, 1)  # Single particle has all weight

        positions, sigmas, weights = head(particles, log_weights)

        assert_no_nan_inf(positions, "positions")
        assert weights[0, 0] == 1.0, "Single particle should have weight 1"


class TestSpatialProjectionHeadDifferentiability:
    """Differentiability invariants ensure gradients flow correctly.

    This is essential for end-to-end training.
    """

    def test_gradients_flow_through_forward(self):
        """Diff D1: Gradients flow through forward pass"""
        head = SpatialProjectionHead(hidden_size=32)

        particles = torch.randn(4, 8, 32, requires_grad=True)
        log_weights = torch.randn(4, 8, requires_grad=True)

        positions, sigmas, weights = head(particles, log_weights)

        # Create scalar loss
        loss = positions.sum() + sigmas.sum() + weights.sum()
        loss.backward()

        # Check gradients exist
        assert particles.grad is not None, "No gradient for particles"
        assert log_weights.grad is not None, "No gradient for log_weights"

        # Check no NaN in gradients
        assert_no_nan_inf(particles.grad, "particles.grad")
        assert_no_nan_inf(log_weights.grad, "log_weights.grad")

    def test_gradients_flow_to_parameters(self):
        """Diff D2: Gradients reach learnable parameters"""
        head = SpatialProjectionHead(hidden_size=32)

        particles = torch.randn(4, 8, 32)
        log_weights = torch.randn(4, 8)

        positions, sigmas, _ = head(particles, log_weights)
        loss = positions.sum() + sigmas.sum()
        loss.backward()

        # Check all parameters received gradients
        for name, param in head.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert_no_nan_inf(param.grad, f"{name}.grad")

    def test_weighted_position_differentiable(self):
        """Diff D3: get_weighted_position is differentiable"""
        head = SpatialProjectionHead(hidden_size=32)

        particles = torch.randn(4, 8, 32, requires_grad=True)
        log_weights = torch.randn(4, 8, requires_grad=True)

        mean_pos = head.get_weighted_position(particles, log_weights)
        loss = mean_pos.sum()
        loss.backward()

        assert particles.grad is not None
        assert log_weights.grad is not None


class TestSpatialProjectionHeadDeterminism:
    """Determinism invariants ensure reproducibility.

    Same inputs should produce same outputs (no hidden randomness).
    """

    def test_deterministic_forward(self):
        """Determinism Det1: Same input gives same output"""
        head = SpatialProjectionHead(hidden_size=32)
        head.eval()

        particles = torch.randn(4, 8, 32)
        log_weights = torch.randn(4, 8)

        pos1, sig1, w1 = head(particles, log_weights)
        pos2, sig2, w2 = head(particles, log_weights)

        assert torch.equal(pos1, pos2), "Positions differ on same input"
        assert torch.equal(sig1, sig2), "Sigmas differ on same input"
        assert torch.equal(w1, w2), "Weights differ on same input"


class TestSpatialProjectionHeadUncertaintySemantics:
    """Semantic invariants for uncertainty estimates.

    These ensure the uncertainty has meaningful interpretations.
    """

    def test_uncertainty_positive(self):
        """Uncertainty U1: Total uncertainty is always positive"""
        head = SpatialProjectionHead(hidden_size=32)

        particles = torch.randn(4, 8, 32)
        log_weights = torch.randn(4, 8)

        _, uncertainty = head.get_position_uncertainty(particles, log_weights)

        assert torch.all(uncertainty > 0), "Uncertainty must be positive"

    def test_peaked_weights_reduce_epistemic_uncertainty(self):
        """Uncertainty U2: Peaked weights → lower epistemic component"""
        head = SpatialProjectionHead(hidden_size=32)

        # Use same particles for both
        particles = torch.randn(4, 8, 32)

        # Uniform weights
        uniform_log_weights = torch.zeros(4, 8)
        _, uniform_unc = head.get_position_uncertainty(particles, uniform_log_weights)

        # Peaked weights (all on first particle)
        peaked_log_weights = torch.full((4, 8), -100.0)
        peaked_log_weights[:, 0] = 0.0
        _, peaked_unc = head.get_position_uncertainty(particles, peaked_log_weights)

        # Peaked should have lower or equal uncertainty on average
        # (less epistemic uncertainty when concentrated)
        assert peaked_unc.mean() <= uniform_unc.mean() + 0.1


# =============================================================================
# SECTION 2: SoftSpatialRenderer Tests
# =============================================================================

class TestSoftSpatialRendererShapeInvariants:
    """Shape invariants for the spatial renderer."""

    @pytest.mark.parametrize("map_size", [32, 64, 128])
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("n_particles", [4, 16])
    def test_output_shape_egocentric(self, map_size, batch_size, n_particles):
        """Shape R1a: Output is [B, 1, H, W] in egocentric mode (no sensor_pos needed)"""
        renderer = SoftSpatialRenderer(map_size=map_size, egocentric=True)

        positions = torch.rand(batch_size, n_particles, 2)
        sigmas = torch.ones(batch_size, n_particles, 2) * 0.1
        weights = torch.ones(batch_size, n_particles) / n_particles

        # Egocentric mode: sensor_pos is optional
        heatmap = renderer(positions, sigmas, weights)

        assert heatmap.shape == (batch_size, 1, map_size, map_size)

    @pytest.mark.parametrize("map_size", [32, 64, 128])
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("n_particles", [4, 16])
    def test_output_shape_world_frame(self, map_size, batch_size, n_particles):
        """Shape R1b: Output is [B, 1, H, W] in world-frame mode (sensor_pos required)"""
        renderer = SoftSpatialRenderer(map_size=map_size, egocentric=False)

        positions = torch.rand(batch_size, n_particles, 2)
        sigmas = torch.ones(batch_size, n_particles, 2) * 0.1
        weights = torch.ones(batch_size, n_particles) / n_particles
        sensor_pos = torch.rand(batch_size, 3)

        heatmap = renderer(positions, sigmas, weights, sensor_pos)

        assert heatmap.shape == (batch_size, 1, map_size, map_size)

    def test_world_frame_requires_sensor_pos(self):
        """Shape R1c: World-frame mode raises error without sensor_pos"""
        renderer = SoftSpatialRenderer(map_size=32, egocentric=False)

        positions = torch.rand(2, 4, 2)
        sigmas = torch.ones(2, 4, 2) * 0.1
        weights = torch.ones(2, 4) / 4

        with pytest.raises(ValueError, match="World-frame rendering requires sensor_pos"):
            renderer(positions, sigmas, weights)


class TestSoftSpatialRendererProbabilityInvariants:
    """Probability distribution invariants for rendered heatmaps."""

    def test_heatmap_sums_to_one_egocentric(self):
        """Prob P1a: Heatmap integrates to 1 (proper probability) - egocentric"""
        renderer = SoftSpatialRenderer(map_size=64, egocentric=True)

        positions = torch.rand(4, 8, 2)
        sigmas = torch.ones(4, 8, 2) * 0.1
        weights = torch.ones(4, 8) / 8

        heatmap = renderer(positions, sigmas, weights)

        # Sum over spatial dimensions
        total = heatmap.sum(dim=(-2, -1))
        expected = torch.ones(4, 1)

        assert torch.allclose(total, expected, atol=1e-5), \
            f"Heatmap sum: {total}"

    def test_heatmap_sums_to_one_world_frame(self):
        """Prob P1b: Heatmap integrates to 1 (proper probability) - world-frame"""
        renderer = SoftSpatialRenderer(map_size=64, egocentric=False)

        positions = torch.rand(4, 8, 2)
        sigmas = torch.ones(4, 8, 2) * 0.1
        weights = torch.ones(4, 8) / 8
        sensor_pos = torch.rand(4, 3)

        heatmap = renderer(positions, sigmas, weights, sensor_pos)

        # Sum over spatial dimensions
        total = heatmap.sum(dim=(-2, -1))
        expected = torch.ones(4, 1)

        assert torch.allclose(total, expected, atol=1e-5), \
            f"Heatmap sum: {total}"

    def test_heatmap_non_negative(self):
        """Prob P2: Heatmap values are non-negative"""
        renderer = SoftSpatialRenderer(map_size=64, egocentric=True)

        positions = torch.rand(4, 8, 2)
        sigmas = torch.ones(4, 8, 2) * 0.1
        weights = torch.ones(4, 8) / 8

        heatmap = renderer(positions, sigmas, weights)

        assert torch.all(heatmap >= 0), "Heatmap must be non-negative"

    @pytest.mark.parametrize("n_particles", [1, 4, 16, 32])
    def test_probability_invariant_various_particles(self, n_particles):
        """Prob P3: Probability axiom holds for any number of particles (egocentric)"""
        renderer = SoftSpatialRenderer(map_size=64, egocentric=True)

        # Use small r values to ensure particles stay on grid
        positions = torch.rand(2, n_particles, 2) * 0.5  # r_norm capped at 0.5
        sigmas = torch.ones(2, n_particles, 2) * 0.1
        weights = torch.ones(2, n_particles) / n_particles

        heatmap = renderer(positions, sigmas, weights)

        assert torch.all(heatmap >= 0)
        total = heatmap.sum(dim=(-2, -1))
        assert torch.allclose(total, torch.ones(2, 1), atol=1e-5)


class TestSoftSpatialRendererCoordinateTransform:
    """Tests for polar to Cartesian coordinate transformation."""

    def test_egocentric_zero_range_at_center(self):
        """Coord C1a: r=0 places particle at grid center in egocentric mode"""
        renderer = SoftSpatialRenderer(map_size=64, r_max=150.0, egocentric=True)

        # Particle at r=0 (at sensor location = grid center)
        positions = torch.tensor([[[0.0, 0.5]]])  # r_norm=0, theta doesn't matter
        sigmas = torch.tensor([[[0.02, 0.02]]])   # Small sigma
        weights = torch.tensor([[1.0]])

        heatmap = renderer(positions, sigmas, weights)

        # Peak should be at center of heatmap (sensor is at center)
        peak_y, peak_x = torch.where(heatmap[0, 0] == heatmap.max())
        center = 32  # map_size // 2

        # Allow some tolerance due to discretization
        assert abs(peak_x[0].item() - center) <= 3, f"Peak x at {peak_x[0]}, expected ~{center}"
        assert abs(peak_y[0].item() - center) <= 3, f"Peak y at {peak_y[0]}, expected ~{center}"

    def test_world_frame_sensor_at_center_zero_range(self):
        """Coord C1b: r=0 places particle at sensor position in world-frame mode"""
        renderer = SoftSpatialRenderer(map_size=64, r_max=150.0, env_size=200.0, egocentric=False)

        # Sensor at center of environment
        sensor_pos = torch.tensor([[0.5, 0.5, 0.0]])

        # Particle at r=0 (at sensor location)
        positions = torch.tensor([[[0.0, 0.5]]])  # r_norm=0, theta doesn't matter
        sigmas = torch.tensor([[[0.02, 0.02]]])   # Small sigma
        weights = torch.tensor([[1.0]])

        heatmap = renderer(positions, sigmas, weights, sensor_pos)

        # Peak should be at center of heatmap
        peak_y, peak_x = torch.where(heatmap[0, 0] == heatmap.max())
        center = 32  # map_size // 2

        # Allow some tolerance due to discretization
        assert abs(peak_x[0].item() - center) <= 3, f"Peak x at {peak_x[0]}, expected ~{center}"
        assert abs(peak_y[0].item() - center) <= 3, f"Peak y at {peak_y[0]}, expected ~{center}"

    def test_egocentric_bearing_direction(self):
        """Coord C2a: Bearing affects particle placement in egocentric mode.

        Forward (θ=0) maps to +X direction (right on grid).
        """
        renderer = SoftSpatialRenderer(map_size=64, r_max=150.0, egocentric=True)

        # Particle at θ_norm=0.5 → θ=0 radians (forward = +X)
        positions_forward = torch.tensor([[[0.5, 0.5]]])  # θ=0 radians
        # Particle at θ_norm=1.0 → θ=π radians (backward = -X)
        positions_backward = torch.tensor([[[0.5, 1.0]]])  # θ=π radians

        sigmas = torch.tensor([[[0.02, 0.02]]])
        weights = torch.tensor([[1.0]])

        heatmap_forward = renderer(positions_forward, sigmas, weights)
        heatmap_backward = renderer(positions_backward, sigmas, weights)

        # Find peaks
        peak_y_f, peak_x_f = torch.where(heatmap_forward[0, 0] == heatmap_forward.max())
        peak_y_b, peak_x_b = torch.where(heatmap_backward[0, 0] == heatmap_backward.max())

        # Forward particle should have x greater than backward particle
        assert peak_x_f[0].item() > peak_x_b[0].item(), \
            f"Expected forward peak ({peak_x_f[0]}) > backward peak ({peak_x_b[0]})"

    def test_world_frame_bearing_direction(self):
        """Coord C2b: Bearing affects particle placement direction in world-frame mode."""
        renderer = SoftSpatialRenderer(map_size=64, r_max=150.0, env_size=200.0, egocentric=False)

        # Sensor at center, heading=0 (pointing right)
        sensor_pos = torch.tensor([[0.5, 0.5, 0.0]])

        # Particle at θ_norm=0.25 → θ=0 radians (ahead in sensor frame)
        # With heading=0, should be to the right
        positions_right = torch.tensor([[[0.5, 0.25]]])  # θ=0 radians
        # Particle at θ_norm=0.75 → θ=π radians (behind sensor)
        # Should be to the left
        positions_left = torch.tensor([[[0.5, 0.75]]])  # θ=π radians

        sigmas = torch.tensor([[[0.02, 0.02]]])
        weights = torch.tensor([[1.0]])

        heatmap_right = renderer(positions_right, sigmas, weights, sensor_pos)
        heatmap_left = renderer(positions_left, sigmas, weights, sensor_pos)

        # Find peaks
        peak_y_r, peak_x_r = torch.where(heatmap_right[0, 0] == heatmap_right.max())
        peak_y_l, peak_x_l = torch.where(heatmap_left[0, 0] == heatmap_left.max())

        # Right particle should have x greater than left particle
        assert peak_x_r[0].item() > peak_x_l[0].item(), \
            f"Expected right peak ({peak_x_r[0]}) > left peak ({peak_x_l[0]})"


class TestSoftSpatialRendererGaussianRendering:
    """Tests for Gaussian blob rendering properties."""

    def test_larger_sigma_spreads_probability(self):
        """Gauss G1: Larger sigma spreads probability more"""
        renderer = SoftSpatialRenderer(map_size=64, egocentric=True)

        positions = torch.tensor([[[0.5, 0.5]]])  # Half distance, θ=0
        weights = torch.tensor([[1.0]])

        # Small sigma
        small_sigma = torch.tensor([[[0.02, 0.02]]])
        heatmap_small = renderer(positions, small_sigma, weights)

        # Large sigma
        large_sigma = torch.tensor([[[0.2, 0.2]]])
        heatmap_large = renderer(positions, large_sigma, weights)

        # Larger sigma should have smaller max (more spread)
        assert heatmap_large.max() < heatmap_small.max()

    def test_single_gaussian_shape(self):
        """Gauss G2: Single particle produces Gaussian-like heatmap"""
        renderer = SoftSpatialRenderer(map_size=64, egocentric=True)

        heatmap = renderer.render_single_gaussian(
            torch.tensor([0.0]),  # x at center
            torch.tensor([0.0]),  # y at center
            torch.tensor([0.1]),  # sigma
        )

        # Should have single peak
        assert heatmap.max() > 0
        # Peak should be at center
        peak_y, peak_x = torch.where(heatmap[0, 0] == heatmap.max())
        assert abs(peak_x[0].item() - 32) <= 1
        assert abs(peak_y[0].item() - 32) <= 1


class TestSoftSpatialRendererNumericalStability:
    """Numerical stability tests for the renderer."""

    def test_extreme_positions(self):
        """Stability RS1: No NaN/Inf for positions at boundaries"""
        renderer = SoftSpatialRenderer(map_size=64, egocentric=True)

        extreme_positions = [
            torch.zeros(2, 4, 2),           # All at origin
            torch.ones(2, 4, 2),            # All at max
            torch.tensor([[[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]]).expand(2, -1, -1),
        ]

        for positions in extreme_positions:
            sigmas = torch.ones(2, 4, 2) * 0.1
            weights = torch.ones(2, 4) / 4

            heatmap = renderer(positions, sigmas, weights)

            assert_no_nan_inf(heatmap, "heatmap")

    def test_very_small_sigma(self):
        """Stability RS2: Handles small sigmas gracefully

        Note: Very small sigmas (< grid spacing) may result in Gaussians
        falling between grid points, which is numerically acceptable but
        produces near-zero probability mass. We test with small but
        reasonable sigmas that still render visible blobs.
        """
        renderer = SoftSpatialRenderer(map_size=64, min_sigma=0.01, egocentric=True)

        # Ensure particles render on grid by using centered positions
        positions = torch.tensor([
            [[0.3, 0.5], [0.2, 0.3], [0.4, 0.7], [0.3, 0.3]],
            [[0.25, 0.6], [0.35, 0.4], [0.3, 0.5], [0.2, 0.7]],
        ])
        sigmas = torch.ones(2, 4, 2) * 0.02  # Small but renderable
        weights = torch.ones(2, 4) / 4

        heatmap = renderer(positions, sigmas, weights)

        assert_no_nan_inf(heatmap, "heatmap")
        # With centered particles, should normalize properly
        assert heatmap.sum(dim=(-2, -1)).allclose(torch.ones(2, 1), atol=1e-3)


class TestSoftSpatialRendererDifferentiability:
    """Differentiability tests for end-to-end training."""

    def test_gradients_flow_through_render_egocentric(self):
        """Diff RD1a: Gradients flow through egocentric rendering"""
        renderer = SoftSpatialRenderer(map_size=32, egocentric=True)

        # Use leaf tensors for gradient checking
        positions = torch.rand(2, 4, 2, requires_grad=True)
        sigmas_base = torch.ones(2, 4, 2, requires_grad=True)
        weights_base = torch.ones(2, 4, requires_grad=True)

        # Scale sigmas and weights (these become non-leaf)
        sigmas = sigmas_base * 0.1
        weights = weights_base / 4

        heatmap = renderer(positions, sigmas, weights)
        loss = heatmap.sum()
        loss.backward()

        # Check leaf tensors have gradients
        assert positions.grad is not None, "positions should have grad"
        assert sigmas_base.grad is not None, "sigmas_base should have grad"
        assert weights_base.grad is not None, "weights_base should have grad"

        assert_no_nan_inf(positions.grad, "positions.grad")
        assert_no_nan_inf(sigmas_base.grad, "sigmas_base.grad")

    def test_gradients_flow_through_render_world_frame(self):
        """Diff RD1b: Gradients flow through world-frame rendering"""
        renderer = SoftSpatialRenderer(map_size=32, egocentric=False)

        # Use leaf tensors for gradient checking
        positions = torch.rand(2, 4, 2, requires_grad=True)
        sigmas_base = torch.ones(2, 4, 2, requires_grad=True)
        weights_base = torch.ones(2, 4, requires_grad=True)
        sensor_pos = torch.rand(2, 3, requires_grad=True)

        # Scale sigmas and weights (these become non-leaf)
        sigmas = sigmas_base * 0.1
        weights = weights_base / 4

        heatmap = renderer(positions, sigmas, weights, sensor_pos)
        loss = heatmap.sum()
        loss.backward()

        # Check leaf tensors have gradients
        assert positions.grad is not None, "positions should have grad"
        assert sigmas_base.grad is not None, "sigmas_base should have grad"
        assert weights_base.grad is not None, "weights_base should have grad"
        assert sensor_pos.grad is not None, "sensor_pos should have grad"

        assert_no_nan_inf(positions.grad, "positions.grad")
        assert_no_nan_inf(sigmas_base.grad, "sigmas_base.grad")


# =============================================================================
# SECTION 3: SpatialPFNCP Integration Tests
# =============================================================================

class TestSpatialPFNCPShapeInvariants:
    """Shape invariants for the complete SpatialPFNCP wrapper."""

    @pytest.fixture
    def wiring(self):
        """Create standard wiring for tests."""
        wiring = AutoNCP(units=32, output_size=2)
        wiring.build(5)
        return wiring

    def test_forward_output_shapes_egocentric(self, wiring):
        """Shape SP1a: Full output shapes in egocentric mode (no sensor_positions needed)"""
        model = SpatialPFNCP(
            wiring=wiring,
            n_particles=8,
            map_size=32,
            egocentric=True,
        )

        batch_size, seq_len = 4, 10
        x = torch.randn(batch_size, seq_len, 5)

        # Egocentric: heatmaps generated without sensor_positions
        outputs, heatmaps, state = model(x)

        assert outputs.shape == (batch_size, seq_len, 2)
        assert heatmaps.shape == (batch_size, seq_len, 1, 32, 32)

        particles, log_weights = state
        assert particles.shape == (batch_size, 8, 32)
        assert log_weights.shape == (batch_size, 8)

    def test_forward_output_shapes_world_frame(self, wiring):
        """Shape SP1b: Full output shapes in world-frame mode (sensor_positions required)"""
        model = SpatialPFNCP(
            wiring=wiring,
            n_particles=8,
            map_size=32,
            egocentric=False,
        )

        batch_size, seq_len = 4, 10
        x = torch.randn(batch_size, seq_len, 5)
        sensor_pos = torch.rand(batch_size, seq_len, 3)

        outputs, heatmaps, state = model(x, sensor_positions=sensor_pos)

        assert outputs.shape == (batch_size, seq_len, 2)
        assert heatmaps.shape == (batch_size, seq_len, 1, 32, 32)

        particles, log_weights = state
        assert particles.shape == (batch_size, 8, 32)
        assert log_weights.shape == (batch_size, 8)

    def test_forward_without_sensor_positions_world_frame(self, wiring):
        """Shape SP2: Heatmaps are None without sensor positions in world-frame mode"""
        model = SpatialPFNCP(
            wiring=wiring,
            n_particles=8,
            map_size=32,
            egocentric=False,
        )

        x = torch.randn(4, 10, 5)

        outputs, heatmaps, state = model(x)

        assert outputs.shape == (4, 10, 2)
        assert heatmaps is None  # World-frame requires sensor_pos

    def test_return_sequences_false_egocentric(self, wiring):
        """Shape SP3: return_sequences=False gives final step only (egocentric)"""
        model = SpatialPFNCP(
            wiring=wiring,
            n_particles=8,
            map_size=32,
            return_sequences=False,
            egocentric=True,
        )

        x = torch.randn(4, 10, 5)

        outputs, heatmaps, state = model(x)

        assert outputs.shape == (4, 2)  # Just final output
        assert heatmaps.shape == (4, 1, 32, 32)  # Just final heatmap


class TestSpatialPFNCPStateHandling:
    """Tests for hidden state initialization and propagation."""

    @pytest.fixture
    def model(self):
        wiring = AutoNCP(units=32, output_size=2)
        wiring.build(5)
        return SpatialPFNCP(wiring=wiring, n_particles=8, map_size=32)

    def test_initial_state_none(self, model):
        """State ST1: Works with initial state = None"""
        x = torch.randn(4, 10, 5)

        outputs, _, state = model(x, hx=None)

        assert outputs is not None
        particles, log_weights = state
        assert particles is not None
        assert log_weights is not None

    def test_state_propagation(self, model):
        """State ST2: State can be passed between calls"""
        x1 = torch.randn(4, 5, 5)
        x2 = torch.randn(4, 5, 5)

        _, _, state1 = model(x1)
        _, _, state2 = model(x2, hx=state1)

        # States should be different (processed more data)
        particles1, _ = state1
        particles2, _ = state2
        assert not torch.equal(particles1, particles2)


class TestSpatialPFNCPBeliefState:
    """Tests for belief state extraction methods."""

    @pytest.fixture
    def model(self):
        wiring = AutoNCP(units=32, output_size=2)
        wiring.build(5)
        return SpatialPFNCP(wiring=wiring, n_particles=8, map_size=32)

    def test_get_belief_state_shapes(self, model):
        """Belief B1: get_belief_state returns correct shapes"""
        x = torch.randn(4, 10, 5)
        _, _, (particles, log_weights) = model(x)

        mean_pos, uncertainty = model.get_belief_state(particles, log_weights)

        assert mean_pos.shape == (4, 2)
        assert uncertainty.shape == (4, 2)

    def test_belief_uncertainty_positive(self, model):
        """Belief B2: Uncertainty is always positive"""
        x = torch.randn(4, 10, 5)
        _, _, (particles, log_weights) = model(x)

        _, uncertainty = model.get_belief_state(particles, log_weights)

        assert torch.all(uncertainty > 0)


class TestSpatialPFNCPApproachCompatibility:
    """Tests for compatibility with different PF approaches.

    Note: Currently SpatialPFNCP only fully supports 'state' and 'sde' approaches,
    which use 2-element state tuples (particles, log_weights). The 'param' and
    'dual' approaches use 3-element state tuples and require additional handling.
    """

    def test_state_approach(self):
        """Approach A1: Works with state approach (primary use case)"""
        wiring = AutoNCP(units=32, output_size=2)
        wiring.build(5)

        model = SpatialPFNCP(
            wiring=wiring,
            n_particles=8,
            map_size=32,
            approach="state",
        )

        x = torch.randn(2, 5, 5)
        sensor_pos = torch.rand(2, 5, 3)

        outputs, heatmaps, state = model(x, sensor_positions=sensor_pos)

        assert outputs.shape == (2, 5, 2)
        assert heatmaps.shape == (2, 5, 1, 32, 32)

    @pytest.mark.skip(reason="param approach returns 3-element state; not yet supported")
    def test_param_approach(self):
        """Approach A2: param approach (not yet supported in SpatialPFNCP)"""
        wiring = AutoNCP(units=32, output_size=2)
        wiring.build(5)

        model = SpatialPFNCP(
            wiring=wiring,
            n_particles=8,
            map_size=32,
            approach="param",
        )

        x = torch.randn(2, 5, 5)
        sensor_pos = torch.rand(2, 5, 3)
        outputs, heatmaps, state = model(x, sensor_positions=sensor_pos)

    @pytest.mark.skip(reason="dual approach returns 3-element state; not yet supported")
    def test_dual_approach(self):
        """Approach A3: dual approach (not yet supported in SpatialPFNCP)"""
        wiring = AutoNCP(units=32, output_size=2)
        wiring.build(5)

        model = SpatialPFNCP(
            wiring=wiring,
            n_particles=8,
            map_size=32,
            approach="dual",
        )

        x = torch.randn(2, 5, 5)
        sensor_pos = torch.rand(2, 5, 3)
        outputs, heatmaps, state = model(x, sensor_positions=sensor_pos)

    def test_sde_approach_with_ltc(self):
        """Approach A2: SDE approach works with LTC cell"""
        wiring = AutoNCP(units=32, output_size=2)
        wiring.build(5)

        model = SpatialPFNCP(
            wiring=wiring,
            n_particles=8,
            map_size=32,
            approach="sde",
            cell_type="ltc",
        )

        x = torch.randn(2, 5, 5)
        outputs, _, state = model(x)

        assert outputs.shape == (2, 5, 2)


class TestSpatialPFNCPDifferentiability:
    """End-to-end differentiability tests."""

    def test_end_to_end_gradients(self):
        """Diff ED1: Gradients flow from heatmap to all parameters"""
        wiring = AutoNCP(units=16, output_size=2)
        wiring.build(5)

        model = SpatialPFNCP(
            wiring=wiring,
            n_particles=4,
            map_size=16,
        )

        x = torch.randn(2, 5, 5)
        sensor_pos = torch.rand(2, 5, 3)

        outputs, heatmaps, _ = model(x, sensor_positions=sensor_pos)

        # Loss from both outputs and heatmaps
        loss = outputs.sum() + heatmaps.sum()
        loss.backward()

        # Check critical parameters have gradients
        # PFNCP parameters
        has_pfncp_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.pfncp.parameters()
        )
        assert has_pfncp_grads, "No gradients in PFNCP"

        # Projection head parameters
        has_proj_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.projection_head.parameters()
        )
        assert has_proj_grads, "No gradients in projection head"


# =============================================================================
# SECTION 4: Property-Based Tests (Hypothesis)
# =============================================================================

class TestSpatialProjectionHeadProperties:
    """Property-based tests using Hypothesis for exhaustive invariant checking."""

    @given(
        batch_size=st.integers(1, 8),
        n_particles=st.integers(1, 32),
        hidden_size=st.integers(8, 64),
    )
    @settings(max_examples=50, deadline=None)
    def test_positions_always_bounded(self, batch_size, n_particles, hidden_size):
        """Property PH1: Positions always in [0, 1]² for any valid input"""
        head = SpatialProjectionHead(hidden_size=hidden_size)

        particles = torch.randn(batch_size, n_particles, hidden_size)
        log_weights = torch.randn(batch_size, n_particles)

        positions, _, _ = head(particles, log_weights)

        assert torch.all(positions >= 0)
        assert torch.all(positions <= 1)

    @given(
        batch_size=st.integers(1, 8),
        n_particles=st.integers(1, 32),
        sigma_min=st.floats(0.001, 0.1, allow_nan=False),
        sigma_max=st.floats(0.2, 1.0, allow_nan=False),
    )
    @settings(max_examples=50, deadline=None)
    def test_sigmas_always_in_range(self, batch_size, n_particles, sigma_min, sigma_max):
        """Property PH2: Sigmas always in [sigma_min, sigma_max] for any configuration"""
        assume(sigma_min < sigma_max)

        head = SpatialProjectionHead(
            hidden_size=32,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
        )

        particles = torch.randn(batch_size, n_particles, 32)
        log_weights = torch.randn(batch_size, n_particles)

        _, sigmas, _ = head(particles, log_weights)

        assert torch.all(sigmas >= sigma_min - 1e-6)
        assert torch.all(sigmas <= sigma_max + 1e-6)

    @given(
        batch_size=st.integers(1, 8),
        n_particles=st.integers(2, 32),
    )
    @settings(max_examples=50, deadline=None)
    def test_weights_always_sum_to_one(self, batch_size, n_particles):
        """Property PH3: Output weights always form valid probability distribution"""
        head = SpatialProjectionHead(hidden_size=32)

        particles = torch.randn(batch_size, n_particles, 32)
        log_weights = torch.randn(batch_size, n_particles) * 10  # Wide range

        _, _, weights = head(particles, log_weights)

        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestSoftSpatialRendererProperties:
    """Property-based tests for renderer invariants."""

    @given(
        batch_size=st.integers(1, 4),
        n_particles=st.integers(1, 16),
        map_size=st.sampled_from([16, 32, 64]),
    )
    @settings(max_examples=30, deadline=None)
    def test_heatmap_is_probability_distribution_egocentric(self, batch_size, n_particles, map_size):
        """Property PR1a: Heatmap is always a valid probability distribution (egocentric)"""
        renderer = SoftSpatialRenderer(map_size=map_size, r_max=150.0, egocentric=True)

        # Use moderate position range to keep particles on grid
        positions = torch.rand(batch_size, n_particles, 2) * 0.5  # r_norm in [0, 0.5]
        sigmas = torch.rand(batch_size, n_particles, 2) * 0.15 + 0.05  # [0.05, 0.2]
        weights = torch.softmax(torch.randn(batch_size, n_particles), dim=-1)

        heatmap = renderer(positions, sigmas, weights)

        # Non-negative
        assert torch.all(heatmap >= 0)

        # Sums to 1 (with tolerance for edge cases)
        total = heatmap.sum(dim=(-2, -1))
        assert torch.allclose(total, torch.ones(batch_size, 1), atol=1e-3), \
            f"Heatmap sums: {total.squeeze()}"

    @given(
        batch_size=st.integers(1, 4),
        n_particles=st.integers(1, 16),
        map_size=st.sampled_from([16, 32, 64]),
    )
    @settings(max_examples=30, deadline=None)
    def test_heatmap_is_probability_distribution_world_frame(self, batch_size, n_particles, map_size):
        """Property PR1b: Heatmap is always a valid probability distribution (world-frame)"""
        renderer = SoftSpatialRenderer(map_size=map_size, r_max=150.0, env_size=200.0, egocentric=False)

        # Use moderate position range to keep particles on grid
        positions = torch.rand(batch_size, n_particles, 2) * 0.6 + 0.2  # [0.2, 0.8]
        sigmas = torch.rand(batch_size, n_particles, 2) * 0.15 + 0.05  # [0.05, 0.2]
        weights = torch.softmax(torch.randn(batch_size, n_particles), dim=-1)
        # Sensor near center for reliable rendering
        sensor_pos = torch.rand(batch_size, 3) * 0.4 + 0.3  # [0.3, 0.7]

        heatmap = renderer(positions, sigmas, weights, sensor_pos)

        # Non-negative
        assert torch.all(heatmap >= 0)

        # Sums to 1 (with tolerance for edge cases)
        total = heatmap.sum(dim=(-2, -1))
        assert torch.allclose(total, torch.ones(batch_size, 1), atol=1e-3), \
            f"Heatmap sums: {total.squeeze()}"

    @given(
        batch_size=st.integers(1, 4),
        n_particles=st.integers(1, 16),
    )
    @settings(max_examples=30, deadline=None)
    def test_no_nan_inf_output(self, batch_size, n_particles):
        """Property PR2: Never produces NaN or Inf"""
        renderer = SoftSpatialRenderer(map_size=32, egocentric=True)

        positions = torch.rand(batch_size, n_particles, 2)
        sigmas = torch.rand(batch_size, n_particles, 2) * 0.3 + 0.01
        weights = torch.softmax(torch.randn(batch_size, n_particles), dim=-1)

        heatmap = renderer(positions, sigmas, weights)

        assert not torch.isnan(heatmap).any()
        assert not torch.isinf(heatmap).any()


# =============================================================================
# SECTION 5: Composition and Functoriality Tests
# =============================================================================

class TestSpatialModuleComposition:
    """Tests that verify the compositional structure of the spatial module.

    The SpatialPFNCP is a composition:
    SpatialPFNCP = PFNCP ∘ SpatialProjectionHead ∘ SoftSpatialRenderer

    This section tests that the composition behaves correctly.
    """

    def test_projection_renderer_composition_egocentric(self):
        """Composition C1a: ProjectionHead → Renderer produces valid heatmap (egocentric)"""
        head = SpatialProjectionHead(hidden_size=32)
        renderer = SoftSpatialRenderer(map_size=32, egocentric=True)

        particles = torch.randn(4, 8, 32)
        log_weights = torch.randn(4, 8)

        # Compose manually
        positions, sigmas, weights = head(particles, log_weights)
        heatmap = renderer(positions, sigmas, weights)

        # Verify composition produces valid output
        assert heatmap.shape == (4, 1, 32, 32)
        assert torch.all(heatmap >= 0)
        assert torch.allclose(
            heatmap.sum(dim=(-2, -1)),
            torch.ones(4, 1),
            atol=1e-5
        )

    def test_projection_renderer_composition_world_frame(self):
        """Composition C1b: ProjectionHead → Renderer produces valid heatmap (world-frame)"""
        head = SpatialProjectionHead(hidden_size=32)
        renderer = SoftSpatialRenderer(map_size=32, egocentric=False)

        particles = torch.randn(4, 8, 32)
        log_weights = torch.randn(4, 8)
        sensor_pos = torch.rand(4, 3)

        # Compose manually
        positions, sigmas, weights = head(particles, log_weights)
        heatmap = renderer(positions, sigmas, weights, sensor_pos)

        # Verify composition produces valid output
        assert heatmap.shape == (4, 1, 32, 32)
        assert torch.all(heatmap >= 0)
        assert torch.allclose(
            heatmap.sum(dim=(-2, -1)),
            torch.ones(4, 1),
            atol=1e-5
        )

    def test_spatial_pfncp_matches_composition_egocentric(self):
        """Composition C2: SpatialPFNCP matches manual composition (egocentric)"""
        wiring = AutoNCP(units=32, output_size=2)
        wiring.build(5)

        model = SpatialPFNCP(
            wiring=wiring,
            n_particles=8,
            map_size=32,
            return_sequences=False,
            egocentric=True,
        )

        x = torch.randn(4, 10, 5)

        # Full model forward (egocentric, no sensor_positions needed)
        outputs, heatmaps, (particles, log_weights) = model(x)

        # Manual composition with same state
        positions, sigmas, weights = model.projection_head(particles, log_weights)
        manual_heatmap = model.renderer(positions, sigmas, weights)

        # Should match
        assert torch.allclose(heatmaps, manual_heatmap, atol=1e-6)


# =============================================================================
# SECTION 6: Edge Cases and Boundary Conditions
# =============================================================================

class TestSpatialModuleEdgeCases:
    """Edge cases that push the module to its limits."""

    def test_single_batch_single_particle(self):
        """Edge E1: Minimal configuration B=1, K=1"""
        wiring = AutoNCP(units=16, output_size=2)
        wiring.build(3)

        model = SpatialPFNCP(
            wiring=wiring,
            n_particles=1,
            map_size=16,
        )

        x = torch.randn(1, 5, 3)
        sensor_pos = torch.rand(1, 5, 3)

        outputs, heatmaps, state = model(x, sensor_positions=sensor_pos)

        assert outputs.shape == (1, 5, 2)
        assert heatmaps.shape == (1, 5, 1, 16, 16)

    def test_sequence_length_one(self):
        """Edge E2: Sequence length = 1"""
        wiring = AutoNCP(units=16, output_size=2)
        wiring.build(3)

        model = SpatialPFNCP(
            wiring=wiring,
            n_particles=4,
            map_size=16,
        )

        x = torch.randn(4, 1, 3)  # Single timestep
        sensor_pos = torch.rand(4, 1, 3)

        outputs, heatmaps, state = model(x, sensor_positions=sensor_pos)

        assert outputs.shape == (4, 1, 2)
        assert heatmaps.shape == (4, 1, 1, 16, 16)

    def test_large_particle_count(self):
        """Edge E3: Large number of particles K=64"""
        wiring = AutoNCP(units=16, output_size=2)
        wiring.build(3)

        model = SpatialPFNCP(
            wiring=wiring,
            n_particles=64,
            map_size=16,
        )

        x = torch.randn(2, 3, 3)
        sensor_pos = torch.rand(2, 3, 3)

        outputs, heatmaps, state = model(x, sensor_positions=sensor_pos)

        particles, log_weights = state
        assert particles.shape == (2, 64, 16)
        assert heatmaps is not None

    def test_timespans_handling(self):
        """Edge E4: Works with explicit timespans"""
        wiring = AutoNCP(units=16, output_size=2)
        wiring.build(3)

        model = SpatialPFNCP(
            wiring=wiring,
            n_particles=4,
            map_size=16,
        )

        x = torch.randn(2, 5, 3)
        timespans = torch.ones(2, 5, 1) * 0.1
        sensor_pos = torch.rand(2, 5, 3)

        outputs, heatmaps, state = model(
            x, timespans=timespans, sensor_positions=sensor_pos
        )

        assert outputs is not None
        assert heatmaps is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
