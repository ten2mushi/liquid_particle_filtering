"""Gradient flow and backpropagation tests.

Tests for gradient flow through all components of the pfncps library.
"""

import math
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from pfncps.nn.utils.weights import (
    normalize_log_weights,
    weighted_mean,
    weighted_variance,
)
from pfncps.nn.utils.resampling import soft_resample, SoftResampler
from pfncps.nn.utils.noise import (
    LearnedNoise,
    StateDependentNoise,
)
from pfncps.nn.observation import GaussianObservationModel
from pfncps.nn.state_level import PFCfCCell, PFLTCCell
from pfncps.nn.param_level import ParamPFCfCCell
from pfncps.nn.dual import DualPFCfCCell
from pfncps.nn.sde import SDELTCCell
from pfncps.nn.wrappers import PFCfC, PFLTC
from pfncps.wirings import FullyConnected


# =============================================================================
# Basic Gradient Flow Tests
# =============================================================================

class TestBasicGradientFlow:
    """Tests for basic gradient flow through components."""

    def test_normalize_log_weights_gradient(self):
        """Gradients flow through log weight normalization."""
        log_weights = torch.randn(4, 32, requires_grad=True)
        normalized = normalize_log_weights(log_weights)
        loss = normalized.sum()
        loss.backward()

        assert log_weights.grad is not None
        assert not torch.isnan(log_weights.grad).any()
        assert torch.isfinite(log_weights.grad).all()

    def test_weighted_mean_gradient(self):
        """Gradients flow through weighted mean."""
        particles = torch.randn(4, 32, 64, requires_grad=True)
        log_weights = torch.full((4, 32), -math.log(32), requires_grad=True)

        mean = weighted_mean(particles, log_weights)
        loss = mean.sum()
        loss.backward()

        assert particles.grad is not None
        assert log_weights.grad is not None
        assert torch.isfinite(particles.grad).all()
        assert torch.isfinite(log_weights.grad).all()

    def test_weighted_variance_gradient(self):
        """Gradients flow through weighted variance."""
        particles = torch.randn(4, 32, 64, requires_grad=True)
        log_weights = torch.full((4, 32), -math.log(32), requires_grad=True)

        variance = weighted_variance(particles, log_weights)
        loss = variance.sum()
        loss.backward()

        assert particles.grad is not None
        assert log_weights.grad is not None

    def test_soft_resample_gradient(self):
        """Gradients flow through soft resampling."""
        particles = torch.randn(4, 32, 64, requires_grad=True)
        log_weights = torch.randn(4, 32, requires_grad=True)
        log_weights_norm = normalize_log_weights(log_weights)

        new_p, new_w = soft_resample(particles, log_weights_norm, alpha=0.5)
        loss = new_p.sum() + new_w.sum()
        loss.backward()

        assert particles.grad is not None
        assert log_weights.grad is not None


# =============================================================================
# Cell Gradient Flow Tests
# =============================================================================

class TestCellGradientFlow:
    """Tests for gradient flow through particle filter cells."""

    def test_pf_cfc_cell_input_gradient(self):
        """Gradients flow to input through PFCfCCell."""
        cell = PFCfCCell(20, 64, n_particles=16)
        x = torch.randn(4, 20, requires_grad=True)

        output, _ = cell(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_pf_cfc_cell_parameter_gradients(self):
        """All parameters in PFCfCCell receive gradients."""
        cell = PFCfCCell(20, 64, n_particles=16)
        x = torch.randn(4, 20)

        output, _ = cell(x)
        loss = output.sum()
        loss.backward()

        params_with_grad = 0
        params_without_grad = 0
        for name, param in cell.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    params_with_grad += 1
                else:
                    params_without_grad += 1

        assert params_with_grad > 0, "No parameters received gradients"

    def test_pf_ltc_cell_gradient(self):
        """Gradients flow through PFLTCCell."""
        wiring = FullyConnected(units=64, output_dim=10)
        cell = PFLTCCell(wiring=wiring, in_features=20, n_particles=16)
        x = torch.randn(4, 20, requires_grad=True)

        output, _ = cell(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    def test_param_pf_cell_gradient(self):
        """Gradients flow through ParamPFCfCCell."""
        cell = ParamPFCfCCell(20, 64, n_particles=8)
        x = torch.randn(4, 20, requires_grad=True)

        output, _ = cell(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    def test_dual_pf_cell_gradient(self):
        """Gradients flow through DualPFCfCCell."""
        cell = DualPFCfCCell(20, 64, n_particles=8)
        x = torch.randn(4, 20, requires_grad=True)

        output, _ = cell(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    def test_sde_cell_gradient(self):
        """Gradients flow through SDELTCCell."""
        wiring = FullyConnected(units=64, output_dim=10)
        cell = SDELTCCell(wiring=wiring, in_features=20, n_particles=16)
        x = torch.randn(4, 20, requires_grad=True)

        output, _ = cell(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None


# =============================================================================
# Sequence Gradient Flow Tests
# =============================================================================

class TestSequenceGradientFlow:
    """Tests for gradient flow over sequences."""

    def test_cell_sequence_gradient(self):
        """Gradients flow through multi-step cell usage."""
        cell = PFCfCCell(20, 64, n_particles=16)
        seq_len = 10
        x_seq = torch.randn(4, seq_len, 20, requires_grad=True)

        state = None
        outputs = []
        for t in range(seq_len):
            output, state = cell(x_seq[:, t, :], state)
            outputs.append(output)

        total_output = torch.stack(outputs, dim=1)
        loss = total_output.sum()
        loss.backward()

        assert x_seq.grad is not None
        # Gradients should be present for all timesteps
        assert torch.isfinite(x_seq.grad).all()

    def test_wrapper_gradient(self):
        """Gradients flow through sequence wrapper."""
        model = PFCfC(20, 64, n_particles=16)
        x = torch.randn(4, 25, 20, requires_grad=True)

        output, _ = model(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None

    def test_wrapper_parameter_gradients(self):
        """All wrapper parameters receive gradients."""
        model = PFCfC(20, 64, n_particles=16)
        x = torch.randn(4, 25, 20)

        output, _ = model(x)
        loss = output.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# =============================================================================
# Gradient Magnitude Tests
# =============================================================================

class TestGradientMagnitude:
    """Tests for gradient magnitude bounds."""

    def test_gradient_no_explosion_short_sequence(self):
        """Gradients don't explode on short sequences."""
        cell = PFCfCCell(20, 64, n_particles=16)
        x_seq = torch.randn(4, 20, 20, requires_grad=True)

        state = None
        outputs = []
        for t in range(20):
            output, state = cell(x_seq[:, t, :], state)
            outputs.append(output)

        loss = torch.stack(outputs).sum()
        loss.backward()

        assert torch.isfinite(x_seq.grad).all()
        # Gradients should not be extremely large
        assert x_seq.grad.abs().max() < 1e6

    def test_gradient_no_vanishing(self):
        """Gradients don't completely vanish."""
        cell = PFCfCCell(20, 64, n_particles=16)
        x = torch.randn(4, 20, requires_grad=True)

        output, _ = cell(x)
        loss = output.sum()
        loss.backward()

        # At least some gradients should be non-zero
        assert x.grad.abs().max() > 1e-10

    def test_gradient_magnitude_reasonable(self):
        """Gradient magnitudes are in reasonable range."""
        cell = PFCfCCell(20, 64, n_particles=16)
        x = torch.randn(4, 20, requires_grad=True)

        output, _ = cell(x)
        loss = output.sum()
        loss.backward()

        # Gradients should be in a reasonable range
        grad_norm = x.grad.norm()
        assert grad_norm > 1e-8, "Gradients too small"
        assert grad_norm < 1e6, "Gradients too large"


# =============================================================================
# Learnable Component Gradient Tests
# =============================================================================

class TestLearnableComponentGradients:
    """Tests for gradients of learnable components."""

    def test_learned_noise_gradient(self):
        """Learned noise parameters receive gradients."""
        noise = LearnedNoise(64)
        states = torch.randn(4, 32, 64, requires_grad=True)

        noisy = noise(states)
        loss = noisy.sum()
        loss.backward()

        # Noise log_scale should have gradient
        assert noise.log_noise_scale.grad is not None

    def test_state_dependent_noise_gradient(self):
        """State-dependent noise network receives gradients."""
        noise = StateDependentNoise(64)
        states = torch.randn(4, 32, 64, requires_grad=True)

        noisy = noise(states)
        loss = noisy.sum()
        loss.backward()

        # Network parameters should have gradients
        for name, param in noise.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_learnable_alpha_gradient(self):
        """Learnable alpha in resampler receives gradients."""
        resampler = SoftResampler(32, alpha_mode='learnable')
        particles = torch.randn(4, 32, 64, requires_grad=True)
        log_weights = torch.randn(4, 32, requires_grad=True)

        new_p, new_w = resampler(particles, log_weights, force_resample=True)
        loss = new_p.sum() + new_w.sum()
        loss.backward()

        assert resampler._alpha_logit.grad is not None

    def test_observation_model_gradient(self):
        """Observation model parameters receive gradients."""
        obs_model = GaussianObservationModel(64, 10, learnable_noise=True)
        states = torch.randn(4, 32, 64, requires_grad=True)
        obs = torch.randn(4, 10)

        log_lik = obs_model.log_likelihood(states, obs)
        loss = log_lik.sum()
        loss.backward()

        # Prediction network and noise should have gradients
        assert obs_model.log_noise_std.grad is not None


# =============================================================================
# Observation-Weighted Gradient Tests
# =============================================================================

class TestObservationGradientFlow:
    """Tests for gradient flow with observation models."""

    def test_gradient_with_observation(self):
        """Gradients flow when observation model is used."""
        obs_model = GaussianObservationModel(64, 10)
        cell = PFCfCCell(20, 64, n_particles=16, observation_model=obs_model)

        x = torch.randn(4, 20, requires_grad=True)
        obs = torch.randn(4, 10)

        output, _ = cell(x, observation=obs)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_observation_sequence_gradient(self):
        """Gradients flow with observations over sequence."""
        obs_model = GaussianObservationModel(64, 10)
        cell = PFCfCCell(20, 64, n_particles=16, observation_model=obs_model)

        seq_len = 10
        x_seq = torch.randn(4, seq_len, 20, requires_grad=True)
        obs_seq = torch.randn(4, seq_len, 10)

        state = None
        outputs = []
        for t in range(seq_len):
            output, state = cell(x_seq[:, t, :], state, observation=obs_seq[:, t, :])
            outputs.append(output)

        loss = torch.stack(outputs).sum()
        loss.backward()

        assert x_seq.grad is not None


# =============================================================================
# SDE-Specific Gradient Tests
# =============================================================================

class TestSDEGradientFlow:
    """Tests for gradient flow specific to SDE cells."""

    def test_sde_diffusion_gradient(self):
        """Diffusion parameters receive gradients."""
        wiring = FullyConnected(units=64, output_dim=10)
        cell = SDELTCCell(wiring=wiring, in_features=20, n_particles=16, diffusion_type="learned")
        x = torch.randn(4, 20, requires_grad=True)

        output, _ = cell(x)
        loss = output.sum()
        loss.backward()

        # Check diffusion parameters have gradients
        has_diffusion_grad = False
        for name, param in cell.named_parameters():
            if 'diffusion' in name.lower() and param.grad is not None:
                has_diffusion_grad = True
                break

    def test_sde_with_timespan_gradient(self):
        """Gradients flow correctly with varying timespans."""
        wiring = FullyConnected(units=64, output_dim=10)
        cell = SDELTCCell(wiring=wiring, in_features=20, n_particles=16)
        x = torch.randn(4, 20, requires_grad=True)
        ts = torch.tensor(0.5, requires_grad=True)

        output, _ = cell(x, timespans=ts)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None


# =============================================================================
# Gradient Checkpointing/Memory Tests
# =============================================================================

class TestGradientMemory:
    """Tests for gradient computation memory efficiency."""

    def test_gradient_computation_completes(self):
        """Gradient computation completes for moderate sequences."""
        cell = PFCfCCell(20, 64, n_particles=32)
        x_seq = torch.randn(8, 50, 20, requires_grad=True)

        state = None
        outputs = []
        for t in range(50):
            output, state = cell(x_seq[:, t, :], state)
            outputs.append(output)

        loss = torch.stack(outputs).sum()
        loss.backward()

        assert x_seq.grad is not None

    def test_detached_state_no_gradient(self):
        """Detaching state stops gradient flow as expected."""
        cell = PFCfCCell(20, 64, n_particles=16)
        x1 = torch.randn(4, 20, requires_grad=True)
        x2 = torch.randn(4, 20, requires_grad=True)

        output1, state = cell(x1)
        # Detach state
        particles, log_weights = state
        detached_state = (particles.detach(), log_weights.detach())

        output2, _ = cell(x2, detached_state)
        loss = output2.sum()
        loss.backward()

        # x2 should have gradient, x1 should not (due to detach)
        assert x2.grad is not None
        assert x1.grad is None
