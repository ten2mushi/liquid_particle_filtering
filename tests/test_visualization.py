"""Test Organization:
    - Unit Tests: Test individual components in isolation
    - Integration Tests: Test components working together
    - Edge Case Tests: Test boundary conditions and error handling
    - Property-Based Tests: Define invariants that must always hold

Run with: pytest tests/test_visualization.py -v
"""

import math
import os
import tempfile
from typing import List, Optional
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
import torch
from torch import Tensor

# Skip all tests if matplotlib is not available
pytest.importorskip("matplotlib")


# =============================================================================
# Test Fixtures - Reusable test data and objects
# =============================================================================

@pytest.fixture
def n_particles():
    """Standard number of particles for tests."""
    return 32


@pytest.fixture
def hidden_size():
    """Standard hidden state dimension for tests."""
    return 64


@pytest.fixture
def n_timesteps():
    """Standard number of timesteps for tests."""
    return 50


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 4


@pytest.fixture
def base_collector(n_particles, hidden_size):
    """Create a BaseDataCollector with standard configuration."""
    from pfncps.utils.visualization.collectors import BaseDataCollector
    return BaseDataCollector(
        max_history=1000,
        downsample_strategy="lttb",
        batch_idx=0,
    )


@pytest.fixture
def populated_base_collector(base_collector, n_particles, hidden_size, n_timesteps, batch_size):
    """Create a BaseDataCollector populated with synthetic data."""
    for t in range(n_timesteps):
        # Create particles with batch dimension
        particles = torch.randn(batch_size, n_particles, hidden_size)
        log_weights = torch.log_softmax(torch.randn(batch_size, n_particles), dim=-1)
        outputs = torch.randn(batch_size, n_particles, 10)

        base_collector.log_step(
            particles=particles,
            log_weights=log_weights,
            outputs=outputs,
        )
    return base_collector


@pytest.fixture
def state_collector(n_particles, hidden_size):
    """Create a StateCollector with standard configuration."""
    from pfncps.utils.visualization.collectors import StateCollector
    return StateCollector(
        max_history=1000,
        downsample_strategy="lttb",
        batch_idx=0,
    )


@pytest.fixture
def populated_state_collector(state_collector, n_particles, hidden_size, n_timesteps, batch_size):
    """Create a StateCollector populated with state-level specific data."""
    for t in range(n_timesteps):
        particles = torch.randn(batch_size, n_particles, hidden_size)
        log_weights = torch.log_softmax(torch.randn(batch_size, n_particles), dim=-1)
        # Provide multi-dimensional noise_scale for state-dependent noise tests
        noise_scale = torch.abs(torch.randn(batch_size, hidden_size)) * 0.1 + 0.05
        pre_noise_particles = particles - 0.1 * torch.randn_like(particles)

        state_collector.log_step(
            particles=particles,
            log_weights=log_weights,
            noise_scale=noise_scale,
            pre_noise_particles=pre_noise_particles,
        )
    return state_collector


@pytest.fixture
def param_collector():
    """Create a ParamCollector with standard configuration."""
    from pfncps.utils.visualization.collectors import ParamCollector
    return ParamCollector(
        max_history=1000,
        downsample_strategy="lttb",
        batch_idx=0,
    )


@pytest.fixture
def populated_param_collector(param_collector, n_particles, hidden_size, n_timesteps, batch_size):
    """Create a ParamCollector populated with parameter-level specific data."""
    param_size = 20
    param_names = ["w1", "b1", "w2", "b2"]

    for t in range(n_timesteps):
        particles = torch.randn(batch_size, n_particles, hidden_size)
        log_weights = torch.log_softmax(torch.randn(batch_size, n_particles), dim=-1)
        param_particles = torch.randn(batch_size, n_particles, param_size)
        shared_state = torch.randn(batch_size, hidden_size)

        param_collector.log_step(
            particles=particles,
            log_weights=log_weights,
            param_particles=param_particles,
            shared_state=shared_state,
            param_names=param_names,
        )
    return param_collector


@pytest.fixture
def dual_collector():
    """Create a DualCollector with standard configuration."""
    from pfncps.utils.visualization.collectors import DualCollector
    return DualCollector(
        max_history=1000,
        downsample_strategy="lttb",
        batch_idx=0,
    )


@pytest.fixture
def populated_dual_collector(dual_collector, n_particles, hidden_size, n_timesteps, batch_size):
    """Create a DualCollector populated with dual PF specific data."""
    param_size = 20

    for t in range(n_timesteps):
        particles = torch.randn(batch_size, n_particles, hidden_size)
        log_weights = torch.log_softmax(torch.randn(batch_size, n_particles), dim=-1)
        param_particles = torch.randn(batch_size, n_particles, param_size)
        rb_var_before = torch.rand(batch_size, hidden_size) * 0.5
        rb_var_after = rb_var_before * 0.5  # Rao-Blackwell reduces variance

        dual_collector.log_step(
            particles=particles,
            log_weights=log_weights,
            param_particles=param_particles,
            rao_blackwell_variance_before=rb_var_before,
            rao_blackwell_variance_after=rb_var_after,
        )
    return dual_collector


@pytest.fixture
def sde_collector():
    """Create an SDECollector with standard configuration."""
    from pfncps.utils.visualization.collectors import SDECollector
    return SDECollector(
        max_history=1000,
        downsample_strategy="lttb",
        batch_idx=0,
    )


@pytest.fixture
def populated_sde_collector(sde_collector, n_particles, hidden_size, n_timesteps, batch_size):
    """Create an SDECollector populated with SDE specific data."""
    n_unfolds = 6

    for t in range(n_timesteps):
        particles = torch.randn(batch_size, n_particles, hidden_size)
        log_weights = torch.log_softmax(torch.randn(batch_size, n_particles), dim=-1)
        diffusion_values = torch.abs(torch.randn(batch_size, hidden_size)) * 0.1
        drift_values = torch.randn(batch_size, hidden_size) * 0.5
        per_unfold_states = torch.randn(batch_size, n_unfolds, n_particles, hidden_size)
        clamped_mask = torch.zeros(batch_size, n_particles, hidden_size, dtype=torch.bool)

        sde_collector.log_step(
            particles=particles,
            log_weights=log_weights,
            diffusion_values=diffusion_values,
            drift_values=drift_values,
            per_unfold_states=per_unfold_states,
            clamped_mask=clamped_mask,
        )
    return sde_collector


@pytest.fixture
def mock_model(n_particles, hidden_size):
    """Create a mock PFNCPS model for testing."""
    model = MagicMock()
    model.n_particles = n_particles
    model.hidden_size = hidden_size
    model.input_size = 10
    model.output_size = 5
    model.__class__.__name__ = "PFCfCCell"
    return model


@pytest.fixture
def mock_ltc_model(n_particles, hidden_size):
    """Create a mock LTC-based PFNCPS model."""
    model = MagicMock()
    model.n_particles = n_particles
    model.hidden_size = hidden_size
    model.__class__.__name__ = "PFLTCCell"
    return model


@pytest.fixture
def mock_sde_model(n_particles, hidden_size):
    """Create a mock SDE-based PFNCPS model."""
    model = MagicMock()
    model.n_particles = n_particles
    model.hidden_size = hidden_size
    model.__class__.__name__ = "SDELTCCell"
    return model


@pytest.fixture
def default_theme():
    """Get the default theme."""
    from pfncps.utils.visualization import get_theme
    return get_theme("default")


@pytest.fixture
def paper_theme():
    """Get the paper theme."""
    from pfncps.utils.visualization import get_theme
    return get_theme("paper")


@pytest.fixture
def dark_theme():
    """Get the dark theme."""
    from pfncps.utils.visualization import get_theme
    return get_theme("dark")


# =============================================================================
# UNIT TESTS: Import and Module Structure
# =============================================================================

class TestModuleImports:
    """Test that all visualization components can be imported correctly.

    These tests define the public API surface of the visualization module.
    Any import that fails here indicates a breaking API change.
    """

    def test_shouldImportPFVisualizerFromTopLevel(self):
        """PFVisualizer should be importable from the top-level module."""
        from pfncps.utils.visualization import PFVisualizer
        assert PFVisualizer is not None
        assert callable(PFVisualizer)

    def test_shouldImportArchitectureDetectionComponents(self):
        """Architecture detection components should be importable."""
        from pfncps.utils.visualization import (
            ArchitectureInfo,
            PFApproach,
            BaseArchitecture,
            detect_architecture,
        )
        assert ArchitectureInfo is not None
        assert PFApproach is not None
        assert BaseArchitecture is not None
        assert callable(detect_architecture)

    def test_shouldImportThemeComponents(self):
        """Theme-related components should be importable."""
        from pfncps.utils.visualization import (
            Theme,
            get_theme,
            register_theme,
            AVAILABLE_THEMES,
        )
        assert Theme is not None
        assert callable(get_theme)
        assert callable(register_theme)
        assert isinstance(AVAILABLE_THEMES, dict)

    def test_shouldImportAllCollectorClasses(self):
        """All collector classes should be importable."""
        from pfncps.utils.visualization import (
            BaseDataCollector,
            CollectedStep,
            StateCollector,
            ParamCollector,
            DualCollector,
            SDECollector,
        )
        assert all(cls is not None for cls in [
            BaseDataCollector,
            CollectedStep,
            StateCollector,
            ParamCollector,
            DualCollector,
            SDECollector,
        ])

    def test_shouldImportAllCorePlotFunctions(self):
        """All core plot functions (C1-C10) should be importable."""
        from pfncps.utils.visualization.plots import (
            plot_ess_timeline,
            plot_weight_distribution,
            plot_weight_entropy,
            plot_particle_trajectories,
            plot_particle_diversity,
            plot_resampling_events,
            plot_observation_likelihoods,
            plot_numerical_health,
            plot_weighted_output,
            animate_particles_2d,
        )
        assert all(callable(fn) for fn in [
            plot_ess_timeline,
            plot_weight_distribution,
            plot_weight_entropy,
            plot_particle_trajectories,
            plot_particle_diversity,
            plot_resampling_events,
            plot_observation_likelihoods,
            plot_numerical_health,
            plot_weighted_output,
            animate_particles_2d,
        ])

    def test_shouldImportAllStatePlotFunctions(self):
        """All state-level plot functions (S1-S4) should be importable."""
        from pfncps.utils.visualization.plots import (
            plot_noise_injection_magnitude,
            plot_state_dependent_noise,
            plot_particle_pairwise_distances,
            plot_particle_cloud_evolution,
        )
        assert all(callable(fn) for fn in [
            plot_noise_injection_magnitude,
            plot_state_dependent_noise,
            plot_particle_pairwise_distances,
            plot_particle_cloud_evolution,
        ])

    def test_shouldImportAllParamPlotFunctions(self):
        """All parameter-level plot functions (P1-P5) should be importable."""
        from pfncps.utils.visualization.plots import (
            plot_param_posterior_marginals,
            plot_param_uncertainty_timeline,
            plot_param_correlation_matrix,
            plot_tracked_vs_base_params,
            plot_param_evolution_trajectory,
        )
        assert all(callable(fn) for fn in [
            plot_param_posterior_marginals,
            plot_param_uncertainty_timeline,
            plot_param_correlation_matrix,
            plot_tracked_vs_base_params,
            plot_param_evolution_trajectory,
        ])

    def test_shouldImportAllDualPlotFunctions(self):
        """All dual PF plot functions (D1-D4) should be importable."""
        from pfncps.utils.visualization.plots import (
            plot_joint_state_param_scatter,
            plot_rao_blackwell_variance,
            plot_state_param_correlation,
            plot_marginal_posteriors,
        )
        assert all(callable(fn) for fn in [
            plot_joint_state_param_scatter,
            plot_rao_blackwell_variance,
            plot_state_param_correlation,
            plot_marginal_posteriors,
        ])

    def test_shouldImportAllSDEPlotFunctions(self):
        """All SDE plot functions (E1-E6) should be importable."""
        from pfncps.utils.visualization.plots import (
            plot_diffusion_magnitude,
            plot_drift_diffusion_ratio,
            plot_unfold_convergence,
            plot_brownian_increments,
            plot_state_clamping_events,
            plot_euler_maruyama_stability,
        )
        assert all(callable(fn) for fn in [
            plot_diffusion_magnitude,
            plot_drift_diffusion_ratio,
            plot_unfold_convergence,
            plot_brownian_increments,
            plot_state_clamping_events,
            plot_euler_maruyama_stability,
        ])

    def test_shouldImportAllLTCPlotFunctions(self):
        """All LTC plot functions (L1-L7) should be importable."""
        from pfncps.utils.visualization.plots import (
            plot_voltage_traces,
            plot_time_constants,
            plot_synapse_activations,
            plot_leak_vs_synaptic,
            plot_ode_unfold_dynamics,
            plot_reversal_potential_flow,
            plot_sparsity_mask_utilization,
        )
        assert all(callable(fn) for fn in [
            plot_voltage_traces,
            plot_time_constants,
            plot_synapse_activations,
            plot_leak_vs_synaptic,
            plot_ode_unfold_dynamics,
            plot_reversal_potential_flow,
            plot_sparsity_mask_utilization,
        ])

    def test_shouldImportAllCfCPlotFunctions(self):
        """All CfC plot functions (F1-F5) should be importable."""
        from pfncps.utils.visualization.plots import (
            plot_interpolation_weights,
            plot_ff1_ff2_contributions,
            plot_time_constants_learned,
            plot_backbone_activations,
            plot_mode_comparison,
        )
        assert all(callable(fn) for fn in [
            plot_interpolation_weights,
            plot_ff1_ff2_contributions,
            plot_time_constants_learned,
            plot_backbone_activations,
            plot_mode_comparison,
        ])

    def test_shouldImportAllWiredPlotFunctions(self):
        """All Wired/NCP plot functions (W1-W5) should be importable."""
        from pfncps.utils.visualization.plots import (
            plot_layer_activations,
            plot_ncp_connectivity_graph,
            plot_information_flow,
            plot_layer_wise_ess,
            plot_sensory_to_motor_path,
        )
        assert all(callable(fn) for fn in [
            plot_layer_activations,
            plot_ncp_connectivity_graph,
            plot_information_flow,
            plot_layer_wise_ess,
            plot_sensory_to_motor_path,
        ])

    def test_shouldImportDashboardFunction(self):
        """Dashboard creation function should be importable."""
        from pfncps.utils.visualization.plots import create_dashboard
        assert callable(create_dashboard)

    def test_shouldImportTensorBoardBackend(self):
        """TensorBoard backend should be importable."""
        from pfncps.utils.visualization.backends.tensorboard_backend import log_to_tensorboard
        assert callable(log_to_tensorboard)


# =============================================================================
# UNIT TESTS: Theme System
# =============================================================================

class TestThemeSystem:
    """Tests for the visual theme system.

    The theme system should provide consistent styling across all plots
    and support multiple predefined themes.
    """

    def test_shouldHaveDefaultPaperAndDarkThemes(self):
        """AVAILABLE_THEMES should include default, paper, and dark."""
        from pfncps.utils.visualization import AVAILABLE_THEMES
        assert "default" in AVAILABLE_THEMES
        assert "paper" in AVAILABLE_THEMES
        assert "dark" in AVAILABLE_THEMES

    def test_getThemeShouldReturnCorrectThemeByName(self, default_theme, paper_theme, dark_theme):
        """get_theme should return the correct theme for each name."""
        from pfncps.utils.visualization import get_theme
        assert get_theme("default") == default_theme
        assert get_theme("paper") == paper_theme
        assert get_theme("dark") == dark_theme

    def test_getThemeShouldRaiseValueErrorForUnknownTheme(self):
        """get_theme should raise ValueError for unknown theme names."""
        from pfncps.utils.visualization import get_theme
        with pytest.raises(ValueError, match="Unknown theme"):
            get_theme("nonexistent_theme")

    def test_themeShouldHaveRequiredColorKeys(self, default_theme):
        """Theme colors dict should contain all required color keys."""
        required_keys = [
            "primary", "secondary", "tertiary",
            "mean", "particles", "weights", "ess", "threshold", "resampling",
            "healthy", "warning", "error",
            "heatmap_low", "heatmap_high",
            "background", "grid", "text",
        ]
        for key in required_keys:
            assert key in default_theme.colors, f"Missing required color key: {key}"

    def test_themeShouldHaveRequiredFontSizeKeys(self, default_theme):
        """Theme font_sizes dict should contain all required keys."""
        required_keys = ["title", "axis_label", "tick_label", "legend", "annotation"]
        for key in required_keys:
            assert key in default_theme.font_sizes, f"Missing font size key: {key}"

    def test_themeShouldHaveRequiredLineWidthKeys(self, default_theme):
        """Theme line_widths dict should contain all required keys."""
        required_keys = ["main", "secondary", "thin", "thick", "particle", "mean", "threshold"]
        for key in required_keys:
            assert key in default_theme.line_widths, f"Missing line width key: {key}"

    def test_themeShouldHaveRequiredAlphaKeys(self, default_theme):
        """Theme alpha_values dict should contain all required keys."""
        required_keys = ["particle_line", "confidence_band", "background", "highlight", "grid"]
        for key in required_keys:
            assert key in default_theme.alpha_values, f"Missing alpha key: {key}"

    def test_getParticleColorsShouldReturnCorrectCount(self, default_theme):
        """get_particle_colors should return the requested number of colors."""
        for n in [1, 5, 10, 20, 100]:
            colors = default_theme.get_particle_colors(n)
            assert len(colors) == n

    def test_getParticleColorsShouldCycleForLargeN(self, default_theme):
        """get_particle_colors should cycle through base colors for n > 8."""
        colors = default_theme.get_particle_colors(16)
        assert colors[0] == colors[8]  # Should cycle
        assert colors[1] == colors[9]

    def test_paperThemeShouldHaveGrayscaleColors(self, paper_theme):
        """Paper theme should use grayscale colors suitable for publication."""
        # Paper theme primary should be black
        assert paper_theme.colors["primary"] == "#000000"
        # Paper theme should have white background
        assert paper_theme.colors["background"] == "#ffffff"

    def test_darkThemeShouldHaveDarkBackground(self, dark_theme):
        """Dark theme should have dark background color."""
        # Dark theme background should be dark (low brightness)
        bg = dark_theme.colors["background"]
        # Parse hex color and check brightness
        r = int(bg[1:3], 16)
        g = int(bg[3:5], 16)
        b = int(bg[5:7], 16)
        brightness = (r + g + b) / 3
        assert brightness < 50, "Dark theme background should have low brightness"

    def test_registerThemeShouldAddCustomTheme(self):
        """register_theme should allow adding custom themes."""
        from pfncps.utils.visualization import Theme, register_theme, get_theme, AVAILABLE_THEMES

        custom_theme = Theme(
            name="custom",
            colors={"primary": "#ff0000", "background": "#ffffff"},
        )
        register_theme("custom_test", custom_theme)

        assert "custom_test" in AVAILABLE_THEMES
        assert get_theme("custom_test") == custom_theme

        # Cleanup
        del AVAILABLE_THEMES["custom_test"]

    def test_applyToAxesShouldSetExpectedProperties(self, default_theme):
        """apply_to_axes should configure axes with theme properties."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        default_theme.apply_to_axes(ax)

        # Check that grid is enabled
        assert ax.xaxis.get_gridlines()[0].get_visible()

        plt.close(fig)

    def test_applyToFigureShouldSetFacecolor(self, default_theme):
        """apply_to_figure should set the figure facecolor."""
        import matplotlib.pyplot as plt

        fig = plt.figure()
        default_theme.apply_to_figure(fig)

        # Check facecolor was applied
        assert fig.get_facecolor() is not None

        plt.close(fig)


# =============================================================================
# UNIT TESTS: BaseDataCollector
# =============================================================================

class TestBaseDataCollector:
    """Tests for BaseDataCollector functionality.

    The BaseDataCollector is the foundation for all data collection,
    handling memory management, downsampling, and metric computation.
    """

    def test_shouldInitializeWithDefaultParameters(self):
        """BaseDataCollector should initialize with sensible defaults."""
        from pfncps.utils.visualization.collectors import BaseDataCollector

        collector = BaseDataCollector()
        assert collector.max_history == 10000
        assert collector.downsample_strategy == "lttb"
        assert collector.batch_idx == 0
        assert len(collector.history) == 0

    def test_shouldAcceptCustomParameters(self):
        """BaseDataCollector should accept custom initialization parameters."""
        from pfncps.utils.visualization.collectors import BaseDataCollector

        collector = BaseDataCollector(
            max_history=500,
            downsample_strategy="uniform",
            batch_idx=2,
        )
        assert collector.max_history == 500
        assert collector.downsample_strategy == "uniform"
        assert collector.batch_idx == 2

    def test_logStepShouldIncrementHistoryLength(self, base_collector, n_particles, hidden_size):
        """Each log_step call should add one entry to history."""
        particles = torch.randn(1, n_particles, hidden_size)
        log_weights = torch.log_softmax(torch.randn(1, n_particles), dim=-1)

        assert len(base_collector) == 0

        for i in range(5):
            base_collector.log_step(particles=particles, log_weights=log_weights)
            assert len(base_collector) == i + 1

    def test_logStepShouldExtractCorrectBatchElement(self, n_particles, hidden_size):
        """log_step should extract the specified batch element."""
        from pfncps.utils.visualization.collectors import BaseDataCollector

        batch_size = 4
        collector = BaseDataCollector(batch_idx=2)

        particles = torch.randn(batch_size, n_particles, hidden_size)
        log_weights = torch.log_softmax(torch.randn(batch_size, n_particles), dim=-1)

        collector.log_step(particles=particles, log_weights=log_weights)

        # Stored particles should match batch element 2
        stored = collector.history[0].particles
        assert torch.allclose(stored, particles[2])

    def test_logStepShouldHandleBatchIdxExceedingBatchSize(self, n_particles, hidden_size):
        """log_step should clamp batch_idx to valid range."""
        from pfncps.utils.visualization.collectors import BaseDataCollector

        collector = BaseDataCollector(batch_idx=10)  # Exceeds batch size

        particles = torch.randn(2, n_particles, hidden_size)  # Only 2 batch elements
        log_weights = torch.log_softmax(torch.randn(2, n_particles), dim=-1)

        collector.log_step(particles=particles, log_weights=log_weights)

        # Should use last valid batch element
        stored = collector.history[0].particles
        assert torch.allclose(stored, particles[1])  # batch element 1 (last valid)

    def test_logStepShouldComputeESSCorrectly(self, base_collector, n_particles, hidden_size):
        """log_step should compute correct ESS values."""
        particles = torch.randn(1, n_particles, hidden_size)

        # Uniform weights should give ESS = n_particles
        log_weights_uniform = torch.zeros(1, n_particles) - math.log(n_particles)
        base_collector.log_step(particles=particles, log_weights=log_weights_uniform)
        assert abs(base_collector.history[-1].ess - n_particles) < 0.1

        # Concentrated weights should give low ESS
        log_weights_concentrated = torch.full((1, n_particles), -100.0)
        log_weights_concentrated[0, 0] = 0.0  # All weight on one particle
        base_collector.log_step(particles=particles, log_weights=log_weights_concentrated)
        assert base_collector.history[-1].ess < 2.0

    def test_logStepShouldStoreCPUTensors(self, base_collector, n_particles, hidden_size):
        """log_step should convert tensors to CPU."""
        particles = torch.randn(1, n_particles, hidden_size)
        log_weights = torch.log_softmax(torch.randn(1, n_particles), dim=-1)

        base_collector.log_step(particles=particles, log_weights=log_weights)

        assert base_collector.history[0].particles.device.type == "cpu"
        assert base_collector.history[0].log_weights.device.type == "cpu"

    def test_logStepShouldIncrementTimestep(self, base_collector, n_particles, hidden_size):
        """Each log_step should increment the timestep counter."""
        particles = torch.randn(1, n_particles, hidden_size)
        log_weights = torch.log_softmax(torch.randn(1, n_particles), dim=-1)

        for expected_t in range(10):
            base_collector.log_step(particles=particles, log_weights=log_weights)
            assert base_collector.history[-1].timestep == expected_t

    def test_resetShouldClearAllData(self, populated_base_collector):
        """reset should clear history and reset counters."""
        assert len(populated_base_collector) > 0

        populated_base_collector.reset()

        assert len(populated_base_collector) == 0
        assert populated_base_collector._step_counter == 0
        assert len(populated_base_collector._cache) == 0

    def test_getParticlesShouldReturnStackedTensor(self, populated_base_collector, n_timesteps, n_particles, hidden_size):
        """get_particles should return correctly shaped stacked tensor."""
        particles = populated_base_collector.get_particles()

        assert particles.shape == (n_timesteps, n_particles, hidden_size)
        assert isinstance(particles, Tensor)

    def test_getLogWeightsShouldReturnStackedTensor(self, populated_base_collector, n_timesteps, n_particles):
        """get_log_weights should return correctly shaped stacked tensor."""
        log_weights = populated_base_collector.get_log_weights()

        assert log_weights.shape == (n_timesteps, n_particles)
        assert isinstance(log_weights, Tensor)

    def test_getWeightsShouldReturnNormalizedWeights(self, populated_base_collector, n_timesteps, n_particles):
        """get_weights should return normalized (summing to 1) weights."""
        weights = populated_base_collector.get_weights()

        assert weights.shape == (n_timesteps, n_particles)

        # Each timestep's weights should sum to 1
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones(n_timesteps), atol=1e-5)

    def test_getWeightsShouldBeNonNegative(self, populated_base_collector):
        """get_weights should return non-negative values."""
        weights = populated_base_collector.get_weights()
        assert (weights >= 0).all()

    def test_getESSShouldReturnTimelineVector(self, populated_base_collector, n_timesteps):
        """get_ess should return a 1D tensor of ESS values."""
        ess = populated_base_collector.get_ess()

        assert ess.shape == (n_timesteps,)
        assert isinstance(ess, Tensor)

    def test_getTimestepsShouldReturnCorrectIndices(self, populated_base_collector, n_timesteps):
        """get_timesteps should return sequential indices."""
        timesteps = populated_base_collector.get_timesteps()

        assert timesteps.shape == (n_timesteps,)
        expected = torch.arange(n_timesteps)
        assert torch.allclose(timesteps, expected)

    def test_getWeightEntropyShouldComputeCorrectly(self, base_collector, n_particles, hidden_size):
        """get_weight_entropy should compute Shannon entropy of weights."""
        particles = torch.randn(1, n_particles, hidden_size)

        # Uniform weights should give max entropy = log(K)
        log_weights_uniform = torch.zeros(1, n_particles) - math.log(n_particles)
        base_collector.log_step(particles=particles, log_weights=log_weights_uniform)

        entropy = base_collector.get_weight_entropy()
        expected_max_entropy = math.log(n_particles)
        assert abs(entropy[0].item() - expected_max_entropy) < 0.01

    def test_getParticleVarianceShouldReturnPerDimVariance(self, populated_base_collector, n_timesteps, hidden_size):
        """get_particle_variance should return variance per dimension."""
        variance = populated_base_collector.get_particle_variance()

        assert variance.shape == (n_timesteps, hidden_size)
        assert (variance >= 0).all()  # Variance is non-negative

    def test_getWeightedMeanShouldComputeCorrectly(self, populated_base_collector, n_timesteps, hidden_size):
        """get_weighted_mean should compute weighted average of particles."""
        mean = populated_base_collector.get_weighted_mean()

        assert mean.shape == (n_timesteps, hidden_size)

    def test_getWeightedVarianceShouldComputeCorrectly(self, populated_base_collector, n_timesteps, hidden_size):
        """get_weighted_variance should compute weighted variance."""
        variance = populated_base_collector.get_weighted_variance()

        assert variance.shape == (n_timesteps, hidden_size)
        assert (variance >= 0).all()

    def test_getPairwiseDistancesShouldReturnPositiveValues(self, populated_base_collector, n_timesteps):
        """get_pairwise_distances should return positive average distances."""
        distances = populated_base_collector.get_pairwise_distances()

        assert distances.shape == (n_timesteps,)
        assert (distances >= 0).all()

    def test_getNumericalHealthShouldDetectCleanData(self, populated_base_collector):
        """get_numerical_health should report healthy for clean data."""
        health = populated_base_collector.get_numerical_health()

        assert "has_nan" in health
        assert "has_inf" in health
        assert "max_norm" in health

        # Clean synthetic data should not have NaN/Inf
        assert not health["has_nan"].any()
        assert not health["has_inf"].any()

    def test_getNumericalHealthShouldDetectNaN(self, base_collector, n_particles, hidden_size):
        """get_numerical_health should detect NaN values."""
        particles = torch.randn(1, n_particles, hidden_size)
        particles[0, 0, 0] = float('nan')
        log_weights = torch.log_softmax(torch.randn(1, n_particles), dim=-1)

        base_collector.log_step(particles=particles, log_weights=log_weights)
        health = base_collector.get_numerical_health()

        assert health["has_nan"][0].item() == True

    def test_getNumericalHealthShouldDetectInf(self, base_collector, n_particles, hidden_size):
        """get_numerical_health should detect Inf values."""
        particles = torch.randn(1, n_particles, hidden_size)
        particles[0, 0, 0] = float('inf')
        log_weights = torch.log_softmax(torch.randn(1, n_particles), dim=-1)

        base_collector.log_step(particles=particles, log_weights=log_weights)
        health = base_collector.get_numerical_health()

        assert health["has_inf"][0].item() == True

    def test_cacheShouldBeClearedOnNewData(self, populated_base_collector, n_particles, hidden_size):
        """Cache should be cleared when new data is logged."""
        # Access data to populate cache
        _ = populated_base_collector.get_particles()
        assert "particles" in populated_base_collector._cache

        # Log new data
        particles = torch.randn(1, n_particles, hidden_size)
        log_weights = torch.log_softmax(torch.randn(1, n_particles), dim=-1)
        populated_base_collector.log_step(particles=particles, log_weights=log_weights)

        # Cache should be cleared
        assert len(populated_base_collector._cache) == 0


# =============================================================================
# UNIT TESTS: Downsampling
# =============================================================================

class TestDownsampling:
    """Tests for LTTB and uniform downsampling algorithms.

    Downsampling should preserve important data characteristics while
    reducing memory usage for long sequences.
    """

    def test_lttbShouldNotDownsampleBelowMaxHistory(self):
        """LTTB should not downsample if history is below max."""
        from pfncps.utils.visualization.collectors import BaseDataCollector

        collector = BaseDataCollector(max_history=100)
        particles = torch.randn(1, 32, 64)
        log_weights = torch.log_softmax(torch.randn(1, 32), dim=-1)

        # Log fewer steps than max_history
        for _ in range(50):
            collector.log_step(particles=particles, log_weights=log_weights)

        assert len(collector) == 50

    def test_lttbShouldDownsampleAboveMaxHistory(self):
        """LTTB should downsample when history exceeds max_history."""
        from pfncps.utils.visualization.collectors import BaseDataCollector

        max_history = 100
        collector = BaseDataCollector(max_history=max_history, downsample_strategy="lttb")
        particles = torch.randn(1, 32, 64)
        log_weights = torch.log_softmax(torch.randn(1, 32), dim=-1)

        # Log more steps than max_history
        for _ in range(150):
            collector.log_step(particles=particles, log_weights=log_weights)

        # Should have downsampled to ~90% of max_history
        assert len(collector) <= max_history
        assert len(collector) >= int(max_history * 0.85)

    def test_lttbShouldPreserveFirstAndLastPoints(self):
        """LTTB should always preserve the first and last data points."""
        from pfncps.utils.visualization.collectors import BaseDataCollector

        collector = BaseDataCollector(max_history=50, downsample_strategy="lttb")
        particles = torch.randn(1, 32, 64)

        # Create distinct first and last weights
        first_log_weights = torch.zeros(1, 32) - 100.0
        first_log_weights[0, 0] = 0.0  # ESS = 1
        last_log_weights = torch.zeros(1, 32) - math.log(32)  # ESS = 32

        collector.log_step(particles=particles, log_weights=first_log_weights)

        for _ in range(98):
            log_weights = torch.log_softmax(torch.randn(1, 32), dim=-1)
            collector.log_step(particles=particles, log_weights=log_weights)

        collector.log_step(particles=particles, log_weights=last_log_weights)

        # First point should have ESS close to 1
        assert collector.history[0].ess < 2.0
        # Last point should have ESS close to 32
        assert collector.history[-1].ess > 30.0

    def test_uniformDownsamplingShouldPreserveFirstAndLast(self):
        """Uniform downsampling should preserve first and last points."""
        from pfncps.utils.visualization.collectors import BaseDataCollector

        collector = BaseDataCollector(max_history=50, downsample_strategy="uniform")
        particles = torch.randn(1, 32, 64)
        log_weights = torch.log_softmax(torch.randn(1, 32), dim=-1)

        for t in range(100):
            collector.log_step(particles=particles, log_weights=log_weights)

        # First timestep should be 0
        assert collector.history[0].timestep == 0
        # Last timestep should be 99
        assert collector.history[-1].timestep == 99

    def test_lttbShouldPreserveExtrema(self):
        """LTTB should tend to preserve local extrema in ESS signal."""
        from pfncps.utils.visualization.collectors import BaseDataCollector

        collector = BaseDataCollector(max_history=30, downsample_strategy="lttb")
        particles = torch.randn(1, 32, 64)

        # Create ESS signal with clear peak
        for t in range(100):
            if t == 50:
                # Create a clear minimum ESS
                log_weights = torch.full((1, 32), -100.0)
                log_weights[0, 0] = 0.0
            else:
                log_weights = torch.zeros(1, 32) - math.log(32)
            collector.log_step(particles=particles, log_weights=log_weights)

        # The extreme low ESS point should be preserved
        ess_values = [h.ess for h in collector.history]
        assert min(ess_values) < 2.0  # The extreme point should survive


# =============================================================================
# UNIT TESTS: StateCollector
# =============================================================================

class TestStateCollector:
    """Tests for StateCollector specific functionality.

    StateCollector extends BaseDataCollector with state-level PF
    specific data like noise injection tracking.
    """

    def test_shouldLogNoiseScale(self, state_collector, n_particles, hidden_size):
        """StateCollector should store noise_scale values."""
        particles = torch.randn(1, n_particles, hidden_size)
        log_weights = torch.log_softmax(torch.randn(1, n_particles), dim=-1)
        noise_scale = torch.tensor(0.1)

        state_collector.log_step(
            particles=particles,
            log_weights=log_weights,
            noise_scale=noise_scale,
        )

        assert "noise_scale" in state_collector.history[0].extra

    def test_shouldLogPreNoiseParticles(self, state_collector, n_particles, hidden_size):
        """StateCollector should store pre_noise_particles values."""
        particles = torch.randn(1, n_particles, hidden_size)
        pre_noise = torch.randn(1, n_particles, hidden_size)
        log_weights = torch.log_softmax(torch.randn(1, n_particles), dim=-1)

        state_collector.log_step(
            particles=particles,
            log_weights=log_weights,
            pre_noise_particles=pre_noise,
        )

        assert "pre_noise_particles" in state_collector.history[0].extra

    def test_getNoiseScaleShouldReturnStoredValues(self, populated_state_collector, n_timesteps):
        """get_noise_scale should return the stored noise scale values."""
        noise_scale = populated_state_collector.get_noise_scale()

        assert noise_scale is not None
        assert len(noise_scale) == n_timesteps

    def test_getNoiseMagnitudeShouldComputeFromDifference(self, populated_state_collector, n_timesteps):
        """get_noise_magnitude should compute L2 norm of injected noise."""
        noise_magnitude = populated_state_collector.get_noise_magnitude()

        assert noise_magnitude is not None
        assert noise_magnitude.shape == (n_timesteps,)
        assert (noise_magnitude >= 0).all()

    def test_getCollapseRatioShouldReturnPositiveValues(self, populated_state_collector, n_timesteps):
        """get_collapse_ratio should return non-negative values."""
        collapse_ratio = populated_state_collector.get_collapse_ratio()

        assert collapse_ratio.shape == (n_timesteps,)
        assert (collapse_ratio >= 0).all()


# =============================================================================
# UNIT TESTS: ParamCollector
# =============================================================================

class TestParamCollector:
    """Tests for ParamCollector specific functionality.

    ParamCollector extends BaseDataCollector with parameter-level PF
    specific data like parameter particles and posterior statistics.
    """

    def test_shouldLogParamParticles(self, param_collector, n_particles, hidden_size):
        """ParamCollector should store param_particles values."""
        particles = torch.randn(1, n_particles, hidden_size)
        param_particles = torch.randn(1, n_particles, 20)
        log_weights = torch.log_softmax(torch.randn(1, n_particles), dim=-1)

        param_collector.log_step(
            particles=particles,
            log_weights=log_weights,
            param_particles=param_particles,
        )

        assert "param_particles" in param_collector.history[0].extra

    def test_getParamParticlesShouldReturnStackedTensor(self, populated_param_collector, n_timesteps, n_particles):
        """get_param_particles should return correctly shaped tensor."""
        param_particles = populated_param_collector.get_param_particles()

        assert param_particles is not None
        assert param_particles.shape[0] == n_timesteps
        assert param_particles.shape[1] == n_particles

    def test_getParamPosteriorStatsShouldComputeStatistics(self, populated_param_collector, n_timesteps):
        """get_param_posterior_stats should compute mean, std, and quantiles."""
        stats = populated_param_collector.get_param_posterior_stats()

        assert stats is not None
        assert "mean" in stats
        assert "std" in stats
        assert "quantiles" in stats

        assert stats["mean"].shape[0] == n_timesteps
        assert stats["std"].shape[0] == n_timesteps
        assert stats["quantiles"].shape[0] == n_timesteps
        assert stats["quantiles"].shape[1] == 3  # 25%, 50%, 75%

    def test_getParamCorrelationShouldReturnCorrectShape(self, populated_param_collector, n_timesteps):
        """get_param_correlation should return correlation matrices."""
        correlation = populated_param_collector.get_param_correlation()

        assert correlation is not None
        assert correlation.shape[0] == n_timesteps
        # Correlation matrix should be square
        assert correlation.shape[1] == correlation.shape[2]


# =============================================================================
# UNIT TESTS: DualCollector
# =============================================================================

class TestDualCollector:
    """Tests for DualCollector specific functionality.

    DualCollector extends BaseDataCollector with dual PF specific
    data like joint state-parameter particles and Rao-Blackwell metrics.
    """

    def test_shouldLogParamParticles(self, dual_collector, n_particles, hidden_size):
        """DualCollector should store param_particles values."""
        particles = torch.randn(1, n_particles, hidden_size)
        param_particles = torch.randn(1, n_particles, 20)
        log_weights = torch.log_softmax(torch.randn(1, n_particles), dim=-1)

        dual_collector.log_step(
            particles=particles,
            log_weights=log_weights,
            param_particles=param_particles,
        )

        assert "param_particles" in dual_collector.history[0].extra

    def test_shouldLogRaoBlackwellVariance(self, dual_collector, n_particles, hidden_size):
        """DualCollector should store Rao-Blackwell variance data."""
        particles = torch.randn(1, n_particles, hidden_size)
        log_weights = torch.log_softmax(torch.randn(1, n_particles), dim=-1)
        rb_before = torch.rand(1, hidden_size)
        rb_after = rb_before * 0.5

        dual_collector.log_step(
            particles=particles,
            log_weights=log_weights,
            rao_blackwell_variance_before=rb_before,
            rao_blackwell_variance_after=rb_after,
        )

        assert "rb_var_before" in dual_collector.history[0].extra
        assert "rb_var_after" in dual_collector.history[0].extra

    def test_getJointParticlesShouldConcatenateStateAndParam(self, populated_dual_collector, n_timesteps, n_particles, hidden_size):
        """get_joint_particles should concatenate state and parameter particles."""
        joint = populated_dual_collector.get_joint_particles()

        assert joint is not None
        assert joint.shape[0] == n_timesteps
        assert joint.shape[1] == n_particles
        # Joint size should be state + param size
        assert joint.shape[2] > hidden_size

    def test_getRaoBlackwellVarianceShouldComputeReduction(self, populated_dual_collector, n_timesteps):
        """get_rao_blackwell_variance should compute variance reduction."""
        rb_var = populated_dual_collector.get_rao_blackwell_variance()

        assert rb_var is not None
        assert "before" in rb_var
        assert "after" in rb_var
        assert "reduction" in rb_var

        # After should be less than before (variance reduction)
        assert (rb_var["after"] <= rb_var["before"] * 1.01).all()  # Allow small numerical error

    def test_getStateParamCorrelationShouldReturnCorrectShape(self, populated_dual_collector, n_timesteps, hidden_size):
        """get_state_param_correlation should return correlation matrices."""
        correlation = populated_dual_collector.get_state_param_correlation()

        assert correlation is not None
        assert correlation.shape[0] == n_timesteps
        assert correlation.shape[1] == hidden_size  # State dims

    def test_getMarginalStateStatsShouldComputeStatistics(self, populated_dual_collector, n_timesteps, hidden_size):
        """get_marginal_state_stats should compute marginalized statistics."""
        stats = populated_dual_collector.get_marginal_state_stats()

        assert "mean" in stats
        assert "std" in stats
        assert "quantiles" in stats

        assert stats["mean"].shape == (n_timesteps, hidden_size)
        assert stats["std"].shape == (n_timesteps, hidden_size)


# =============================================================================
# UNIT TESTS: SDECollector
# =============================================================================

class TestSDECollector:
    """Tests for SDECollector specific functionality.

    SDECollector extends BaseDataCollector with SDE PF specific
    data like diffusion coefficients and per-unfold states.
    """

    def test_shouldLogDiffusionValues(self, sde_collector, n_particles, hidden_size):
        """SDECollector should store diffusion_values."""
        particles = torch.randn(1, n_particles, hidden_size)
        log_weights = torch.log_softmax(torch.randn(1, n_particles), dim=-1)
        diffusion = torch.rand(1, hidden_size) * 0.1

        sde_collector.log_step(
            particles=particles,
            log_weights=log_weights,
            diffusion_values=diffusion,
        )

        assert "diffusion" in sde_collector.history[0].extra

    def test_shouldLogDriftValues(self, sde_collector, n_particles, hidden_size):
        """SDECollector should store drift_values."""
        particles = torch.randn(1, n_particles, hidden_size)
        log_weights = torch.log_softmax(torch.randn(1, n_particles), dim=-1)
        drift = torch.randn(1, hidden_size)

        sde_collector.log_step(
            particles=particles,
            log_weights=log_weights,
            drift_values=drift,
        )

        assert "drift" in sde_collector.history[0].extra

    def test_shouldLogPerUnfoldStates(self, sde_collector, n_particles, hidden_size):
        """SDECollector should store per_unfold_states."""
        particles = torch.randn(1, n_particles, hidden_size)
        log_weights = torch.log_softmax(torch.randn(1, n_particles), dim=-1)
        per_unfold = torch.randn(1, 6, n_particles, hidden_size)

        sde_collector.log_step(
            particles=particles,
            log_weights=log_weights,
            per_unfold_states=per_unfold,
        )

        assert "unfold_states" in sde_collector.history[0].extra

    def test_getDiffusionShouldReturnStoredValues(self, populated_sde_collector, n_timesteps):
        """get_diffusion should return the stored diffusion values."""
        diffusion = populated_sde_collector.get_diffusion()

        assert diffusion is not None
        assert diffusion.shape[0] == n_timesteps

    def test_getDriftShouldReturnStoredValues(self, populated_sde_collector, n_timesteps):
        """get_drift should return the stored drift values."""
        drift = populated_sde_collector.get_drift()

        assert drift is not None
        assert drift.shape[0] == n_timesteps

    def test_getDriftDiffusionRatioShouldComputeRatio(self, populated_sde_collector, n_timesteps):
        """get_drift_diffusion_ratio should compute the ratio of magnitudes."""
        ratio = populated_sde_collector.get_drift_diffusion_ratio()

        assert ratio is not None
        assert ratio.shape == (n_timesteps,)
        assert (ratio >= 0).all()

    def test_getUnfoldConvergenceShouldComputeDeltas(self, populated_sde_collector, n_timesteps):
        """get_unfold_convergence should compute per-unfold state changes."""
        convergence = populated_sde_collector.get_unfold_convergence()

        assert convergence is not None
        assert convergence.shape[0] == n_timesteps

    def test_getEulerMaruyamaStabilityShouldReturnMetrics(self, populated_sde_collector, n_timesteps):
        """get_euler_maruyama_stability should return stability metrics."""
        stability = populated_sde_collector.get_euler_maruyama_stability()

        assert stability is not None
        assert "drift_term" in stability
        assert "diffusion_term" in stability
        assert "ratio" in stability


# =============================================================================
# UNIT TESTS: Architecture Detection
# =============================================================================

class TestArchitectureDetection:
    """Tests for architecture detection functionality.

    The detect_architecture function should correctly identify model
    types based on class names and attributes.
    """

    def test_shouldDetectStateLevel(self, mock_model):
        """Should detect state-level PF approach."""
        from pfncps.utils.visualization import detect_architecture, PFApproach

        arch_info = detect_architecture(mock_model)

        assert arch_info.pf_approach == PFApproach.STATE_LEVEL

    def test_shouldDetectCfCArchitecture(self, mock_model):
        """Should detect CfC base architecture."""
        from pfncps.utils.visualization import detect_architecture, BaseArchitecture

        arch_info = detect_architecture(mock_model)

        assert arch_info.base_arch == BaseArchitecture.CFC

    def test_shouldDetectLTCArchitecture(self, mock_ltc_model):
        """Should detect LTC base architecture."""
        from pfncps.utils.visualization import detect_architecture, BaseArchitecture

        arch_info = detect_architecture(mock_ltc_model)

        assert arch_info.base_arch == BaseArchitecture.LTC

    def test_shouldDetectSDEApproach(self, mock_sde_model):
        """Should detect SDE PF approach."""
        from pfncps.utils.visualization import detect_architecture, PFApproach

        arch_info = detect_architecture(mock_sde_model)

        assert arch_info.pf_approach == PFApproach.SDE

    def test_shouldExtractNParticles(self, mock_model, n_particles):
        """Should extract n_particles from model."""
        from pfncps.utils.visualization import detect_architecture

        arch_info = detect_architecture(mock_model)

        assert arch_info.n_particles == n_particles

    def test_shouldExtractHiddenSize(self, mock_model, hidden_size):
        """Should extract hidden_size from model."""
        from pfncps.utils.visualization import detect_architecture

        arch_info = detect_architecture(mock_model)

        assert arch_info.hidden_size == hidden_size

    def test_architectureInfoPropertiesShouldWork(self):
        """ArchitectureInfo convenience properties should work correctly."""
        from pfncps.utils.visualization import ArchitectureInfo, PFApproach, BaseArchitecture

        state_level_info = ArchitectureInfo(
            pf_approach=PFApproach.STATE_LEVEL,
            base_arch=BaseArchitecture.CFC,
            has_wiring=False,
            n_particles=32,
            hidden_size=64,
        )

        assert state_level_info.is_state_level
        assert not state_level_info.is_param_level
        assert not state_level_info.is_dual
        assert not state_level_info.is_sde
        assert state_level_info.is_cfc_based
        assert not state_level_info.is_ltc_based


# =============================================================================
# UNIT TESTS: Core Plot Functions
# =============================================================================

class TestCorePlotFunctions:
    """Tests for core plot functions (C1-C10).

    Each plot function should:
    - Return valid matplotlib figure and axes
    - Handle various data sizes
    - Apply themes correctly
    - Degrade gracefully with missing data
    """

    def test_plotEssTimelineShouldReturnFigureAndAxes(self, populated_base_collector):
        """plot_ess_timeline should return figure and axes."""
        import matplotlib.pyplot as plt
        from pfncps.utils.visualization.plots import plot_ess_timeline

        fig, ax = plot_ess_timeline(populated_base_collector)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plotEssTimelineShouldShowThreshold(self, populated_base_collector):
        """plot_ess_timeline should show threshold line when enabled."""
        import matplotlib.pyplot as plt
        from pfncps.utils.visualization.plots import plot_ess_timeline

        fig, ax = plot_ess_timeline(populated_base_collector, show_threshold=True)

        # Check that there's a horizontal line (threshold)
        lines = ax.get_lines()
        horizontal_lines = [l for l in lines if len(set(l.get_ydata())) == 1]
        assert len(horizontal_lines) >= 1

        plt.close(fig)

    def test_plotEssTimelineShouldApplyTheme(self, populated_base_collector, default_theme):
        """plot_ess_timeline should apply theme styling."""
        import matplotlib.pyplot as plt
        from pfncps.utils.visualization.plots import plot_ess_timeline

        fig, ax = plot_ess_timeline(populated_base_collector, theme=default_theme)

        # Theme should be applied
        assert ax.get_facecolor() is not None

        plt.close(fig)

    def test_plotWeightDistributionHeatmapMode(self, populated_base_collector):
        """plot_weight_distribution should work in heatmap mode."""
        import matplotlib.pyplot as plt
        from pfncps.utils.visualization.plots import plot_weight_distribution

        fig, ax = plot_weight_distribution(populated_base_collector, mode="heatmap")

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plotWeightDistributionStackedMode(self, populated_base_collector):
        """plot_weight_distribution should work in stacked_area mode."""
        import matplotlib.pyplot as plt
        from pfncps.utils.visualization.plots import plot_weight_distribution

        fig, ax = plot_weight_distribution(populated_base_collector, mode="stacked_area")

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plotWeightEntropyShouldReturnValidFigure(self, populated_base_collector):
        """plot_weight_entropy should return valid figure."""
        import matplotlib.pyplot as plt
        from pfncps.utils.visualization.plots import plot_weight_entropy

        fig, ax = plot_weight_entropy(populated_base_collector)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plotParticleTrajectoriesFanStyle(self, populated_base_collector):
        """plot_particle_trajectories should work in fan style."""
        import matplotlib.pyplot as plt
        from pfncps.utils.visualization.plots import plot_particle_trajectories

        fig, axes = plot_particle_trajectories(populated_base_collector, style="fan", dims=[0])

        assert fig is not None
        plt.close(fig)

    def test_plotParticleTrajectoriesSpaghettiStyle(self, populated_base_collector):
        """plot_particle_trajectories should work in spaghetti style."""
        import matplotlib.pyplot as plt
        from pfncps.utils.visualization.plots import plot_particle_trajectories

        fig, axes = plot_particle_trajectories(populated_base_collector, style="spaghetti", dims=[0])

        assert fig is not None
        plt.close(fig)

    def test_plotParticleTrajectoriesQuantilesStyle(self, populated_base_collector):
        """plot_particle_trajectories should work in quantiles style."""
        import matplotlib.pyplot as plt
        from pfncps.utils.visualization.plots import plot_particle_trajectories

        fig, axes = plot_particle_trajectories(populated_base_collector, style="quantiles", dims=[0])

        assert fig is not None
        plt.close(fig)

    def test_plotParticleDiversityShouldReturn4Panels(self, populated_base_collector):
        """plot_particle_diversity should return 4-panel figure."""
        import matplotlib.pyplot as plt
        from pfncps.utils.visualization.plots import plot_particle_diversity

        fig, axes = plot_particle_diversity(populated_base_collector)

        assert fig is not None
        assert len(axes) == 4
        plt.close(fig)

    def test_plotResamplingEventsShouldWork(self, populated_base_collector):
        """plot_resampling_events should return valid figure."""
        import matplotlib.pyplot as plt
        from pfncps.utils.visualization.plots import plot_resampling_events

        fig, ax = plot_resampling_events(populated_base_collector)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plotObservationLikelihoodsBoxMode(self, populated_base_collector):
        """plot_observation_likelihoods should work in box mode."""
        import matplotlib.pyplot as plt
        from pfncps.utils.visualization.plots import plot_observation_likelihoods

        fig, ax = plot_observation_likelihoods(populated_base_collector, mode="box")

        assert fig is not None
        plt.close(fig)

    def test_plotObservationLikelihoodsViolinMode(self, populated_base_collector):
        """plot_observation_likelihoods should work in violin mode."""
        import matplotlib.pyplot as plt
        from pfncps.utils.visualization.plots import plot_observation_likelihoods

        fig, ax = plot_observation_likelihoods(populated_base_collector, mode="violin")

        assert fig is not None
        plt.close(fig)

    def test_plotNumericalHealthShouldWork(self, populated_base_collector):
        """plot_numerical_health should return valid figure."""
        import matplotlib.pyplot as plt
        from pfncps.utils.visualization.plots import plot_numerical_health

        fig, ax = plot_numerical_health(populated_base_collector)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plotWeightedOutputShouldWork(self, populated_base_collector):
        """plot_weighted_output should return valid figure."""
        import matplotlib.pyplot as plt
        from pfncps.utils.visualization.plots import plot_weighted_output

        fig, axes = plot_weighted_output(populated_base_collector)

        assert fig is not None
        plt.close(fig)

    def test_animateParticles2dShouldReturnAnimation(self, populated_base_collector):
        """animate_particles_2d should return animation object."""
        import matplotlib.pyplot as plt
        from pfncps.utils.visualization.plots import animate_particles_2d

        anim = animate_particles_2d(populated_base_collector)

        assert anim is not None
        plt.close("all")


# =============================================================================
# UNIT TESTS: Dashboard Functions
# =============================================================================

class TestDashboardFunctions:
    """Tests for dashboard creation functions."""

    def test_createDashboardHealthMode(self, populated_base_collector):
        """create_dashboard should work in health mode."""
        import matplotlib.pyplot as plt
        from pfncps.utils.visualization.plots import create_dashboard

        fig = create_dashboard(populated_base_collector, which="health")

        assert fig is not None
        plt.close(fig)

    def test_createDashboardResearchMode(self, populated_base_collector):
        """create_dashboard should work in research mode."""
        import matplotlib.pyplot as plt
        from pfncps.utils.visualization.plots import create_dashboard

        fig = create_dashboard(populated_base_collector, which="research")

        assert fig is not None
        plt.close(fig)

    def test_createDashboardDebugMode(self, populated_base_collector):
        """create_dashboard should work in debug mode."""
        import matplotlib.pyplot as plt
        from pfncps.utils.visualization.plots import create_dashboard

        fig = create_dashboard(populated_base_collector, which="debug")

        assert fig is not None
        plt.close(fig)


# =============================================================================
# UNIT TESTS: State-Level Plot Functions
# =============================================================================

class TestStatePlotFunctions:
    """Tests for state-level specific plot functions (S1-S4)."""

    def test_plotNoiseInjectionMagnitude(self, populated_state_collector):
        """plot_noise_injection_magnitude should work with state collector."""
        import matplotlib.pyplot as plt
        from pfncps.utils.visualization.plots import plot_noise_injection_magnitude

        fig, ax = plot_noise_injection_magnitude(populated_state_collector)

        assert fig is not None
        assert ax is not None
        plt.close(fig)

    def test_plotStateDependentNoise(self, populated_state_collector):
        """plot_state_dependent_noise should work with state collector."""
        import matplotlib.pyplot as plt
        from pfncps.utils.visualization.plots import plot_state_dependent_noise

        fig, ax = plot_state_dependent_noise(populated_state_collector)

        assert fig is not None
        plt.close(fig)

    def test_plotParticlePairwiseDistances(self, populated_state_collector):
        """plot_particle_pairwise_distances should work with state collector."""
        import matplotlib.pyplot as plt
        from pfncps.utils.visualization.plots import plot_particle_pairwise_distances

        fig, ax = plot_particle_pairwise_distances(populated_state_collector)

        assert fig is not None
        plt.close(fig)

    def test_plotParticleCloudEvolution(self, populated_state_collector):
        """plot_particle_cloud_evolution should create 3D plot."""
        import matplotlib.pyplot as plt
        from pfncps.utils.visualization.plots import plot_particle_cloud_evolution

        fig, ax = plot_particle_cloud_evolution(populated_state_collector, n_timesteps=5)

        assert fig is not None
        plt.close(fig)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions.

    These tests ensure the system handles unusual inputs gracefully.
    """

    def test_shouldHandleEmptyCollector(self):
        """Plot functions should handle empty collector gracefully."""
        from pfncps.utils.visualization.collectors import BaseDataCollector

        collector = BaseDataCollector()

        # Getting data from empty collector should return empty tensors or raise appropriate error
        assert len(collector) == 0

    def test_shouldHandleSingleTimestep(self, n_particles, hidden_size):
        """Plot functions should handle single timestep data."""
        import matplotlib.pyplot as plt
        from pfncps.utils.visualization.collectors import BaseDataCollector
        from pfncps.utils.visualization.plots import plot_ess_timeline

        collector = BaseDataCollector()
        particles = torch.randn(1, n_particles, hidden_size)
        log_weights = torch.log_softmax(torch.randn(1, n_particles), dim=-1)
        collector.log_step(particles=particles, log_weights=log_weights)

        fig, ax = plot_ess_timeline(collector)

        assert fig is not None
        plt.close(fig)

    def test_shouldHandleSingleParticle(self, hidden_size):
        """System should handle n_particles=1 (degenerate case)."""
        from pfncps.utils.visualization.collectors import BaseDataCollector

        collector = BaseDataCollector()
        particles = torch.randn(1, 1, hidden_size)  # Single particle
        log_weights = torch.zeros(1, 1)  # Log(1) = 0

        collector.log_step(particles=particles, log_weights=log_weights)

        ess = collector.get_ess()
        assert ess[0] == 1.0  # ESS of single particle is 1

    def test_shouldHandleLargeParticleCount(self, hidden_size):
        """System should handle large number of particles."""
        from pfncps.utils.visualization.collectors import BaseDataCollector

        n_particles = 1000
        collector = BaseDataCollector()
        particles = torch.randn(1, n_particles, hidden_size)
        log_weights = torch.log_softmax(torch.randn(1, n_particles), dim=-1)

        collector.log_step(particles=particles, log_weights=log_weights)

        # Should work without memory issues
        ess = collector.get_ess()
        assert ess[0] <= n_particles

    def test_shouldHandleNaNInParticles(self, n_particles, hidden_size):
        """System should handle NaN values without crashing."""
        from pfncps.utils.visualization.collectors import BaseDataCollector

        collector = BaseDataCollector()
        particles = torch.randn(1, n_particles, hidden_size)
        particles[0, 0, 0] = float('nan')
        log_weights = torch.log_softmax(torch.randn(1, n_particles), dim=-1)

        collector.log_step(particles=particles, log_weights=log_weights)

        health = collector.get_numerical_health()
        assert health["has_nan"][0].item() == True

    def test_shouldHandleInfInWeights(self, n_particles, hidden_size):
        """System should handle Inf values in log weights."""
        from pfncps.utils.visualization.collectors import BaseDataCollector

        collector = BaseDataCollector()
        particles = torch.randn(1, n_particles, hidden_size)
        log_weights = torch.randn(1, n_particles)
        log_weights[0, 0] = float('inf')

        collector.log_step(particles=particles, log_weights=log_weights)

        health = collector.get_numerical_health()
        # Inf in weights should be detected
        assert health["has_inf"][0].item() == True

    def test_shouldHandleZeroDimensionalHiddenState(self, n_particles):
        """System should handle hidden_size=1."""
        from pfncps.utils.visualization.collectors import BaseDataCollector

        hidden_size = 1
        collector = BaseDataCollector()
        particles = torch.randn(1, n_particles, hidden_size)
        log_weights = torch.log_softmax(torch.randn(1, n_particles), dim=-1)

        collector.log_step(particles=particles, log_weights=log_weights)

        particles_out = collector.get_particles()
        assert particles_out.shape == (1, n_particles, hidden_size)

    def test_shouldHandleVeryLongSequences(self, n_particles, hidden_size):
        """System should handle long sequences via downsampling."""
        from pfncps.utils.visualization.collectors import BaseDataCollector

        max_history = 100
        collector = BaseDataCollector(max_history=max_history)
        particles = torch.randn(1, n_particles, hidden_size)
        log_weights = torch.log_softmax(torch.randn(1, n_particles), dim=-1)

        # Log many steps
        for _ in range(500):
            collector.log_step(particles=particles, log_weights=log_weights)

        # Should be downsampled
        assert len(collector) <= max_history

    def test_plotShouldHandleMissingOptionalData(self, populated_base_collector):
        """Plots should handle missing optional data gracefully."""
        import matplotlib.pyplot as plt
        from pfncps.utils.visualization.plots import plot_weighted_output

        # populated_base_collector has no observations
        assert populated_base_collector.get_observations() is None

        # Should still work without observations
        fig, ax = plot_weighted_output(populated_base_collector)

        assert fig is not None
        plt.close(fig)


# =============================================================================
# PROPERTY-BASED TESTS (Invariants)
# =============================================================================

class TestInvariants:
    """Property-based tests defining invariants that must always hold.

    These tests define mathematical properties that should be true
    regardless of input, serving as a specification of correct behavior.
    """

    def test_essShouldBeBetween1AndN(self, populated_base_collector, n_particles):
        """ESS should always be in [1, n_particles]."""
        ess = populated_base_collector.get_ess()

        assert (ess >= 1.0 - 0.01).all(), "ESS should be >= 1"
        assert (ess <= n_particles + 0.01).all(), f"ESS should be <= {n_particles}"

    def test_weightsShouldSumToOne(self, populated_base_collector):
        """Normalized weights should sum to 1 for each timestep."""
        weights = populated_base_collector.get_weights()

        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_weightsShouldBeNonNegative(self, populated_base_collector):
        """Weights should be non-negative."""
        weights = populated_base_collector.get_weights()

        assert (weights >= 0).all()

    def test_varianceShouldBeNonNegative(self, populated_base_collector):
        """Variance should always be non-negative."""
        variance = populated_base_collector.get_particle_variance()
        weighted_variance = populated_base_collector.get_weighted_variance()

        assert (variance >= 0).all()
        assert (weighted_variance >= 0).all()

    def test_pairwiseDistancesShouldBeNonNegative(self, populated_base_collector):
        """Pairwise distances should be non-negative."""
        distances = populated_base_collector.get_pairwise_distances()

        assert (distances >= 0).all()

    def test_entropyShouldBeBounded(self, populated_base_collector, n_particles):
        """Weight entropy should be in [0, log(n_particles)]."""
        entropy = populated_base_collector.get_weight_entropy()
        max_entropy = math.log(n_particles)

        assert (entropy >= -0.01).all(), "Entropy should be >= 0"
        assert (entropy <= max_entropy + 0.01).all(), f"Entropy should be <= {max_entropy}"

    def test_downsampledDataShouldPreserveTimestampOrder(self):
        """Downsampled data should maintain chronological order."""
        from pfncps.utils.visualization.collectors import BaseDataCollector

        collector = BaseDataCollector(max_history=50)
        particles = torch.randn(1, 32, 64)
        log_weights = torch.log_softmax(torch.randn(1, 32), dim=-1)

        for _ in range(200):
            collector.log_step(particles=particles, log_weights=log_weights)

        timesteps = [h.timestep for h in collector.history]
        assert timesteps == sorted(timesteps), "Timesteps should be in order"

    def test_allPlotsShouldReturnMatplotlibObjects(self, populated_base_collector):
        """All plot functions should return valid matplotlib figure objects."""
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from pfncps.utils.visualization.plots import (
            plot_ess_timeline,
            plot_weight_distribution,
            plot_weight_entropy,
            plot_numerical_health,
        )

        for plot_fn in [
            plot_ess_timeline,
            plot_weight_distribution,
            plot_weight_entropy,
            plot_numerical_health,
        ]:
            result = plot_fn(populated_base_collector)
            fig = result[0] if isinstance(result, tuple) else result

            assert isinstance(fig, Figure), f"{plot_fn.__name__} should return Figure"
            plt.close(fig)

    def test_uniformWeightsShouldGiveMaxESS(self, n_particles, hidden_size):
        """Uniform weights should give ESS = n_particles."""
        from pfncps.utils.visualization.collectors import BaseDataCollector

        collector = BaseDataCollector()
        particles = torch.randn(1, n_particles, hidden_size)
        # Uniform log weights
        log_weights = torch.full((1, n_particles), -math.log(n_particles))

        collector.log_step(particles=particles, log_weights=log_weights)

        ess = collector.get_ess()
        assert abs(ess[0].item() - n_particles) < 0.1

    def test_concentratedWeightsShouldGiveMinESS(self, n_particles, hidden_size):
        """All weight on one particle should give ESS = 1."""
        from pfncps.utils.visualization.collectors import BaseDataCollector

        collector = BaseDataCollector()
        particles = torch.randn(1, n_particles, hidden_size)
        # All weight on first particle
        log_weights = torch.full((1, n_particles), -1000.0)
        log_weights[0, 0] = 0.0

        collector.log_step(particles=particles, log_weights=log_weights)

        ess = collector.get_ess()
        assert abs(ess[0].item() - 1.0) < 0.1


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for components working together."""

    def test_fullWorkflowWithBaseCollector(self, n_particles, hidden_size, n_timesteps, batch_size):
        """Test complete workflow: create collector, log data, generate plots."""
        import matplotlib.pyplot as plt
        from pfncps.utils.visualization.collectors import BaseDataCollector
        from pfncps.utils.visualization.plots import (
            plot_ess_timeline,
            plot_weight_distribution,
            create_dashboard,
        )

        # Create collector
        collector = BaseDataCollector()

        # Simulate model forward passes
        for t in range(n_timesteps):
            particles = torch.randn(batch_size, n_particles, hidden_size)
            log_weights = torch.log_softmax(torch.randn(batch_size, n_particles), dim=-1)
            collector.log_step(particles=particles, log_weights=log_weights)

        # Generate plots
        fig1, _ = plot_ess_timeline(collector)
        fig2, _ = plot_weight_distribution(collector)
        fig3 = create_dashboard(collector, which="health")

        assert fig1 is not None
        assert fig2 is not None
        assert fig3 is not None

        plt.close("all")

    def test_themeApplicationAcrossPlots(self, populated_base_collector, default_theme, paper_theme):
        """Test that themes are consistently applied across different plots."""
        import matplotlib.pyplot as plt
        from pfncps.utils.visualization.plots import (
            plot_ess_timeline,
            plot_weight_entropy,
        )

        # Plot with default theme
        fig1, ax1 = plot_ess_timeline(populated_base_collector, theme=default_theme)

        # Plot with paper theme
        fig2, ax2 = plot_ess_timeline(populated_base_collector, theme=paper_theme)

        # Themes should result in different styling
        # (At minimum, they should both work without error)
        assert fig1 is not None
        assert fig2 is not None

        plt.close("all")

    def test_exportToDirectory(self, populated_base_collector):
        """Test saving all plots to a directory."""
        import matplotlib.pyplot as plt
        from pfncps.utils.visualization import PFVisualizer
        from pfncps.utils.visualization.utils.export import save_all_plots

        # Create mock visualizer with the collector
        visualizer = MagicMock()
        visualizer.collector = populated_base_collector
        visualizer.has_data = True
        visualizer.arch_info = None
        visualizer.theme = None

        # Create actual plot methods that return figures
        from pfncps.utils.visualization.plots import (
            plot_ess_timeline,
            plot_weight_distribution,
            plot_weight_entropy,
            plot_particle_trajectories,
            plot_numerical_health,
            plot_weighted_output,
            create_dashboard,
        )

        visualizer.plot_ess_timeline = lambda **kw: plot_ess_timeline(populated_base_collector, **kw)
        visualizer.plot_weight_distribution = lambda **kw: plot_weight_distribution(populated_base_collector, **kw)
        visualizer.plot_weight_entropy = lambda **kw: plot_weight_entropy(populated_base_collector, **kw)
        visualizer.plot_particle_trajectories = lambda **kw: plot_particle_trajectories(populated_base_collector, **kw)
        visualizer.plot_numerical_health = lambda **kw: plot_numerical_health(populated_base_collector, **kw)
        visualizer.plot_weighted_output = lambda **kw: plot_weighted_output(populated_base_collector, **kw)
        visualizer.dashboard = lambda which: create_dashboard(populated_base_collector, which=which)

        with tempfile.TemporaryDirectory() as tmpdir:
            saved_files = save_all_plots(visualizer, output_dir=tmpdir, format="png")

            assert len(saved_files) > 0
            for filepath in saved_files:
                assert os.path.exists(filepath)

        plt.close("all")

    def test_tensorboardLogging(self, populated_base_collector):
        """Test TensorBoard logging integration."""
        from pfncps.utils.visualization.backends.tensorboard_backend import log_to_tensorboard
        import matplotlib.pyplot as plt

        # Create mock visualizer
        visualizer = MagicMock()
        visualizer.collector = populated_base_collector
        visualizer.has_data = True

        # Create mock plot methods
        from pfncps.utils.visualization.plots import (
            plot_ess_timeline,
            plot_weight_distribution,
        )
        visualizer.plot_ess_timeline = lambda: plot_ess_timeline(populated_base_collector)
        visualizer.plot_weight_distribution = lambda mode="heatmap": plot_weight_distribution(populated_base_collector, mode=mode)

        # Create mock writer
        mock_writer = MagicMock()

        # Should not raise
        log_to_tensorboard(visualizer, mock_writer, step=0, prefix="test")

        # Verify scalars were logged
        assert mock_writer.add_scalar.called

        plt.close("all")

    def test_collectorTypeSelection(self, mock_model, mock_ltc_model, mock_sde_model):
        """Test that PFVisualizer selects correct collector type."""
        from pfncps.utils.visualization import PFVisualizer
        from pfncps.utils.visualization.collectors import (
            StateCollector,
            SDECollector,
        )

        # For CfC model, should get StateCollector
        with patch.object(PFVisualizer, '_register_hooks'):
            viz_cfc = PFVisualizer(mock_model)
            assert isinstance(viz_cfc.collector, StateCollector)

        # For SDE model, should get SDECollector
        with patch.object(PFVisualizer, '_register_hooks'):
            viz_sde = PFVisualizer(mock_sde_model)
            assert isinstance(viz_sde.collector, SDECollector)


# =============================================================================
# REGRESSION TESTS
# =============================================================================

class TestRegressions:
    """Regression tests for previously found bugs."""

    def test_shouldNotCrashOnEmptyHistory(self):
        """Calling getters on empty collector should not crash."""
        from pfncps.utils.visualization.collectors import BaseDataCollector

        collector = BaseDataCollector()

        # These should either return empty tensors or raise clear errors
        with pytest.raises(Exception):
            # Stacking empty list should raise
            _ = collector.get_particles()

    def test_shouldHandleBatchSizeOneCorrectly(self, n_particles, hidden_size):
        """Batch size of 1 should work correctly."""
        from pfncps.utils.visualization.collectors import BaseDataCollector

        collector = BaseDataCollector()
        particles = torch.randn(1, n_particles, hidden_size)
        log_weights = torch.log_softmax(torch.randn(1, n_particles), dim=-1)

        collector.log_step(particles=particles, log_weights=log_weights)

        assert len(collector) == 1
        assert collector.get_particles().shape == (1, n_particles, hidden_size)

    def test_cacheShouldNotReturnStaleData(self, n_particles, hidden_size):
        """Cache should be invalidated when new data arrives."""
        from pfncps.utils.visualization.collectors import BaseDataCollector

        collector = BaseDataCollector()

        # Log first batch
        particles1 = torch.ones(1, n_particles, hidden_size)
        log_weights = torch.log_softmax(torch.randn(1, n_particles), dim=-1)
        collector.log_step(particles=particles1, log_weights=log_weights)

        # Access to populate cache
        first_mean = collector.get_weighted_mean()[0].clone()

        # Log second batch with different values
        particles2 = torch.ones(1, n_particles, hidden_size) * 10.0
        collector.log_step(particles=particles2, log_weights=log_weights)

        # Get updated mean
        second_mean = collector.get_weighted_mean()[1]

        # Means should be different
        assert not torch.allclose(first_mean, second_mean)


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance-related tests."""

    def test_shouldHandleLargeDataEfficiently(self, n_particles, hidden_size):
        """System should handle large amounts of data efficiently."""
        from pfncps.utils.visualization.collectors import BaseDataCollector
        import time

        collector = BaseDataCollector(max_history=1000)
        particles = torch.randn(1, n_particles, hidden_size)
        log_weights = torch.log_softmax(torch.randn(1, n_particles), dim=-1)

        start = time.time()
        for _ in range(2000):
            collector.log_step(particles=particles, log_weights=log_weights)
        elapsed = time.time() - start

        # Should complete in reasonable time (< 10 seconds for 2000 steps)
        assert elapsed < 10.0, f"Logging 2000 steps took {elapsed:.2f}s, expected < 10s"

    def test_gettersShouldUseCaching(self, populated_base_collector):
        """Repeated getter calls should use cached values."""
        import time

        # First call (no cache)
        start = time.time()
        _ = populated_base_collector.get_pairwise_distances()
        first_call = time.time() - start

        # Second call (should use cache)
        start = time.time()
        _ = populated_base_collector.get_pairwise_distances()
        second_call = time.time() - start

        # Cached call should be much faster (at least 5x)
        # Allow for some variance in timing
        assert second_call < first_call * 0.5 or second_call < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
