"""Shared fixtures for pfncps test suite."""

import math
from typing import Tuple

import pytest
import torch
from torch import Tensor


# =============================================================================
# Device Configuration
# =============================================================================

def get_available_device() -> torch.device:
    """Get the best available device for testing."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def device() -> torch.device:
    """Primary test device (MPS > CUDA > CPU)."""
    return get_available_device()


@pytest.fixture
def cpu_device() -> torch.device:
    """CPU device for explicit CPU tests."""
    return torch.device("cpu")


# =============================================================================
# Dimension Fixtures (Parameterized)
# =============================================================================

@pytest.fixture(params=[1, 4, 16])
def batch_size(request) -> int:
    """Test with various batch sizes."""
    return request.param


@pytest.fixture(params=[1, 8, 32])
def n_particles(request) -> int:
    """Test with various particle counts."""
    return request.param


@pytest.fixture(params=[16, 64])
def hidden_size(request) -> int:
    """Test with various hidden dimensions."""
    return request.param


@pytest.fixture(params=[8, 20])
def input_size(request) -> int:
    """Test with various input dimensions."""
    return request.param


@pytest.fixture(params=[4, 10])
def obs_size(request) -> int:
    """Test with various observation dimensions."""
    return request.param


@pytest.fixture(params=[10, 50])
def seq_len(request) -> int:
    """Test with various sequence lengths."""
    return request.param


# =============================================================================
# Fixed Dimension Fixtures (Non-parameterized)
# =============================================================================

@pytest.fixture
def batch_size_fixed() -> int:
    """Fixed batch size for simpler tests."""
    return 4


@pytest.fixture
def n_particles_fixed() -> int:
    """Fixed particle count for simpler tests."""
    return 16


@pytest.fixture
def hidden_size_fixed() -> int:
    """Fixed hidden size for simpler tests."""
    return 32


@pytest.fixture
def input_size_fixed() -> int:
    """Fixed input size for simpler tests."""
    return 20


@pytest.fixture
def obs_size_fixed() -> int:
    """Fixed observation size for simpler tests."""
    return 10


# =============================================================================
# Tensor Fixtures
# =============================================================================

@pytest.fixture
def uniform_log_weights(batch_size: int, n_particles: int) -> Tensor:
    """Create uniform log weights: log(1/K) for all particles."""
    return torch.full((batch_size, n_particles), -math.log(n_particles))


@pytest.fixture
def uniform_log_weights_fixed(batch_size_fixed: int, n_particles_fixed: int) -> Tensor:
    """Create uniform log weights with fixed dimensions."""
    return torch.full(
        (batch_size_fixed, n_particles_fixed),
        -math.log(n_particles_fixed)
    )


@pytest.fixture
def peaked_log_weights(batch_size: int, n_particles: int) -> Tensor:
    """Create log weights where first particle dominates.

    Creates near-degenerate distribution where particle 0 has
    almost all weight.
    """
    log_weights = torch.full((batch_size, n_particles), -100.0)
    log_weights[:, 0] = 0.0
    # Normalize in log space
    log_normalizer = torch.logsumexp(log_weights, dim=-1, keepdim=True)
    return log_weights - log_normalizer


@pytest.fixture
def peaked_log_weights_fixed(batch_size_fixed: int, n_particles_fixed: int) -> Tensor:
    """Create peaked log weights with fixed dimensions."""
    log_weights = torch.full((batch_size_fixed, n_particles_fixed), -100.0)
    log_weights[:, 0] = 0.0
    log_normalizer = torch.logsumexp(log_weights, dim=-1, keepdim=True)
    return log_weights - log_normalizer


@pytest.fixture
def random_log_weights(batch_size: int, n_particles: int) -> Tensor:
    """Create random normalized log weights."""
    log_weights = torch.randn(batch_size, n_particles)
    log_normalizer = torch.logsumexp(log_weights, dim=-1, keepdim=True)
    return log_weights - log_normalizer


@pytest.fixture
def random_particles(batch_size: int, n_particles: int, hidden_size: int) -> Tensor:
    """Create random particle states."""
    return torch.randn(batch_size, n_particles, hidden_size)


@pytest.fixture
def random_particles_fixed(
    batch_size_fixed: int,
    n_particles_fixed: int,
    hidden_size_fixed: int
) -> Tensor:
    """Create random particle states with fixed dimensions."""
    return torch.randn(batch_size_fixed, n_particles_fixed, hidden_size_fixed)


@pytest.fixture
def random_input(batch_size: int, input_size: int) -> Tensor:
    """Create random input tensor."""
    return torch.randn(batch_size, input_size)


@pytest.fixture
def random_observation(batch_size: int, obs_size: int) -> Tensor:
    """Create random observation tensor."""
    return torch.randn(batch_size, obs_size)


@pytest.fixture
def random_sequence(batch_size: int, seq_len: int, input_size: int) -> Tensor:
    """Create random input sequence."""
    return torch.randn(batch_size, seq_len, input_size)


# =============================================================================
# Tolerance Fixtures
# =============================================================================

@pytest.fixture
def tolerance() -> dict:
    """Default numerical tolerances for floating point comparisons."""
    return {"atol": 1e-5, "rtol": 1e-4}


@pytest.fixture
def loose_tolerance() -> dict:
    """Looser tolerances for stochastic operations."""
    return {"atol": 1e-3, "rtol": 1e-2}


@pytest.fixture
def strict_tolerance() -> dict:
    """Stricter tolerances for precise operations."""
    return {"atol": 1e-7, "rtol": 1e-6}


# =============================================================================
# Wiring Fixtures (for NCP integration)
# =============================================================================

@pytest.fixture
def auto_wiring(hidden_size, input_size):
    """Create an AutoNCP wiring for tests."""
    try:
        from pfncps.wirings import AutoNCP
        wiring = AutoNCP(units=hidden_size, output_dim=min(hidden_size, 10))
        wiring.build(input_size)
        return wiring
    except ImportError:
        pytest.skip("pfncps.wirings not available")


@pytest.fixture
def fully_connected_wiring(hidden_size, input_size):
    """Create a FullyConnected wiring for tests."""
    from pfncps.wirings import FullyConnected
    wiring = FullyConnected(units=hidden_size, output_dim=min(hidden_size, 10))
    wiring.build(input_size)
    return wiring


@pytest.fixture
def fully_connected_wiring_fixed(hidden_size_fixed, input_size_fixed):
    """Create a FullyConnected wiring for tests with fixed dimensions."""
    from pfncps.wirings import FullyConnected
    wiring = FullyConnected(units=hidden_size_fixed, output_dim=10)
    wiring.build(input_size_fixed)
    return wiring


# =============================================================================
# Alpha Mode Fixtures
# =============================================================================

@pytest.fixture(params=["fixed", "adaptive", "learnable"])
def alpha_mode(request) -> str:
    """Test all alpha modes."""
    return request.param


@pytest.fixture(params=[0.0, 0.25, 0.5, 0.75, 1.0])
def alpha_value(request) -> float:
    """Test various alpha values."""
    return request.param


# =============================================================================
# Noise Type Fixtures
# =============================================================================

@pytest.fixture(params=["constant", "time_scaled", "learned", "state_dependent"])
def noise_type(request) -> str:
    """Test all noise injection types."""
    return request.param


# =============================================================================
# Observation Model Fixtures
# =============================================================================

@pytest.fixture
def gaussian_observation_model(hidden_size_fixed: int, obs_size_fixed: int):
    """Create a GaussianObservationModel for tests."""
    from pfncps.nn import GaussianObservationModel
    return GaussianObservationModel(hidden_size_fixed, obs_size_fixed)


@pytest.fixture
def classification_observation_model(hidden_size_fixed: int):
    """Create a ClassificationObservationModel for tests."""
    from pfncps.nn import ClassificationObservationModel
    n_classes = 5
    return ClassificationObservationModel(hidden_size_fixed, n_classes)


# =============================================================================
# Resampler Fixtures
# =============================================================================

@pytest.fixture
def soft_resampler_fixed(n_particles_fixed: int):
    """Create a SoftResampler with fixed alpha."""
    from pfncps.nn import SoftResampler
    return SoftResampler(n_particles_fixed, alpha_mode="fixed", alpha_init=0.5)


@pytest.fixture
def soft_resampler_adaptive(n_particles_fixed: int):
    """Create a SoftResampler with adaptive alpha."""
    from pfncps.nn import SoftResampler
    return SoftResampler(n_particles_fixed, alpha_mode="adaptive")


@pytest.fixture
def soft_resampler_learnable(n_particles_fixed: int):
    """Create a SoftResampler with learnable alpha."""
    from pfncps.nn import SoftResampler
    return SoftResampler(n_particles_fixed, alpha_mode="learnable")


# =============================================================================
# Cell Fixtures
# =============================================================================

@pytest.fixture
def pf_cfc_cell(input_size_fixed: int, hidden_size_fixed: int, n_particles_fixed: int):
    """Create a PFCfCCell for tests."""
    from pfncps.nn import PFCfCCell
    return PFCfCCell(input_size_fixed, hidden_size_fixed, n_particles_fixed)


@pytest.fixture
def param_pf_cfc_cell(input_size_fixed: int, hidden_size_fixed: int, n_particles_fixed: int):
    """Create a ParamPFCfCCell for tests."""
    from pfncps.nn import ParamPFCfCCell
    return ParamPFCfCCell(input_size_fixed, hidden_size_fixed, n_particles_fixed)


@pytest.fixture
def dual_pf_cfc_cell(input_size_fixed: int, hidden_size_fixed: int, n_particles_fixed: int):
    """Create a DualPFCfCCell for tests."""
    from pfncps.nn import DualPFCfCCell
    return DualPFCfCCell(input_size_fixed, hidden_size_fixed, n_particles_fixed)


# =============================================================================
# Approach Fixtures
# =============================================================================

@pytest.fixture(params=["state", "param", "dual", "sde"])
def approach(request) -> str:
    """Test all PF approaches."""
    return request.param


# =============================================================================
# Helper Functions (Not fixtures)
# =============================================================================

def assert_no_nan_inf(tensor: Tensor, name: str = "tensor") -> None:
    """Assert that a tensor contains no NaN or Inf values."""
    assert not torch.isnan(tensor).any(), f"NaN detected in {name}"
    assert not torch.isinf(tensor).any(), f"Inf detected in {name}"


def assert_valid_log_weights(log_weights: Tensor, atol: float = 1e-4) -> None:
    """Assert that log weights form a valid probability distribution."""
    assert_no_nan_inf(log_weights, "log_weights")
    sums = torch.exp(log_weights).sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=atol), \
        f"Log weights do not sum to 1: {sums}"


def assert_shape(tensor: Tensor, expected_shape: Tuple[int, ...], name: str = "tensor") -> None:
    """Assert that a tensor has the expected shape."""
    assert tensor.shape == expected_shape, \
        f"{name} has shape {tensor.shape}, expected {expected_shape}"


def assert_device(tensor: Tensor, device: torch.device, name: str = "tensor") -> None:
    """Assert that a tensor is on the expected device."""
    assert tensor.device.type == device.type, \
        f"{name} is on {tensor.device}, expected {device}"


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "hypothesis: marks property-based tests")
