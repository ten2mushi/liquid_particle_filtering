"""Unit tests for classification observation models (nn/observation/classification.py).

Tests for:
- ClassificationObservationModel
- MLPClassificationObservationModel
- OrdinalObservationModel
"""

import math
import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from pfncps.nn.observation.classification import (
    ClassificationObservationModel,
    MLPClassificationObservationModel,
    OrdinalObservationModel,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def n_classes() -> int:
    """Number of classes for classification."""
    return 5


@pytest.fixture
def observation_labels(batch_size: int, n_classes: int) -> Tensor:
    """Random class labels [batch]."""
    return torch.randint(0, n_classes, (batch_size,))


# =============================================================================
# Tests for ClassificationObservationModel
# =============================================================================

class TestClassificationObservationModel:
    """Tests for the ClassificationObservationModel class."""

    def test_output_shape(self, batch_size, n_particles, hidden_size, n_classes):
        """Log-likelihood has shape [batch, K]."""
        model = ClassificationObservationModel(hidden_size, n_classes)
        states = torch.randn(batch_size, n_particles, hidden_size)
        labels = torch.randint(0, n_classes, (batch_size,))

        log_lik = model.log_likelihood(states, labels)
        assert log_lik.shape == (batch_size, n_particles)

    def test_log_likelihood_non_positive(self, batch_size, n_particles, hidden_size, n_classes):
        """Classification log-likelihood <= 0 (log of probability)."""
        model = ClassificationObservationModel(hidden_size, n_classes)
        states = torch.randn(batch_size, n_particles, hidden_size)
        labels = torch.randint(0, n_classes, (batch_size,))

        log_lik = model.log_likelihood(states, labels)
        assert torch.all(log_lik <= 0)

    def test_log_likelihood_bounded_below(self, batch_size, n_particles, hidden_size, n_classes):
        """Log-likelihood bounded below by log(1/n_classes) approximately."""
        model = ClassificationObservationModel(hidden_size, n_classes)
        states = torch.randn(batch_size, n_particles, hidden_size)
        labels = torch.randint(0, n_classes, (batch_size,))

        log_lik = model.log_likelihood(states, labels)
        # Should be >= log(epsilon) for numerical stability
        assert torch.all(log_lik > -100)

    def test_get_logits_shape(self, batch_size, n_particles, hidden_size, n_classes):
        """Logits have shape [batch, K, n_classes]."""
        model = ClassificationObservationModel(hidden_size, n_classes)
        states = torch.randn(batch_size, n_particles, hidden_size)

        logits = model.get_logits(states)
        assert logits.shape == (batch_size, n_particles, n_classes)

    def test_get_probs_shape(self, batch_size, n_particles, hidden_size, n_classes):
        """Probabilities have shape [batch, K, n_classes]."""
        model = ClassificationObservationModel(hidden_size, n_classes)
        states = torch.randn(batch_size, n_particles, hidden_size)

        probs = model.get_probs(states)
        assert probs.shape == (batch_size, n_particles, n_classes)

    def test_probs_sum_to_one(self, batch_size, n_particles, hidden_size, n_classes):
        """Probabilities sum to 1 across classes."""
        model = ClassificationObservationModel(hidden_size, n_classes)
        states = torch.randn(batch_size, n_particles, hidden_size)

        probs = model.get_probs(states)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_probs_non_negative(self, batch_size, n_particles, hidden_size, n_classes):
        """Probabilities are non-negative."""
        model = ClassificationObservationModel(hidden_size, n_classes)
        states = torch.randn(batch_size, n_particles, hidden_size)

        probs = model.get_probs(states)
        assert torch.all(probs >= 0)

    def test_temperature_effect(self, batch_size, n_particles, hidden_size, n_classes):
        """Higher temperature = more uniform probabilities."""
        model_low_temp = ClassificationObservationModel(hidden_size, n_classes, temperature=0.1)
        model_high_temp = ClassificationObservationModel(hidden_size, n_classes, temperature=10.0)

        # Use same weights for both
        model_high_temp.classifier.weight.data = model_low_temp.classifier.weight.data.clone()
        model_high_temp.classifier.bias.data = model_low_temp.classifier.bias.data.clone()

        states = torch.randn(batch_size, n_particles, hidden_size)

        probs_low = model_low_temp.get_probs(states)
        probs_high = model_high_temp.get_probs(states)

        # High temp should have higher entropy (more uniform)
        entropy_low = -(probs_low * torch.log(probs_low + 1e-8)).sum(dim=-1).mean()
        entropy_high = -(probs_high * torch.log(probs_high + 1e-8)).sum(dim=-1).mean()

        assert entropy_high > entropy_low

    def test_learnable_temperature(self, hidden_size, n_classes):
        """Learnable temperature has gradient."""
        model = ClassificationObservationModel(
            hidden_size, n_classes, temperature=1.0, learnable_temperature=True
        )
        states = torch.randn(4, 8, hidden_size)
        labels = torch.randint(0, n_classes, (4,))

        log_lik = model.log_likelihood(states, labels)
        loss = -log_lik.mean()
        loss.backward()

        assert model.log_temperature.grad is not None

    def test_label_smoothing(self, batch_size, n_particles, hidden_size, n_classes):
        """Label smoothing modifies log-likelihood."""
        model_no_smooth = ClassificationObservationModel(
            hidden_size, n_classes, label_smoothing=0.0
        )
        model_smooth = ClassificationObservationModel(
            hidden_size, n_classes, label_smoothing=0.1
        )

        # Copy weights
        model_smooth.classifier.weight.data = model_no_smooth.classifier.weight.data.clone()
        model_smooth.classifier.bias.data = model_no_smooth.classifier.bias.data.clone()

        states = torch.randn(batch_size, n_particles, hidden_size)
        labels = torch.randint(0, n_classes, (batch_size,))

        log_lik_no_smooth = model_no_smooth.log_likelihood(states, labels)
        log_lik_smooth = model_smooth.log_likelihood(states, labels)

        # Should be different
        assert not torch.allclose(log_lik_no_smooth, log_lik_smooth)

    def test_predict_class_shape(self, batch_size, n_particles, hidden_size, n_classes):
        """predict_class returns [batch] tensor of class indices."""
        model = ClassificationObservationModel(hidden_size, n_classes)
        states = torch.randn(batch_size, n_particles, hidden_size)

        predicted = model.predict_class(states)
        assert predicted.shape == (batch_size,)
        assert predicted.dtype == torch.int64

    def test_predict_class_with_weights(self, batch_size, n_particles, hidden_size, n_classes):
        """predict_class uses weights when provided."""
        model = ClassificationObservationModel(hidden_size, n_classes)
        states = torch.randn(batch_size, n_particles, hidden_size)
        log_weights = torch.randn(batch_size, n_particles)

        predicted = model.predict_class(states, log_weights)
        assert predicted.shape == (batch_size,)
        assert torch.all(predicted >= 0) and torch.all(predicted < n_classes)

    def test_sample_shape(self, batch_size, n_particles, hidden_size, n_classes):
        """sample returns [batch, K] tensor of class indices."""
        model = ClassificationObservationModel(hidden_size, n_classes)
        states = torch.randn(batch_size, n_particles, hidden_size)

        samples = model.sample(states)
        assert samples.shape == (batch_size, n_particles)
        assert torch.all(samples >= 0) and torch.all(samples < n_classes)

    def test_no_nan_inf(self, batch_size, n_particles, hidden_size, n_classes):
        """No NaN or Inf in outputs."""
        model = ClassificationObservationModel(hidden_size, n_classes)
        states = torch.randn(batch_size, n_particles, hidden_size)
        labels = torch.randint(0, n_classes, (batch_size,))

        log_lik = model.log_likelihood(states, labels)
        probs = model.get_probs(states)

        assert not torch.isnan(log_lik).any()
        assert not torch.isinf(log_lik).any()
        assert not torch.isnan(probs).any()
        assert not torch.isinf(probs).any()


# =============================================================================
# Tests for MLPClassificationObservationModel
# =============================================================================

class TestMLPClassificationObservationModel:
    """Tests for the MLPClassificationObservationModel class."""

    def test_output_shape(self, batch_size, n_particles, hidden_size, n_classes):
        """Log-likelihood has shape [batch, K]."""
        model = MLPClassificationObservationModel(hidden_size, n_classes)
        states = torch.randn(batch_size, n_particles, hidden_size)
        labels = torch.randint(0, n_classes, (batch_size,))

        log_lik = model.log_likelihood(states, labels)
        assert log_lik.shape == (batch_size, n_particles)

    def test_log_likelihood_non_positive(self, batch_size, n_particles, hidden_size, n_classes):
        """Classification log-likelihood <= 0."""
        model = MLPClassificationObservationModel(hidden_size, n_classes)
        states = torch.randn(batch_size, n_particles, hidden_size)
        labels = torch.randint(0, n_classes, (batch_size,))

        log_lik = model.log_likelihood(states, labels)
        assert torch.all(log_lik <= 0)

    def test_custom_hidden_sizes(self, batch_size, n_particles, hidden_size, n_classes):
        """Custom MLP hidden sizes work."""
        model = MLPClassificationObservationModel(
            hidden_size, n_classes, mlp_hidden_sizes=[128, 64, 32]
        )
        states = torch.randn(batch_size, n_particles, hidden_size)
        labels = torch.randint(0, n_classes, (batch_size,))

        log_lik = model.log_likelihood(states, labels)
        assert log_lik.shape == (batch_size, n_particles)

    def test_different_activations(self, batch_size, n_particles, hidden_size, n_classes):
        """Different activation functions work."""
        for activation in ["tanh", "relu", "gelu"]:
            model = MLPClassificationObservationModel(
                hidden_size, n_classes, activation=activation
            )
            states = torch.randn(batch_size, n_particles, hidden_size)
            labels = torch.randint(0, n_classes, (batch_size,))

            log_lik = model.log_likelihood(states, labels)
            assert log_lik.shape == (batch_size, n_particles)

    def test_dropout(self, batch_size, n_particles, hidden_size, n_classes):
        """Model with dropout works (training vs eval mode)."""
        model = MLPClassificationObservationModel(
            hidden_size, n_classes, dropout=0.5
        )
        states = torch.randn(batch_size, n_particles, hidden_size)
        labels = torch.randint(0, n_classes, (batch_size,))

        model.train()
        log_lik_train1 = model.log_likelihood(states, labels)
        log_lik_train2 = model.log_likelihood(states, labels)

        model.eval()
        log_lik_eval1 = model.log_likelihood(states, labels)
        log_lik_eval2 = model.log_likelihood(states, labels)

        # Eval mode should be deterministic
        assert torch.allclose(log_lik_eval1, log_lik_eval2)

    def test_predict_shape(self, batch_size, n_particles, hidden_size, n_classes):
        """predict returns probabilities [batch, K, n_classes]."""
        model = MLPClassificationObservationModel(hidden_size, n_classes)
        states = torch.randn(batch_size, n_particles, hidden_size)

        probs = model.predict(states)
        assert probs.shape == (batch_size, n_particles, n_classes)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size, n_particles), atol=1e-5)

    def test_gradient_flow(self, hidden_size, n_classes):
        """Gradients flow through MLP."""
        model = MLPClassificationObservationModel(hidden_size, n_classes)
        states = torch.randn(4, 8, hidden_size, requires_grad=True)
        labels = torch.randint(0, n_classes, (4,))

        log_lik = model.log_likelihood(states, labels)
        loss = -log_lik.mean()
        loss.backward()

        assert states.grad is not None
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# =============================================================================
# Tests for OrdinalObservationModel
# =============================================================================

class TestOrdinalObservationModel:
    """Tests for the OrdinalObservationModel class."""

    def test_output_shape(self, batch_size, n_particles, hidden_size, n_classes):
        """Log-likelihood has shape [batch, K]."""
        model = OrdinalObservationModel(hidden_size, n_classes)
        states = torch.randn(batch_size, n_particles, hidden_size)
        labels = torch.randint(0, n_classes, (batch_size,))

        log_lik = model.log_likelihood(states, labels)
        assert log_lik.shape == (batch_size, n_particles)

    def test_log_likelihood_negative(self, batch_size, n_particles, hidden_size, n_classes):
        """Ordinal log-likelihood < 0 (log of probability)."""
        model = OrdinalObservationModel(hidden_size, n_classes)
        states = torch.randn(batch_size, n_particles, hidden_size)
        labels = torch.randint(0, n_classes, (batch_size,))

        log_lik = model.log_likelihood(states, labels)
        assert torch.all(log_lik < 0)

    def test_cumulative_probs_shape(self, batch_size, n_particles, hidden_size, n_classes):
        """Cumulative probabilities have shape [batch, K, n_classes - 1]."""
        model = OrdinalObservationModel(hidden_size, n_classes)
        states = torch.randn(batch_size, n_particles, hidden_size)

        cum_probs = model.get_cumulative_probs(states)
        assert cum_probs.shape == (batch_size, n_particles, n_classes - 1)

    def test_cumulative_probs_in_01(self, batch_size, n_particles, hidden_size, n_classes):
        """Cumulative probabilities in [0, 1]."""
        model = OrdinalObservationModel(hidden_size, n_classes)
        states = torch.randn(batch_size, n_particles, hidden_size)

        cum_probs = model.get_cumulative_probs(states)
        assert torch.all(cum_probs >= 0)
        assert torch.all(cum_probs <= 1)

    def test_cumulative_probs_monotonic(self, batch_size, n_particles, hidden_size, n_classes):
        """Cumulative P(Y > k) should decrease as k increases."""
        model = OrdinalObservationModel(hidden_size, n_classes)
        states = torch.randn(batch_size, n_particles, hidden_size)

        cum_probs = model.get_cumulative_probs(states)  # [batch, K, n_classes - 1]

        # P(Y > k) should be >= P(Y > k+1)
        for k in range(cum_probs.shape[-1] - 1):
            assert torch.all(cum_probs[..., k] >= cum_probs[..., k + 1] - 1e-5)

    def test_class_probs_shape(self, batch_size, n_particles, hidden_size, n_classes):
        """Class probabilities have shape [batch, K, n_classes]."""
        model = OrdinalObservationModel(hidden_size, n_classes)
        states = torch.randn(batch_size, n_particles, hidden_size)

        probs = model.get_probs(states)
        assert probs.shape == (batch_size, n_particles, n_classes)

    def test_class_probs_sum_to_one(self, batch_size, n_particles, hidden_size, n_classes):
        """Class probabilities sum to 1."""
        model = OrdinalObservationModel(hidden_size, n_classes)
        states = torch.randn(batch_size, n_particles, hidden_size)

        probs = model.get_probs(states)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_class_probs_non_negative(self, batch_size, n_particles, hidden_size, n_classes):
        """Class probabilities are non-negative."""
        model = OrdinalObservationModel(hidden_size, n_classes)
        states = torch.randn(batch_size, n_particles, hidden_size)

        probs = model.get_probs(states)
        assert torch.all(probs >= 0)

    def test_thresholds_learnable(self, hidden_size, n_classes):
        """Thresholds have gradients."""
        model = OrdinalObservationModel(hidden_size, n_classes)
        states = torch.randn(4, 8, hidden_size)
        labels = torch.randint(0, n_classes, (4,))

        log_lik = model.log_likelihood(states, labels)
        loss = -log_lik.mean()
        loss.backward()

        assert model.thresholds.grad is not None

    def test_thresholds_ordered(self, hidden_size, n_classes):
        """Effective thresholds are ordered (via cumsum of softplus)."""
        model = OrdinalObservationModel(hidden_size, n_classes)

        # Get effective thresholds
        with torch.no_grad():
            effective = torch.cumsum(F.softplus(model.thresholds), dim=0)

        # Should be monotonically increasing
        for i in range(len(effective) - 1):
            assert effective[i] < effective[i + 1]

    def test_predict_returns_probs(self, batch_size, n_particles, hidden_size, n_classes):
        """predict returns class probabilities."""
        model = OrdinalObservationModel(hidden_size, n_classes)
        states = torch.randn(batch_size, n_particles, hidden_size)

        probs = model.predict(states)
        assert probs.shape == (batch_size, n_particles, n_classes)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size, n_particles), atol=1e-4)

    def test_custom_score_hidden_size(self, batch_size, n_particles, hidden_size, n_classes):
        """Custom score network hidden size works."""
        model = OrdinalObservationModel(
            hidden_size, n_classes, score_hidden_size=128
        )
        states = torch.randn(batch_size, n_particles, hidden_size)
        labels = torch.randint(0, n_classes, (batch_size,))

        log_lik = model.log_likelihood(states, labels)
        assert log_lik.shape == (batch_size, n_particles)

    def test_no_nan_inf(self, batch_size, n_particles, hidden_size, n_classes):
        """No NaN or Inf in outputs."""
        model = OrdinalObservationModel(hidden_size, n_classes)
        states = torch.randn(batch_size, n_particles, hidden_size)
        labels = torch.randint(0, n_classes, (batch_size,))

        log_lik = model.log_likelihood(states, labels)
        probs = model.get_probs(states)
        cum_probs = model.get_cumulative_probs(states)

        assert not torch.isnan(log_lik).any()
        assert not torch.isinf(log_lik).any()
        assert not torch.isnan(probs).any()
        assert not torch.isnan(cum_probs).any()


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestClassificationEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_binary_classification(self, batch_size, n_particles, hidden_size):
        """Binary classification (n_classes=2) works."""
        model = ClassificationObservationModel(hidden_size, n_classes=2)
        states = torch.randn(batch_size, n_particles, hidden_size)
        labels = torch.randint(0, 2, (batch_size,))

        log_lik = model.log_likelihood(states, labels)
        assert log_lik.shape == (batch_size, n_particles)

    def test_many_classes(self, batch_size, n_particles, hidden_size):
        """Large number of classes works."""
        n_classes = 100
        model = ClassificationObservationModel(hidden_size, n_classes)
        states = torch.randn(batch_size, n_particles, hidden_size)
        labels = torch.randint(0, n_classes, (batch_size,))

        log_lik = model.log_likelihood(states, labels)
        assert log_lik.shape == (batch_size, n_particles)

    def test_single_particle(self, batch_size, hidden_size, n_classes):
        """K=1 works."""
        model = ClassificationObservationModel(hidden_size, n_classes)
        states = torch.randn(batch_size, 1, hidden_size)
        labels = torch.randint(0, n_classes, (batch_size,))

        log_lik = model.log_likelihood(states, labels)
        assert log_lik.shape == (batch_size, 1)

    def test_single_batch(self, n_particles, hidden_size, n_classes):
        """Batch size 1 works."""
        model = ClassificationObservationModel(hidden_size, n_classes)
        states = torch.randn(1, n_particles, hidden_size)
        labels = torch.randint(0, n_classes, (1,))

        log_lik = model.log_likelihood(states, labels)
        assert log_lik.shape == (1, n_particles)

    def test_ordinal_binary(self, batch_size, n_particles, hidden_size):
        """Binary ordinal classification (n_classes=2) works."""
        model = OrdinalObservationModel(hidden_size, n_classes=2)
        states = torch.randn(batch_size, n_particles, hidden_size)
        labels = torch.randint(0, 2, (batch_size,))

        log_lik = model.log_likelihood(states, labels)
        assert log_lik.shape == (batch_size, n_particles)

    def test_very_low_temperature(self, batch_size, n_particles, hidden_size, n_classes):
        """Very low temperature doesn't cause numerical issues."""
        model = ClassificationObservationModel(hidden_size, n_classes, temperature=0.01)
        states = torch.randn(batch_size, n_particles, hidden_size)
        labels = torch.randint(0, n_classes, (batch_size,))

        log_lik = model.log_likelihood(states, labels)
        assert not torch.isnan(log_lik).any()
        assert not torch.isinf(log_lik).any()

    def test_very_high_temperature(self, batch_size, n_particles, hidden_size, n_classes):
        """Very high temperature doesn't cause numerical issues."""
        model = ClassificationObservationModel(hidden_size, n_classes, temperature=100.0)
        states = torch.randn(batch_size, n_particles, hidden_size)
        labels = torch.randint(0, n_classes, (batch_size,))

        log_lik = model.log_likelihood(states, labels)
        probs = model.get_probs(states)

        assert not torch.isnan(log_lik).any()
        # High temp should give nearly uniform probabilities
        expected_prob = 1.0 / n_classes
        assert torch.allclose(probs, torch.full_like(probs, expected_prob), atol=0.1)

    def test_extreme_state_values(self, batch_size, n_particles, hidden_size, n_classes):
        """Extreme state values handled gracefully."""
        model = ClassificationObservationModel(hidden_size, n_classes)
        states = torch.randn(batch_size, n_particles, hidden_size) * 100
        labels = torch.randint(0, n_classes, (batch_size,))

        log_lik = model.log_likelihood(states, labels)
        assert not torch.isnan(log_lik).any()
