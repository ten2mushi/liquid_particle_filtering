"""Classification observation models for particle filters.

Provides observation models for discrete/categorical observations,
enabling particle filtering for classification tasks.
"""

from typing import Optional, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import ObservationModel


class ClassificationObservationModel(ObservationModel):
    """Categorical observation model for classification tasks.

    Maps hidden states to class probabilities via a linear layer
    with softmax, then computes log p(class | h).

    log p(y | h) = log softmax(Wh + b)[y]

    where y is the class index.
    """

    def __init__(
        self,
        hidden_size: int,
        n_classes: int,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
        label_smoothing: float = 0.0,
    ):
        """Initialize classification observation model.

        Args:
            hidden_size: Dimension of hidden states
            n_classes: Number of classes
            temperature: Softmax temperature (higher = more uniform)
            learnable_temperature: If True, temperature is learnable
            label_smoothing: Label smoothing factor (0 = no smoothing)
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.label_smoothing = label_smoothing

        # Classification head
        self.classifier = nn.Linear(hidden_size, n_classes)

        # Temperature
        if learnable_temperature:
            self.log_temperature = nn.Parameter(
                torch.tensor(temperature).log()
            )
        else:
            self.register_buffer(
                "log_temperature",
                torch.tensor(temperature).log()
            )

    @property
    def temperature(self) -> Tensor:
        """Get current temperature."""
        return torch.exp(self.log_temperature)

    def get_logits(self, states: Tensor) -> Tensor:
        """Get class logits from states.

        Args:
            states: Particle states [batch, K, hidden_size]

        Returns:
            logits: Class logits [batch, K, n_classes]
        """
        batch, K, H = states.shape
        states_flat = states.reshape(-1, H)
        logits_flat = self.classifier(states_flat)
        return logits_flat.reshape(batch, K, self.n_classes)

    def get_probs(self, states: Tensor) -> Tensor:
        """Get class probabilities from states.

        Args:
            states: Particle states [batch, K, hidden_size]

        Returns:
            probs: Class probabilities [batch, K, n_classes]
        """
        logits = self.get_logits(states)
        return F.softmax(logits / self.temperature, dim=-1)

    def log_likelihood(
        self,
        states: Tensor,
        observations: Tensor,
        **kwargs,
    ) -> Tensor:
        """Compute classification log-likelihood.

        Args:
            states: Particle states [batch, K, hidden_size]
            observations: Class labels [batch] (integer indices)

        Returns:
            log_likelihoods: [batch, K]
        """
        batch, K, H = states.shape

        # Get log probabilities
        logits = self.get_logits(states)  # [batch, K, n_classes]
        log_probs = F.log_softmax(logits / self.temperature, dim=-1)

        # Get log prob of observed class
        # observations: [batch] -> [batch, K, 1]
        labels_expanded = observations.long().unsqueeze(1).unsqueeze(2)
        labels_expanded = labels_expanded.expand(-1, K, 1)

        # Gather log prob of true class
        log_lik = log_probs.gather(dim=-1, index=labels_expanded).squeeze(-1)

        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            smooth_loss = -log_probs.mean(dim=-1)  # [batch, K]
            log_lik = (1 - self.label_smoothing) * log_lik + self.label_smoothing * smooth_loss

        return log_lik

    def predict(self, states: Tensor, **kwargs) -> Tensor:
        """Predict class probabilities.

        Args:
            states: Particle states [batch, K, hidden_size]

        Returns:
            probs: Class probabilities [batch, K, n_classes]
        """
        return self.get_probs(states)

    def predict_class(self, states: Tensor, log_weights: Optional[Tensor] = None) -> Tensor:
        """Predict most likely class.

        If log_weights provided, uses weighted average of probabilities.

        Args:
            states: Particle states [batch, K, hidden_size]
            log_weights: Optional log weights [batch, K]

        Returns:
            predicted_class: Class indices [batch]
        """
        probs = self.get_probs(states)  # [batch, K, n_classes]

        if log_weights is not None:
            # Weighted average of probabilities
            weights = F.softmax(log_weights, dim=-1)  # [batch, K]
            weights = weights.unsqueeze(-1)  # [batch, K, 1]
            avg_probs = (weights * probs).sum(dim=1)  # [batch, n_classes]
        else:
            # Simple average
            avg_probs = probs.mean(dim=1)

        return avg_probs.argmax(dim=-1)

    def sample(self, states: Tensor, **kwargs) -> Tensor:
        """Sample class labels from the model.

        Args:
            states: Particle states [batch, K, hidden_size]

        Returns:
            samples: Sampled class indices [batch, K]
        """
        probs = self.get_probs(states)  # [batch, K, n_classes]
        batch, K, C = probs.shape

        # Reshape for multinomial sampling
        probs_flat = probs.reshape(-1, C)  # [batch * K, n_classes]
        samples_flat = torch.multinomial(probs_flat, 1).squeeze(-1)  # [batch * K]

        return samples_flat.reshape(batch, K)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, n_classes={self.n_classes}, "
            f"temperature={self.temperature.item():.4f}"
        )


class MLPClassificationObservationModel(ObservationModel):
    """MLP-based classification observation model.

    Uses a multi-layer MLP instead of a single linear layer for
    more expressive class probability prediction.
    """

    def __init__(
        self,
        hidden_size: int,
        n_classes: int,
        mlp_hidden_sizes: Optional[List[int]] = None,
        activation: str = "relu",
        dropout: float = 0.0,
        temperature: float = 1.0,
    ):
        """Initialize MLP classification observation model.

        Args:
            hidden_size: Dimension of hidden states
            n_classes: Number of classes
            mlp_hidden_sizes: List of hidden layer sizes
            activation: Activation function
            dropout: Dropout probability
            temperature: Softmax temperature
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.temperature = temperature

        if mlp_hidden_sizes is None:
            mlp_hidden_sizes = [64]

        # Activation
        activations = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "gelu": nn.GELU,
        }
        act_fn = activations.get(activation, nn.ReLU)

        # Build MLP
        layers = []
        in_dim = hidden_size
        for hidden_dim in mlp_hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, n_classes))
        self.mlp = nn.Sequential(*layers)

    def get_logits(self, states: Tensor) -> Tensor:
        """Get class logits from states."""
        batch, K, H = states.shape
        states_flat = states.reshape(-1, H)
        logits_flat = self.mlp(states_flat)
        return logits_flat.reshape(batch, K, self.n_classes)

    def log_likelihood(
        self,
        states: Tensor,
        observations: Tensor,
        **kwargs,
    ) -> Tensor:
        """Compute classification log-likelihood.

        Args:
            states: Particle states [batch, K, hidden_size]
            observations: Class labels [batch]

        Returns:
            log_likelihoods: [batch, K]
        """
        batch, K, H = states.shape

        logits = self.get_logits(states)
        log_probs = F.log_softmax(logits / self.temperature, dim=-1)

        labels_expanded = observations.long().unsqueeze(1).unsqueeze(2)
        labels_expanded = labels_expanded.expand(-1, K, 1)

        log_lik = log_probs.gather(dim=-1, index=labels_expanded).squeeze(-1)

        return log_lik

    def predict(self, states: Tensor, **kwargs) -> Tensor:
        """Predict class probabilities."""
        logits = self.get_logits(states)
        return F.softmax(logits / self.temperature, dim=-1)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, n_classes={self.n_classes}"


class OrdinalObservationModel(ObservationModel):
    """Ordinal observation model for ordered categorical data.

    Uses cumulative logits to model ordinal outcomes, which is
    appropriate when classes have a natural ordering (e.g., ratings).

    P(Y > k | h) = sigmoid(theta_k - f(h))

    where theta_k are learned thresholds and f(h) is a latent score.
    """

    def __init__(
        self,
        hidden_size: int,
        n_classes: int,
        score_hidden_size: Optional[int] = None,
    ):
        """Initialize ordinal observation model.

        Args:
            hidden_size: Dimension of hidden states
            n_classes: Number of ordinal classes
            score_hidden_size: Hidden size for score network
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.n_classes = n_classes

        if score_hidden_size is None:
            score_hidden_size = hidden_size // 2

        # Network to compute latent score
        self.score_net = nn.Sequential(
            nn.Linear(hidden_size, score_hidden_size),
            nn.Tanh(),
            nn.Linear(score_hidden_size, 1),
        )

        # Learnable thresholds (n_classes - 1 thresholds)
        # Initialize to be evenly spaced
        init_thresholds = torch.linspace(-2, 2, n_classes - 1)
        self.thresholds = nn.Parameter(init_thresholds)

    def get_cumulative_probs(self, states: Tensor) -> Tensor:
        """Get cumulative probabilities P(Y > k | h).

        Args:
            states: Particle states [batch, K, hidden_size]

        Returns:
            cum_probs: Cumulative probabilities [batch, K, n_classes - 1]
        """
        batch, K, H = states.shape

        # Compute latent scores
        states_flat = states.reshape(-1, H)
        scores_flat = self.score_net(states_flat)  # [batch * K, 1]
        scores = scores_flat.reshape(batch, K, 1)  # [batch, K, 1]

        # Ensure thresholds are ordered
        thresholds = torch.cumsum(F.softplus(self.thresholds), dim=0)
        thresholds = thresholds.reshape(1, 1, -1)  # [1, 1, n_classes - 1]

        # P(Y > k) = sigmoid(score - theta_k)
        # Higher score -> higher probability of exceeding threshold k
        cum_probs = torch.sigmoid(scores - thresholds)

        return cum_probs

    def get_probs(self, states: Tensor) -> Tensor:
        """Get class probabilities from cumulative probabilities.

        P(Y = k) = P(Y > k-1) - P(Y > k)

        Args:
            states: Particle states [batch, K, hidden_size]

        Returns:
            probs: Class probabilities [batch, K, n_classes]
        """
        cum_probs = self.get_cumulative_probs(states)  # [batch, K, n_classes - 1]
        batch, K, _ = cum_probs.shape

        # Add boundaries: P(Y > -1) = 1, P(Y > n_classes - 1) = 0
        ones = torch.ones(batch, K, 1, device=cum_probs.device)
        zeros = torch.zeros(batch, K, 1, device=cum_probs.device)
        cum_probs_full = torch.cat([ones, cum_probs, zeros], dim=-1)  # [batch, K, n_classes + 1]

        # P(Y = k) = P(Y > k-1) - P(Y > k)
        probs = cum_probs_full[..., :-1] - cum_probs_full[..., 1:]

        # Clamp for numerical stability
        probs = probs.clamp(min=1e-8)

        return probs

    def log_likelihood(
        self,
        states: Tensor,
        observations: Tensor,
        **kwargs,
    ) -> Tensor:
        """Compute ordinal classification log-likelihood.

        Args:
            states: Particle states [batch, K, hidden_size]
            observations: Ordinal class labels [batch]

        Returns:
            log_likelihoods: [batch, K]
        """
        probs = self.get_probs(states)  # [batch, K, n_classes]

        labels_expanded = observations.long().unsqueeze(1).unsqueeze(2)
        labels_expanded = labels_expanded.expand(-1, probs.shape[1], 1)

        prob_of_label = probs.gather(dim=-1, index=labels_expanded).squeeze(-1)
        log_lik = torch.log(prob_of_label + 1e-8)

        return log_lik

    def predict(self, states: Tensor, **kwargs) -> Tensor:
        """Predict class probabilities."""
        return self.get_probs(states)

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, n_classes={self.n_classes}"
