"""Feature embedding model for the SF agent."""
import torch
import math
from typing import Dict, Tuple

from agents.fb.base import AbstractMLP


class AbstractPhi(AbstractMLP):
    """
    Embeds states in feature space.
    """

    def __init__(
        self,
        observation_dim: int,
        z_dimension: int,
        hidden_dimension: int,
        hidden_layers: int,
        device: torch.device,
        activation: str,
        layernorm: bool,
    ):
        self._z_dimension = z_dimension
        super().__init__(
            input_dimension=observation_dim,
            output_dimension=z_dimension,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            device=device,
            activation=activation,
            layernorm=layernorm,
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Embeds observation in z space and L2 normalizes.
        Args:
            observation: tensor of shape [batch_dim, observation_dim]
        Returns:
            z: embedded feature tensor of shape [batch_dim, z_dimension]
        """

        z = self.trunk(observation)  # pylint: disable=E1102

        # L2 normalize then scale to radius sqrt(z_dimension)
        z = math.sqrt(self._z_dimension) * torch.nn.functional.normalize(z, dim=1)

        return z


class LaplacianFeatures(torch.nn.Module):
    """
    Feature embedding learned with Laplacian eigenfunctions.
    """

    def __init__(
        self,
        observation_dim: int,
        z_dimension: int,
        hidden_dimension: int,
        hidden_layers: int,
        device: torch.device,
        activation: str,
        layernorm: bool,
    ):
        super().__init__()

        self.phi = AbstractPhi(
            observation_dim=observation_dim,
            z_dimension=z_dimension,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            device=device,
            activation=activation,
            layernorm=layernorm,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Taken an observation and returns the feature embedding.
        Args:
            observations: tensor of shape [batch_dim, observation_dim]
        Returns:
            phi: embedded feature tensor of shape [batch_dim, z_dimension]
        """

        return self.phi.forward(observation=observations)

    def get_loss(
        self,
        observations: torch.Tensor,
        next_observations: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Get loss for laplacian features update.
        Args:
            observations:
            next_observations:
        Returns:
            metrics: dictionary of metrics for logging
        """

        z = self.phi.forward(observation=observations)
        next_z = self.phi.forward(observation=next_observations)

        loss = (z - next_z).pow(2).mean()
        Cov = torch.matmul(z, z.T)

        I = torch.eye(*Cov.size(), device=Cov.device)
        off_diag = ~I.bool()
        orth_loss_diag = -2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        loss += orth_loss

        metrics = {
            "train/laplacian_features_loss": loss,
            "train/laplacian_features_orth_loss": orth_loss,
        }

        return loss, metrics
