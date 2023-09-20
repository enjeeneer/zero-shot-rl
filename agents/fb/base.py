"""Module defining base classed for forward-backward agent."""

import abc
import torch

from agents.base import AbstractMLP, AbstractActor, AbstractGaussianActor
from typing import Tuple


class AbstractPreprocessor(AbstractMLP, metaclass=abc.ABCMeta):
    """Preprocesses an observation concatenated with another variable
    into a feature space."""

    def __init__(
        self,
        observation_length: int,
        concatenated_variable_length: int,
        hidden_dimension: int,
        feature_space_dimension: int,
        hidden_layers: int,
        activation: str,
        device: torch.device,
    ):
        super().__init__(
            input_dimension=observation_length + concatenated_variable_length,
            output_dimension=feature_space_dimension,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
            preprocessor=True,
        )

    def forward(self, concatenation: torch.tensor) -> torch.tensor:
        """
        Passes concatenation through network to predict feature space
        Args:
            concatenation: tensor of shape
                        [batch_dim, observation_length + concatenated_variable_length]

        Returns:
            features: feature space tensor of shape [batch_dim, feature_space_dimension]
        """
        features = self.trunk(concatenation)  # pylint: disable=E1102

        return features


class ForwardModel(AbstractMLP):
    """
    Predicts the expected future states (measure) given an
    embedding of a current state-action pair and policy parameterised by z.
    """

    def __init__(
        self,
        preprocessor_feature_space_dimension: int,
        number_of_preprocessed_features: int,
        z_dimension: int,
        hidden_dimension: int,
        hidden_layers: int,
        device: torch.device,
        activation: str,
    ):
        super().__init__(
            input_dimension=preprocessor_feature_space_dimension
            * number_of_preprocessed_features,
            output_dimension=z_dimension,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Takes observation, action, z, preprocesses and forwards into z-space.
        Args:
            h: preprocessed/embedded obs/action/z tensor of shape
             [batch_dim, preprocessor_feature_space_dimension * number_of_features]
        Returns:
            z_: embedded tensor of shape [batch_dim, z_dimension]
        """

        z_ = self.trunk(h)  # pylint: disable=E1102

        return z_


class BackwardModel(AbstractMLP):
    """Backwards model--can be interpreted as the number of ways
    of reaching the observation."""

    def __init__(
        self,
        observation_length: int,
        z_dimension: int,
        hidden_dimension: int,
        hidden_layers: int,
        device: torch.device,
        activation: str,
    ):
        super().__init__(
            input_dimension=observation_length,
            output_dimension=z_dimension,
            hidden_dimension=hidden_dimension,
            hidden_layers=hidden_layers,
            activation=activation,
            device=device,
        )
        self._z_dimension = z_dimension

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Takes observation, action, z, preprocesses and forwards into z-space.
        Args:
            observation: state tensor of shape [batch_dim, observation_length]
        Returns:
            z: embedded tensor of shape [batch_dim, z_dimension]
        """
        z = self.trunk(observation)  # pylint: disable=E1102

        # L2 normalize then scale to radius sqrt(z_dimension)
        z = torch.sqrt(
            torch.tensor(self._z_dimension, dtype=torch.int, device=self.device)
        ) * torch.nn.functional.normalize(z, dim=1)

        return z


class ActorModel(torch.nn.Module):
    """FB agent actor model."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        preprocessor_hidden_dimension: int,
        preprocessor_feature_space_dimension: int,
        number_of_features: int,
        preprocessor_hidden_layers: int,
        preprocessor_activation: torch.nn,
        z_dimension: int,
        actor_hidden_dimension: int,
        actor_hidden_layers: int,
        gaussian_actor: bool,
        actor_activation: torch.nn,
        std_dev_clip: float,
        device: torch.device,
    ):
        super().__init__()

        self.actor = (AbstractGaussianActor if gaussian_actor else AbstractActor)(
            observation_length=preprocessor_feature_space_dimension
            * number_of_features,
            action_length=action_length,
            hidden_dimension=actor_hidden_dimension,
            hidden_layers=actor_hidden_layers,
            activation=actor_activation,
            device=device,
        )

        # pre-procossors
        self.obs_preprocessor = AbstractPreprocessor(
            observation_length=observation_length,
            concatenated_variable_length=0,  # preprocess observation alone
            hidden_dimension=preprocessor_hidden_dimension,
            feature_space_dimension=preprocessor_feature_space_dimension,
            hidden_layers=preprocessor_hidden_layers,
            device=device,
            activation=preprocessor_activation,
        )

        self.obs_z_preprocessor = AbstractPreprocessor(
            observation_length=observation_length,
            concatenated_variable_length=z_dimension,
            hidden_dimension=preprocessor_hidden_dimension,
            feature_space_dimension=preprocessor_feature_space_dimension,
            hidden_layers=preprocessor_hidden_layers,
            device=device,
            activation=preprocessor_activation,
        )

        self._std_dev_clip = std_dev_clip

    def forward(
        self,
        observation: torch.Tensor,
        z: torch.Tensor,
        std: float,
        sample: bool = False,
    ) -> Tuple[torch.Tensor, torch.distributions.Distribution]:
        """
        Takes observation, action, z, preprocesses and forwards into z-space.
        Args:
            observation: state tensor of shape [batch_dim, observation_length]
            z: policy parameter tensor of shape [batch_dim, z_dimension]
            std: standard deviation of the policy
            sample: whether to sample from the policy or not
        Returns:
            action: action tensor of shape [batch_dim, action_length]
        """

        obs_embedding = self.obs_preprocessor(observation)
        obs_z_embedding = self.obs_z_preprocessor(torch.cat([observation, z], dim=-1))
        h = torch.cat([obs_embedding, obs_z_embedding], dim=-1)

        action_dist = (
            self.actor(h)
            if type(self.actor)  # pylint: disable=unidiomatic-typecheck
            == AbstractGaussianActor
            else self.actor(h, std)
        )

        if sample:
            action = (
                action_dist.rsample()
                if type(self.actor)  # pylint: disable=unidiomatic-typecheck
                == AbstractGaussianActor
                else action_dist.sample(clip=self._std_dev_clip)
            )

        else:
            action = action_dist.mean

        return action.clip(-1, 1), action_dist
