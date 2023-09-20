"""Module for fully defined models used by the forward-backward agent. """

from agents.fb.base import (
    ForwardModel,
    BackwardModel,
    ActorModel,
    AbstractPreprocessor,
)
from typing import Tuple

import torch


class ForwardRepresentation(torch.nn.Module):
    """Forward representation network."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        preprocessor_hidden_dimension: int,
        preprocessor_feature_space_dimension: int,
        preprocessor_hidden_layers: int,
        preprocessor_activation: torch.nn,
        number_of_features: int,
        z_dimension: int,
        forward_hidden_dimension: int,
        forward_hidden_layers: int,
        device: torch.device,
        forward_activation: str,
    ):
        super().__init__()

        # pre-processors
        self.obs_action_preprocessor = AbstractPreprocessor(
            observation_length=observation_length,
            concatenated_variable_length=action_length,
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

        self.F1 = ForwardModel(
            preprocessor_feature_space_dimension=preprocessor_feature_space_dimension,
            number_of_preprocessed_features=number_of_features,
            z_dimension=z_dimension,
            hidden_dimension=forward_hidden_dimension,
            hidden_layers=forward_hidden_layers,
            device=device,
            activation=forward_activation,
        )

        self.F2 = ForwardModel(
            preprocessor_feature_space_dimension=preprocessor_feature_space_dimension,
            number_of_preprocessed_features=number_of_features,
            z_dimension=z_dimension,
            hidden_dimension=forward_hidden_dimension,
            hidden_layers=forward_hidden_layers,
            device=device,
            activation=forward_activation,
        )

    def forward(
        self, observation: torch.Tensor, action: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Passes observations, actions, and zs through both forward models."""
        obs_action_embedding = self.obs_action_preprocessor(
            torch.cat([observation, action], dim=-1)
        )
        obs_z_embedding = self.obs_z_preprocessor(torch.cat([observation, z], dim=-1))

        h = torch.cat([obs_action_embedding, obs_z_embedding], dim=-1)

        return self.F1(h), self.F2(h)


class BackwardRepresentation(torch.nn.Module):
    """Backward representation network."""

    def __init__(
        self,
        observation_length: int,
        z_dimension: int,
        backward_hidden_dimension: int,
        backward_hidden_layers: int,
        device: torch.device,
        backward_activation: torch.nn,
    ):
        super().__init__()

        self.B = BackwardModel(
            observation_length=observation_length,
            z_dimension=z_dimension,
            hidden_dimension=backward_hidden_dimension,
            hidden_layers=backward_hidden_layers,
            device=device,
            activation=backward_activation,
        )

    def forward(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """Estimates routes to observation via backwards model."""

        return self.B(observation)


class ForwardBackwardRepresentation(torch.nn.Module):
    """Combined Forward-backward representation network."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        preprocessor_hidden_dimension: int,
        preprocessor_feature_space_dimension: int,
        preprocessor_hidden_layers: int,
        preprocessor_activation: str,
        number_of_features: int,
        z_dimension: int,
        forward_hidden_dimension: int,
        forward_hidden_layers: int,
        backward_hidden_dimension: int,
        backward_hidden_layers: int,
        forward_activation: str,
        backward_activation: str,
        orthonormalisation_coefficient: float,
        discount: float,
        device: torch.device,
    ):
        super().__init__()
        self.forward_representation = ForwardRepresentation(
            observation_length=observation_length,
            action_length=action_length,
            preprocessor_hidden_dimension=preprocessor_hidden_dimension,
            preprocessor_feature_space_dimension=preprocessor_feature_space_dimension,
            preprocessor_hidden_layers=preprocessor_hidden_layers,
            preprocessor_activation=preprocessor_activation,
            number_of_features=number_of_features,
            z_dimension=z_dimension,
            forward_hidden_dimension=forward_hidden_dimension,
            forward_hidden_layers=forward_hidden_layers,
            device=device,
            forward_activation=forward_activation,
        )

        self.backward_representation = BackwardRepresentation(
            observation_length=observation_length,
            z_dimension=z_dimension,
            backward_hidden_dimension=backward_hidden_dimension,
            backward_hidden_layers=backward_hidden_layers,
            device=device,
            backward_activation=backward_activation,
        )

        self.forward_representation_target = ForwardRepresentation(
            observation_length=observation_length,
            action_length=action_length,
            preprocessor_hidden_dimension=preprocessor_hidden_dimension,
            preprocessor_feature_space_dimension=preprocessor_feature_space_dimension,
            preprocessor_hidden_layers=preprocessor_hidden_layers,
            preprocessor_activation=preprocessor_activation,
            number_of_features=number_of_features,
            z_dimension=z_dimension,
            forward_hidden_dimension=forward_hidden_dimension,
            forward_hidden_layers=forward_hidden_layers,
            device=device,
            forward_activation=forward_activation,
        )

        self.backward_representation_target = BackwardRepresentation(
            observation_length=observation_length,
            z_dimension=z_dimension,
            backward_hidden_dimension=backward_hidden_dimension,
            backward_hidden_layers=backward_hidden_layers,
            device=device,
            backward_activation=backward_activation,
        )

        self._discount = discount
        self.orthonormalisation_coefficient = orthonormalisation_coefficient
        self._device = device


class Actor(torch.nn.Module):
    """Action selecting network."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        preprocessor_hidden_dimension: int,
        preprocessor_feature_space_dimension: int,
        preprocessor_hidden_layers: int,
        preprocessor_activation: torch.nn,
        z_dimension: int,
        actor_hidden_dimension: int,
        actor_hidden_layers: int,
        actor_activation: torch.nn,
        std_dev_schedule: str,
        std_dev_clip: float,
        device: torch.device,
        gaussian_actor: bool,
    ):
        super().__init__()

        self.actor = ActorModel(
            observation_length=observation_length,
            action_length=action_length,
            preprocessor_hidden_dimension=preprocessor_hidden_dimension,
            preprocessor_feature_space_dimension=preprocessor_feature_space_dimension,
            preprocessor_hidden_layers=preprocessor_hidden_layers,
            preprocessor_activation=preprocessor_activation,
            number_of_features=2,
            z_dimension=z_dimension,
            actor_hidden_dimension=actor_hidden_dimension,
            actor_hidden_layers=actor_hidden_layers,
            actor_activation=actor_activation,
            std_dev_clip=std_dev_clip,
            device=device,
            gaussian_actor=gaussian_actor,
        )

        self._std_dev_schedule = std_dev_schedule

    def forward(
        self,
        observation: torch.Tensor,
        z: torch.Tensor,
        std: float,
        sample: bool = False,
    ) -> Tuple[torch.Tensor, torch.distributions.Distribution]:
        """Returns actions from actor model."""
        action, action_dist = self.actor(observation, z, std, sample)

        return action, action_dist
