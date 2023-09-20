"""Module for TD3 implementation."""

import torch
import numpy as np
from agents.base import AbstractAgent, AbstractActor, DoubleQCritic, Batch
from agents.utils import schedule
from typing import Dict
from pathlib import Path


class TD3(AbstractAgent):
    """TD3 Agent: https://arxiv.org/pdf/1802.09477.pdf"""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        device: torch.device,
        name: str,
        critic_hidden_dimension: int,
        critic_hidden_layers: int,
        critic_learning_rate: float,
        critic_activation: str,
        actor_hidden_dimension: int,
        actor_hidden_layers: int,
        actor_learning_rate: float,
        actor_activation: str,
        std_dev_clip: float,
        std_dev_schedule: str,
        batch_size: int,
        discount: float,
        tau: float,
    ):
        super().__init__(
            observation_length=observation_length,
            action_length=action_length,
            name=name,
        )

        self.std_dev_clip = std_dev_clip
        self.std_dev_schedule = std_dev_schedule
        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau
        self.device = device

        self.actor = AbstractActor(
            observation_length=observation_length,
            action_length=action_length,
            hidden_dimension=actor_hidden_dimension,
            hidden_layers=actor_hidden_layers,
            activation=actor_activation,
            device=device,
        )
        self.critic = DoubleQCritic(
            observation_length=observation_length,
            action_length=action_length,
            hidden_dimension=critic_hidden_dimension,
            hidden_layers=critic_hidden_layers,
            activation=critic_activation,
            device=device,
            layernorm=True,
        )
        self.critic_target = DoubleQCritic(
            observation_length=observation_length,
            action_length=action_length,
            hidden_dimension=critic_hidden_dimension,
            hidden_layers=critic_hidden_layers,
            activation=critic_activation,
            device=device,
            layernorm=True,
        )
        self.critic_target.load_state_dict(self.critic.state_dict())

        # --- optimisers
        self.actor_optimiser = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_learning_rate,
        )
        self.critic_optimiser = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_learning_rate,
        )

    def act(
        self,
        observation: np.ndarray,
        step: int,
        sample=False,
    ) -> np.ndarray:
        """
        Takes observation array from environment and selects action from actor.
        Args:
            observation: observation array of shape [observation_length]
            step: current step in training loop
            sample: True if in training loop, False if in eval or production
        Returns:
            action: action array of shape [action_length]
        """

        observation = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        )
        observation = observation.unsqueeze(0)

        # get action from actor
        std_dev = schedule(self.std_dev_schedule, step)
        action_dist = self.actor(observation, std_dev)

        if sample:
            action = action_dist.sample(clip=self.std_dev_clip)
        else:
            action = action_dist.mean

        return action.detach().cpu().numpy()[0]

    def update(self, batch: Batch, step: int) -> Dict[str, float]:
        """
        Updates actor and critic networks.
        Args:
            batch: batch of transitions from replay buffer
            step: current step in training loop
        Returns:
            metrics: dictionary of metrics for logging
        """

        (observations, actions, rewards, next_observations, not_dones) = (
            batch.observations,
            batch.actions,
            batch.rewards,
            batch.next_observations,
            batch.not_dones,
        )

        critic_metrics = self._update_critic(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            not_dones=not_dones,
            step=step,
        )

        actor_metrics = self._update_actor(observations=observations, step=step)

        # polyak critic target update
        self._soft_critic_target_update()

        metrics = {**critic_metrics, **actor_metrics}

        return metrics

    def _update_critic(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        not_dones: torch.Tensor,
        step: int,
    ) -> Dict[str, float]:
        """
        Updates critic as per orginal online TD3 implementation.
        Args:
            observations: tensor of shape [batch_dim, observation_length]
            actions: tensor of shape [batch_dim, action_length]
            rewards: tensor of shape [batch_dim, 1]
            next_observations: tensor of shape [batch_dim, observation_length]
            not_dones: tensor of shape [batch_dim, 1]
        Returns:
            metrics: dictionary of metrics for logging
        """

        # get next actions and evaluate log prob
        with torch.no_grad():
            std_dev = schedule(self.std_dev_schedule, step)
            next_actions_dist = self.actor(next_observations, std_dev)
            next_actions = next_actions_dist.sample(clip=self.std_dev_clip)

            # get Q targets via soft policy evaluation
            target_Q1, target_Q2 = self.critic_target(next_observations, next_actions)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (not_dones * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(observations, actions)
        critic_loss = torch.nn.functional.mse_loss(
            current_Q1, target_Q
        ) + torch.nn.functional.mse_loss(current_Q2, target_Q)

        # optimize the critic
        self.critic_optimiser.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimiser.step()

        metrics = {
            "train/critic_loss": critic_loss.detach().cpu().numpy(),
            "train/Q1": current_Q1.mean().detach().cpu().numpy(),
            "train/Q2": current_Q2.mean().detach().cpu().numpy(),
        }

        return metrics

    def _update_actor(self, observations: torch.Tensor, step: int) -> Dict[str, float]:
        """
        Updates actor as per TD3 paper.
        Args:
            observations: tensor of shape [batch_dim, observation_length.
            step: current step in training loop
        Returns:
            metrics: dictionary of metrics for logging
        """
        stddev = schedule(self.std_dev_schedule, step)
        policy = self.actor(observations, stddev)

        Q1, Q2 = self.critic(observations, policy.sample(clip=self.std_dev_clip))
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize the actor
        self.actor_optimiser.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimiser.step()

        metrics = {
            "train/actor_loss": actor_loss.detach().cpu().numpy(),
            "train/Q": Q.mean().detach().cpu().numpy(),
            "train/actor_entropy": policy.entropy()
            .sum(dim=-1)
            .mean()
            .detach()
            .cpu()
            .numpy(),
        }

        return metrics

    def _soft_critic_target_update(self):
        """
        Moves target critics weights towards the critics.
        """
        for critic_params, critic_target_params in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            critic_target_params.data.copy_(
                self.tau * critic_params.data
                + (1 - self.tau) * critic_target_params.data
            )

    def load(self, filepath: Path):
        return torch.load(filepath.with_suffix(".pickle"))
