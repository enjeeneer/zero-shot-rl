# pylint: disable=W0613

"""Module for GC-IQl implementation."""

import torch
import numpy as np
from agents.base import (
    AbstractAgent,
    AbstractGaussianActor,
    DoubleVCritic,
    Batch,
)
from typing import Dict, Tuple, Optional
from pathlib import Path


class GCIQL(AbstractAgent):
    """Goal-conditioned IQL agent"""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        device: torch.device,
        name: str,
        critic_hidden_dimension: int,
        critic_hidden_layers: int,
        critic_activation: str,
        value_learning_rate: float,
        critic_target_update_frequency: int,
        actor_update_frequency: int,
        actor_hidden_dimension: int,
        actor_hidden_layers: int,
        actor_learning_rate: float,
        actor_activation: str,
        batch_size: int,
        discount: float,
        expectile: float,
        temperature: float,
        tau: float,
    ):
        super().__init__(
            observation_length=observation_length,
            action_length=action_length,
            name=name,
        )

        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau
        self.device = device
        self.critic_target_update_frequency = critic_target_update_frequency
        self.actor_update_frequency = actor_update_frequency
        self.expectile = expectile
        self.temperature = temperature
        goal_length = observation_length

        self.actor = AbstractGaussianActor(
            observation_length=observation_length + goal_length,
            action_length=action_length,
            hidden_dimension=actor_hidden_dimension,
            hidden_layers=actor_hidden_layers,
            activation=actor_activation,
            device=device,
        )
        self.value_critic = DoubleVCritic(
            observation_length=observation_length + goal_length,
            hidden_dimension=critic_hidden_dimension,
            hidden_layers=critic_hidden_layers,
            activation=critic_activation,
            device=device,
            layernorm=True,
        )
        self.value_critic_target = DoubleVCritic(
            observation_length=observation_length + goal_length,
            hidden_dimension=critic_hidden_dimension,
            hidden_layers=critic_hidden_layers,
            activation=critic_activation,
            device=device,
            layernorm=True,
        )
        self.value_critic_target.load_state_dict(self.value_critic.state_dict())

        # --- optimisers
        self.actor_optimiser = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_learning_rate,
        )
        self.value_critic_optimiser = torch.optim.Adam(
            self.value_critic.parameters(),
            lr=value_learning_rate,
        )

    def act(
        self,
        observation: np.ndarray,
        task: np.ndarray,
        step: int,
        sample=False,
    ) -> Tuple[np.ndarray, None]:
        """
        Takes observation array from environment and selects action from actor.
        Args:
            observation: observation array of shape [observation_length]
            task: goal array of shape [observation_length]
            step: current step in training loop
            sample: True if in training loop, False if in eval or production
        Returns:
            action: action array of shape [action_length]
        """

        observation = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        )
        goal = torch.as_tensor(task, dtype=torch.float32, device=self.device)
        obs_goal = torch.cat([observation, goal], dim=-1).unsqueeze(0)

        # get action dist from actor
        dist = self.actor.forward(obs_goal)

        if sample:
            action = dist.sample()
        else:
            action = dist.mean

        return action.detach().cpu().numpy()[0], None

    def update(self, batch: Batch, step: int) -> Dict[str, float]:
        """
        Updates actor and critic networks.
        Args:
            batch: batch of transitions from replay buffer
            step: current step in training loop
        Returns:
            metrics: dictionary of metrics for logging
        """

        (observations, actions, goals, next_observations) = (
            batch.observations,
            batch.actions,
            batch.gciql_goals,
            batch.next_observations,
        )

        dones = (observations - goals).abs().sum(dim=-1, keepdim=True) < 1e-8
        rewards = dones.to(torch.float32) - 1
        masks = torch.where(rewards < 0, 1, 0)

        # update value net
        value_loss, value_metrics = self._value_loss(
            observations=observations,
            next_observations=next_observations,
            goals=goals,
            rewards=rewards,
            masks=masks,
        )
        self.value_critic_optimiser.zero_grad(set_to_none=True)
        value_loss.backward()
        self.value_critic_optimiser.step()

        # update actor net
        actor_loss, actor_metrics = self._actor_loss(
            observations=observations,
            next_observations=next_observations,
            goals=goals,
            actions=actions,
        )
        self.actor_optimiser.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimiser.step()

        # polyak value target update
        if step % self.critic_target_update_frequency == 0:
            self._soft_value_target_update()

        metrics = {**actor_metrics, **value_metrics}

        return metrics

    def _actor_loss(
        self,
        observations: torch.Tensor,
        next_observations: torch.Tensor,
        actions: torch.Tensor,
        goals: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Updates actor as per TD3 paper.
        Args:
            observations: tensor of shape [batch_dim, observation_length.
            step: current step in training loop
        Returns:
            metrics: dictionary of metrics for logging
        """
        obs_goals = torch.cat([observations, goals], dim=-1)
        next_obs_goals = torch.cat([next_observations, goals], dim=-1)

        v1, v2 = self.value_critic.forward(obs_goals)
        next_v1, next_v2 = self.value_critic_target.forward(next_obs_goals)
        v = (v1 + v2) / 2
        next_v = (next_v1 + next_v2) / 2

        advantage = next_v - v
        exp_adv = torch.exp(advantage * self.temperature)
        exp_adv = torch.minimum(exp_adv, torch.tensor(100.0, device=self.device))

        dist = self.actor.forward(obs_goals)
        log_probs = dist.log_prob(actions)
        actor_loss = -(exp_adv * log_probs).mean()

        metrics = {
            "actor_loss": actor_loss,
            "adv": advantage.mean(),
            "bc_log_probs": log_probs.mean(),
            "adv_median": torch.median(advantage),
        }

        return actor_loss, metrics

    def _value_loss(
        self,
        observations: torch.Tensor,
        next_observations: torch.Tensor,
        goals: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:

        obs_goals = torch.cat([observations, goals], dim=-1)
        next_obs_goals = torch.cat([next_observations, goals], dim=-1)

        next_v1, next_v2 = self.value_critic_target.forward(next_obs_goals)
        next_v = torch.minimum(next_v1, next_v2)
        q = rewards + self.discount * masks * next_v

        v1_target, v2_target = self.value_critic_target.forward(obs_goals)
        v_target = (v1_target + v2_target) / 2
        advantage = q - v_target

        q1 = rewards + self.discount * masks * next_v1
        q2 = rewards + self.discount * masks * next_v2
        v1, v2 = self.value_critic.forward(obs_goals)

        value_loss1 = self._expectile_loss(advantage, q1 - v1, self.expectile).mean()
        value_loss2 = self._expectile_loss(advantage, q2 - v2, self.expectile).mean()
        value_loss = value_loss1 + value_loss2

        metrics = {
            "value_loss": value_loss,
            "v max": v1.max(),
            "v min": v1.min(),
            "v mean": v1.mean(),
            "abs adv mean": torch.abs(advantage).mean(),
            "adv mean": advantage.mean(),
            "adv max": advantage.max(),
            "adv min": advantage.min(),
        }

        return value_loss, metrics

    def _soft_value_target_update(self):
        """
        Moves target critics weights towards the critics.
        """
        for value_critic_params, value_critic_target_params in zip(
            self.value_critic.parameters(), self.value_critic_target.parameters()
        ):
            value_critic_target_params.data.copy_(
                self.tau * value_critic_params.data
                + (1 - self.tau) * value_critic_target_params.data
            )

    def infer_z(
        self,
        goals: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Infers goal as the observation with highest rewqrd in sample if rewards
        are passed, otherwise returns the goal state which is assumed to have been
        passed.
        Args:
            goals: tensor of shape [inference_steps, observation_length]
            rewards: tensor of shape [inference_steps, 1]
        """
        if rewards is not None:
            return goals[rewards.argmax()].squeeze().cpu().numpy()
        else:
            return goals.squeeze().cpu().numpy()

    @staticmethod
    def _expectile_loss(adv, diff, expectile=0.8):
        weight = torch.where(adv > 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def load(self, filepath: Path):
        return torch.load(filepath.with_suffix(".pickle"))
