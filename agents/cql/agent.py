# pylint: disable=W0237

"""Module for the Conservative Soft Actor Critic agent."""

import torch
import numpy as np
from typing import List, Dict, Tuple

from agents.sac.agent import SAC


class CQL(SAC):
    """Conservative Q-learning (Kumar et al., 2020) applied to SAC."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        device: torch.device,
        name: str,
        critic_hidden_dimension: int,
        critic_hidden_layers: int,
        critic_learning_rate: float,
        critic_betas: List[float],
        actor_hidden_dimension: int,
        actor_hidden_layers: int,
        actor_learning_rate: float,
        actor_betas: List[float],
        actor_log_std_bounds: List[float],
        alpha_learning_rate: float,
        alpha_betas: List[float],
        init_temperature: float,
        learnable_temperature: bool,
        critic_tau: float,
        discount: float,
        actor_update_frequency: int,
        critic_target_update_frequency: int,
        batch_size: int,
        activation: str,
        action_range: List[np.ndarray],
        normalisation_samples: None,
        cql_n_samples: int,
        cql_alpha: float,
        cql_target_penalty: float,
        cql_lagrange: bool = False,
    ):
        super().__init__(
            observation_length=observation_length,
            action_length=action_length,
            device=device,
            name=name,
            critic_hidden_dimension=critic_hidden_dimension,
            critic_hidden_layers=critic_hidden_layers,
            critic_learning_rate=critic_learning_rate,
            critic_betas=critic_betas,
            actor_hidden_dimension=actor_hidden_dimension,
            actor_hidden_layers=actor_hidden_layers,
            actor_learning_rate=actor_learning_rate,
            actor_betas=actor_betas,
            actor_log_std_bounds=actor_log_std_bounds,
            alpha_learning_rate=alpha_learning_rate,
            alpha_betas=alpha_betas,
            init_temperature=init_temperature,
            learnable_temperature=learnable_temperature,
            critic_tau=critic_tau,
            discount=discount,
            actor_update_frequency=actor_update_frequency,
            critic_target_update_frequency=critic_target_update_frequency,
            batch_size=batch_size,
            activation=activation,
            action_range=action_range,
            normalisation_samples=normalisation_samples,
        )

        self.cql_n_samples = cql_n_samples
        self.cql_target_penalty = cql_target_penalty
        self.cql_lagrange = cql_lagrange
        self.cql_alpha = cql_alpha

        # lagrange multiplier for the CQL constraint
        self.log_cql_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.cql_alpha_optimiser = torch.optim.Adam(
            [self.log_cql_alpha], lr=critic_learning_rate
        )

    def _repeated_critic_apply(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Used for passing CQL sampled actions through the critic.
        Args:
            observations: observation tensor of shape [batch_size, observation_length]
            actions: action tensor of shape [cql_n_samples, batch_size, action_length]
        Returns:
            qs: tensor of shape [cql_n_samples, batch_size, 1]
        """

        reshaped_actions = actions.reshape((self.cql_n_samples * self.batch_size, -1))
        repeated_observations = observations.unsqueeze(0).repeat(
            (self.cql_n_samples, 1, 1)
        )
        repeated_observations = repeated_observations.reshape(
            (self.cql_n_samples * self.batch_size, -1)
        )

        Q1, Q2 = self.critic(repeated_observations, reshaped_actions)
        Q1 = Q1.reshape((self.cql_n_samples, self.batch_size, 1))
        Q2 = Q2.reshape((self.cql_n_samples, self.batch_size, 1))

        return Q1, Q2

    def _update_critic(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        not_dones: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Updates critic using the normal SAC update (equation 7 in
        Haarnoja et. al (2018)) and the CQL term from
        Kumar et. al (2020).
        Args:
            observations: tensor of shape [batch_dim, observation_length]
            actions: tensor of shape [batch_dim, action_length]
            rewards: tensor of shape [batch_dim, 1]
            next_observations: tensor of shape [batch_dim, observation_length]
            not_dones: tensor of shape [batch_dim, 1]

        Returns:
            metrics: dictionary of metrics for logging
        """

        # --- Standard SAC loss ---
        # get next actions and evaluate log prob
        with torch.no_grad():
            next_actions, next_actions_log_probs = self.actor(
                next_observations, sample=True
            )
            # get Q targets via soft policy evaluation
            target_Q1, target_Q2 = self.critic_target(next_observations, next_actions)
            target_V = (
                torch.min(target_Q1, target_Q2)
                - self.alpha.detach() * next_actions_log_probs
            )
            target_Q = rewards + (not_dones * self.gamma * target_V)
            target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(observations, actions)
        critic_loss = torch.nn.functional.mse_loss(
            current_Q1, target_Q
        ) + torch.nn.functional.mse_loss(current_Q2, target_Q)

        # --- CQL loss ---
        with torch.no_grad():
            cql_random_actions = torch.empty(
                size=(self.cql_n_samples, self.batch_size, self.action_length),
                device=self.device,
            ).uniform_(-1, 1)
            cql_sampled_actions, _ = self.actor(
                observations.repeat(self.cql_n_samples, 1, 1), sample=True
            )  # [cql_n_samples, batch_size, action_length]
            cql_sampled_next_actions, _ = self.actor(
                next_observations.repeat(self.cql_n_samples, 1, 1), sample=True
            )  # [cql_n_samples, batch_size, action_length]

        # get cql q estimates
        cql_random_q1, cql_random_q2 = self._repeated_critic_apply(
            observations, cql_random_actions
        )
        cql_sampled_q1, cql_sampled_q2 = self._repeated_critic_apply(
            observations, cql_sampled_actions
        )
        cql_sampled_next_q1, cql_sampled_next_q2 = self._repeated_critic_apply(
            observations, cql_sampled_next_actions
        )

        # concatenate
        cql_cat_q1 = torch.cat(
            [
                cql_random_q1,
                cql_sampled_q1,
                cql_sampled_next_q1,
                current_Q1.unsqueeze(0),
            ],
            dim=0,
        )
        cql_cat_q2 = torch.cat(
            [
                cql_random_q2,
                cql_sampled_q2,
                cql_sampled_next_q2,
                current_Q2.unsqueeze(0),
            ],
            dim=0,
        )
        cql_logsumexp = (
            torch.logsumexp(cql_cat_q1, dim=0).mean()
            + torch.logsumexp(cql_cat_q2, dim=0).mean()
        )

        cql_penalty = cql_logsumexp - (current_Q1 + current_Q2).mean()

        if self.cql_lagrange:
            cql_alpha = torch.clamp(self.log_cql_alpha.exp(), min=0.0, max=1e6)
            cql_alpha_loss = -0.5 * cql_alpha * (cql_penalty - self.cql_target_penalty)

            # step alpha prime optimiser
            self.cql_alpha_optimiser.zero_grad()
            cql_alpha_loss.backward(retain_graph=True)
            self.cql_alpha_optimiser.step()
            cql_alpha = torch.clamp(self.log_cql_alpha.exp(), min=0.0, max=1e6).detach()

        else:
            cql_alpha = self.cql_alpha
            cql_alpha_loss = torch.tensor(0.0)

        conservative_penalty = cql_alpha * cql_penalty

        critic_loss = critic_loss + conservative_penalty

        # optimize the critic
        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        # for param in self.critic.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.critic_optimiser.step()

        metrics = {
            "train/critic_loss": critic_loss.detach().cpu().numpy(),
            "train/Q1": current_Q1.mean().detach().cpu().numpy(),
            "train/Q2": current_Q2.mean().detach().cpu().numpy(),
            "train/cql_q1_rand": cql_random_q1.mean().detach().cpu().numpy(),
            "train/cql_q2_rand": cql_random_q2.mean().detach().cpu().numpy(),
            "train/cql_q1_current_actions": cql_sampled_q1.mean()
            .detach()
            .cpu()
            .numpy(),
            "train/cql_q2_current_actions": cql_sampled_q2.mean()
            .detach()
            .cpu()
            .numpy(),
            "train/cql_q1_next_actions": cql_sampled_next_q1.mean()
            .detach()
            .cpu()
            .numpy(),
            "train/cql_q2_next_actions": cql_sampled_next_q2.mean()
            .detach()
            .cpu()
            .numpy(),
            "train/cql_alpha": cql_alpha,
            "train/cql_alpha_loss": cql_alpha_loss.detach().cpu().numpy(),
            "train/cql_penalty": cql_penalty.detach().cpu().numpy(),
            "train/conservative_penalty": conservative_penalty.detach().cpu().numpy(),
        }

        return metrics
