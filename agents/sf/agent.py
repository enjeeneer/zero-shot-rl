"""Module defining the Successor Feature Agent."""

import math
from pathlib import Path
from typing import Tuple, Dict, Optional

import torch
import numpy as np

from agents.fb.models import ForwardRepresentation, ActorModel
from agents.sf.models import LaplacianFeatures
from agents.base import AbstractAgent, Batch, AbstractGaussianActor
from agents.utils import schedule


class SF(AbstractAgent):
    """
    Successor feature agent.
    """

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        preprocessor_hidden_dimension: int,
        preprocessor_output_dimension: int,
        preprocessor_hidden_layers: int,
        preprocessor_activation: str,
        z_dimension: int,
        forward_hidden_dimension: int,
        forward_hidden_layers: int,
        forward_number_of_features: int,
        features_hidden_dimension: int,
        features_hidden_layers: int,
        features_activation: str,
        actor_hidden_dimension: int,
        actor_hidden_layers: int,
        forward_activation: str,
        actor_activation: str,
        actor_learning_rate: float,
        sf_learning_rate: float,
        feature_learning_rate: float,
        batch_size: int,
        gaussian_actor: bool,
        std_dev_clip: float,
        std_dev_schedule: str,
        tau: float,
        device: torch.device,
        name: str,
        z_mix_ratio: float,
        q_loss: bool,
    ):
        super().__init__(
            observation_length=observation_length,
            action_length=action_length,
            name=name + "-lap",
        )

        self.successor_net_target = ForwardRepresentation(
            observation_length=observation_length,
            action_length=action_length,
            preprocessor_hidden_dimension=preprocessor_hidden_dimension,
            preprocessor_feature_space_dimension=preprocessor_output_dimension,
            preprocessor_hidden_layers=preprocessor_hidden_layers,
            preprocessor_activation=preprocessor_activation,
            number_of_features=forward_number_of_features,
            z_dimension=z_dimension,
            forward_hidden_dimension=forward_hidden_dimension,
            forward_hidden_layers=forward_hidden_layers,
            forward_activation=forward_activation,
            device=device,
        )
        self.successor_net = ForwardRepresentation(
            observation_length=observation_length,
            action_length=action_length,
            preprocessor_hidden_dimension=preprocessor_hidden_dimension,
            preprocessor_feature_space_dimension=preprocessor_output_dimension,
            preprocessor_hidden_layers=preprocessor_hidden_layers,
            preprocessor_activation=preprocessor_activation,
            number_of_features=forward_number_of_features,
            z_dimension=z_dimension,
            forward_hidden_dimension=forward_hidden_dimension,
            forward_hidden_layers=forward_hidden_layers,
            forward_activation=forward_activation,
            device=device,
        )
        self.successor_net_target.load_state_dict(self.successor_net.state_dict())

        self.feature_net = LaplacianFeatures(
            observation_dim=observation_length,
            z_dimension=z_dimension,
            hidden_dimension=features_hidden_dimension,
            hidden_layers=features_hidden_layers,
            device=device,
            activation=features_activation,
            layernorm=True,
        )
        self.actor = ActorModel(
            observation_length=observation_length,
            action_length=action_length,
            preprocessor_hidden_dimension=preprocessor_hidden_dimension,
            number_of_features=forward_number_of_features,
            preprocessor_feature_space_dimension=preprocessor_output_dimension,
            preprocessor_hidden_layers=preprocessor_hidden_layers,
            preprocessor_activation=preprocessor_activation,
            z_dimension=z_dimension,
            actor_hidden_dimension=actor_hidden_dimension,
            actor_hidden_layers=actor_hidden_layers,
            actor_activation=actor_activation,
            gaussian_actor=gaussian_actor,
            std_dev_clip=std_dev_clip,
            device=device,
        )

        self.encoder = torch.nn.Identity()  # TODO: add encoder for pixels
        self.augmentation = torch.nn.Identity()  # TODO: add augmentation for pixels

        # optimisers
        self.sf_optimizer = torch.optim.Adam(
            self.successor_net.parameters(),
            lr=sf_learning_rate,
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_learning_rate
        )
        self.feature_optimizer = torch.optim.Adam(
            self.feature_net.parameters(), lr=feature_learning_rate
        )

        self._device = device
        self._q_loss = q_loss
        self.batch_size = batch_size
        self._tau = tau
        self._z_dimension = z_dimension
        self._z_mix_ratio = z_mix_ratio
        self.std_dev_schedule = std_dev_schedule
        self.inv_cov = torch.eye(z_dimension, dtype=torch.float32, device=device)

    @torch.no_grad()
    def act(
        self,
        observation: Dict[str, np.ndarray],
        task: np.array,
        step: int,
        sample: bool = False,
    ) -> Tuple[np.array, float]:
        """
        Used at test time to perform zero-shot rollouts.
        Takes observation array from environment, encodes, and selects
        action from actor.
        Args:
            observation: observation array of shape [observation_length]
            task: task array of shape [z_dimension]
            step: current step in env  # TODO: check this is indeed used in rollouts?
            sample: whether to sample action from actor distribution
        Returns:
            action: action array of shape [action_length]
            std_dev: current actor standard deviation
        """

        observation = torch.as_tensor(
            observation, dtype=torch.float32, device=self._device
        ).unsqueeze(0)
        h = self.encoder(observation)
        z = torch.as_tensor(task, dtype=torch.float32, device=self._device).unsqueeze(0)

        # get action from actor
        std_dev = schedule(self.std_dev_schedule, step)
        action, _ = self.actor(observation=h, z=z, std=std_dev, sample=sample)

        return action.detach().cpu().numpy()[0], std_dev

    def update(self, batch: Batch, step: int) -> Dict[str, float]:
        """
        Updates agent's networks given a batch_size sample from the replay buffer.
        Args:
            batch: memory buffer containing transitions
            step: no. of steps taken in the environment
        Returns:
            metrics: dictionary of metrics for logging
        """

        # sample zs and betas if appropriate
        zs = self._sample_z(size=self.batch_size)

        if self._z_mix_ratio > 0:
            perm = torch.randperm(self.batch_size)
            phi_input = batch.next_observations[perm]

            with torch.no_grad():
                phi = self.feature_net.forward(phi_input)

            # compute inverse of cov of phi
            cov = torch.matmul(phi.T, phi) / phi.shape[0]
            inv_cov = torch.linalg.pinv(cov)

            mix_idxs = np.where(
                np.random.uniform(size=self.batch_size) < self._z_mix_ratio
            )[0]

            with torch.no_grad():
                new_z = phi[mix_idxs]

            new_z = torch.matmul(new_z, inv_cov)  # batch_size x z_dim
            new_z = math.sqrt(self._z_dimension) * torch.nn.functional.normalize(
                new_z, dim=1
            )
            zs[mix_idxs] = new_z

        actor_zs = zs.clone().requires_grad_(True)
        actor_observations = batch.observations.clone().requires_grad_(True)

        # update successor nets
        sf_metrics = self.update_successor_features(
            observations=batch.observations,
            next_observations=batch.next_observations,
            actions=batch.actions,
            discounts=batch.discounts,
            zs=zs,
            step=step,
        )

        # update actor
        actor_metrics = self.update_actor(
            observation=actor_observations, z=actor_zs, step=step
        )

        # update target networks for successor features
        self.soft_update_params(
            network=self.successor_net,
            target_network=self.successor_net_target,
            tau=self._tau,
        )

        metrics = {
            **sf_metrics,
            **actor_metrics,
        }

        return metrics

    def update_successor_features(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        discounts: torch.Tensor,
        zs: torch.Tensor,
        step: int,
    ) -> Dict[str, float]:
        """
        Updates the successor features.
        Args:
            observations: observation tensor of shape [batch_size, observation_length]
            actions: action tensor of shape [batch_size, action_length]
            next_observations: next observation tensor of
                                shape [batch_size, observation_length]
            discounts: discount tensor of shape [batch_size, 1]
            zs: policy tensor of shape [batch_size, z_dimension]
            step: current training step
        Returns:
            metrics: dictionary of metrics for logging
        """

        sf_loss, sf_metrics = self._get_successor_loss(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            discounts=discounts,
            zs=zs,
            step=step,
        )

        phi_loss, phi_metrics = self.feature_net.get_loss(
            observations=observations, next_observations=next_observations
        )

        # step optimisers
        self.sf_optimizer.zero_grad(set_to_none=True)
        sf_loss.backward()
        for param in self.successor_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.sf_optimizer.step()

        self.feature_optimizer.zero_grad(set_to_none=True)
        phi_loss.backward(retain_graph=True)
        for param in self.feature_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.feature_optimizer.step()

        metrics = {
            **sf_metrics,
            **phi_metrics,
        }

        return metrics

    def _get_successor_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        discounts: torch.Tensor,
        zs: torch.Tensor,
        step: int,
    ):
        """
        Loss computation common to FB and all child classes. All equation references
        are to the appendix of the FB paper (Touati et. al (2022)).
        The loss contains two components:
            1. Forward-backward representation loss: a Bellman update on the successor
                measure (equation 24, Appendix B)
            2. Orthonormalisation loss: constrains backward function such that the
                measure of state s from state s = 1 (equation 26, Appendix B)
            Note: Q loss (Equation 9) is not implemented.
        Args:
            observations: observation tensor of shape [batch_size, observation_length]
            actions: action tensor of shape [batch_size, action_length]
            next_observations: next observation tensor of
                                shape [batch_size, observation_length]
            discounts: discount tensor of shape [batch_size, 1]
            zs: policy tensor of shape [batch_size, z_dimension]
            step: current training step
        Returns:
            total_loss: total loss for FB
            metrics: dictionary of metrics for logging
        """

        with torch.no_grad():
            actor_std_dev = schedule(self.std_dev_schedule, step)
            next_actions, _ = self.actor(
                next_observations, zs, actor_std_dev, sample=True
            )

            next_F1, next_F2 = self.successor_net_target(
                observation=next_observations, z=zs, action=next_actions
            )

            target_phi = self.feature_net(observations=next_observations).detach()
            next_Q1, next_Q2 = [
                torch.einsum("sd, sd -> s", next_Fi, zs)
                for next_Fi in [next_F1, next_F2]
            ]
            next_F = torch.where((next_Q1 < next_Q2).reshape(-1, 1), next_F1, next_F2)
            target_F = target_phi + discounts * next_F

        # --- successor net loss ---
        F1, F2 = self.successor_net.forward(
            observation=observations, action=actions, z=zs
        )
        if self._q_loss:
            Q1, Q2 = [torch.einsum("sd, sd -> s", Fi, zs) for Fi in [F1, F2]]
            target_Q = torch.einsum("sd, sd -> s", target_F, zs)
            sf_loss = torch.nn.functional.mse_loss(
                Q1, target_Q
            ) + torch.nn.functional.mse_loss(Q2, target_Q)
        else:
            sf_loss = torch.nn.functional.mse_loss(
                F1, target_F
            ) + torch.nn.functional.mse_loss(F2, target_F)

        metrics = {
            "train/sf_loss": sf_loss.item(),
            "train/target_F": target_F.mean().item(),
            "train/F": F1.mean().item(),
        }

        return sf_loss, metrics

    def update_actor(
        self, observation: torch.Tensor, z: torch.Tensor, step: int
    ) -> Dict[str, float]:
        """
        Computes the actor loss.
        Args:
            observation: tensor of shape [batch_size, observation_length]
            z: tensor of shape [batch_size, z_dimension]
            betas: SF augmentation tensor of shape [batch_size, z_dimension];
                    ones if beta is not enabled.
            step: current training step
        Returns:
            metrics: dictionary of metrics for logging
        """
        std = schedule(self.std_dev_schedule, step)
        action, action_dist = self.actor(
            observation=observation,
            z=z,
            std=std,
            sample=True,
        )

        # with torch.no_grad():
        F1, F2 = self.successor_net(observation=observation, z=z, action=action)

        # get Qs from F and z
        Q1 = torch.einsum("sd, sd -> s", F1, z)
        Q2 = torch.einsum("sd, sd -> s", F2, z)
        Q = torch.min(Q1, Q2)

        # update actor towards action that maximise Q (minimise -Q)
        actor_loss = -Q

        if (
            type(self.actor.actor)  # pylint: disable=unidiomatic-typecheck
            == AbstractGaussianActor
        ):
            # add an entropy regularisation term
            log_prob = action_dist.log_prob(action).sum(-1)
            actor_loss += 0.1 * log_prob  # NOTE: currently hand-coded weight!
            mean_log_prob = log_prob.mean().item()
        else:
            mean_log_prob = 0.0

        actor_loss = actor_loss.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.actor_optimizer.step()

        metrics = {
            "train/actor_loss": actor_loss.item(),
            "train/actor_Q": Q.mean().item(),
            "train/actor_log_prob": mean_log_prob,
        }

        return metrics

    def load(self, filepath: Path):
        """Loads model."""
        pass

    def _sample_z(self, size: int) -> torch.Tensor:
        """Samples z in the sphere of radius sqrt(D)."""
        gaussian_random_variable = torch.randn(
            size, self._z_dimension, dtype=torch.float32, device=self._device
        )
        gaussian_random_variable = torch.nn.functional.normalize(
            gaussian_random_variable, dim=1
        )
        z = math.sqrt(self._z_dimension) * gaussian_random_variable

        return z

    def infer_z(
        self, observations: torch.Tensor, rewards: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Infers z from observations and rewards. Follows page 17 from:
        https://arxiv.org/pdf/2209.14935.pdf. If no rewards are passed, we
        assume we're inferring z from a goal-reaching task, and thus set z
        directly from the goal state.
        Args:
            observations: tensor of shape [inference_steps, observation_length]
            rewards (optional): tensor of shape [inference_steps, 1]
        Returns:
            z: tensor of shape [z_dimension,]
        """

        with torch.no_grad():
            phi = self.feature_net(observations=observations)

        if rewards is not None:
            z = torch.linalg.lstsq(phi, rewards).solution
            z = math.sqrt(self._z_dimension) * torch.nn.functional.normalize(z, dim=0)

        # goal reaching
        else:
            z = torch.matmul(phi, self.inv_cov)
            z = math.sqrt(self._z_dimension) * torch.nn.functional.normalize(z, dim=1)

        z = z.squeeze().cpu().numpy()

        return z

    def infer_z_from_goal(
        self, observation: np.ndarray, goal_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Infers z w.r.t. a goal state and the current observation
        Args:
            observation: array of shape [observation_length]
            goal_state: tensor of shape [observation_length]
            step: current step in env
        Returns:
            z: tensor of shape [z_dimension]
        """
        obs = torch.as_tensor(
            observation, dtype=torch.float32, device=self._device
        ).unsqueeze(0)

        with torch.no_grad():
            obs = self.encoder(obs)
            desired_goal = self.encoder(goal_state)

        with torch.no_grad():
            z_g = self.feature_net(desired_goal)
            z_s = self.feature_net(obs)

        z = z_g - z_s
        z = math.sqrt(self._z_dimension) * torch.nn.functional.normalize(z, dim=1)

        z = z.squeeze().cpu().numpy()

        return z

    def compute_cov(
        self,
        observations: torch.Tensor,
    ) -> None:
        """
        Computes the inverse of the covariance matrix of the features and
        stores internally. This is performed at the beginning of each
        evaluation on goal reaching tasks.
        Args:
            observations: tensor of shape [inference_steps, observation_length]
        Returns:
            None
        """

        with torch.no_grad():
            phi = self.feature_net(observations=observations)

        cov = torch.matmul(phi.T, phi) / phi.shape[0]
        inv_cov = torch.linalg.pinv(cov)

        self.inv_cov = inv_cov

    def predict_q(
        self, observation: torch.Tensor, z: torch.Tensor, action: torch.Tensor
    ):
        """
        Predicts the value of a state-action pair w.r.t. a task. Used as a utility
        function for downstream analysis.
        Args:
            observation: tensor of shape [N, observation_length]
            z: tensor of shape [N, z_dimension]
            action: tensor of shape [N, action_length]
        Returns:
            Qs
        """
        F1, F2 = self.successor_net(observation=observation, z=z, action=action)

        # get Qs from F and z
        Q1 = torch.einsum("sd, sd -> s", F1, z)
        Q2 = torch.einsum("sd, sd -> s", F2, z)
        Q = torch.min(Q1, Q2)

        return Q

    @staticmethod
    def soft_update_params(
        network: torch.nn.Sequential, target_network: torch.nn.Sequential, tau: float
    ) -> None:
        """
        Soft updates the target network parameters via Polyak averaging.
        Args:
            network: Online network.
            target_network: Target network.
            tau: Interpolation parameter.
        """

        for param, target_param in zip(
            network.parameters(), target_network.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
