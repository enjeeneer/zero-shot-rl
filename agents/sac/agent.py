"""Module implementing the Soft Actor Critic (SAC) algorithm."""

import abc
from pathlib import Path
from typing import Dict, List, Union, Optional

import numpy as np
import torch
import torch.nn.functional

from agents.base import AbstractAgent, AbstractGaussianActor, DoubleQCritic, Batch
from agents.sac.replay_buffer import SoftActorCriticReplayBuffer


class SAC(AbstractAgent, metaclass=abc.ABCMeta):
    """
    Paper: Haarjona et al. (2018) [https://arxiv.org/pdf/1801.01290.pdf]
    Torch Implementation: Yarats (2019) [https://github.com/denisyarats/pytorch_sac]
    """

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
        normalisation_samples: int = None,
    ):
        super().__init__(
            observation_length=observation_length,
            action_length=action_length,
            name=name,
        )

        # --- networks
        self.actor = AbstractGaussianActor(
            observation_length=observation_length,
            action_length=action_length,
            hidden_dimension=actor_hidden_dimension,
            hidden_layers=actor_hidden_layers,
            log_std_bounds=actor_log_std_bounds,
            device=device,
            activation=activation,
        )
        self.critic = DoubleQCritic(
            observation_length=observation_length,
            action_length=action_length,
            hidden_dimension=critic_hidden_dimension,
            hidden_layers=critic_hidden_layers,
            activation=activation,
            device=device,
        )
        self.critic_target = DoubleQCritic(
            observation_length=observation_length,
            action_length=action_length,
            hidden_dimension=critic_hidden_dimension,
            hidden_layers=critic_hidden_layers,
            activation=activation,
            device=device,
        )
        self.critic_target.load_state_dict(self.critic.state_dict())

        # --- misc
        self.device = device
        self.log_alpha = torch.tensor(
            np.log(init_temperature), dtype=torch.float32, device=self.device
        )
        self.log_alpha.requires_grad = True
        self._critic_tau = critic_tau
        self._gamma = discount
        self._actor_update_frequency = actor_update_frequency
        self._critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self._learnable_temperature = learnable_temperature
        self._action_length = action_length
        self._action_range = action_range

        # normalisation parameters
        self._normalisation_samples = normalisation_samples
        if self._normalisation_samples is not None:
            self._normalise = True
        else:
            self._normalise = False
        self.running_mean_numpy = None
        self.running_std_numpy = None
        self.running_mean_torch = None
        self.running_std_torch = None

        # --- optimisers
        self.actor_optimiser = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_learning_rate,
            betas=actor_betas,
        )
        self.critic_optimiser = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_learning_rate,
            betas=critic_betas,
        )
        self.log_alpha_optimiser = torch.optim.Adam(
            [self.log_alpha],
            lr=alpha_learning_rate,
            betas=alpha_betas,
        )

    def act(
        self,
        observation: np.ndarray,
        sample: bool = False,
        step: int = None,  # pylint: disable=unused-argument
    ) -> np.ndarray:
        """
        Takes observation array from environment and selects action from actor.
        Args:
            observation: observation array of shape [observation_length]
            sample: True if in training loop, False if in eval or production
            step: current step in training loop
        Returns:
            action: action array of shape [action_length]
        """

        observation = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        )
        observation = observation.unsqueeze(0)

        action, _ = self.actor(observation, sample=sample)

        return action.detach().cpu().numpy()

    def update(self, batch: Batch, step: int) -> Dict[str, float]:
        """
        Updates agent's networks given a batch_size sample from the replay buffer.
        Args:
            batch: memory buffer containing transitions
            step: no. of steps taken in the environment
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

        # if self._normalise:
        #     observations = self.normalise_observation(
        #         observations, replay_buffer=replay_buffer
        #     )
        #     next_observations = self.normalise_observation(
        #         next_observations, replay_buffer=replay_buffer
        #     )

        critic_metrics = self._update_critic(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            not_dones=not_dones,
        )

        actor_metrics = {}
        if step % self.actor_update_frequency == 0:
            actor_metrics = self._update_actor(observations)

        if step % self.critic_target_update_frequency == 0:
            self._soft_critic_target_update()

        metrics = {**actor_metrics, **critic_metrics}

        return metrics

    def _update_critic(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        not_dones: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Updates critic. See equation 7 in Haarnoja et. al (2018).
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
            next_actions, log_prob = self.actor(next_observations, sample=True)

            # get Q targets via soft policy evaluation
            target_Q1, target_Q2 = self.critic_target(next_observations, next_actions)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            target_Q = rewards + (not_dones * self.gamma * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(observations, actions)
        critic_loss = torch.nn.functional.mse_loss(
            current_Q1, target_Q
        ) + torch.nn.functional.mse_loss(current_Q2, target_Q)

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
        }

        return metrics

    def _update_actor(self, observations: torch.Tensor) -> Dict[str, float]:
        """
        Updates actor via double Q learning and maximum entropy RL.
        See equations 1 and 12 in Haarnoja et. al (2018).
        Args:
            observations: tensor of shape [batch_dim, observation_length]
        Returns:
            metrics: dictionary of metrics for logging
        """

        actions, log_prob = self.actor(observations, sample=True)

        actor_Q1, actor_Q2 = self.critic(observations, actions)

        # get loss (a combination of maximising reward and acting randomly)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        # optimize the actor
        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        # for param in self.actor.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.actor_optimiser.step()

        alpha_metrics = {}
        if self._learnable_temperature:
            alpha_metrics = self._update_alpha(log_prob)

        actor_metrics = {
            "train/actor_loss": actor_loss.detach().cpu().numpy(),
            "train/actor_Q": actor_Q.mean().detach().cpu().numpy(),
            "train/log_prob": log_prob.mean().detach().cpu().numpy(),
        }

        metrics = {**actor_metrics, **alpha_metrics}

        return metrics

    def _update_alpha(self, log_prob: torch.Tensor) -> Dict[str, float]:
        """
        Updates policy temperature (balances maximising reward vs. acting randomly
        in actor loss)
        Args:
            log_prob: log probability of actions sampled from current policy
                        conditioned on observations from memory.
        Returns:
            metrics: dictionary of metrics for logging
        """

        self.log_alpha_optimiser.zero_grad()
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimiser.step()

        metrics = {
            "train/alpha": self.alpha.detach().cpu().numpy(),
            "train/alpha_loss": alpha_loss.detach().cpu().numpy(),
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

    @property
    def target_entropy(self) -> float:
        """
        Target entropy shoud be proportional to action dimension (i.e. larger action
        space requires more total randomness0. Haarnoja et. al (2018) set it to the
        length of the action space directly.
        """
        return -self._action_length

    @property
    def alpha(self) -> torch.Tensor:
        """
        Balances reward maximisation vs
        entropy maximisation in policy.
        """
        return self.log_alpha.exp()

    @property
    def tau(self) -> float:
        """
        Weights how much we move the target critics weights towards the critics
        during the soft value function update.
        """
        return self._critic_tau

    @property
    def gamma(self) -> float:
        """
        Discounts future returns during updates
        """
        return self._gamma

    @property
    def actor_update_frequency(self) -> int:
        """
        The step interval after which the actor updates
        """
        return self._actor_update_frequency

    @property
    def critic_target_update_frequency(self) -> int:
        """
        The step interval after which the target critics parameters
        are moved toward the online critics parameter values
        """
        return self._critic_target_update_frequency

    def normalise_observation(
        self,
        observation: Union[np.ndarray, torch.tensor],
        replay_buffer: Optional[SoftActorCriticReplayBuffer] = None,
    ):
        """
        Z-normalises observation by calculating running mean and std
        of observations in replay buffer.
        Args:
            observation: observation to be normalised
            replay_buffer: buffer to sample observations from.
        Returns:
            normalised observation
        """

        if replay_buffer is not None:
            samples_idxs = np.random.randint(
                low=0,
                high=replay_buffer.current_memory_index,
                size=self._normalisation_samples,
            )

            samples = replay_buffer.observations[samples_idxs]

            running_mean = np.mean(
                samples,
                axis=0,
            )
            running_std = np.std(
                samples,
                axis=0,
            )
            self.running_mean_numpy = running_mean
            self.running_std_numpy = running_std

        if torch.is_tensor(observation):
            running_mean_torch = torch.as_tensor(
                self.running_mean_numpy, dtype=torch.float32, device=self.device
            )
            running_std_torch = torch.as_tensor(
                self.running_std_numpy, dtype=torch.float32, device=self.device
            )

            self.running_mean_torch = running_mean_torch
            self.running_std_torch = running_std_torch

            return (observation - self.running_mean_torch) / self.running_std_torch

        else:
            return (observation - self.running_mean_numpy) / self.running_std_numpy
