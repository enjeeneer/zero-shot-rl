"""Module defining the replay buffer for the FB agent."""
from pathlib import Path
import torch
import numpy as np
from rewards import RewardFunctionConstructor
from typing import Dict, Tuple

from agents.base import OfflineReplayBuffer, Batch


class FBReplayBuffer(OfflineReplayBuffer):
    """Forward backward replay buffer."""

    def __init__(
        self,
        reward_constructor: RewardFunctionConstructor,
        dataset_path: Path,
        discount: float,
        transitions: int,
        device: torch.device,
        relabel: bool = False,
        task: str = None,
        action_condition: dict = None,
    ):
        super().__init__(
            reward_constructor=reward_constructor,
            dataset_path=dataset_path,
            discount=discount,
            transitions=transitions,
            device=device,
            relabel=relabel,
            task=task,
            action_condition=action_condition,
        )
        self.reward_constructor = reward_constructor

    def sample_task_inference_transitions(
        self,
        inference_steps: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Sample transitions from the replay buffer for FB task inference.
        Args:
            inference_steps: number of transitions to sample
        Returns:
            z_inf_observations: observations for task inference
            z_inf_rewards: rewards for task inference
        """

        if len(self.storage) == 0:
            raise RuntimeError(
                "The replay buffer is empty. Task inference sampling"
                "can only be performed after the replay buffer has been"
                "loaded."
            )

        assert inference_steps <= len(self.storage["observations"])

        # sample transitions from the replay buffer for processing
        batch_indices = torch.randint(
            0, len(self.storage["observations"]), (inference_steps,)
        )

        batch_physics = self.storage["physics"][batch_indices]
        batch_actions = self.storage["actions"][batch_indices].detach().cpu().numpy()

        z_inf_observations, z_inf_rewards = self.reward_constructor.process_timesteps(
            batch_physics=batch_physics, batch_actions=batch_actions
        )

        return z_inf_observations, z_inf_rewards


class OnlineFBReplayBuffer(FBReplayBuffer):
    """
    Replay buffer for FB agent that initially loads an offline
    dataset, but allows data obtained from online interaction
    to augment it.
    """

    def __init__(
        self,
        offline_data_ratio: float,
        capacity: int,
        observation_length: int,
        action_length: int,
        reward_constructor: RewardFunctionConstructor,
        dataset_path: Path,
        discount: float,
        transitions: int,
        device: torch.device,
        relabel: bool = False,
        task: str = None,
        action_condition: dict = None,
    ):
        super().__init__(
            reward_constructor=reward_constructor,
            dataset_path=dataset_path,
            discount=discount,
            transitions=transitions,
            device=device,
            relabel=relabel,
            task=task,
            action_condition=action_condition,
        )

        self.online_observations = np.zeros(
            (capacity, observation_length),
            dtype=np.float32,
        )

        self.online_next_observations = np.zeros(
            (capacity, observation_length),
            dtype=np.float32,
        )

        self.online_actions = np.empty(
            (
                capacity,
                action_length,
            ),
            dtype=np.float32,
        )

        self.online_rewards = np.empty((capacity, int(1)), dtype=np.float32)

        self.online_not_dones = np.empty((capacity, int(1)), dtype=np.float32)

        self.online_discounts = np.ones((capacity, int(1)), dtype=np.float32) * discount

        self.current_memory_index = int(0)
        self.full_memory = False
        self.capacity = capacity
        self.last_offline_data_index = int(len(self.storage["observations"]))
        self.offline_data_ratio = offline_data_ratio

    def add(
        self,
        observation: np.array,
        next_observation: np.array,
        action: np.array,
        reward: np.array,
        done: np.array,
    ) -> None:
        """
        Stores online transition in memory.
        Args:
            observation: array of shape [observation_length]
            next_observation: array of shape [observation_length]
            action: array of shape [action_length]
            reward: array of shape [1]
            done: array of shape [1]

        Returns:
            None
        """

        np.copyto(self.online_observations[self.current_memory_index], observation)
        np.copyto(
            self.online_next_observations[self.current_memory_index], next_observation
        )
        np.copyto(self.online_actions[self.current_memory_index], action)
        np.copyto(self.online_rewards[self.current_memory_index], reward)
        np.copyto(self.online_not_dones[self.current_memory_index], not done)

        # update index
        self.current_memory_index = int((self.current_memory_index + 1) % self.capacity)
        self.full_memory = self.full_memory or self.current_memory_index == 0

    def sample(self, batch_size: int):
        """
        Sample a mix of transitions from the online and offline
        replay buffers.
        Args:
            batch_size: number of transitions to sample
        Returns:
            Batch: batch of transitions
        """
        # get offline and online sample indices
        offline_samples_number = int(batch_size * self.offline_data_ratio)
        online_samples_number = batch_size - offline_samples_number
        assert offline_samples_number + online_samples_number == batch_size

        offline_sample_indices = torch.randint(
            0, len(self.storage["observations"]), (offline_samples_number,)
        )
        online_sample_indices = np.random.randint(
            0, self.current_memory_index, size=online_samples_number
        )

        # combine offline and online transitions
        observations = torch.cat(
            (
                self.storage["observations"][offline_sample_indices],
                torch.as_tensor(
                    self.online_observations[online_sample_indices], device=self.device
                ),
            ),
        )
        next_observations = torch.cat(
            (
                self.storage["next_observations"][offline_sample_indices],
                torch.as_tensor(
                    self.online_next_observations[online_sample_indices],
                    device=self.device,
                ),
            ),
        )
        actions = torch.cat(
            (
                self.storage["actions"][offline_sample_indices],
                torch.as_tensor(
                    self.online_actions[online_sample_indices], device=self.device
                ),
            ),
        )
        rewards = torch.cat(
            (
                self.storage["rewards"][offline_sample_indices],
                torch.as_tensor(
                    self.online_rewards[online_sample_indices], device=self.device
                ),
            ),
        )
        not_dones = torch.cat(
            (
                self.storage["not_dones"][offline_sample_indices],
                torch.as_tensor(
                    self.online_not_dones[online_sample_indices], device=self.device
                ),
            ),
        )
        discounts = torch.cat(
            (
                self.storage["discounts"][offline_sample_indices],
                torch.as_tensor(
                    self.online_discounts[online_sample_indices], device=self.device
                ),
            ),
        )

        return Batch(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            not_dones=not_dones,
            discounts=discounts,
        )
