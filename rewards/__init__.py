"""Module for constructing reward functions."""
import importlib
import torch
from typing import List, Dict, Tuple
import numpy as np
import custom_dmc_tasks as cdmc
import dmc
from utils import set_seed_everywhere

# NOTE: These default tasks aren't actually used;
# we just need to specify one to get the env to load
DEFAULT_TASKS = {
    "walker": "flip",
    "point_mass_maze": "reach_bottom_left",
    "cheetah": "run_backward",
    "quadruped": "jump",
}


class RewardFunctionConstructor:
    """
    Given a domain and tasks, constructs a set of reward functions.
    """

    def __init__(
        self,
        domain_name: str,
        task_names: List[str],
        seed: int,
        device: torch.device,
        hardcode_seed: int = None,
    ):
        if hardcode_seed is not None:
            set_seed_everywhere(hardcode_seed)
        if domain_name == "jaco":
            self._env = dmc.make(name="jaco_reach_bottom_left")
        else:
            self._env = cdmc.make(
                domain=domain_name,
                task=DEFAULT_TASKS[domain_name],
                environment_kwargs={"flat_observation": True},
                task_kwargs={"random": seed},
            )

        self.domain_name = domain_name
        self.task_names = task_names
        self.device = device
        self.reward_functions = {}
        for task in task_names:
            self.reward_functions[task] = importlib.import_module(
                f"rewards.{domain_name}.{task}"
            ).reward_function

    def __call__(self, physics):
        return [self.reward_functions[task](physics) for task in self.task_names]

    def process_episode(
        self, episode: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, Dict]:
        """Given an episode from an offline dataset, return observations and rewards
        for all tasks."""
        observations, rewards = [], []

        for physics, action in zip(episode["physics"][:-1], episode["action"][1:]):
            self._env.physics.set_state(physics)
            assert (physics == self._env.physics.get_state()).all()
            timestep = self._env.step(action)
            observations.append(timestep.observation["observations"])
            rewards.append(self(self._env.physics))

        rewards_dict = {}
        for i, task in enumerate(self.task_names):
            rewards_dict[task] = np.array(rewards)[:, i]

        return np.array(observations), rewards_dict

    def process_timesteps(
        self, batch_physics: np.ndarray, batch_actions: np.ndarray
    ) -> Tuple[torch.tensor, Dict[str, torch.tensor]]:
        """
        Given a set of timesteps from an offline dataset, return observations and
        rewards for all tasks.
        """
        observations, rewards = [], []

        for physics, action in zip(batch_physics, batch_actions):
            self._env.physics.set_state(physics)
            assert (physics == self._env.physics.get_state()).all()
            timestep = self._env.step(action)
            observations.append(timestep.observation["observations"])
            rewards.append(self(self._env.physics))

        observations = np.array(observations)
        observations = torch.tensor(
            observations, dtype=torch.float32, device=self.device
        )

        rewards_dict = {}
        for i, task in enumerate(self.task_names):
            rewards_dict[task] = torch.tensor(
                rewards, dtype=torch.float32, device=self.device
            )[:, i].unsqueeze(-1)

        return observations, rewards_dict

    def rollout_agent(
        self, agent, zs: Dict, obs_flag: bool = False
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Given an agent and z vector, rollout an episode and return rewards and
        (potentially) observations.
        Args:
            agent: Agent for rollout
            zs: Dictionary mapping task names to z vectors
            obs_flag: Whether to return observations
        Returns:
            rewards: Dictionary mapping task names to rewards
            observations: Dictionary mapping task names to observations
        """
        rewards = {}
        observations = {}
        actions = {}
        # loop through zs specified for each task
        for task, z in zs.items():
            assert task in self.task_names, f"Env not built for task name: {task}"
            task_rewards = 0.0
            obs_list = []
            action_list = []
            reward_list = []

            timestep = self._env.reset()
            obs_list.append(timestep.observation["observations"])

            while not timestep.last():
                action, _ = agent.act(
                    timestep.observation["observations"],
                    task=z,
                    step=None,
                    sample=False,
                )
                timestep = self._env.step(action)
                task_rewards += self.reward_functions[task](self._env.physics)
                if obs_flag:
                    obs_list.append(timestep.observation["observations"])
                    action_list.append(action)
                    reward_list.append(self.reward_functions[task](self._env.physics))

            rewards[task] = task_rewards
            if obs_flag:
                observations[task] = np.array(obs_list)[:-1]
                actions[task] = np.array(action_list)
                rewards[task] = np.array(reward_list)

        return rewards, observations, actions
