"""Logger for SAC."""
import wandb
from typing import Dict, List
from pathlib import Path

from agents.base import AbstractLogger, AbstractAgent


class SACLogger(AbstractLogger):
    """Logs FB agent metrics during training and eval."""

    def __init__(
        self, agent_config: Dict, use_wandb: bool = False, wandb_tags: List[str] = None
    ):
        super().__init__(agent_config, use_wandb, wandb_tags)

        self.use_wandb = use_wandb

        self.metrics = {
            "train/critic_loss": [],
            "train/Q1": [],
            "train/Q2": [],
            "train/actor_loss": [],
            "train/actor_Q": [],
            "train/alpha": [],
            "train/alpha_loss": [],
            "eval/step_reward": [],
            "eval/actor_std_dev": [],
            "eval/mean_episode_reward": [],
        }

    def save_model(self, agent: AbstractAgent, model_path: Path) -> None:
        """Saves model."""
        path = agent.save(model_path)

        if self.use_wandb:
            wandb.save(path.as_posix())
