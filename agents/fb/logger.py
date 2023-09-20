"""Logger for FB agent."""
import wandb
from typing import Dict, List
from pathlib import Path

from agents.base import AbstractLogger, AbstractAgent


class FBLogger(AbstractLogger):
    """Logs FB agent metrics during training and eval."""

    def __init__(
        self, agent_config: Dict, use_wandb: bool = False, wandb_tags: List[str] = None
    ):
        super().__init__(agent_config, use_wandb, wandb_tags)

        self.use_wandb = use_wandb

        self.metrics = {
            "train/forward_backward_total_loss": [],
            "train/forward_backward_fb_loss": [],
            "train/forward_backward_fb_diag_loss": [],
            "train/forward_backward_fb_off_diag_loss": [],
            "train/actor_loss": [],
            "train/actor_std_dev": [],
            "train/actor_loss_Q1": [],
            "train/actor_loss_Q2": [],
            "train/fb_loss": [],
            "train/ortho_diag_loss": [],
            "train/ortho_off_diag_loss": [],
            "train/target_M": [],
            "train/M": [],
            "train/F": [],
            "train/B": [],
            "train/actor_Q": [],
            "eval/step_reward": [],
            "eval/actor_std_dev": [],
            "eval/mean_episode_reward": [],
        }

    def save_model(self, agent: AbstractAgent, model_path: Path) -> None:
        """Saves model."""
        path = agent.save(model_path)

        if self.use_wandb:
            wandb.save(path.as_posix())
