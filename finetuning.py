# pylint: disable=protected-access

"""Finetunes pre-trained agents on a specified task."""
import yaml
import torch
from argparse import ArgumentParser
import datetime
from pathlib import Path

from agents.workspaces import FinetuningWorkspace
from agents.fb.replay_buffer import FBReplayBuffer, OnlineFBReplayBuffer
from rewards import RewardFunctionConstructor
from utils import set_seed_everywhere, download_from_gcs_bucket, pull_model_from_wandb

parser = ArgumentParser()
parser.add_argument("algorithm", type=str)
parser.add_argument("domain_name", type=str)
parser.add_argument("exploration_algorithm", type=str)
parser.add_argument("--eval_tasks", nargs="+", required=True)
parser.add_argument("--online_finetuning", type=str, default="False")
parser.add_argument("--critic_tuning", type=str, default="False")
parser.add_argument("--wandb_logging", type=str, default="True")
parser.add_argument("--episodes", type=int, default=500)
parser.add_argument("--offline_data_ratio", type=float, default=0.0)
parser.add_argument("--dataset_transitions", type=int, default=100000)
parser.add_argument("--learning_steps", type=int, default=500000)
parser.add_argument("--wandb_run_id", type=str, required=True)
parser.add_argument("--wandb_model_id", type=str, required=True)
parser.add_argument("--lagrange", type=str, default="True")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--target_conservative_penalty", type=float, default=50.0)
args = parser.parse_args()

if args.wandb_logging == "True":
    args.wandb_logging = True
elif args.wandb_logging == "False":
    args.wandb_logging = False
else:
    raise ValueError("wandb_logging must be either True or False")

if args.online_finetuning == "True":
    args.online_finetuning = True
elif args.online_finetuning == "False":
    args.online_finetuning = False

if args.critic_tuning == "True":
    args.critic_tuning = True
elif args.critic_tuning == "False":
    args.critic_tuning = False

if args.algorithm in ("vcfb", "vcalfb"):
    args.vcfb = True
    args.mcfb = False
elif args.algorithm in ("mcfb", "mcalfb"):
    args.vcfb = False
    args.mcfb = True

if args.lagrange == "True":
    args.lagrange = True
elif args.lagrange == "False":
    args.lagrange = False

working_dir = Path.cwd()
if args.algorithm in ("vcfb", "mcfb", "vcalfb", "mcalfb"):
    algo_dir = "cfb"
    config_path = working_dir / "agents" / algo_dir / "config.yaml"
    model_dir = working_dir / "agents" / algo_dir / "saved_models"
else:
    config_path = working_dir / "agents" / args.algorithm / "config.yaml"
    model_dir = working_dir / "agents" / args.algorithm / "saved_models"

time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

with open(config_path, "rb") as f:
    config = yaml.safe_load(f)

config.update(vars(args))
config["device"] = torch.device(
    "cuda" if torch.cuda.is_available() else ("mps" if torch.has_mps else "cpu")
)
config["train_task"] = config["eval_tasks"][0]

set_seed_everywhere(config["seed"])

# pull data from GCS
dataset_path = download_from_gcs_bucket(
    domain_name=config["domain_name"],
    exploration_algorithm=config["exploration_algorithm"],
)
relabel = False

reward_constructor = RewardFunctionConstructor(
    domain_name=config["domain_name"],
    task_names=config["eval_tasks"],
    seed=config["seed"],
    device=config["device"],
)

if config["domain_name"] == "jaco":
    observation_length = reward_constructor._env.observation_spec().shape[
        0
    ]  # pylint: disable=protected-access
    action_length = reward_constructor._env.action_spec().shape[
        0
    ]  # pylint: disable=protected-access
    action_range = [-1.0, 1.0]
else:
    observation_length = reward_constructor._env.observation_spec()[
        "observations"
    ].shape[
        0
    ]  # pylint: disable=protected-access
    action_length = reward_constructor._env.action_spec().shape[
        0
    ]  # pylint: disable=protected-access
    action_range = [
        reward_constructor._env.action_spec().minimum[
            0
        ],  # pylint: disable=protected-access
        reward_constructor._env.action_spec().maximum[
            0
        ],  # pylint: disable=protected-access
    ]

agent = pull_model_from_wandb(
    algorithm=config["algorithm"],
    domain_name=config["domain_name"],
    wandb_run_id=config["wandb_run_id"],
    wandb_model_id=config["wandb_model_id"],
    observation_length=observation_length,
    action_length=action_length,
    config=config,
)

if config["online_finetuning"]:
    capacity = (config["episodes"] + 1) * 1000
    replay_buffer = OnlineFBReplayBuffer(
        offline_data_ratio=config["offline_data_ratio"],
        capacity=capacity,
        observation_length=observation_length,
        action_length=action_length,
        reward_constructor=reward_constructor,
        dataset_path=dataset_path,
        transitions=config["dataset_transitions"],
        relabel=relabel,
        task=config["train_task"],
        device=config["device"],
        discount=config["discount"],
    )
else:
    replay_buffer = FBReplayBuffer(
        reward_constructor=reward_constructor,
        dataset_path=dataset_path,
        transitions=config["dataset_transitions"],
        relabel=relabel,
        task=config["train_task"],
        device=config["device"],
        discount=config["discount"],
    )

z_inference_steps = config["z_inference_steps"]
train_std = config["std_dev_schedule"]
eval_std = config["std_dev_eval"]

workspace = FinetuningWorkspace(
    reward_constructor=reward_constructor,
    learning_steps=config["learning_steps"],
    model_dir=model_dir,
    eval_frequency=config["eval_frequency"],
    eval_rollouts=config["eval_rollouts"],
    z_inference_steps=config["z_inference_steps"],
    train_std=train_std,
    eval_std=eval_std,
    wandb_logging=config["wandb_logging"],
    device=config["device"],
    online=config["online_finetuning"],
    critic_tuning=config["critic_tuning"],
)

if __name__ == "__main__":

    workspace.train(
        agent=agent,
        tasks=config["eval_tasks"],
        agent_config=config,
        replay_buffer=replay_buffer,
        episodes=config["episodes"],
    )
