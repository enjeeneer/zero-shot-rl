# pylint: disable=protected-access

"""
Trains agents on a static, offline dataset and
evaluates their performance periodically.
"""

import yaml
import torch
from argparse import ArgumentParser
import datetime
from pathlib import Path

from agents.base import OfflineReplayBuffer
from agents.workspaces import OfflineRLWorkspace
from agents.cql.agent import CQL
from agents.sac.agent import SAC
from agents.fb.agent import FB
from agents.cfb.agent import CFB
from agents.calfb.agent import CalFB
from agents.td3.agent import TD3
from agents.fb.replay_buffer import FBReplayBuffer
from rewards import RewardFunctionConstructor
from utils import set_seed_everywhere, BASE_DIR

parser = ArgumentParser()
parser.add_argument("algorithm", type=str)
parser.add_argument("domain_name", type=str)
parser.add_argument("exploration_algorithm", type=str)
parser.add_argument("--wandb_logging", type=str, default="True")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--alpha", type=float, default=0.01)
parser.add_argument("--discount", type=float, default=0.98)
parser.add_argument("--z_dimension", type=int, default=50)
parser.add_argument("--weighted_cml", type=bool, default=False)
parser.add_argument("--total_action_samples", type=int, default=12)
parser.add_argument("--ood_action_weight", type=float, default=0.5)
parser.add_argument("--train_task", type=str)
parser.add_argument("--dataset_transitions", type=int, default=100000)
parser.add_argument("--eval_tasks", nargs="+", required=True)
parser.add_argument("--learning_steps", type=int, default=1000000)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--model_name", type=str, default=None)
parser.add_argument("--lagrange", type=str, default="True")
parser.add_argument("--target_conservative_penalty", type=float, default=50.0)
parser.add_argument("--action_condition_index", type=int)
parser.add_argument("--action_condition_value", type=float)
parser.add_argument("--cql_alpha", type=float, default=0.01)
args = parser.parse_args()

if args.wandb_logging == "True":
    args.wandb_logging = True
elif args.wandb_logging == "False":
    args.wandb_logging = False
else:
    raise ValueError("wandb_logging must be either True or False")

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

# action condition for subsampling dataset
if args.action_condition_index is not None:
    args.action_condition = {args.action_condition_index: args.action_condition_value}
else:
    args.action_condition = None

working_dir = Path.cwd()
if args.algorithm in ("vcfb", "mcfb", "vcalfb", "mcalfb"):
    algo_dir = "calfb" if "cal" in args.algorithm else "cfb"
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

set_seed_everywhere(config["seed"])

# setup dataset
dataset_path = (
    BASE_DIR
    / "datasets"
    / config["domain_name"]
    / config["exploration_algorithm"]
    / "dataset.npz"
)
if config["algorithm"] in ("fb", "vcfb", "mcfb", "vcalfb", "mcalfb"):
    relabel = False
else:
    relabel = True

# setup multi-reward environment
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

if config["algorithm"] == "cql":
    agent = CQL(
        observation_length=observation_length,
        action_length=action_length,
        device=config["device"],
        name=config["name"],
        batch_size=config["batch_size"],
        discount=config["discount"],
        critic_hidden_dimension=config["critic_hidden_dimension"],
        critic_hidden_layers=config["critic_hidden_layers"],
        critic_betas=config["critic_betas"],
        critic_tau=config["critic_tau"],
        critic_learning_rate=config["critic_learning_rate"],
        critic_target_update_frequency=config["critic_target_update_frequency"],
        actor_hidden_dimension=config["actor_hidden_dimension"],
        actor_hidden_layers=config["actor_hidden_layers"],
        actor_betas=config["actor_betas"],
        actor_learning_rate=config["actor_learning_rate"],
        actor_log_std_bounds=config["actor_log_std_bounds"],
        alpha_learning_rate=config["alpha_learning_rate"],
        alpha_betas=config["alpha_betas"],
        actor_update_frequency=config["actor_update_frequency"],
        init_temperature=config["init_temperature"],
        learnable_temperature=config["learnable_temperature"],
        activation=config["activation"],
        action_range=action_range,
        normalisation_samples=None,
        cql_n_samples=config["cql_n_samples"],
        cql_lagrange=config["lagrange"],
        cql_alpha=config["cql_alpha"],
        cql_target_penalty=config["target_conservative_penalty"],
    )

    replay_buffer = OfflineReplayBuffer(
        reward_constructor=reward_constructor,
        dataset_path=dataset_path,
        transitions=config["dataset_transitions"],
        relabel=relabel,
        task=config["train_task"],
        device=config["device"],
        discount=config["discount"],
    )

    z_inference_steps = None
    train_std = None
    eval_std = None

elif config["algorithm"] == "td3":
    agent = TD3(
        observation_length=observation_length,
        action_length=action_length,
        device=config["device"],
        name=config["name"],
        critic_hidden_dimension=config["critic_hidden_dimension"],
        critic_hidden_layers=config["critic_hidden_layers"],
        critic_learning_rate=config["critic_learning_rate"],
        critic_activation=config["activation"],
        actor_hidden_dimension=config["actor_hidden_dimension"],
        actor_hidden_layers=config["actor_hidden_layers"],
        actor_learning_rate=config["actor_learning_rate"],
        actor_activation=config["activation"],
        std_dev_clip=config["std_dev_clip"],
        std_dev_schedule=config["std_dev_schedule"],
        batch_size=config["batch_size"],
        discount=config["discount"],
        tau=config["critic_tau"],
    )

    replay_buffer = OfflineReplayBuffer(
        reward_constructor=reward_constructor,
        dataset_path=dataset_path,
        transitions=config["dataset_transitions"],
        relabel=relabel,
        task=config["train_task"],
        device=config["device"],
        discount=config["discount"],
    )

    z_inference_steps = None
    train_std = None
    eval_std = None


elif config["algorithm"] == "sac":
    agent = SAC(
        observation_length=observation_length,
        action_length=action_length,
        device=config["device"],
        name=config["name"],
        batch_size=config["batch_size"],
        discount=config["discount"],
        critic_hidden_dimension=config["critic_hidden_dimension"],
        critic_hidden_layers=config["critic_hidden_layers"],
        critic_betas=config["critic_betas"],
        critic_tau=config["critic_tau"],
        critic_learning_rate=config["critic_learning_rate"],
        critic_target_update_frequency=config["critic_target_update_frequency"],
        actor_hidden_dimension=config["actor_hidden_dimension"],
        actor_hidden_layers=config["actor_hidden_layers"],
        actor_betas=config["actor_betas"],
        actor_learning_rate=config["actor_learning_rate"],
        actor_log_std_bounds=config["actor_log_std_bounds"],
        alpha_learning_rate=config["alpha_learning_rate"],
        alpha_betas=config["alpha_betas"],
        actor_update_frequency=config["actor_update_frequency"],
        init_temperature=config["init_temperature"],
        learnable_temperature=config["learnable_temperature"],
        activation=config["activation"],
        action_range=action_range,
        normalisation_samples=None,
    )

    # load buffer
    replay_buffer = OfflineReplayBuffer(
        reward_constructor=reward_constructor,
        dataset_path=dataset_path,
        transitions=config["dataset_transitions"],
        relabel=relabel,
        task=config["train_task"],
        device=config["device"],
        discount=config["discount"],
    )

    z_inference_steps = None
    train_std = None
    eval_std = None

elif config["algorithm"] == "fb":

    if config["domain_name"] == "point_mass_maze":
        config["discount"] = 0.99
        config["z_dimension"] = 100

    agent = FB(
        observation_length=observation_length,
        action_length=action_length,
        preprocessor_hidden_dimension=config["preprocessor_hidden_dimension"],
        preprocessor_output_dimension=config["preprocessor_output_dimension"],
        preprocessor_hidden_layers=config["preprocessor_hidden_layers"],
        forward_hidden_dimension=config["forward_hidden_dimension"],
        forward_hidden_layers=config["forward_hidden_layers"],
        forward_number_of_features=config["forward_number_of_features"],
        backward_hidden_dimension=config["backward_hidden_dimension"],
        backward_hidden_layers=config["backward_hidden_layers"],
        actor_hidden_dimension=config["actor_hidden_dimension"],
        actor_hidden_layers=config["actor_hidden_layers"],
        preprocessor_activation=config["preprocessor_activation"],
        forward_activation=config["forward_activation"],
        backward_activation=config["backward_activation"],
        actor_activation=config["actor_activation"],
        z_dimension=config["z_dimension"],
        critic_learning_rate=config["critic_learning_rate"],
        actor_learning_rate=config["actor_learning_rate"],
        learning_rate_coefficient=config["learning_rate_coefficient"],
        orthonormalisation_coefficient=config["orthonormalisation_coefficient"],
        discount=config["discount"],
        batch_size=config["batch_size"],
        z_mix_ratio=config["z_mix_ratio"],
        gaussian_actor=config["gaussian_actor"],
        std_dev_clip=config["std_dev_clip"],
        std_dev_schedule=config["std_dev_schedule"],
        tau=config["tau"],
        device=config["device"],
        name=config["name"],
    )

    replay_buffer = FBReplayBuffer(
        reward_constructor=reward_constructor,
        dataset_path=dataset_path,
        transitions=config["dataset_transitions"],
        relabel=relabel,
        task=config["train_task"],
        device=config["device"],
        discount=config["discount"],
        action_condition=config["action_condition"],
    )

    z_inference_steps = config["z_inference_steps"]
    train_std = config["std_dev_schedule"]
    eval_std = config["std_dev_eval"]


elif config["algorithm"] in ("vcfb", "mcfb", "vcalfb", "mcalfb"):
    if config["domain_name"] == "point_mass_maze":
        config["discount"] = 0.99
        config["z_dimension"] = 100

    agent = (CalFB if "cal" in config["algorithm"] else CFB)(
        observation_length=observation_length,
        action_length=action_length,
        preprocessor_hidden_dimension=config["preprocessor_hidden_dimension"],
        preprocessor_output_dimension=config["preprocessor_output_dimension"],
        preprocessor_hidden_layers=config["preprocessor_hidden_layers"],
        forward_hidden_dimension=config["forward_hidden_dimension"],
        forward_hidden_layers=config["forward_hidden_layers"],
        forward_number_of_features=config["forward_number_of_features"],
        backward_hidden_dimension=config["backward_hidden_dimension"],
        backward_hidden_layers=config["backward_hidden_layers"],
        actor_hidden_dimension=config["actor_hidden_dimension"],
        actor_hidden_layers=config["actor_hidden_layers"],
        preprocessor_activation=config["preprocessor_activation"],
        forward_activation=config["forward_activation"],
        backward_activation=config["backward_activation"],
        actor_activation=config["actor_activation"],
        z_dimension=config["z_dimension"],
        actor_learning_rate=config["actor_learning_rate"],
        critic_learning_rate=config["critic_learning_rate"],
        learning_rate_coefficient=config["learning_rate_coefficient"],
        orthonormalisation_coefficient=config["orthonormalisation_coefficient"],
        discount=config["discount"],
        batch_size=config["batch_size"],
        z_mix_ratio=config["z_mix_ratio"],
        gaussian_actor=config["gaussian_actor"],
        std_dev_clip=config["std_dev_clip"],
        std_dev_schedule=config["std_dev_schedule"],
        tau=config["tau"],
        device=config["device"],
        vcfb=config["vcfb"],
        mcfb=config["mcfb"],
        total_action_samples=config["total_action_samples"],
        ood_action_weight=config["ood_action_weight"],
        alpha=config["alpha"],
        target_conservative_penalty=config["target_conservative_penalty"],
        lagrange=config["lagrange"],
    )

    replay_buffer = FBReplayBuffer(
        reward_constructor=reward_constructor,
        dataset_path=dataset_path,
        transitions=config["dataset_transitions"],
        relabel=relabel,
        task=config["train_task"],
        device=config["device"],
        discount=config["discount"],
        action_condition=config["action_condition"],
    )

    z_inference_steps = config["z_inference_steps"]
    train_std = config["std_dev_schedule"]
    eval_std = config["std_dev_eval"]

else:
    raise NotImplementedError(f"Algorithm {config['algorithm']} not implemented")

workspace = OfflineRLWorkspace(
    reward_constructor=reward_constructor,
    learning_steps=config["learning_steps"],
    model_dir=model_dir,
    eval_frequency=config["eval_frequency"],
    eval_rollouts=config["eval_rollouts"],
    z_inference_steps=z_inference_steps,
    train_std=train_std,
    eval_std=eval_std,
    wandb_logging=config["wandb_logging"],
    device=config["device"],
)

if __name__ == "__main__":

    workspace.train(
        agent=agent,
        tasks=config["eval_tasks"],
        agent_config=config,
        replay_buffer=replay_buffer,
    )
