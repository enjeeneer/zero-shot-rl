# pylint: disable=protected-access

"""Trains and evaluates agents on D4RL."""
import yaml
import gym
import torch
import argparse
from argparse import ArgumentParser
import datetime
from pathlib import Path

from agents.workspaces import D4RLWorkspace
from agents.fb.agent import FB
from agents.cfb.agent import CFB
from agents.cql.agent import CQL
from agents.sf.agent import SF
from agents.base import D4RLReplayBuffer
from utils import set_seed_everywhere

parser = ArgumentParser()
parser.add_argument("algorithm", type=str)
parser.add_argument("domain_name", type=str)
parser.add_argument("dataset", type=str)
parser.add_argument(
    "--wandb_logging", default=True, action=argparse.BooleanOptionalAction
)
parser.add_argument("--wandb_entity", type=str, default="enjeeneer")
parser.add_argument("--wandb_project", type=str, default="zsrl-delta")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default=None)
parser.add_argument("--learning_steps", type=int, default=1000000)
parser.add_argument("--target_conservative_penalty", type=float, default=50)
args = parser.parse_args()

if args.algorithm in ("vcfb"):
    args.vcfb = True
    args.mcfb = False
    args.dvcfb = False
elif args.algorithm in ("mcfb"):
    args.vcfb = False
    args.mcfb = True
    args.dvcfb = False
elif args.algorithm == "dvcfb":
    args.vcfb = False
    args.mcfb = False
    args.dvcfb = True

working_dir = Path.cwd()

if args.algorithm in ("vcfb", "mcfb", "dvcfb"):
    algo_dir = "cfb"
    config_path = working_dir / "agents" / algo_dir / "config.yaml"
    model_dir = working_dir / "agents" / algo_dir / "saved_models"

elif args.algorithm in ("sf-lap", "sf-hilp"):
    algo_dir = "sf"
    config_path = working_dir / "agents" / algo_dir / "config.yaml"
    model_dir = working_dir / "agents" / algo_dir / "saved_models"
else:
    config_path = working_dir / "agents" / args.algorithm / "config.yaml"
    model_dir = working_dir / "agents" / args.algorithm / "saved_models"

with open(config_path, "rb") as f:
    config = yaml.safe_load(f)

time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

config.update(vars(args))

if config["device"] is None:
    config["device"] = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

set_seed_everywhere(config["seed"])

# pull data from GCS
dataset_path = (
    working_dir
    / "datasets"
    / f"{config['domain_name']}"
    / f"{config['dataset']}"
    / "dataset.hdf5"
)

# create env
if args.domain_name == "walker":
    gym_name = "Walker2d-v4"
elif args.domain_name == "cheetah":
    gym_name = "HalfCheetah-v4"
else:
    raise ValueError(f"Unknown domain {args.domain_name}")

env = gym.make(gym_name)
observation_length = env.observation_space.shape[0]
action_length = env.action_space.shape[0]
action_range = [min(env.action_space.low), max(env.action_space.high)]

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
        cql_lagrange=False,
        cql_alpha=config["cql_alpha"],
        cql_target_penalty=config["target_conservative_penalty"],
    )
    z_inference_steps = None
    train_std = None
    eval_std = None

elif config["algorithm"] == "fb":

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

    z_inference_steps = config["z_inference_steps"]
    train_std = config["std_dev_schedule"]
    eval_std = config["std_dev_eval"]

elif config["algorithm"] in ["sf-lap", "sf-hilp"]:

    if config["algorithm"] == "sf-lap":
        config["sf_features"] = "laplacian"
        config["hilp_p_random_goal"] = 0
    elif config["algorithm"] == "sf-hilp":
        config["sf_features"] = "hilp"
        config["hilp_p_random_goal"] = 0.375
    else:
        raise ValueError(f"Unknown algorithm {config['algorithm']}")

    agent = SF(
        observation_length=observation_length,
        action_length=action_length,
        preprocessor_hidden_dimension=config["preprocessor_hidden_dimension"],
        preprocessor_output_dimension=config["preprocessor_output_dimension"],
        preprocessor_hidden_layers=config["preprocessor_hidden_layers"],
        forward_hidden_dimension=config["forward_hidden_dimension"],
        forward_hidden_layers=config["forward_hidden_layers"],
        forward_number_of_features=config["forward_number_of_features"],
        features_hidden_dimension=config["features_hidden_dimension"],
        features_hidden_layers=config["features_hidden_layers"],
        features_activation=config["features_activation"],
        actor_hidden_dimension=config["actor_hidden_dimension"],
        actor_hidden_layers=config["actor_hidden_layers"],
        preprocessor_activation=config["preprocessor_activation"],
        forward_activation=config["forward_activation"],
        actor_activation=config["actor_activation"],
        z_dimension=config["z_dimension"],
        sf_learning_rate=config["sf_learning_rate"],
        feature_learning_rate=config["feature_learning_rate"],
        actor_learning_rate=config["actor_learning_rate"],
        batch_size=config["batch_size"],
        gaussian_actor=config["gaussian_actor"],
        std_dev_clip=config["std_dev_clip"],
        std_dev_schedule=config["std_dev_schedule"],
        tau=config["tau"],
        device=config["device"],
        name=config["name"],
        z_mix_ratio=config["z_mix_ratio"],
        q_loss=True,
    )

    z_inference_steps = config["z_inference_steps"]
    train_std = config["std_dev_schedule"]
    eval_std = config["std_dev_eval"]


elif config["algorithm"] in ("vcfb", "mcfb"):
    agent = CFB(
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

    z_inference_steps = config["z_inference_steps"]
    train_std = config["std_dev_schedule"]
    eval_std = config["std_dev_eval"]

else:
    raise NotImplementedError(f"Algorithm {config['algorithm']} not implemented")

# load buffer
replay_buffer = D4RLReplayBuffer(
    dataset_path=dataset_path,
    device=config["device"],
    discount=config["discount"],
)

workspace = D4RLWorkspace(
    env=env,
    domain_name=config["domain_name"],
    learning_steps=config["learning_steps"],
    model_dir=model_dir,
    eval_frequency=config["eval_frequency"],
    eval_rollouts=config["eval_rollouts"],
    z_inference_steps=z_inference_steps,
    wandb_logging=config["wandb_logging"],
    device=config["device"],
    wandb_project=config["wandb_project"],
    wandb_entity=config["wandb_entity"],
)

if __name__ == "__main__":

    workspace.train(
        agent=agent,
        agent_config=config,
        replay_buffer=replay_buffer,
    )
