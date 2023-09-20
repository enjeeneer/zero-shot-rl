import pathlib

import torch
from dm_control import suite

from agents.fb.replay_buffer import FBReplayBuffer

_base = pathlib.Path(__file__).parent.parent.parent
_data_dir = _base / "datasets" / "walker" / "diayn" / "buffer"

env = suite.load(
    domain_name="walker",
    task_name="walk",
)

buffer = FBReplayBuffer(
    env=env,

    data_dir=_data_dir,
    max_episodes=1000,
    discount=0.99,
    device=torch.device("cpu"),
)


batch = buffer.sample(batch_size=32)
print('here')

