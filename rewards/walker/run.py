"""module for retrieving run reward function from dm_control suite walker"""

from dm_control.suite.walker import run

reward_function = run()._task.get_reward  # pylint: disable=protected-access
