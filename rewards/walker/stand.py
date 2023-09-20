"""module for retrieving run reward function from dm_control suite walker"""

from dm_control.suite.walker import stand

reward_function = stand()._task.get_reward  # pylint: disable=protected-access
