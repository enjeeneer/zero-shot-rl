"""module for retrieving run reward function from dm_control suite walker"""

from dm_control.suite.walker import walk

reward_function = walk()._task.get_reward  # pylint: disable=protected-access
