"""module for retrieving flip reward function from dm_control suite walker"""

from custom_dmc_tasks.walker import flip

reward_function = flip()._task.get_reward  # pylint: disable=protected-access
