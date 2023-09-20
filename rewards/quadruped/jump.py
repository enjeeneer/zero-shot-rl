"""module for retrieving jump reward function from dm_control suite quadruped"""

from custom_dmc_tasks.quadruped import jump

reward_function = jump()._task.get_reward  # pylint: disable=protected-access
