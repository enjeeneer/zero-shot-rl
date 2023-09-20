"""module for retrieving stand reward function from dm_control suite quadruped"""

from custom_dmc_tasks.quadruped import stand

reward_function = stand()._task.get_reward  # pylint: disable=protected-access
