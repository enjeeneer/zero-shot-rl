"""module for retrieving roll reward function from dm_control suite quadruped"""

from custom_dmc_tasks.quadruped import roll

reward_function = roll()._task.get_reward  # pylint: disable=protected-access
