"""module for retrieving fetch reward function from dm_control suite quadruped"""

from custom_dmc_tasks.quadruped import fetch

reward_function = fetch()._task.get_reward  # pylint: disable=protected-access
