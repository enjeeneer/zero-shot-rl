"""module for retrieving escape reward function from dm_control suite quadruped"""

from custom_dmc_tasks.quadruped import escape

reward_function = escape()._task.get_reward  # pylint: disable=protected-access
