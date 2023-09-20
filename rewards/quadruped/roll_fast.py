"""module for retrieving roll_fast reward function from dm_control suite quadruped"""

from custom_dmc_tasks.quadruped import roll_fast

reward_function = roll_fast()._task.get_reward  # pylint: disable=protected-access
