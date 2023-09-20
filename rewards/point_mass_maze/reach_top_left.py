"""
module for retrieving reach_top_left reward function
from dm_control suite point_mass_maze
"""

from custom_dmc_tasks.point_mass_maze import reach_top_left

reward_function = reach_top_left()._task.get_reward  # pylint: disable=protected-access
