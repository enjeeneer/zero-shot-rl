"""
module for retrieving reach_bottom_left reward function
from dm_control suite point_mass_maze
"""

from custom_dmc_tasks.cheetah import flip_backward

reward_function = flip_backward()._task.get_reward  # pylint: disable=protected-access
