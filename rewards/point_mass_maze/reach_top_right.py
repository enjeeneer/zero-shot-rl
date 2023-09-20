"""
module for retrieving reach_top_right reward function
 from dm_control suite point_mass_maze
 """

from custom_dmc_tasks.point_mass_maze import reach_top_right

reward_function = reach_top_right()._task.get_reward  # pylint: disable=protected-access
