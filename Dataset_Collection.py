import torch
from lerobot.common.robot_devices.robots import SO100Robot
from lerobot.common.datasets.utils import record_episode

robot = SO100Robot(fps=30, camera_type="opencv")
task_description = "Pick up the red cup and place it on the table"
episode_data = record_episode(
    robot=robot,
    task_description=task_description,
    output_dir="outputs/vla_red_cup",
    max_duration=60
)
episode_data.save("outputs/vla_red_cup/episode_1")
