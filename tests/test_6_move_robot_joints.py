import numpy as np
import os
import sys
import time

from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.gino.kinematics.kinematics import RobotKinematics
from src.gino.kinematics.robot_visualization import RobotVisualisation
from src.gino.kinematics.move_robot import MoveRobot

robot_config = SO101FollowerConfig(
    port="/dev/ttyACM0",
    id="toni",
)

robot = SO101Follower(robot_config)
robot.connect()

# --- Robot model setup ---
urdf_name = "so100"
urdf_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'urdf', f'{urdf_name}.urdf')
urdf_path = os.path.abspath(urdf_path)
kin = RobotKinematics(urdf_path)
viz = RobotVisualisation(kin, urdf_name, trajectory_viz=True)
move = MoveRobot(kin, robot=robot, visualization=viz, use_sim_time=False)

# --- Trajectory parameters ---
num_steps = 500
end_effector_name = viz.link_names[5]  # 6th link is end-effector

target_point = np.array([0.15, 0.0, 0.1])  # Define your target point here

move.move_to_target(target_point, end_effector_name, num_steps)

time.sleep(2)

move.home()

robot.disconnect()
