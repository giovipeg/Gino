import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.gino.kinematics.kinematics import RobotKinematics
from src.gino.kinematics.robot_visualization import RobotVisualisation
from src.gino.kinematics.move_robot import MoveRobot

# --- Robot model setup ---
urdf_name = "so100"
urdf_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'urdf', f'{urdf_name}.urdf')
urdf_path = os.path.abspath(urdf_path)
kin = RobotKinematics(urdf_path)
viz = RobotVisualisation(kin, urdf_name, trajectory_viz=True)
move = MoveRobot(kin, robot=None, visualization=viz, use_sim_time=True)

# --- Trajectory parameters ---
num_steps = 100
start_q = np.array([0.0, -1.5708, 1.5708, 0.0, 0.0])  # Define your start joint configuration here
move.current_q = start_q
end_effector_name = viz.link_names[5]  # 6th link is end-effector

target_point = np.array([0.15, 0.0, 0.15])  # Define your target point here

move.move_to_target(target_point, end_effector_name, num_steps)
