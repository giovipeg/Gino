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

end_effector_name = viz.link_names[5]  # 6th link is end-effector

# --- Trajectory parameters ---
center = np.array([0.20, 0.0, 0.15])  # Center of the circle (x, y, z)
radius = 0.05  # 5 cm
num_points = 300  # Number of points along the circle
normal = np.array([0, 0, 1])  # Circle in the XY plane

# Generate circle points in the XY plane, then shift to center
angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
circle_points = np.zeros((num_points, 3))
circle_points[:, 0] = center[0] + radius * np.cos(angles)
circle_points[:, 1] = center[1] + radius * np.sin(angles)
circle_points[:, 2] = center[2]  # Constant height

move.home()

time.sleep(1)

# --- Move to starting point ---
start_point = circle_points[0]
frame = end_effector_name  # Use end-effector frame from kinematics
num_steps_to_start = 500
print(f"Moving to start point: {start_point}")
move.move_to_target(start_point, frame, num_steps_to_start)

# --- Execute circular trajectory ---
print("Executing circular trajectory...")
for idx, pos in enumerate(circle_points):
    # Use current end-effector orientation (keep constant)
    rot = move._get_current_end_effector_orientation(frame)
    q_sol = move.get_ik_solution(rot, pos, frame)
    q_sol_deg = np.degrees(q_sol)
    action_dict = move._create_action_dict(q_sol_deg)
    robot.send_action(action_dict)
    time.sleep(0.05)  # 50 ms between points
    print(f"Point {idx+1}/{num_points}: {pos}")

print("Trajectory complete.")
