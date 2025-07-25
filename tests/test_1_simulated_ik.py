import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

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
move = MoveRobot(kin, viz, use_sim_time=True)

# --- Trajectory parameters ---
num_steps = 300
radius = 0.12
center = np.array([0.22, 0.0, 0.12])
traj_points = []
actual_ee_points = []

start_q = np.array([0.0, 3.14, 3.14, 0.0, 0.0])  # Define your start joint configuration here
move.current_q = start_q

# Fixed orientation for the end-effector (pointing downwards)
default_rot = [0, 90, 0]

# Initial guess for joint angles (in radians)
q_guess = np.zeros(5)
end_effector_name = viz.link_names[5]  # 6th link is end-effector

for t in range(num_steps):
    # Desired end-effector position: circle in YZ plane
    theta = 2 * np.pi * t / num_steps
    pos = center + radius * np.array([0, np.cos(theta), np.sin(theta)])
    traj_points.append(pos)

    q_sol = move.get_ik_solution(default_rot, pos, end_effector_name)

    # Draw robot
    viz.draw(np.concatenate([q_sol, [np.radians(45)]]))

    # Draw desired and actual trajectory so far
    traj_array = np.array(traj_points)
    if len(traj_array) > 1:
        viz.ax.plot(traj_array[:, 0], traj_array[:, 1], traj_array[:, 2], 'g--', label='Desired Trajectory' if t == 0 else "")
    if t == 0:
        viz.ax.legend()

    viz.ax.set_xlim([0, 0.4])
    viz.ax.set_ylim([-0.25, 0.25])
    viz.ax.set_zlim([0, 0.3])
    viz.ax.set_title("Robot Arm Following End-Effector Trajectory (IK)")
    plt.pause(0.02)

plt.show()
