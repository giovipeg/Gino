import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.gino.kinematics.kinematics import RobotKinematics
from src.gino.kinematics.robot_visualization import RobotVisualisation

# --- Robot model setup ---
urdf_name = "so100"
urdf_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'urdf', f'{urdf_name}.urdf')
urdf_path = os.path.abspath(urdf_path)
kin = RobotKinematics(urdf_path)
viz = RobotVisualisation(kin, urdf_name)

# --- Visualization setup ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# --- Trajectory parameters ---
num_steps = 300
radius = 0.12
center = np.array([0.22, 0.0, 0.12])
traj_points = []
actual_ee_points = []

# Fixed orientation for the end-effector (pointing downwards)
default_rot = R.from_euler("ZYX", [0, 90, 0], degrees=True).as_matrix()

# Initial guess for joint angles (in radians)
q_guess = np.zeros(5)
end_effector_name = viz.link_names[5]  # 6th link is end-effector

for t in range(num_steps):
    # Desired end-effector position: circle in YZ plane
    theta = 2 * np.pi * t / num_steps
    pos = center + radius * np.array([0, np.cos(theta), np.sin(theta)])
    traj_points.append(pos)

    # Build target SE(3) pose
    T = np.eye(4)
    T[:3, :3] = default_rot
    T[:3, 3] = pos

    # Solve IK for joint angles (in radians)
    q_sol = kin.ik(q_guess, T, frame=end_effector_name, max_iters=10)
    q_guess = q_sol.copy()  # Use solution as next guess for smoothness

    # Prepare joint vector for visualization (degrees + gripper)
    q_vis = np.zeros(6)
    q_vis[:5] = np.degrees(q_sol)
    q_vis[5] = 45  # Fixed gripper open

    # Compute actual end-effector position from FK
    ee_actual = kin.fk(q_sol, end_effector_name)[:3, 3]
    actual_ee_points.append(ee_actual)

    # Draw robot
    viz.draw(ax, q_vis)

    # Draw desired and actual trajectory so far
    traj_array = np.array(traj_points)
    actual_array = np.array(actual_ee_points)
    if len(traj_array) > 1:
        ax.plot(traj_array[:, 0], traj_array[:, 1], traj_array[:, 2], 'g--', label='Desired Trajectory' if t == 0 else "")
        ax.plot(actual_array[:, 0], actual_array[:, 1], actual_array[:, 2], 'b-', label='Actual Trajectory' if t == 0 else "")
    if t == 0:
        ax.legend()

    ax.set_xlim([0, 0.4])
    ax.set_ylim([-0.25, 0.25])
    ax.set_zlim([0, 0.3])
    ax.set_title("Robot Arm Following End-Effector Trajectory (IK)")
    plt.pause(0.02)

plt.show() 