import cv2
import numpy as np
import time
import subprocess
import os
import sys
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.gino.aruco.visualization_utils as viz_utils
from src.gino.aruco.cube_detection import ArucoCubeTracker
from src.gino.kinematics.kinematics import RobotKinematics
from src.gino.kinematics.robot_visualization import RobotVisualisation

# --- Robot model setup ---
urdf_name = "so100"
urdf_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'urdf', f'{urdf_name}.urdf')
urdf_path = os.path.abspath(urdf_path)
kin = RobotKinematics(urdf_path)
robot_viz = RobotVisualisation(kin, urdf_name)

# --- Visualization setup ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.ion()
plt.show()

# Fixed orientation for the end-effector (pointing downwards)
default_rot = R.from_euler("ZYX", [0, 90, 0], degrees=True).as_matrix()

aruco = ArucoCubeTracker()

cube_positions = []
cube_orientations = []
actual_ee_points = []
desired_ee_points = []
window_size = 5

# Initial guess for joint angles (in radians)
q_guess = np.zeros(5)
end_effector_name = robot_viz.link_names[5]  # 6th link is end-effector

# Offset logic
offset = None

# Main script logic for video file
start_script_time = time.time()
#cap = cv2.VideoCapture('data/vid3.avi')
cap = cv2.VideoCapture(0)

frame_count = 0
key = None
while key != ord('q') and key != 27:
    start_time = time.time()

    ret, image = cap.read()
    if not ret:
        break
    frame_count += 1

    smoothed_position, rotation_matrix_plot, cube_markers = aruco.pose_estimation(image)

    if smoothed_position is not None:
        if offset is None:
            # Get initial robot EE position
            q_init = np.zeros(5)  # or your actual initial joint config
            initial_ee_position = kin.fk(q_init, end_effector_name)[:3, 3]
            offset = initial_ee_position - smoothed_position
        # Apply offset to trajectory
        adjusted_position = smoothed_position + offset
        cube_positions.append(smoothed_position)
        cube_orientations.append(rotation_matrix_plot)
        desired_ee_points.append(adjusted_position)

        # Build target SE(3) pose for IK
        T = np.eye(4)
        T[:3, :3] = default_rot
        T[:3, 3] = adjusted_position

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
        robot_viz.draw(ax, q_vis)

        # Draw desired and actual trajectory so far
        traj_array = np.array(desired_ee_points)
        actual_array = np.array(actual_ee_points)
        if len(traj_array) > 1:
            ax.plot(traj_array[:, 0], traj_array[:, 1], traj_array[:, 2], 'g--', label='Desired Trajectory' if frame_count == 1 else "")
            ax.plot(actual_array[:, 0], actual_array[:, 1], actual_array[:, 2], 'b-', label='Actual Trajectory' if frame_count == 1 else "")
        if frame_count == 1:
            ax.legend()

        ax.set_xlim([0, 0.4])
        ax.set_ylim([-0.25, 0.25])
        ax.set_zlim([0, 0.3])
        ax.set_title("Robot Arm Following Detected Cube Trajectory (IK)")
        plt.pause(0.001)

    cv2.imshow('ArUco Cube Tracking', image)
    key = cv2.waitKey(1) & 0xFF

    print(f"fps: {1 / (time.time() - start_time)}")

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()

total_exec_time = time.time() - start_script_time
print(f"Tracking completed. Total frames processed: {frame_count}")
print(f"Cube positions recorded: {len(cube_positions)}")
print(f"Total execution time: {total_exec_time:.2f} seconds")

# Save trajectory to file
np.savez('data/cube_trajectory1.npz', positions=np.array(cube_positions), orientations=np.array(cube_orientations))
print("Cube trajectory saved to 'data/cube_trajectory1.npz'")
