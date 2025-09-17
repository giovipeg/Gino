import cv2
import numpy as np
import time
import subprocess
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.gino.aruco.visualization_utils as viz
from src.gino.aruco.cube_detection import ArucoCubeTracker
from src.gino.mcu.mcu import MCU

# Visualization toggles
VISUALIZE_POSITION = False  # Set to True to visualize position
VISUALIZE_POSE = False      # Set to True to visualize pose

# 3D plot setup
if VISUALIZE_POSITION or VISUALIZE_POSE:
    fig_pos, ax_pos, fig_pose, ax_pose = viz.setup_visualization(VISUALIZE_POSITION, VISUALIZE_POSE)
else:
    fig_pos = ax_pos = fig_pose = ax_pose = None

mcu = MCU('/dev/ttyACM1', 500000)
# Connect to MCU
if mcu.connect():
    print("MCU connected successfully!")
    
    # Wait for Arduino to be ready
    print("Waiting for Arduino to send first data...")
    while not mcu.is_arduino_ready():
        time.sleep(0.1)
    print("Arduino is ready!")
else:
    print("Failed to connect to MCU!")

aruco = ArucoCubeTracker()

cube_positions = []
cube_orientations = []
window_size = 5

# Main script logic for video file
start_script_time = time.time()
cap = cv2.VideoCapture(0)

frame_count = 0
key = None
while key != ord('q') and key != 27:
    start_time = time.time()

    ret, image = cap.read()
    if not ret:
        break
    
    frame_count += 1

    smoothed_position, rotation_matrix_plot, euler_angles, cube_markers = aruco.pose_estimation(image)
    aruco.outlier_detection(mcu.get_last_pose_data(), euler_angles)
    
    if smoothed_position is not None:
            cube_positions.append(smoothed_position)
            cube_orientations.append(rotation_matrix_plot)

            # Update plots
            if VISUALIZE_POSITION or VISUALIZE_POSE:
                viz.update_visualization(
                    VISUALIZE_POSITION, VISUALIZE_POSE,
                    ax_pos, ax_pose,
                    cube_positions, smoothed_position,
                    rotation_matrix_plot, cube_markers,
                    aruco.cube_marker_positions, aruco.cube_size
                )
    cv2.imshow('ArUco Cube Tracking', image)
    key = cv2.waitKey(17) & 0xFF

cap.release()
cv2.destroyAllWindows()
plt.ioff()
if VISUALIZE_POSITION or VISUALIZE_POSE:
    viz.close_visualization()

total_exec_time = time.time() - start_script_time
print(f"Tracking completed. Total frames processed: {frame_count}")
print(f"Cube positions recorded: {len(cube_positions)}")
print(f"Total execution time: {total_exec_time:.2f} seconds")

# Save trajectory to file
np.savez('data/cube_trajectory2.npz', positions=np.array(cube_positions), orientations=np.array(cube_orientations))
print("Cube trajectory saved to 'data/cube_trajectory2.npz'")
