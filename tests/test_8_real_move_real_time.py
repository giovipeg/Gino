import numpy as np
import os
import sys
import time
import cv2
from scipy.spatial.transform import Rotation as R

from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.gino.kinematics.kinematics import RobotKinematics
from src.gino.kinematics.robot_visualization import RobotVisualisation
from src.gino.kinematics.move_robot import MoveRobot
from src.gino.aruco.cube_detection import ArucoCubeTracker

aruco = ArucoCubeTracker()

robot_config = SO101FollowerConfig(
    port="/dev/ttyACM0",
    id="toni",
)

robot = SO101Follower(robot_config)
robot.connect()

cap = cv2.VideoCapture(0)

# --- Robot model setup ---
urdf_name = "so100"
urdf_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'urdf', f'{urdf_name}.urdf')
urdf_path = os.path.abspath(urdf_path)
kin = RobotKinematics(urdf_path)
viz = RobotVisualisation(kin, urdf_name, trajectory_viz=True)
move = MoveRobot(kin, robot=robot, visualization=viz, use_sim_time=False)

frame = viz.link_names[5]  # 6th link is end-effector

# State tracking variables for relative movement
movement_active = False
reference_arm_position = None
reference_cube_position = None

# Tilt angle oscillation variables
tilt_start_time = time.time()
tilt_period = 8.0  # Time in seconds for one complete cycle (0->90->0)

print("Controls:")
print("  'm' - Start/stop relative movement")
print("  'q' - Quit")

move.home()

while True:
    ret, image = cap.read()
    if not ret:
        break

    smoothed_position, rotation_matrix, euler_angles, cube_markers = aruco.pose_estimation(image)
    print(euler_angles)

    cv2.imshow('ArUco Cube Tracking', image)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('m'):
        # Toggle movement mode
        if not movement_active:
            # Start movement - record reference positions
            if smoothed_position is not None:
                reference_arm_position = move._get_current_end_effector_position(frame)                
                reference_cube_position = smoothed_position.copy()
                movement_active = True
                print(f"Movement started. Reference arm pos: {reference_arm_position}, Reference cube pos: {reference_cube_position}")
            else:
                print("No cube detected. Cannot start movement.")
        else:
            # Stop movement
            movement_active = False
            print("Movement stopped.")
    
    # Execute real-time relative movement
    if movement_active and smoothed_position is not None and reference_cube_position is not None:
        # Calculate relative movement of the cube
        cube_delta = smoothed_position - reference_cube_position
        
        # Calculate target arm position (relative to reference arm position)
        target_arm_position = reference_arm_position + cube_delta
        
        # Execute the movement
        # Create a rotation matrix with Z up and X pointing to target projection
        # Z axis always points up
        z_axis = np.array([0.0, 0.0, 1.0])
        
        # X axis points to the horizontal projection of the target
        target_horizontal = np.array([target_arm_position[0], target_arm_position[1], 0.0])
        if np.linalg.norm(target_horizontal) < 1e-6:  # If target is directly above/below
            x_axis = np.array([1.0, 0.0, 0.0])  # Default to pointing forward
        else:
            x_axis = target_horizontal / np.linalg.norm(target_horizontal)
            
        # Y axis is determined by cross product to maintain right-handed system
        y_axis = np.cross(z_axis, x_axis)
        
        # Build base rotation matrix
        rot_base = np.column_stack([x_axis, y_axis, z_axis])
        
        # Add end-effector rotation around Y axis (in degrees)
        # Oscillate tilt angle between 0 and 90 degrees
        elapsed_time = time.time() - tilt_start_time
        # Use sine wave to create smooth oscillation: sin goes from -1 to 1, so we map it to 0-90
        tilt_angle = 45 * (1 + np.sin(2 * np.pi * elapsed_time / tilt_period))
        tilt_rot = R.from_euler('y', tilt_angle, degrees=True).as_matrix()
        
        # Apply tilt rotation to base rotation
        rot_to_target = rot_base @ tilt_rot
        
        # Build target SE(3) pose for IK
        T = np.eye(4)
        T[:3, :3] = rot_to_target
        T[:3, 3] = target_arm_position
        
        q_sol = move.kin.ik(move.get_q_guess(), T, frame=frame, max_iters=10)
        q_sol_deg = np.degrees(q_sol)
        q_sol_deg[4] = 0.0
        action_dict = move._create_action_dict(q_sol_deg)
        print(f"Action dict: {action_dict}")
        robot.send_action(action_dict)
        
        # Display target position
        cv2.putText(image, f"Target: {target_arm_position}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        #time.sleep(0.02)  # 50 ms between points

move.home()

robot.disconnect()

cap.release()
cv2.destroyAllWindows()
