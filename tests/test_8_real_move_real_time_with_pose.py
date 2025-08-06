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
from src.gino.mcu.mcu import MCU
from src.gino.kinematics.orientation_fusion import OrientationFusion

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

# Initialize the orientation fusion system
orientation_fusion = OrientationFusion(
    complementary_alpha=0.90,  # 90% gyro, 10% ArUco
    lpf_alpha=0.35,           # Strong smoothing for noise reduction
    dt=0.010                  # 10ms time step
)

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

    # Get gyroscope data
    try:
        gyro_roll, gyro_pitch, gyro_yaw = mcu.get_last_gyro_data()
        orientation_fusion.update_gyro_data(gyro_roll, gyro_pitch, gyro_yaw)
    except Exception as e:
        print(f"Warning: Could not read gyroscope data: {e}")
        orientation_fusion.update_gyro_data(0, 0, 0)

    smoothed_position, rotation_matrix, euler_angles, cube_markers = aruco.pose_estimation(image)
    
    # Update ArUco data for orientation fusion
    if smoothed_position is not None and rotation_matrix is not None and euler_angles is not None:
        aruco_roll, aruco_pitch, aruco_yaw = euler_angles
        orientation_fusion.update_aruco_data(aruco_roll, aruco_pitch, aruco_yaw)
    else:
        orientation_fusion.update_aruco_data(None, None, None)
    
    # Process the sensor fusion
    filtered_roll, filtered_pitch, filtered_yaw = orientation_fusion.process()
    
    # Get current state for display
    current_state = orientation_fusion.get_current_state()
    
    # Get potentiometer value
    input_data = mcu.get_last_input_data()
    potentiometer_value = input_data['potentiometer_value'] if input_data else None
    
    outlier = aruco.outlier_detection(mcu.get_last_pose_data(), euler_angles, threshold=15)

    if outlier:
        print("Outlier detected")
        #continue

    # Print pose values in a formatted way
    print("\033[2J\033[H", end="")  # Clear screen and move cursor to top
    print("=" * 80)
    print("           REAL-TIME ROBOT MOVEMENT WITH POSE TRACKING")
    print("=" * 80)
    print(f"Movement Active: {movement_active}")
    print("-" * 80)
    print("SENSOR          ROLL (°)    PITCH (°)    YAW (°)")
    print("-" * 80)
    
    # Gyroscope values
    gyro = current_state['gyro']
    print(f"Gyroscope       {gyro['roll']:8.1f}    {gyro['pitch']:8.1f}    {gyro['yaw']:8.1f}")
    
    # ArUco values
    aruco_state = current_state['aruco']
    if aruco_state['roll'] is not None:
        print(f"ArUco           {aruco_state['roll']:8.1f}    {aruco_state['pitch']:8.1f}    {aruco_state['yaw']:8.1f}")
    else:
        print(f"ArUco           {'N/A':>8}    {'N/A':>8}    {'N/A':>8}")
    
    # Filtered values
    filtered = current_state['filtered']
    print(f"Filtered        {filtered['roll']:8.1f}    {filtered['pitch']:8.1f}    {filtered['yaw']:8.1f}")
    print("-" * 80)
    
    # Print position information
    if smoothed_position is not None:
        print(f"Cube Position:  X={smoothed_position[0]:6.3f}, Y={smoothed_position[1]:6.3f}, Z={smoothed_position[2]:6.3f}")
    else:
        print("Cube Position:  N/A")
    
    if reference_cube_position is not None:
        print(f"Ref Cube Pos:   X={reference_cube_position[0]:6.3f}, Y={reference_cube_position[1]:6.3f}, Z={reference_cube_position[2]:6.3f}")
    
    if reference_arm_position is not None:
        print(f"Ref Arm Pos:    X={reference_arm_position[0]:6.3f}, Y={reference_arm_position[1]:6.3f}, Z={reference_arm_position[2]:6.3f}")
    
    # Print potentiometer value
    if potentiometer_value is not None:
        print(f"Potentiometer:  {potentiometer_value:3d} (0-1023)")
    else:
        print("Potentiometer:  N/A")
    
    print("-" * 80)
    print("Controls: 'm' - Start/stop movement, 'q' - Quit")
    print("=" * 80)

    cv2.imshow('ArUco Cube Tracking', image)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('m') or key == ord('h'):
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
            if key == ord('h'):
                move.home()
    
    # Execute real-time relative movement
    if movement_active and smoothed_position is not None and reference_cube_position is not None:
        # Calculate relative movement of the cube
        cube_delta = smoothed_position - reference_cube_position
        
        # --- Axis remapping: Map cube's Z (forward) to robot's X (forward) ---
        # Adjust this mapping if needed for your robot's coordinate system
        # Example: remapped_delta = [cube Z, cube X, cube Y]
        remapped_delta = np.array([
            cube_delta[1],  # Cube Z (forward) -> Robot X (forward)
            -cube_delta[0],  # Cube X (right)   -> Robot Y (sideways)
            cube_delta[2],  # Cube Y (up)      -> Robot Z (up)
        ])
        
        # Calculate target arm position (relative to reference arm position)
        target_arm_position = reference_arm_position + remapped_delta
        
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
        #tilt_angle = 45 * (1 + np.sin(2 * np.pi * elapsed_time / tilt_period))
        tilt_angle = filtered['pitch']
        tilt_rot = R.from_euler('y', tilt_angle, degrees=True).as_matrix()
        
        # Apply tilt rotation to base rotation
        rot_to_target = rot_base @ tilt_rot
        
        # Build target SE(3) pose for IK
        T = np.eye(4)
        T[:3, :3] = rot_to_target
        T[:3, 3] = target_arm_position
        
        q_sol = move.kin.ik(move.get_q_guess(), T, frame=frame, max_iters=10)
        q_sol_deg = np.degrees(q_sol)
        q_sol_deg[4] = filtered['roll']
        q_sol_deg = np.append(q_sol_deg, potentiometer_value / 15 - 30)
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