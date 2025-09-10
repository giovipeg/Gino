import numpy as np
import os
import sys
from scipy.spatial.transform import Rotation as R
import json
import matplotlib.pyplot as plt

from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.gino.kinematics.kinematics import RobotKinematics
from src.gino.kinematics.robot_visualization import RobotVisualisation
from src.gino.kinematics.move_robot import MoveRobot
from src.gino.udp.receiver import Receiver
from test_kinematics import rk

def remap_convert(unity_position):
    return np.array([-unity_position[0] / 100, -unity_position[2] / 100, unity_position[1] / 100])

def save_positions_to_file(positions, filename):
    with open(filename, 'w') as f:
        json.dump(positions, f)

def read_positions_from_file(filename):
    with open(filename, 'r') as f:
        return json.load(f)

receiver = Receiver(udp_port=5005)

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
move.set_ik_weights([1.0, 1.0, 1.0, 1.0, 1.0, 0])
frame = viz.link_names[5]  # 6th link is end-effector

move.home()

start_position, start_quaternion, _ = receiver.receive()
start_position = remap_convert(start_position)
current_pos = move.compute_ee_pos(frame=frame)

position = None
positions_log = []
output_file = "positions_log.json"

try:
    i = 0
    while True:
        position, quaternion, controls = receiver.receive()
        print(quaternion)
        position = remap_convert(position)

        target_pos = position - start_position + np.array(current_pos)
        positions_log.append(target_pos.tolist())

        save_positions_to_file(positions_log, output_file)
        
        # Remap Unity rotation to robot frame: R_robot = C * R_unity * C^T
        
        C = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0]
        ])
        
        D = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0]
        ])

        # --- Method A: Matrix route ---
        R_unity = R.from_quat(quaternion).as_matrix()
        R_robot = C @ R_unity @ C.T
        q_robot = R.from_matrix(R_robot).as_quat()

        rr = D @ R_robot @ D.T

        print("Remapped quaternion (matrix method):", q_robot)







        # Build base rotation matrix
        target_horizontal = np.array([target_pos[0], target_pos[1], 0.0])
        x_axis = target_horizontal / np.linalg.norm(target_horizontal)
        z_axis = np.array([0, 0, 1])
        y_axis = np.cross(z_axis, x_axis)
        rot_base = np.column_stack([x_axis, y_axis, z_axis])

        # Add end-effector rotations (in degrees)
        tilt_angle = 90   # rotation about local Y (pitch)
        roll_angle = -90   # rotation about local X (roll)
        tilt_rot = R.from_euler('y', tilt_angle, degrees=True).as_matrix()
        roll_rot = R.from_euler('x', roll_angle, degrees=True).as_matrix()

        # Apply rotations to base rotation (order: tilt then roll in EE frame)
        rot_to_target = rot_base @ tilt_rot @ roll_rot

        R_ee = R.from_quat(quaternion).as_matrix()
        rot_to_target = rot_base @ R_robot

        # Build target SE(3) pose
        T = np.eye(4)
        T[:3, :3] = R_robot
        T[:3, 3] = target_pos

        # Solve IK for joint angles (in radians)
        q_sol = kin.ik(move.get_q_guess(), T, frame=frame, max_iters=10, weights6=move.default_weights6)
        current_q = q_sol.copy()  # Update current joint state with best effort
        q_sol = q_sol


        q_sol = np.degrees(q_sol)
        action_dict = move._create_action_dict(q_sol)
        move.robot.send_action(action_dict)

except KeyboardInterrupt:
    save_positions_to_file(positions_log, output_file)
    move.home()
    robot.disconnect()
    receiver.disconnect()
