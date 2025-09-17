import numpy as np
import os
import sys
from scipy.spatial.transform import Rotation as R
import json

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
frame = viz.link_names[5]  # 6th link is end-effector

move.home()

start_position, start_quaternion, _ = receiver.receive()
start_position = remap_convert(start_position)
current_pos = move.compute_ee_pos(frame=frame)

position = None
positions_log = []
output_file = "positions_log.json"

try:
    while True:
        position, quaternion, controls = receiver.receive()
        position = remap_convert(position)

        target_pos = position - start_position + np.array(current_pos)
        print(f"Target Position: {target_pos}, position: {position}, start_position: {start_position}, current_pos: {current_pos}")
        positions_log.append(target_pos.tolist())

        save_positions_to_file(positions_log, output_file)
        
        #q_sol = move.get_ik_solution(move._get_current_end_effector_orientation(frame), target_pos, frame)

        # START
        # Remap Unity rotation to robot frame: R_robot = C * R_unity * C^T
        C = np.array([
            [0, 1, 0],  # x_g = -x_u
            [0, 0, -1],  # y_g = -z_u
            [1, 0, 0],  # z_g =  y_u
        ])
        rot_matrix_unity = R.from_quat(quaternion).as_matrix()
        rot_matrix = C @ rot_matrix_unity @ C.T

        # Build target SE(3) pose
        T = np.eye(4)
        T[:3, :3] = rot_matrix
        T[:3, 3] = target_pos

        q_guess = move.get_q_guess()

        # Solve IK for joint angles (in radians)
        q_sol = kin.ik(q_guess, T, frame=frame, max_iters=10)
        current_q = q_sol.copy()  # Update current joint state with best effort
        q_sol = q_sol
        # END

        q_sol = np.degrees(q_sol)
        action_dict = move._create_action_dict(q_sol)
        move.robot.send_action(action_dict)

except KeyboardInterrupt:
    save_positions_to_file(positions_log, output_file)
    move.home()
    robot.disconnect()
    receiver.disconnect()
