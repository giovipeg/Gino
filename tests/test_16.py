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

controller_zero, _, _ = receiver.receive()
controller_zero = remap_convert(controller_zero)
start_position = np.array(move.compute_ee_pos(frame=frame))

position = None
positions_log = []
output_file = "positions_log.json"

C = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0]
])
angles_thresh = 2
prev_q = None

try:
    while True:
        position, quaternion, controls = receiver.receive()

        if controls["button2"] is True:
            move.home()

            while controls["button2"] is True:
                position, quaternion, controls = receiver.receive()

            controller_zero = remap_convert(position)
            start_position = np.array(move.compute_ee_pos(frame=frame))

            prev_q = None
        elif controls["button1"] is not True:
            position = remap_convert(position)

            target_pos = position - controller_zero + start_position
            positions_log.append(target_pos.tolist())
            save_positions_to_file(positions_log, output_file)
            
            # Remap Unity rotation to robot frame
            R_unity = R.from_quat(quaternion).as_matrix()
            R_robot = C @ R_unity @ C.T

            # Build target SE(3) pose
            T = np.eye(4)
            T[:3, :3] = R_robot
            T[:3, 3] = target_pos

            # Solve IK for joint angles (in radians)
            q_sol = kin.ik(move.get_q_guess(), T, frame=frame, max_iters=10, weights6=move.default_weights6)
            q_sol = np.degrees(q_sol)
            
            # Clip q_sol values to prevent erratic behaviour
            if prev_q is not None:
                q_sol = np.clip(q_sol, prev_q - angles_thresh, prev_q + angles_thresh)

            prev_q = q_sol
            print(q_sol)
            #q_sol[4] = 90 - q_sol[4]
            q_sol = np.append(q_sol, 90 - controls["slider"] * 90)
            action_dict = move._create_action_dict(q_sol)
            move.robot.send_action(action_dict)
        else:
            controller_zero = remap_convert(position)
            start_position = np.array(move.compute_ee_pos(frame=frame))

            prev_q = None

except KeyboardInterrupt:
    save_positions_to_file(positions_log, output_file)
    move.home()
    robot.disconnect()
    receiver.disconnect()
