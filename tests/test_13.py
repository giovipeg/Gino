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
from src.gino.udp.receiver import Receiver

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

# start_position, start_quaternion, _ = receiver.receive()
start_position = [0, 0, 0]

position = None
try:
    for z in range(20):
        # position, quaternion, controls = receiver.receive()
        position = [0, 0, z/1000]

        current_pos = move.compute_ee_pos(frame=frame)

        target_pos = np.array(position) - np.array(start_position) + np.array(current_pos)

        q_sol = move.get_ik_solution(move._get_current_end_effector_orientation(frame), target_pos, frame)
        q_sol = np.degrees(q_sol)
        action_dict = move._create_action_dict(q_sol)
        move.robot.send_action(action_dict)

        input("Press Enter to continue...")

except KeyboardInterrupt:
    robot.disconnect()
    receiver.disconnect()
