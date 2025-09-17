import os
import sys
import time

from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.gino.kinematics.kinematics import RobotKinematics
from src.gino.kinematics.robot_visualization import RobotVisualisation
from src.gino.kinematics.move_robot import MoveRobot

urdf_name = "so100"
urdf_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'urdf', f'{urdf_name}.urdf')
urdf_path = os.path.abspath(urdf_path)
kin = RobotKinematics(urdf_path)
move = MoveRobot(kin, use_sim_time=False)
robot_config = SO101FollowerConfig(
    port="/dev/ttyACM0",
    id="toni",
)

robot = SO101Follower(robot_config)
robot.connect()

print("Press Ctrl+C to exit...")

try:
    while True:
        print(robot.get_observation())
        time.sleep(1)
except KeyboardInterrupt:
    print("\nExiting...")

robot.disconnect()
