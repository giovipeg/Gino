import sys
import os

from lerobot.robots.so100_follower import SO100FollowerConfig, SO100Follower
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from src.gino.controller.gino_controller import GinoController


robot_config = SO101FollowerConfig(
    port="/dev/ttyACM0",
    id="toni",
)
robot = SO101Follower(robot_config)
robot.connect()

controller = GinoController(robot, udp_port=5005)

try:
    while True:
        action = controller.get_action()
        robot.send_action(action)

except KeyboardInterrupt:
    controller.disconnect()
    robot.disconnect()
