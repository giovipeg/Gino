import sys
import os
import argparse

from lerobot.robots.so100_follower import SO100FollowerConfig, SO100Follower
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from src.gino.controller.gino_controller import GinoController


def parse_args():
    parser = argparse.ArgumentParser(description="Run Gino controller with configurable robot and UDP settings.")
    parser.add_argument("--robot", choices=["so100", "so101"], default="so101", help="Robot type to use.")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port for the robot bus.")
    parser.add_argument("--id", dest="robot_id", default="toni", help="Robot identifier (used by follower config).")
    parser.add_argument("--udp-port", type=int, default=5005, help="UDP port to listen for commands.")
    return parser.parse_args()

def build_robot(robot_type: str, port: str, robot_id: str):
    if robot_type == "so100":
        config = SO100FollowerConfig(port=port, id=robot_id)
        follower = SO100Follower(config)
    else:
        config = SO101FollowerConfig(port=port, id=robot_id)
        follower = SO101Follower(config)
    return follower

args = parse_args()

robot = build_robot(args.robot, args.port, args.robot_id)
robot.connect()

controller = GinoController(robot, udp_port=args.udp_port)

try:
    while True:
        action = controller.get_action()
        robot.send_action(action)

except KeyboardInterrupt:
    controller.disconnect()
    robot.disconnect()
