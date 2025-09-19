import numpy as np
import os
import sys
from scipy.spatial.transform import Rotation as R
import json
import matplotlib.pyplot as plt
import socket

from lerobot.robots.so100_follower import SO100FollowerConfig, SO100Follower
from lerobot.robots.so101_follower import SO101FollowerConfig, SO101Follower

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from src.gino.kinematics.kinematics import RobotKinematics
from src.gino.kinematics.robot_visualization import RobotVisualisation
from src.gino.kinematics.move_robot import MoveRobot
from src.gino.udp.receiver import Receiver


class GinoController:
    def __init__(self, robot: SO101Follower | SO100Follower, udp_port: int = 5005):
        # --- Robot model setup ---
        # Note: so_100.urdf is ok for so101 as well
        urdf_name = "so100"
        urdf_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'urdf', f'so_100.urdf')
        urdf_path = os.path.abspath(urdf_path)
        
        self.robot = robot
        self.kin = RobotKinematics(urdf_path)
        self.viz = RobotVisualisation(self.kin, urdf_name, trajectory_viz=True)
        self.move = MoveRobot(self.kin, robot=self.robot, visualization=self.viz, use_sim_time=False)
        self.receiver = Receiver(udp_port=udp_port)

        # Remove z-axis rotation constrain since the robot has 5 DoF
        self.ik_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0])
        self.frame = self.viz.link_names[5]  # 6th link is end-effector
        self.C = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0]
        ])

        # Configuration values
        self.angles_thresh = 3
        self.prev_q = None
        deg_offset = 45
        self.rad_offset = np.deg2rad(deg_offset)

        self._display_ip_and_port()

        self.controller_zero, self.start_position = self._get_zero()

        self.move.home()

    def _display_ip_and_port(self):
        try:
            # Connect to a remote address to determine the local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
            print(f"UDP Server listening on IP: {local_ip}")
        except Exception:
            print("Failed to get local IP")

        print(f"UDP Port: {self.receiver.udp_port}")

    def _remap_convert(self, unity_position):
        return np.array([-unity_position[0] / 100, -unity_position[2] / 100, unity_position[1] / 100])

    def _get_zero(self):
        controller_zero, _, _ = self.receiver.receive()
        controller_zero = self._remap_convert(controller_zero)
        start_position = np.array(self.move.compute_ee_pos(frame=self.frame))

        return controller_zero, start_position

    def get_action(self):
        position, quaternion, controls = self.receiver.receive()

        if controls["button2"] is True:
            self.move.home()

            while controls["button2"] is True:
                position, quaternion, controls = self.receiver.receive()

            self.controller_zero = remap_convert(position)
            self.start_position = np.array(self.move.compute_ee_pos(frame=self.frame))

            self.prev_q = None
        elif controls["button1"] is not True:
            position = remap_convert(position)

            target_pos = self.start_position + (position - self.controller_zero) * 0.6
            positions_log.append(target_pos.tolist())
            save_positions_to_file(positions_log, output_file)
            
            # Remap Unity rotation to robot frame
            R_unity = R.from_quat(quaternion).as_matrix()
            R_robot = self.C @ R_unity @ self.C.T

            # Build target SE(3) pose
            T = np.eye(4)
            T[:3, :3] = R_robot
            T[:3, 3] = target_pos

            # Solve IK for joint angles (in radians)
            current_q = self.move.get_q_guess()
            current_q[4] = self.rad_offset - current_q[4]
            q_sol = self.kin.ik(current_q, T, frame=self.frame, max_iters=10, weights6=self.ik_weights)
            q_sol[4] = self.rad_offset - q_sol[4]
            q_sol = np.degrees(q_sol)
            
            # Clip q_sol values to prevent erratic behaviour
            if self.prev_q is not None:
                q_sol = np.clip(q_sol, self.prev_q - self.angles_thresh, self.prev_q + self.angles_thresh)

            self.prev_q = q_sol
            q_sol = np.append(q_sol, 90 - controls["slider"] * 90)
            action_dict = self.move._create_action_dict(q_sol)
            self.move.robot.send_action(action_dict)
            # return action_dict
        else:
            self.controller_zero = remap_convert(position)
            self.start_position = np.array(self.move.compute_ee_pos(frame=self.frame))

            self.prev_q = None


def main():
    def get_local_ip():
        """Get the local IP address of this machine"""
        try:
            # Connect to a remote address to determine the local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
            return local_ip
        except Exception:
            return "127.0.0.1"  # fallback to localhost

    def remap_convert(unity_position):
        return np.array([-unity_position[0] / 100, -unity_position[2] / 100, unity_position[1] / 100])

    def save_positions_to_file(positions, filename):
        with open(filename, 'w') as f:
            json.dump(positions, f)

    def read_positions_from_file(filename):
        with open(filename, 'r') as f:
            return json.load(f)

    receiver = Receiver(udp_port=5005)

    # Print IP address and UDP port information
    local_ip = get_local_ip()
    print(f"UDP Server listening on IP: {local_ip}")
    print(f"UDP Port: {receiver.udp_port}")

    robot_config = SO100FollowerConfig(
        port="/dev/ttyACM0",
        id="toni",
    )
    robot = SO100Follower(robot_config)
    robot.connect()

    # --- Robot model setup ---
    urdf_name = "so100"
    urdf_path = os.path.join(os.path.dirname(__file__), 'data', 'urdf', f'{urdf_name}.urdf')
    urdf_path = os.path.abspath(urdf_path)
    kin = RobotKinematics(urdf_path)
    viz = RobotVisualisation(kin, urdf_name, trajectory_viz=True)
    move = MoveRobot(kin, robot=robot, visualization=viz, use_sim_time=False)
    # Remove z-axis rotation constrain since the robot has 5 DoF
    ik_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0])
    frame = viz.link_names[5]  # 6th link is end-effector

    move.home()

    controller_zero, _, _ = receiver.receive()
    print(controller_zero)
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
    angles_thresh = 3
    prev_q = None
    deg_offset = 45
    rad_offset = np.deg2rad(deg_offset)

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

                target_pos = start_position + (position - controller_zero) * 0.6
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
                current_q = move.get_q_guess()
                current_q[4] = rad_offset - current_q[4]
                q_sol = kin.ik(current_q, T, frame=frame, max_iters=10, weights6=ik_weights)
                q_sol[4] = rad_offset - q_sol[4]
                q_sol = np.degrees(q_sol)
                
                # Clip q_sol values to prevent erratic behaviour
                if prev_q is not None:
                    q_sol = np.clip(q_sol, prev_q - angles_thresh, prev_q + angles_thresh)

                prev_q = q_sol
                q_sol = np.append(q_sol, 90 - controls["slider"] * 90)
                action_dict = move._create_action_dict(q_sol)
                move.robot.send_action(action_dict)
                # return action_dict
            else:
                controller_zero = remap_convert(position)
                start_position = np.array(move.compute_ee_pos(frame=frame))

                prev_q = None

    except KeyboardInterrupt:
        save_positions_to_file(positions_log, output_file)
        move.home()
        robot.disconnect()
        receiver.disconnect()


if __name__ == "__main__":
    main()
