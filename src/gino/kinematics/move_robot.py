import os
import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from lerobot.robots.so101_follower import SO101Follower

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from kinematics import RobotKinematics
from robot_visualization import RobotVisualisation

# TO-DO: add cross robot support
class MoveRobot:
    def __init__(self, kinematics: RobotKinematics, robot: SO101Follower = None, visualization: RobotVisualisation = None, use_sim_time: bool = True):
        self.kin = kinematics
        self.robot = robot
        self.viz = visualization
        self.use_sim_time = use_sim_time
        
        # Initialize motor names mapping for action dictionary
        self.motor_names = []
        if not self.use_sim_time and self.robot is not None:
            obs = self.robot.get_observation()
            self.motor_names = [key.removesuffix('.pos') for key in obs.keys() if key.endswith('.pos')]
        
        self.current_q = self.get_q_guess()

        self.home_dict = {
            'shoulder_pan': (self.robot.bus.calibration['shoulder_pan'].range_max + self.robot.bus.calibration['shoulder_pan'].range_min) / 2,
            'shoulder_lift': self.robot.bus.calibration['shoulder_lift'].range_min,
            'elbow_flex': self.robot.bus.calibration['elbow_flex'].range_max,
            'wrist_flex': (self.robot.bus.calibration['wrist_flex'].range_max + self.robot.bus.calibration['wrist_flex'].range_min) / 2,
            'wrist_roll': (self.robot.bus.calibration['wrist_roll'].range_max + self.robot.bus.calibration['wrist_roll'].range_min) / 2,
            'gripper': self.robot.bus.calibration['gripper'].range_min
        }

    def get_q_guess(self):
        if not self.use_sim_time:
            # Get observation from real robot and extract joint positions
            obs = self.robot.get_observation()
            # Extract joint positions from observation dictionary
            # Assuming joint positions are stored with keys like "motor_name.pos"
            joint_positions = []
            for key, value in obs.items():
                if key.endswith('.pos'):
                    joint_positions.append(value)
            # Convert to radians if needed (assuming the values are in degrees)
            joint_positions = np.array(joint_positions)
            # Convert from degrees to radians
            #joint_positions = np.radians(joint_positions)
            
            # We only need the first 5 joints for the arm kinematics
            joint_positions = joint_positions[:self.kin.model.nq]

            return np.radians(joint_positions)
        else:
            try:
                return self.current_q.copy()
            except:
                # If no previous q is set, return zero guess
                # The initial joint configuration shall be set from the caller
                return np.zeros(self.kin.model.nq)

    def _get_current_end_effector_orientation(self, frame=None):
        """
        Retrieve the current end effector orientation as Euler angles (ZYX, degrees).
        If frame is None, use the default frame from kinematics.
        """
        q = self.get_q_guess()
        pose = self.kin.fk(q, frame)
        rot_matrix = pose[:3, :3]
        euler_angles = R.from_matrix(rot_matrix).as_euler("ZYX", degrees=True)
        return euler_angles

    def get_ik_solution(self, rot, pos, frame):
        rot_matrix = R.from_euler("ZYX", rot, degrees=True).as_matrix()

        # Build target SE(3) pose
        T = np.eye(4)
        T[:3, :3] = rot_matrix
        T[:3, 3] = pos

        q_guess = self.get_q_guess()

        # Solve IK for joint angles (in radians)
        q_sol = self.kin.ik(q_guess, T, frame=frame, max_iters=10)
        self.current_q = q_sol.copy()  # Update current joint state with best effort
        return q_sol
    
    def _create_action_dict(self, q_sol):
        # Create action dictionary with joint positions using pre-initialized motor names
        action_dict = {}
        for i, motor_name in enumerate(self.motor_names[:len(q_sol)]):
            action_dict[f"{motor_name}.pos"] = q_sol[i]
        return action_dict

    def move_to_target(self, target_pos, frame, num_steps):
        # Compute start point using forward kinematics
        q_guess = self.get_q_guess()
        
        start_pose = self.kin.fk(q_guess, frame)
        start_point = start_pose[:3, 3]

        # Generate linear trajectory from start to target
        traj_points = np.linspace(start_point, target_pos, num_steps)

        # TO-DO: use rot value at start of the movement
        for t, pos in enumerate(traj_points):
            q_sol = self.get_ik_solution(self._get_current_end_effector_orientation(frame), pos, frame)

            if not self.use_sim_time:
                q_sol = np.degrees(q_sol)
                action_dict = self._create_action_dict(q_sol)
                self.robot.send_action(action_dict)
            else:
                # Draw robot
                self.viz.draw(np.concatenate([q_sol, [np.radians(45)]]))

                # Draw desired and actual trajectory so far
                traj_array = traj_points[:t+1]
                if len(traj_array) > 1:
                    self.viz.ax.plot(traj_array[:, 0], traj_array[:, 1], traj_array[:, 2], 'g--', label='Desired Trajectory' if t == 0 else "")
                if t == 0:
                    self.viz.ax.legend()

                self.viz.ax.set_xlim([0, 0.4])
                self.viz.ax.set_ylim([-0.25, 0.25])
                self.viz.ax.set_zlim([0, 0.3])
                self.viz.ax.set_title("Robot Arm Moving End-Effector from Start to Target (IK)")
                plt.pause(0.02)

        if self.use_sim_time:
            plt.show()

    def home(self, num_steps=300):
        """
        Move robot to home1 position smoothly over specified number of steps.
        
        Args:
            num_steps (int): Number of steps to transition from current to home position
        """
        if self.use_sim_time or self.robot is None:
            print("Warning: home1 called in simulation mode or with no robot")
            return
        
        # Get current motor positions using the highlighted read method
        current_positions = {}
        for motor_name in self.home_dict.keys():
            current_positions[motor_name] = self.robot.bus.read("Present_Position", motor_name, normalize=False)
        
        # Create smooth trajectory for each motor
        for step in range(num_steps):
            # Calculate interpolation factor (0 to 1)
            alpha = step / (num_steps - 1) if num_steps > 1 else 1.0
            
            # Interpolate between current and target positions
            for motor_name in self.home_dict.keys():
                current_pos = current_positions[motor_name]
                target_pos = self.home_dict[motor_name]
                interpolated_pos = current_pos + alpha * (target_pos - current_pos)
                
                # Send interpolated position to robot using the highlighted write method
                self.robot.bus.write("Goal_Position", motor_name, int(interpolated_pos), normalize=False)
            
            # Small delay to allow robot to process the command
            time.sleep(0.01)  # 100ms delay between steps
