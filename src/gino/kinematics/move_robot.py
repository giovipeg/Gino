import os
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from kinematics import RobotKinematics
from robot_visualization import RobotVisualisation

class MoveRobot:
    def __init__(self, kinematics: RobotKinematics, visualization: RobotVisualisation, use_sim_time: bool = True):
        self.kin = kinematics
        self.viz = visualization
        self.use_sim_time = use_sim_time
        self.current_q = self._get_q_guess()

    def _get_q_guess(self):
        if not self.use_sim_time:
            # TO-DO: implement real robot q guess
            return
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
        q = self._get_q_guess()
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

        q_guess = self._get_q_guess()

        # Solve IK for joint angles (in radians)
        q_sol = self.kin.ik(q_guess, T, frame=frame, max_iters=10)
        self.current_q = q_sol.copy()  # Update current joint state with best effort
        return q_sol

    def move_to_target(self, target_pos, frame, num_steps):
        # Compute start point using forward kinematics
        start_pose = self.kin.fk(self._get_q_guess(), frame)
        start_point = start_pose[:3, 3]

        # Generate linear trajectory from start to target
        traj_points = np.linspace(start_point, target_pos, num_steps)

        # TO-DO: use rot value at start of the movement
        for t, pos in enumerate(traj_points):
            q_sol = self.get_ik_solution(self._get_current_end_effector_orientation(frame), pos, frame)

            if not self.use_sim_time:
                return
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
