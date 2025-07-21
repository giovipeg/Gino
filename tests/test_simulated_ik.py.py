import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from src.gino.kinematics.kinematics import RobotKinematics
from src.gino.kinematics.robot_visualization import RobotVisualisation
#from src.gino.kinematics.tracker import HandTracker

if __name__ == "__main__":
    urdf_name = "so100"
    urdf_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'urdf', f'{urdf_name}.urdf')
    urdf_path = os.path.abspath(urdf_path)
    kin = RobotKinematics(urdf_path)
    viz = RobotVisualisation(kin, urdf_name)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    q = np.zeros(6)

    trajectory_points = []
    end_effector_name = viz.link_names[5]  # Assuming 6th link is end-effector

    for t in range(200):
        # Virtual joint movement to calculate trajectory
        q_joints = 45 * np.sin(np.linspace(0, np.pi * 2, 5) + 0.1 * t)
        
        q_deg = np.zeros(6)
        q_deg[:5] = q_joints
        q_deg[5] = 45 # Keep gripper open
        q_rad = np.radians(q_deg)

        ee_pos = kin.fk(q_rad[:5], end_effector_name)[:3, 3]
        trajectory_points.append(ee_pos)

        # Update robot visualization
        viz.draw(ax, q_deg)
        
        # Re-plot the trajectory since viz.draw() clears the axes
        traj_array = np.array(trajectory_points)
        ax.plot(traj_array[:, 0], traj_array[:, 1], traj_array[:, 2], 'b-', label='End-Effector Trajectory')
        ax.legend()
        
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([0, 0.6])

        plt.pause(0.01)

    plt.show() # Keep the window open after the loop
