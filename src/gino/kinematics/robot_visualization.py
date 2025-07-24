import os
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

class RobotVisualisation:
    def __init__(self, kinematics, urdf_name: str, trajectory_viz: bool = True):
        self.kinematics = kinematics
        self.urdf_path = self._resolve_urdf_path(urdf_name)
        self.link_names, self.link_pairs = self._parse_urdf(self.urdf_path)
        self._inset_ax = None
        self._arc_line = None
        self._angle_line = None
        self._angle_text = None

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.trajectory_viz = trajectory_viz
        self.end_effector_name = self.link_names[5]
        self.actual_ee_points = []

    # Untouched
    def _resolve_urdf_path(self, name):
        # Use the directory of this file to construct the URDF path
        if not name.endswith(".urdf"):
            name += ".urdf"

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        data_urdf = os.path.join(project_root, 'data', 'urdf', name)
        if os.path.exists(data_urdf):
            return data_urdf
        raise FileNotFoundError(f"URDF file not found: {name} in {data_urdf}")

    # Untouched
    def _parse_urdf(self, path):
        root = ET.parse(path).getroot()
        links = [l.attrib['name'] for l in root.findall('link')]
        joints = [(j.find('parent').attrib['link'], j.find('child').attrib['link'])
                  for j in root.findall('joint')]
        return links, joints
    
    def draw_trajectory(self, q_sol):
        # Compute actual end-effector position from FK
        ee_actual = self.kinematics.fk(q_sol[:5], self.end_effector_name)[:3, 3]
        self.actual_ee_points.append(ee_actual)

        actual_array = np.array(self.actual_ee_points)
        self.ax.plot(actual_array[:, 0], actual_array[:, 1], actual_array[:, 2], 'b-', label='Actual Trajectory')

    # Untouched
    def draw(self, q):
        """
        Visualize the robot in 3D using joint configuration.

        Expects a 6-element joint array in radians:
        [j1, j2, j3, j4, j5, gripper_open]
        """
        poses = {name: self.kinematics.fk(q[:5], name)[:3, 3] for name in self.link_names[:6]}

        self.ax.clear()
        self.ax.set_xlim([-0.3, 0.3])
        self.ax.set_ylim([-0.3, 0.3])
        self.ax.set_zlim([0, 0.6])
        for parent, child in self.link_pairs:
            if parent in poses and child in poses:
                self.ax.plot(*zip(poses[parent], poses[child]), c='k')
        for pos in poses.values():
            self.ax.scatter(*pos, s=30, c='r')

        # Gripper angle inset
        fig = self.ax.get_figure()
        if self._inset_ax is None:
            self._inset_ax = fig.add_axes((0.78, 0.78, 0.2, 0.2))
            self._inset_ax.set_aspect('equal')
            self._inset_ax.axis('off')
            self._inset_ax.add_artist(plt.Circle((0, 0), 1, fill=False))
            self._inset_ax.plot([0, 1], [0, 0], color='gray', linestyle=':')
            self._arc_line, = self._inset_ax.plot([], [], ls='--')
            self._angle_line, = self._inset_ax.plot([], [], lw=2)
            self._angle_text = self._inset_ax.text(0, -1.4, "", ha='center', fontsize=9)
            self._inset_ax.set_xlim(-1.1, 1.1)
            self._inset_ax.set_ylim(-1.3, 1.1)

        arc = np.linspace(0, q[-1], 64)
        self._arc_line.set_data(np.cos(arc), np.sin(arc))
        self._angle_line.set_data([0, np.cos(q[-1])], [0, np.sin(q[-1])])
        self._angle_text.set_text(f"{np.degrees(q[-1]):.1f}°")

        if self.trajectory_viz:
            self.draw_trajectory(q)
        
