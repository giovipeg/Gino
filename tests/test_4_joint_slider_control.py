import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.gino.kinematics.kinematics import RobotKinematics
from src.gino.kinematics.robot_visualization import RobotVisualisation

# --- Robot model setup ---
urdf_name = "so100"
urdf_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'urdf', f'{urdf_name}.urdf')
urdf_path = os.path.abspath(urdf_path)
kin = RobotKinematics(urdf_path)
viz = RobotVisualisation(kin, urdf_name)

# --- Visualization setup ---
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.25, bottom=0.35)

# --- Joint slider setup ---
joint_names = [f"Joint {i+1}" for i in range(5)] + ["Gripper"]
joint_limits = [(-180, 180)] * 5 + [(0, 90)]  # degrees for all joints, gripper 0-90
init_vals = [0] * 5 + [45]  # initial values in degrees

sliders = []
slider_axes = []

for i, (name, lim, val) in enumerate(zip(joint_names, joint_limits, init_vals)):
    ax_slider = plt.axes([0.25, 0.25 - i*0.04, 0.65, 0.03])
    slider = Slider(ax_slider, name, lim[0], lim[1], valinit=val)
    sliders.append(slider)
    slider_axes.append(ax_slider)

# --- Update function ---
def update(val=None):
    q_vis = np.array([slider.val for slider in sliders])
    viz.draw(ax, q_vis)
    ax.set_title("Interactive Robot Joint Control (Sliders)")
    plt.draw()

# Initial draw
update()

# Connect sliders to update function
for slider in sliders:
    slider.on_changed(update)

plt.show() 