import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.gino.kinematics.kinematics import RobotKinematics
from src.gino.kinematics.robot_visualization import RobotVisualisation
from src.gino.kinematics.move_robot import MoveRobot

# --- Robot model setup ---
urdf_name = "so100"
urdf_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'urdf', f'{urdf_name}.urdf')
urdf_path = os.path.abspath(urdf_path)
kin = RobotKinematics(urdf_path)
viz = RobotVisualisation(kin, urdf_name, trajectory_viz=True)
move = MoveRobot(kin, robot=None, visualization=viz, use_sim_time=True)

viz.draw(np.array([0.0, 0.0, 0.0, 0.0, 0.0]))

# --- Create a separate window with 6 sliders ---
fig_sliders, ax_sliders = plt.subplots(figsize=(6, 4))
plt.subplots_adjust(left=0.25, bottom=0.4)
ax_sliders.set_axis_off()

slider_axes = []
sliders = []
slider_values = [0] * 6  # Initialize all joint values to 0

def update_joint(val, joint_idx):
    """Callback function to update joint values when slider changes"""
    slider_values[joint_idx] = val
    print(f"Joint {joint_idx + 1}: {val:.3f}")
    # Update robot visualization with new joint values
    viz.draw(np.array(slider_values))
    # Use canvas.draw() instead of plt.draw() to avoid recursion issues
    viz.ax.figure.canvas.draw_idle()

for i in range(6):
    ax_slider = plt.axes([0.25, 0.35 - i*0.05, 0.65, 0.03])
    slider = Slider(ax_slider, f'Joint {i+1}', -np.pi, np.pi, valinit=0)
    slider.on_changed(lambda val, idx=i: update_joint(val, idx))  # Connect callback
    sliders.append(slider)
    slider_axes.append(ax_slider)

plt.show()
