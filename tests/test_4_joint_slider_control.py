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
viz = RobotVisualisation(kin, urdf_name)
move = MoveRobot(kin)

# --- Plotting setup ---
# Adjust layout to accommodate both robot visualization and joint states
plt.subplots_adjust(left=0.25, bottom=0.35, right=0.75)

# Joint configuration
joint_names = [f"Joint {i+1}" for i in range(5)] + ["Gripper"]
joint_limits = [(-np.pi, np.pi)] * 5 + [(0, np.pi/2)]  # degrees for all joints, gripper 0-90
init_vals = [0] * 5 + [np.pi/4]  # initial values in degrees

# Create sliders
sliders = []
slider_axes = []
for i, (name, lim, val) in enumerate(zip(joint_names, joint_limits, init_vals)):
    ax_slider = plt.axes([0.25, 0.25 - i*0.04, 0.45, 0.03])
    slider = Slider(ax_slider, name, lim[0], lim[1], valinit=val)
    sliders.append(slider)
    slider_axes.append(ax_slider)

# Create joint states display
joint_states_ax = plt.axes([0.75, 0.25, 0.20, 0.35])
joint_states_ax.set_xlim(0, 1)
joint_states_ax.set_ylim(0, 1)
joint_states_ax.axis('off')

joint_states_text = joint_states_ax.text(0.05, 0.95, "Joint States:", 
                                        fontsize=10, fontweight='bold',
                                        transform=joint_states_ax.transAxes,
                                        verticalalignment='top')

# --- Update function ---
def update(val=None):
    q_vis = np.array([slider.val for slider in sliders])
    viz.draw(q_vis)
    viz.ax.set_title("Interactive Robot Joint Control (Sliders)")
    
    # Update joint states display
    states_text = "Joint States:\n\n"
    q_so101 = move.convert_to_so101_joint_angles(q_vis)
    for i, (name, val) in enumerate(zip(joint_names, q_so101)):
        states_text += f"{name}: {val:.2f}°\n\n"
    
    joint_states_text.set_text(states_text)
    plt.draw()

# --- Execution ---
# Initial draw
update()

# Connect sliders to update function
for slider in sliders:
    slider.on_changed(update)

plt.show()
