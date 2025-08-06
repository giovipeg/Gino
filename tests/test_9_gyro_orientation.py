import math
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.gino.mcu.mcu import MCU

def draw_arrow(ax, roll, pitch, yaw):
    # Arrow parameters
    length = 0.8
    width = 0.2
    # Arrow base shape (triangle)
    base = [
        [0, 0],
        [length, -width/2],
        [length, width/2],
        [0, 0]
    ]
    # Convert to numpy array
    base = np.array(base)
    # Apply roll and pitch as color (or thickness)
    # For now, use color: more red for roll, more blue for pitch
    roll_norm = min(max((roll+90)/180, 0), 1)  # normalize to 0-1
    pitch_norm = min(max((pitch+90)/180, 0), 1)
    color = (roll_norm, 0.2, pitch_norm)
    # Rotate by yaw
    theta = math.radians(yaw)
    rot = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)]
    ])
    arrow = base @ rot.T
    # Draw
    ax.clear()
    ax.fill(arrow[:,0], arrow[:,1], color=color, alpha=0.8)
    ax.plot([0, length*math.cos(theta)], [0, length*math.sin(theta)], color='k', lw=2)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.set_title(f"Yaw: {yaw:.1f}°, Roll: {roll:.1f}°, Pitch: {pitch:.1f}°")
    ax.grid(True)
    ax.set_xticks([])
    ax.set_yticks([])
    # Draw a circle for reference
    circle = plt.Circle((0,0), 1, color='gray', fill=False, linestyle='--', alpha=0.3)
    ax.add_patch(circle)

def draw_horizon(ax, roll, pitch):
    ax.clear()
    # Artificial horizon: blue for sky, brown for ground
    # The horizon line tilts with roll and shifts up/down with pitch
    # Roll: rotation, Pitch: vertical shift
    width = 2
    height = 1.2
    # Calculate horizon line
    roll_rad = math.radians(roll)
    pitch_shift = pitch / 90 * (height/2)  # max pitch = 90deg = half window
    # Draw sky
    sky = plt.Rectangle((-width/2, 0), width, height/2 + pitch_shift, color='#87ceeb', zorder=0)
    ax.add_patch(sky)
    # Draw ground
    ground = plt.Rectangle((-width/2, -height/2 + pitch_shift), width, height/2 - pitch_shift, color='#c2b280', zorder=0)
    ax.add_patch(ground)
    # Draw horizon line
    x = np.linspace(-width/2, width/2, 100)
    y = np.tan(-roll_rad) * x + pitch_shift
    ax.plot(x, y, color='k', lw=3)
    # Draw center marker
    ax.plot(0, 0, marker='+', color='k', markersize=12, mew=2)
    ax.set_xlim(-width/2, width/2)
    ax.set_ylim(-height/2, height/2)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Artificial Horizon (Roll: {roll:.1f}°, Pitch: {pitch:.1f}°)")
    ax.grid(False)

def draw_acceleration(ax, lin_ax, lin_ay, lin_az):
    ax.clear()
    # Plot acceleration values as bars
    axes = ['X', 'Y', 'Z']
    values = [lin_ax, lin_ay, lin_az]
    colors = ['red', 'green', 'blue']
    
    # Calculate dynamic range with padding
    max_val = max(abs(v) for v in values)
    if max_val == 0:
        max_val = 1.0  # Default range if all values are zero
    
    # Add 20% padding
    y_range = max_val * 1.2
    y_lim = (-y_range, y_range)
    
    # Create bars
    bars = ax.bar(axes, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (0.05 * y_range if height >= 0 else -0.05 * y_range),
            f'{value:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax.set_ylim(y_lim)
    ax.set_ylabel('Acceleration (m/s²)')
    ax.set_title('Linear Acceleration (Gravity Compensated)')
    ax.grid(True, alpha=0.3)
    
    # Add some styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def main():
    import matplotlib.pyplot as plt
    mcu = MCU('/dev/ttyACM0', 500000)

    # Connect to MCU
    if mcu.connect():
        print("MCU connected successfully!")
        
        # Wait for Arduino to be ready
        print("Waiting for Arduino to send first data...")
        while not mcu.is_arduino_ready():
            time.sleep(0.1)
        print("Arduino is ready!")
    else:
        print("Failed to connect to MCU!")

    plt.ion()
    fig, (ax_arrow, ax_horizon, ax_accel) = plt.subplots(3, 1, figsize=(6,12))
    plt.subplots_adjust(hspace=0.4)
    try:
        while True:
            roll, pitch, yaw = mcu.get_last_pose_data()
            lin_ax, lin_ay, lin_az = mcu.get_last_imu_data()
            
            # Draw arrow: roll, pitch, yaw
            draw_arrow(ax_arrow, roll, pitch, yaw)
            # Draw artificial horizon
            draw_horizon(ax_horizon, roll, pitch)
            # Draw acceleration
            draw_acceleration(ax_accel, lin_ax, lin_ay, lin_az)
            
            plt.pause(0.001)
    except KeyboardInterrupt:
        mcu.disconnect()

if __name__ == "__main__":
    main()
