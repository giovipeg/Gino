import cv2
import numpy as np
import time
import subprocess
import os
import sys
import matplotlib.pyplot as plt
import math

# Toggle for matplotlib visualization
ENABLE_VISUALIZATION = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.gino.aruco.visualization_utils as viz
from src.gino.aruco.cube_detection import ArucoCubeTracker

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

def draw_cube_position(ax, position):
    ax.clear()
    if position is not None:
        x, y, z = position
        # Create a 2D scatter plot (X vs Y)
        ax.scatter([x], [y], c='red', s=100, marker='o')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Cube Position: ({x:.3f}, {y:.3f}, {z:.3f})')
        
        # Set reasonable limits
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        
        # Add grid
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, 'No cube detected', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Cube Position: Not detected')

def rotation_matrix_to_euler(rotation_matrix):
    """Convert rotation matrix to Euler angles (roll, pitch, yaw)"""
    if rotation_matrix is None:
        return 0, 0, 0
    
    # Extract Euler angles from rotation matrix
    # Assuming rotation matrix is 3x3
    sy = math.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
    
    singular = sy < 1e-6
    
    if not singular:
        roll = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = math.atan2(-rotation_matrix[2, 0], sy)
        yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = math.atan2(-rotation_matrix[2, 0], sy)
        yaw = 0
    
    # Convert to degrees
    roll = math.degrees(roll)
    pitch = math.degrees(pitch)
    yaw = math.degrees(yaw)
    
    return roll, pitch, yaw

def main():
    # Initialize ArUco tracker
    aruco = ArucoCubeTracker()
    
    if ENABLE_VISUALIZATION:
        # Setup matplotlib for real-time plotting
        plt.ion()
        fig, (ax_arrow, ax_horizon, ax_position) = plt.subplots(3, 1, figsize=(8, 12))
        plt.subplots_adjust(hspace=0.4)
    else:
        ax_arrow = ax_horizon = ax_position = None
    
    # Open video file
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video file 'data/vid2.avi'")
        return
    
    frame_count = 0
    key = None
    
    try:
        while key != ord('q') and key != 27:
            ret, image = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Get cube pose estimation
            smoothed_position, rotation_matrix_plot, euler_angles, cube_markers = aruco.pose_estimation(image)
            
            if smoothed_position is not None and rotation_matrix_plot is not None:
                # Convert rotation matrix to Euler angles
                roll, pitch, yaw = euler_angles
                
                # Draw orientation visualizations
                if ENABLE_VISUALIZATION:
                    draw_arrow(ax_arrow, roll, pitch, yaw)
                    draw_horizon(ax_horizon, roll, pitch)
                    draw_cube_position(ax_position, smoothed_position)
                print(f"Frame: {frame_count}, Roll: {roll:.1f}°, Pitch: {pitch:.1f}°, Yaw: {yaw:.1f}°")
            else:
                print(f"Frame: {frame_count}, No cube detected")
            
            # Update matplotlib plots
            if ENABLE_VISUALIZATION:
                plt.pause(0.001)
            
            # Display the image
            cv2.imshow('ArUco Cube Tracking', image)
            key = cv2.waitKey(0) & 0xFF  # 30ms delay for ~33 FPS
            
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        if ENABLE_VISUALIZATION:
            plt.ioff()
            plt.close('all')
        print(f"Tracking completed. Total frames processed: {frame_count}")

if __name__ == "__main__":
    main() 