#!/usr/bin/env python3
"""
Example demonstrating the OrientationFusion class with real-time visualization.

This example shows how to use the OrientationFusion class to combine
gyroscope and ArUco marker data for robust orientation tracking.
"""

import cv2
import numpy as np
import time
import subprocess
import os
import sys
import matplotlib.pyplot as plt
import math
from collections import deque

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import src.gino.aruco.visualization_utils as viz
from src.gino.aruco.cube_detection import ArucoCubeTracker
from src.gino.mcu.mcu import MCU
from src.gino.kinematics.orientation_fusion import OrientationFusion


def draw_arrow(ax, roll, pitch, yaw, title_prefix=""):
    """Draw an arrow visualization showing orientation."""
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
    ax.set_title(f"{title_prefix}Yaw: {yaw:.1f}°, Roll: {roll:.1f}°, Pitch: {pitch:.1f}°")
    ax.grid(True)
    ax.set_xticks([])
    ax.set_yticks([])
    # Draw a circle for reference
    circle = plt.Circle((0,0), 1, color='gray', fill=False, linestyle='--', alpha=0.3)
    ax.add_patch(circle)


def draw_horizon(ax, roll, pitch, title_prefix=""):
    """Draw an artificial horizon visualization."""
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
    ax.set_title(f"{title_prefix}Artificial Horizon (Roll: {roll:.1f}°, Pitch: {pitch:.1f}°)")
    ax.grid(False)


def print_sensor_values(frame_count, state):
    """Print sensor values in a stationary format with updating numbers."""
    # Clear screen and move cursor to top (works on most terminals)
    print("\033[2J\033[H", end="")
    
    print("=" * 60)
    print("           REAL-TIME ORIENTATION SENSOR FUSION")
    print("=" * 60)
    print(f"Frame: {frame_count}")
    print("-" * 60)
    print("SENSOR          ROLL (°)    PITCH (°)    YAW (°)")
    print("-" * 60)
    
    # Gyroscope values
    gyro = state['gyro']
    print(f"Gyroscope       {gyro['roll']:8.1f}    {gyro['pitch']:8.1f}    {gyro['yaw']:8.1f}")
    
    # ArUco values
    aruco = state['aruco']
    if aruco['roll'] is not None:
        print(f"ArUco           {aruco['roll']:8.1f}    {aruco['pitch']:8.1f}    {aruco['yaw']:8.1f}")
    else:
        print(f"ArUco           {'N/A':>8}    {'N/A':>8}    {'N/A':>8}")
    
    # Filtered values
    filtered = state['filtered']
    print(f"Filtered        {filtered['roll']:8.1f}    {filtered['pitch']:8.1f}    {filtered['yaw']:8.1f}")
    print("-" * 60)
    print("Press 'q' or ESC to quit")
    print("=" * 60)


def main():
    """Main function demonstrating the OrientationFusion class."""
    print("Initializing Orientation Fusion Example...")
    
    # Initialize components
    aruco = ArucoCubeTracker()
    mcu = MCU('/dev/ttyACM0', 500000)
    
    # Initialize the orientation fusion system
    # Parameters can be tuned based on your specific needs:
    # - complementary_alpha: Higher = more weight to gyroscope (good for fast movements)
    # - lpf_alpha: Lower = more smoothing (good for reducing noise)
    # - dt: Time step (should match your actual update rate)
    orientation_fusion = OrientationFusion(
        complementary_alpha=0.90,  # 90% gyro, 10% ArUco
        lpf_alpha=0.35,           # Strong smoothing for noise reduction
        dt=0.010                  # 10ms time step
    )
    
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
        return
    
    # Setup matplotlib for real-time plotting
    plt.ion()
    fig, (ax_arrow, ax_horizon) = plt.subplots(2, 1, figsize=(8, 12))
    plt.subplots_adjust(hspace=0.4)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera!")
        return
    
    frame_count = 0
    key = None
    
    print("Starting real-time orientation tracking...")
    print("Press 'q' or ESC to quit")
    
    try:
        while key != ord('q') and key != 27:
            ret, image = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Get gyroscope data
            try:
                gyro_roll, gyro_pitch, gyro_yaw = mcu.get_last_gyro_data()
                orientation_fusion.update_gyro_data(gyro_roll, gyro_pitch, gyro_yaw)
            except Exception as e:
                print(f"Warning: Could not read gyroscope data: {e}")
                orientation_fusion.update_gyro_data(0, 0, 0)
            
            # Get ArUco cube pose estimation
            smoothed_position, rotation_matrix_plot, euler_angles, cube_markers = aruco.pose_estimation(image)
            
            if smoothed_position is not None and rotation_matrix_plot is not None and euler_angles is not None:
                # Convert rotation matrix to Euler angles and update ArUco data
                aruco_roll, aruco_pitch, aruco_yaw = euler_angles
                orientation_fusion.update_aruco_data(aruco_roll, aruco_pitch, aruco_yaw)
            else:
                # No ArUco detection
                orientation_fusion.update_aruco_data(None, None, None)
            
            # Process the sensor fusion
            filtered_roll, filtered_pitch, filtered_yaw = orientation_fusion.process()
            
            # Get current state for display
            current_state = orientation_fusion.get_current_state()

            # Print sensor values in stationary format
            print_sensor_values(frame_count, current_state)
            
            # Draw visualizations
            draw_arrow(ax_arrow, filtered_roll, filtered_pitch, filtered_yaw, "Final: ")
            draw_horizon(ax_horizon, filtered_roll, filtered_pitch, "Final: ")
            
            # Display the image
            cv2.imshow('Orientation Fusion Example', image)
            key = cv2.waitKey(30) & 0xFF  # 30ms delay for ~33 FPS
            
            # Update matplotlib plots
            plt.pause(0.001)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        mcu.disconnect()
        plt.ioff()
        plt.close('all')
        print(f"Tracking completed. Total frames processed: {frame_count}")


if __name__ == "__main__":
    main() 