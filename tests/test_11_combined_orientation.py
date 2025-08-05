import cv2
import numpy as np
import time
import subprocess
import os
import sys
import matplotlib.pyplot as plt
import math
from collections import deque

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.gino.aruco.visualization_utils as viz
from src.gino.aruco.cube_detection import ArucoCubeTracker
from src.gino.mcu.mcu import MCU

class LowPassFilter:
    def __init__(self, alpha=0.1):
        """
        Simple low-pass filter
        alpha: filter coefficient (0-1), lower = more smoothing
        """
        self.alpha = alpha
        self.filtered_roll = 0.0
        self.filtered_pitch = 0.0
        self.filtered_yaw = 0.0
        self.first_measurement = True
        
    def update(self, roll, pitch, yaw):
        """
        Update filter with new measurements
        Returns: filtered_roll, filtered_pitch, filtered_yaw
        """
        if self.first_measurement:
            self.filtered_roll = roll
            self.filtered_pitch = pitch
            self.filtered_yaw = yaw
            self.first_measurement = False
            return self.filtered_roll, self.filtered_pitch, self.filtered_yaw
        
        # Simple low-pass filter: y[n] = α * x[n] + (1-α) * y[n-1]
        self.filtered_roll = self.alpha * roll + (1.0 - self.alpha) * self.filtered_roll
        self.filtered_pitch = self.alpha * pitch + (1.0 - self.alpha) * self.filtered_pitch
        self.filtered_yaw = self.alpha * yaw + (1.0 - self.alpha) * self.filtered_yaw
        
        return self.filtered_roll, self.filtered_pitch, self.filtered_yaw

class ComplementaryFilter:
    def __init__(self, alpha=0.98, dt=0.033):
        """
        Initialize complementary filter
        alpha: complementary filter coefficient (0-1)
               Higher alpha = more weight to gyroscope (high frequency)
               Lower alpha = more weight to ArUco (low frequency)
        dt: time step in seconds
        """
        self.alpha = alpha
        self.dt = dt
        
        # Initialize filtered angles
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        
        # Previous gyroscope readings for integration
        self.prev_gyro_roll = 0.0
        self.prev_gyro_pitch = 0.0
        self.prev_gyro_yaw = 0.0
        
        # First measurement flag
        self.first_measurement = True
        
    def update(self, gyro_roll, gyro_pitch, gyro_yaw, aruco_roll, aruco_pitch, aruco_yaw):
        """
        Update filter with new gyroscope and ArUco measurements
        gyro_*: raw angular velocities in degrees/second
        aruco_*: absolute angles in degrees
        Returns: filtered_roll, filtered_pitch, filtered_yaw
        """
        if self.first_measurement:
            # Initialize with ArUco measurement if available, otherwise use gyro
            if aruco_roll is not None and aruco_pitch is not None and aruco_yaw is not None:
                self.roll = aruco_roll
                self.pitch = aruco_pitch
                self.yaw = aruco_yaw
            else:
                self.roll = 0.0  # Initialize to zero if no ArUco available
                self.pitch = 0.0
                self.yaw = 0.0
            
            self.prev_gyro_roll = gyro_roll
            self.prev_gyro_pitch = gyro_pitch
            self.prev_gyro_yaw = gyro_yaw
            self.first_measurement = False
            return self.roll, self.pitch, self.yaw
        
        # Calculate gyroscope angle changes (proper integration of angular velocity)
        gyro_roll_change = gyro_roll * self.dt  # angular_velocity * time = angle_change
        gyro_pitch_change = gyro_pitch * self.dt
        gyro_yaw_change = gyro_yaw * self.dt
        
        # Update previous gyroscope readings
        self.prev_gyro_roll = gyro_roll
        self.prev_gyro_pitch = gyro_pitch
        self.prev_gyro_yaw = gyro_yaw
        
        # Apply complementary filter
        if aruco_roll is not None and aruco_pitch is not None and aruco_yaw is not None:
            # ArUco measurement available - use complementary filter
            self.roll = self.alpha * (self.roll + gyro_roll_change) + (1 - self.alpha) * aruco_roll
            self.pitch = self.alpha * (self.pitch + gyro_pitch_change) + (1 - self.alpha) * aruco_pitch
            self.yaw = self.alpha * (self.yaw + gyro_yaw_change) + (1 - self.alpha) * aruco_yaw
        else:
            # No ArUco measurement - use only gyroscope integration
            self.roll += gyro_roll_change
            self.pitch += gyro_pitch_change
            self.yaw += gyro_yaw_change
        
        return self.roll, self.pitch, self.yaw

def draw_arrow(ax, roll, pitch, yaw, title_prefix=""):
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

def print_sensor_values(frame_count, gyro_roll, gyro_pitch, gyro_yaw, aruco_roll, aruco_pitch, aruco_yaw, filtered_roll, filtered_pitch, filtered_yaw):
    """Print sensor values in a stationary format with updating numbers"""
    # Clear screen and move cursor to top (works on most terminals)
    print("\033[2J\033[H", end="")
    
    print("=" * 60)
    print("           REAL-TIME ORIENTATION SENSOR FUSION (WITH LPF)")
    print("=" * 60)
    print(f"Frame: {frame_count}")
    print("-" * 60)
    print("SENSOR          ROLL (°)    PITCH (°)    YAW (°)")
    print("-" * 60)
    
    # Gyroscope values
    print(f"Gyroscope       {gyro_roll:8.1f}    {gyro_pitch:8.1f}    {gyro_yaw:8.1f}")
    
    # ArUco values
    if aruco_roll is not None:
        print(f"ArUco           {aruco_roll:8.1f}    {aruco_pitch:8.1f}    {aruco_yaw:8.1f}")
    else:
        print(f"ArUco           {'N/A':>8}    {'N/A':>8}    {'N/A':>8}")
    
    # Filtered values
    print(f"Filtered (LPF)  {filtered_roll:8.1f}    {filtered_pitch:8.1f}    {filtered_yaw:8.1f}")
    print("-" * 60)
    print("Press 'q' or ESC to quit")
    print("=" * 60)

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
    # Initialize components
    aruco = ArucoCubeTracker()
    mcu = MCU('/dev/ttyACM0', 500000)
    complementary_filter = ComplementaryFilter(alpha=0.90, dt=0.010)  # 90% gyro, 10% ArUco
    
    # Initialize simple low-pass filter for noise reduction
    lpf = LowPassFilter(alpha=0.35)  # 10% new data, 90% previous (strong smoothing)
    
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
    
    # Initialize orientation variables
    gyro_roll = gyro_pitch = gyro_yaw = 0
    aruco_roll = aruco_pitch = aruco_yaw = None
    filtered_roll = filtered_pitch = filtered_yaw = 0
    
    try:
        while key != ord('q') and key != 27:
            ret, image = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Get gyroscope data
            try:
                gyro_roll, gyro_pitch, gyro_yaw = mcu.get_last_gyro_data()
            except:
                print("Warning: Could not read gyroscope data")
                gyro_roll = gyro_pitch = gyro_yaw = 0
            
            # Get ArUco cube pose estimation
            smoothed_position, rotation_matrix_plot, euler_angles, cube_markers = aruco.pose_estimation(image)
            
            if smoothed_position is not None and rotation_matrix_plot is not None and euler_angles is not None:
                # Convert rotation matrix to Euler angles
                aruco_roll, aruco_pitch, aruco_yaw = euler_angles
            else:
                # No ArUco detection
                aruco_roll = aruco_pitch = aruco_yaw = None
            
            # Apply complementary filter to combine gyro and ArUco data
            raw_filtered_roll, raw_filtered_pitch, raw_filtered_yaw = complementary_filter.update(
                gyro_roll, gyro_pitch, gyro_yaw, 
                aruco_roll, aruco_pitch, aruco_yaw
            )
            
            # Apply simple low-pass filter to reduce noise
            filtered_roll, filtered_pitch, filtered_yaw = lpf.update(
                raw_filtered_roll, raw_filtered_pitch, raw_filtered_yaw
            )

            # Print sensor values in stationary format
            print_sensor_values(frame_count, gyro_roll, gyro_pitch, gyro_yaw, 
                              aruco_roll, aruco_pitch, aruco_yaw,
                              filtered_roll, filtered_pitch, filtered_yaw)
            
            # Draw visualizations
            draw_arrow(ax_arrow, filtered_roll, filtered_pitch, filtered_yaw, "Final: ")
            draw_horizon(ax_horizon, filtered_roll, filtered_pitch, "Final: ")
            
            # Display the image
            cv2.imshow('Combined Orientation Tracking', image)
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