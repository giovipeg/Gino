#!/usr/bin/env python3
"""
Simple example demonstrating the OrientationFusion class with simulated data.

This example shows how to use the OrientationFusion class without requiring
hardware (camera and MCU). It uses simulated sensor data to demonstrate
the sensor fusion capabilities.
"""

import numpy as np
import time
import os
import sys
import matplotlib.pyplot as plt
import math

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gino.kinematics.orientation_fusion import OrientationFusion


def generate_simulated_data(t, noise_level=0.1):
    """
    Generate simulated sensor data.
    
    Args:
        t: Time in seconds
        noise_level: Amount of noise to add
        
    Returns:
        Tuple of (gyro_data, aruco_data) where each is (roll, pitch, yaw)
    """
    # Simulate a rotating motion
    frequency = 0.5  # Hz
    amplitude = 30   # degrees
    
    # True orientation (what we're trying to estimate)
    true_roll = amplitude * np.sin(2 * np.pi * frequency * t)
    true_pitch = amplitude * 0.5 * np.cos(2 * np.pi * frequency * t)
    true_yaw = amplitude * 0.3 * np.sin(2 * np.pi * frequency * 0.3 * t)
    
    # Gyroscope data (angular velocities with some drift)
    gyro_roll = 2 * np.pi * frequency * amplitude * np.cos(2 * np.pi * frequency * t) + np.random.normal(0, noise_level)
    gyro_pitch = -2 * np.pi * frequency * amplitude * 0.5 * np.sin(2 * np.pi * frequency * t) + np.random.normal(0, noise_level)
    gyro_yaw = 2 * np.pi * frequency * 0.3 * amplitude * 0.3 * np.cos(2 * np.pi * frequency * 0.3 * t) + np.random.normal(0, noise_level)
    
    # Add some drift to gyroscope
    gyro_roll += 0.1 * t  # Gradual drift
    gyro_pitch += 0.05 * t
    gyro_yaw += 0.02 * t
    
    # ArUco data (absolute angles with noise, sometimes missing)
    if np.random.random() > 0.1:  # 90% detection rate
        aruco_roll = true_roll + np.random.normal(0, noise_level * 2)
        aruco_pitch = true_pitch + np.random.normal(0, noise_level * 2)
        aruco_yaw = true_yaw + np.random.normal(0, noise_level * 2)
    else:
        aruco_roll = aruco_pitch = aruco_yaw = None
    
    return (gyro_roll, gyro_pitch, gyro_yaw), (aruco_roll, aruco_pitch, aruco_yaw)


def plot_results(times, true_data, gyro_data, aruco_data, filtered_data):
    """Plot the results showing all sensor data and filtered output."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Orientation Fusion Results', fontsize=16)
    
    # Plot roll
    axes[0].plot(times, true_data[:, 0], 'g-', label='True', linewidth=2)
    axes[0].plot(times, gyro_data[:, 0], 'r--', label='Gyroscope', alpha=0.7)
    axes[0].plot(times, filtered_data[:, 0], 'b-', label='Filtered', linewidth=2)
    
    # Plot ArUco data (only when available)
    aruco_times = [t for t, (r, p, y) in zip(times, aruco_data) if r is not None]
    aruco_rolls = [r for r, p, y in aruco_data if r is not None]
    if aruco_times:
        axes[0].scatter(aruco_times, aruco_rolls, color='orange', label='ArUco', alpha=0.7, s=20)
    
    axes[0].set_ylabel('Roll (°)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot pitch
    axes[1].plot(times, true_data[:, 1], 'g-', label='True', linewidth=2)
    axes[1].plot(times, gyro_data[:, 1], 'r--', label='Gyroscope', alpha=0.7)
    axes[1].plot(times, filtered_data[:, 1], 'b-', label='Filtered', linewidth=2)
    
    aruco_pitches = [p for r, p, y in aruco_data if p is not None]
    if aruco_times:
        axes[1].scatter(aruco_times, aruco_pitches, color='orange', label='ArUco', alpha=0.7, s=20)
    
    axes[1].set_ylabel('Pitch (°)')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot yaw
    axes[2].plot(times, true_data[:, 2], 'g-', label='True', linewidth=2)
    axes[2].plot(times, gyro_data[:, 2], 'r--', label='Gyroscope', alpha=0.7)
    axes[2].plot(times, filtered_data[:, 2], 'b-', label='Filtered', linewidth=2)
    
    aruco_yaws = [y for r, p, y in aruco_data if y is not None]
    if aruco_times:
        axes[2].scatter(aruco_times, aruco_yaws, color='orange', label='ArUco', alpha=0.7, s=20)
    
    axes[2].set_ylabel('Yaw (°)')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()


def main():
    """Main function demonstrating the OrientationFusion class with simulated data."""
    print("Orientation Fusion Example with Simulated Data")
    print("=" * 50)
    
    # Initialize the orientation fusion system
    orientation_fusion = OrientationFusion(
        complementary_alpha=0.95,  # 95% gyro, 5% ArUco (high gyro weight for smooth tracking)
        lpf_alpha=0.3,            # Strong smoothing for noise reduction
        dt=0.01                   # 10ms time step
    )
    
    # Simulation parameters
    duration = 10.0  # seconds
    dt = 0.01        # time step
    times = np.arange(0, duration, dt)
    
    # Storage for plotting
    true_data = []
    gyro_data = []
    aruco_data = []
    filtered_data = []
    
    print(f"Running simulation for {duration} seconds...")
    print("Press Ctrl+C to stop early")
    
    try:
        for i, t in enumerate(times):
            # Generate simulated sensor data
            gyro, aruco = generate_simulated_data(t)
            
            # Update the fusion system
            orientation_fusion.update_gyro_data(*gyro)
            orientation_fusion.update_aruco_data(*aruco)
            
            # Process the sensor fusion
            filtered = orientation_fusion.process()
            
            # Store data for plotting
            true_roll = 30 * np.sin(2 * np.pi * 0.5 * t)
            true_pitch = 30 * 0.5 * np.cos(2 * np.pi * 0.5 * t)
            true_yaw = 30 * 0.3 * np.sin(2 * np.pi * 0.5 * 0.3 * t)
            
            true_data.append([true_roll, true_pitch, true_yaw])
            gyro_data.append(gyro)
            aruco_data.append(aruco)
            filtered_data.append(filtered)
            
            # Print progress every second
            if i % 100 == 0:
                state = orientation_fusion.get_current_state()
                print(f"Time: {t:.1f}s - Filtered: R={filtered[0]:.1f}°, P={filtered[1]:.1f}°, Y={filtered[2]:.1f}°")
            
            # Small delay to simulate real-time processing
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    
    # Convert to numpy arrays
    true_data = np.array(true_data)
    gyro_data = np.array(gyro_data)
    filtered_data = np.array(filtered_data)
    
    print("\nSimulation completed!")
    print("Plotting results...")
    
    # Plot the results
    plot_results(times, true_data, gyro_data, aruco_data, filtered_data)
    
    # Print some statistics
    print("\nStatistics:")
    print(f"Total samples: {len(times)}")
    print(f"ArUco detection rate: {sum(1 for a in aruco_data if a[0] is not None) / len(aruco_data) * 100:.1f}%")
    
    # Calculate RMSE for filtered vs true
    rmse_roll = np.sqrt(np.mean((filtered_data[:, 0] - true_data[:, 0])**2))
    rmse_pitch = np.sqrt(np.mean((filtered_data[:, 1] - true_data[:, 1])**2))
    rmse_yaw = np.sqrt(np.mean((filtered_data[:, 2] - true_data[:, 2])**2))
    
    print(f"RMSE - Roll: {rmse_roll:.2f}°, Pitch: {rmse_pitch:.2f}°, Yaw: {rmse_yaw:.2f}°")


if __name__ == "__main__":
    main() 