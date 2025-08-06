import numpy as np
import math
from typing import Optional, Tuple


class LowPassFilter:
    """Simple low-pass filter for noise reduction in orientation data."""
    
    def __init__(self, alpha: float = 0.1):
        """
        Initialize low-pass filter.
        
        Args:
            alpha: Filter coefficient (0-1), lower = more smoothing
        """
        self.alpha = alpha
        self.filtered_roll = 0.0
        self.filtered_pitch = 0.0
        self.filtered_yaw = 0.0
        self.first_measurement = True
        
    def update(self, roll: float, pitch: float, yaw: float) -> Tuple[float, float, float]:
        """
        Update filter with new measurements.
        
        Args:
            roll: Roll angle in degrees
            pitch: Pitch angle in degrees
            yaw: Yaw angle in degrees
            
        Returns:
            Tuple of (filtered_roll, filtered_pitch, filtered_yaw)
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
    """Complementary filter for fusing gyroscope and ArUco orientation data."""
    
    def __init__(self, alpha: float = 0.98, dt: float = 0.033):
        """
        Initialize complementary filter.
        
        Args:
            alpha: Complementary filter coefficient (0-1)
                   Higher alpha = more weight to gyroscope (high frequency)
                   Lower alpha = more weight to ArUco (low frequency)
            dt: Time step in seconds
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
        
    def update(self, gyro_roll: float, gyro_pitch: float, gyro_yaw: float,
               aruco_roll: Optional[float], aruco_pitch: Optional[float], 
               aruco_yaw: Optional[float]) -> Tuple[float, float, float]:
        """
        Update filter with new gyroscope and ArUco measurements.
        
        Args:
            gyro_roll: Raw angular velocity in degrees/second
            gyro_pitch: Raw angular velocity in degrees/second
            gyro_yaw: Raw angular velocity in degrees/second
            aruco_roll: Absolute angle in degrees (can be None)
            aruco_pitch: Absolute angle in degrees (can be None)
            aruco_yaw: Absolute angle in degrees (can be None)
            
        Returns:
            Tuple of (filtered_roll, filtered_pitch, filtered_yaw)
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


class OrientationFusion:
    """Main class for combining gyroscope and ArUco orientation data using sensor fusion."""
    
    def __init__(self, complementary_alpha: float = 0.90, lpf_alpha: float = 0.35, dt: float = 0.010):
        """
        Initialize the orientation fusion system.
        
        Args:
            complementary_alpha: Complementary filter coefficient (0-1)
            lpf_alpha: Low-pass filter coefficient (0-1)
            dt: Time step in seconds
        """
        self.complementary_filter = ComplementaryFilter(alpha=complementary_alpha, dt=dt)
        self.lpf = LowPassFilter(alpha=lpf_alpha)
        
        # Current orientation state
        self.gyro_roll = 0.0
        self.gyro_pitch = 0.0
        self.gyro_yaw = 0.0
        self.aruco_roll = None
        self.aruco_pitch = None
        self.aruco_yaw = None
        self.filtered_roll = 0.0
        self.filtered_pitch = 0.0
        self.filtered_yaw = 0.0
        
    def update_gyro_data(self, roll: float, pitch: float, yaw: float):
        """
        Update gyroscope data.
        
        Args:
            roll: Angular velocity in degrees/second
            pitch: Angular velocity in degrees/second
            yaw: Angular velocity in degrees/second
        """
        self.gyro_roll = roll
        self.gyro_pitch = pitch
        self.gyro_yaw = yaw
        
    def update_aruco_data(self, roll: Optional[float], pitch: Optional[float], yaw: Optional[float]):
        """
        Update ArUco orientation data.
        
        Args:
            roll: Absolute angle in degrees (can be None if no detection)
            pitch: Absolute angle in degrees (can be None if no detection)
            yaw: Absolute angle in degrees (can be None if no detection)
        """
        self.aruco_roll = roll
        self.aruco_pitch = pitch
        self.aruco_yaw = yaw
        
    def process(self) -> Tuple[float, float, float]:
        """
        Process the sensor fusion and return filtered orientation.
        
        Returns:
            Tuple of (filtered_roll, filtered_pitch, filtered_yaw) in degrees
        """
        # Apply complementary filter to combine gyro and ArUco data
        raw_filtered_roll, raw_filtered_pitch, raw_filtered_yaw = self.complementary_filter.update(
            self.gyro_roll, self.gyro_pitch, self.gyro_yaw,
            self.aruco_roll, self.aruco_pitch, self.aruco_yaw
        )
        
        # Apply simple low-pass filter to reduce noise
        self.filtered_roll, self.filtered_pitch, self.filtered_yaw = self.lpf.update(
            raw_filtered_roll, raw_filtered_pitch, raw_filtered_yaw
        )
        
        return self.filtered_roll, self.filtered_pitch, self.filtered_yaw
    
    def get_current_state(self) -> dict:
        """
        Get the current state of all sensors and filtered data.
        
        Returns:
            Dictionary containing all current orientation values
        """
        return {
            'gyro': {
                'roll': self.gyro_roll,
                'pitch': self.gyro_pitch,
                'yaw': self.gyro_yaw
            },
            'aruco': {
                'roll': self.aruco_roll,
                'pitch': self.aruco_pitch,
                'yaw': self.aruco_yaw
            },
            'filtered': {
                'roll': self.filtered_roll,
                'pitch': self.filtered_pitch,
                'yaw': self.filtered_yaw
            }
        }
    
    @staticmethod
    def rotation_matrix_to_euler(rotation_matrix: Optional[np.ndarray]) -> Tuple[float, float, float]:
        """
        Convert rotation matrix to Euler angles (roll, pitch, yaw).
        
        Args:
            rotation_matrix: 3x3 rotation matrix (can be None)
            
        Returns:
            Tuple of (roll, pitch, yaw) in degrees
        """
        if rotation_matrix is None:
            return 0.0, 0.0, 0.0
        
        # Extract Euler angles from rotation matrix
        # Assuming rotation matrix is 3x3
        sy = math.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + 
                      rotation_matrix[1, 0] * rotation_matrix[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            roll = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            pitch = math.atan2(-rotation_matrix[2, 0], sy)
            yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            roll = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            pitch = math.atan2(-rotation_matrix[2, 0], sy)
            yaw = 0.0
        
        # Convert to degrees
        roll = math.degrees(roll)
        pitch = math.degrees(pitch)
        yaw = math.degrees(yaw)
        
        return roll, pitch, yaw 