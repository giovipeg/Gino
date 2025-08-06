import cv2
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from src.gino.mcu.mcu import MCU

class ArucoCubeTracker:
    def __init__(self, calib_file='data/camera_calib.npz', 
                 cube_size=0.04, marker_size=0.03):
        self.prev_gt_euler = None
        self.prev_euler = None

        # Camera calibration parameters
        calib = np.load(calib_file)
        self.camera_matrix = calib['camera_matrix']
        self.dist_coeffs = calib['dist_coeffs']

        # Cube configuration
        self.cube_size = cube_size
        self.marker_size = marker_size
        self.cube_marker_ids = [0, 1, 2, 3, 4, 5]

        # Define 3D positions of markers on cube faces (in cube's local coordinate system -> as seen from camera)
        # Assuming markers are centered on each face
        self.cube_marker_positions = {
            2: np.array([self.cube_size / 2, 0, 0]),   # Right face (+X)
            4: np.array([-self.cube_size / 2, 0, 0]),  # Left face (-X)
            3: np.array([0, self.cube_size / 2, 0]),   # Top face (+Y)
            1: np.array([0, -self.cube_size / 2, 0]),  # Bottom face (-Y)
            0: np.array([0, 0, self.cube_size / 2]),   # Front face (+Z)
            5: np.array([0, 0, -self.cube_size / 2])   # Back face (-Z)
        }

        # Define rotation transformations for each marker to align with cube coordinate system
        # Cube's coordinate system is the same as marker 0's
        # Each transformation aligns the marker's Z axis with the outward normal of its face
        # Columns are the destination axes (x, y, z order)
        # Rows are the source axes (x, y, z order)
        # Multiply an axis by -1 to flip it
        self.marker_rotations = {
            0: np.eye(3),  # Front face: no rotation needed (Z already points outward)
            5: np.array([[-1, 0, 0],  # Back face: 180° around Y
                        [0, 1, 0],
                        [0, 0, -1]]),
            2: np.array([[0, -1, 0],    # x is cube's y flipped
                        [0, 0, -1],      # y is cube's z flipped
                        [1, 0, 0]]),   # z is cube's x
            4: np.array([[0, -1, 0],    # x is cube's y
                        [0, 0, 1],      # y is cube's z flipped
                        [-1, 0, 0]]),    # z is cube's x flipped
            3: np.array([[-1, 0, 0],    # x is cube's x flipped
                        [0, 0, 1],      # y is cube's z flipped
                        [0, 1, 0]]),   # z is cube's y flipped
            1: np.array([[1, 0, 0],    # x is cube's x
                        [0, 0, 1],    # y is cube's z flipped
                        [0, -1, 0]])    # z is cube's y flipped
        }

        # Minimal rest orientation compensation: identity by default
        self.marker_rest_rotations = {marker_id: np.eye(3) for marker_id in self.cube_marker_ids}

        # 180° rotation about Y
        self.R_flip = np.array([[-1, 0,  0],
                                [ 0, 1,  0],
                                [ 0, 0, -1]])


        # ArUco detector setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
        # Trajectory and smoothing attributes
        self.cube_positions = []
        self.cube_orientations = []
        self.window_size = 5
    
    @staticmethod
    def _moving_average_filter(data, window_size=5):
        if len(data) < window_size:
            return data[-1] if data else np.array([0, 0, 0])
        arr = np.array(data[-window_size:])
        return np.mean(arr, axis=0)

    def euler_from_matrix(self, R_matrix):
        # Convert rotation matrix to Euler angles (ZYX convention)
        # This gives us: [yaw, pitch, roll] in degrees
        roll, yaw, _ = R.from_matrix(R_matrix).as_euler("ZYX", degrees=True)
        
        # 180° rotation about Y to get pitch base value at 0 deg
        # Orientation that a rear-facing camera would see
        R_rear_view = self.R_flip @ R_matrix

        # Convert to ZYX Euler angles (yaw, pitch, roll)
        _, _, pitch = R.from_matrix(R_rear_view).as_euler("ZYX", degrees=True)
        
        return roll, pitch, yaw
    
    def return_avg_pose(self, cube_markers, corners, detected_ids):
        # Estimate pose for each marker, compute cube center from each, average centers
        centers = []
        rotations = []
        for marker_id in cube_markers:
            idx = np.where(detected_ids == marker_id)[0][0]
            marker_corners_single = corners[idx]
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                marker_corners_single, self.marker_size, self.camera_matrix, self.dist_coeffs
            )
            rvec = rvecs[0][0]
            tvec = tvecs[0][0]
            offset = self.cube_marker_positions[marker_id]
            R, _ = cv2.Rodrigues(rvec)

            # Apply marker-specific rotation to align coordinate systems
            R_aligned = R @ self.marker_rotations[marker_id]

            # --- Rest orientation compensation ---
            R_rest = self.marker_rest_rotations[marker_id]
            R_relative = R_aligned @ np.linalg.inv(R_rest)
            # -------------------------------------

            rvec_aligned, _ = cv2.Rodrigues(R_relative)

            # Returns roll, pitch, yaw with input R
            euler_angles = self.euler_from_matrix(R_relative)

            # print(f"R_relative: {euler_angles}")

            cube_center = tvec - R_aligned @ offset
            centers.append(cube_center)
            rotations.append(R_relative)

        # Average the centers and rotations
        avg_center = np.mean(centers, axis=0)
        
        # Properly average rotation matrices using SVD orthogonalization
        if len(rotations) == 1:
            R = rotations[0]
        else:
            # Average the rotation matrices
            R_avg = np.mean(rotations, axis=0)
            # Orthogonalize using SVD to ensure it's a valid rotation matrix
            U, _, Vt = np.linalg.svd(R_avg)
            R = U @ Vt
            # Ensure right-handed coordinate system
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = U @ Vt

        return avg_center, R
    
    def apply_transformation(self, avg_center, R):
        # For visualization, transform coordinates
        tvec_plot = np.array([-avg_center[0], -avg_center[2], -avg_center[1]])
        rotation_matrix_plot = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]) @ R @ np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]).T

        return tvec_plot, rotation_matrix_plot

    def apply_moving_average(self, tvec, rotation_matrix):
        # Apply moving average filter
        if len(self.cube_positions) > 0:
            smoothed_position = self._moving_average_filter(
                self.cube_positions + [tvec], self.window_size)
        else:
            smoothed_position = tvec
        
        self.cube_positions.append(smoothed_position)
        self.cube_orientations.append(rotation_matrix)

        return smoothed_position, rotation_matrix

    def pose_estimation(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None:
            return None, None, None, None

        # Draw all detected markers
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        # Add text overlay with detection info
        y_offset = 30
        for i, marker_id in enumerate(ids.flatten()):
            color = (0, 255, 0) if int(marker_id) in self.cube_marker_ids else (255, 255, 255)
            cv2.putText(image, f"ID: {marker_id}", (10, y_offset + i*20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        detected_ids = ids.flatten()
        cube_markers = [int(id) for id in detected_ids if int(id) in self.cube_marker_ids]

        if not cube_markers:
            return None, None, None, None

        avg_center, R = self.return_avg_pose(cube_markers, corners, detected_ids)

        # Draw cube axes on image
        cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, 
                        R, avg_center, self.cube_size)

        # Returns roll, pitch, yaw with input R
        euler_angles = self.euler_from_matrix(R)

        tvec_plot, rotation_matrix_plot = self.apply_transformation(avg_center, R)

        smoothed_position, rotation_matrix_plot = self.apply_moving_average(tvec_plot, rotation_matrix_plot)

        return smoothed_position, rotation_matrix_plot, euler_angles, cube_markers

    def outlier_detection(self, gt_euler, euler, threshold=10):
        """
        Detect outliers in Euler angle measurements.
        
        Args:
            gt_euler: Ground truth Euler angles as tuple (roll, pitch, yaw) or array [roll, pitch, yaw] or None
            euler: Measured Euler angles as tuple (roll, pitch, yaw) or array [roll, pitch, yaw]
            threshold: Threshold for outlier detection (degrees)
            
        Returns:
            bool: True if outlier detected, False otherwise
        """
        # Handle case where ground truth data is not available yet
        if euler is None:
            if self.prev_euler is None:
                self.prev_euler = euler
            return False
        
        if self.prev_gt_euler is None or self.prev_euler is None:
            self.prev_gt_euler = gt_euler
            self.prev_euler = euler
            return False
        
        # Convert to numpy arrays if they aren't already
        # Handle both tuples and arrays
        gt_euler = np.array(gt_euler)
        euler = np.array(euler)
        
        delta_gt = gt_euler - self.prev_gt_euler
        delta_euler = euler - self.prev_euler

        self.prev_gt_euler = gt_euler.copy()
        self.prev_euler = euler.copy()

        # Check if any component exceeds the threshold
        component_outlier = np.any(np.abs(delta_euler) > np.abs(delta_gt) + threshold)
        
        # Use component-wise check as it's more sensitive to individual axis changes
        if component_outlier:
            print(f"Outlier detected: delta_euler={delta_euler}, delta_gt={delta_gt}, threshold={threshold}")
            return True
        else:
            return False
