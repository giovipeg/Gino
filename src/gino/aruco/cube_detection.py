import cv2
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

class ArucoCubeTracker:
    def __init__(self, calib_file='data/camera_calib.npz', 
                 cube_size=0.04, marker_size=0.03):

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

    def pose_estimation(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        corners, ids, _ = self.detector.detectMarkers(gray)
        
        rvec = None
        tvec = None
        rotation_matrix_plot = None
        
        if ids is not None:
            detected_ids = ids.flatten()
            cube_markers = [int(id) for id in detected_ids if int(id) in self.cube_marker_ids]
            
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
                rvec_aligned, _ = cv2.Rodrigues(R_aligned)

                cube_center = tvec - R_aligned @ offset
                centers.append(cube_center)
                rotations.append(R_aligned)

            if not centers:
                return None, None, None
            
            # Average the centers
            avg_center = np.mean(centers, axis=0)
            # Use the first marker's rotation for now
            R = np.mean(rotations, axis=0)
            # For visualization, transform coordinates
            tvec_plot = np.array([-avg_center[0], -avg_center[2], -avg_center[1]])
            rotation_matrix_plot = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]) @ R @ np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]).T

            # Returns roll, pitch, yaw with input R
            euler_angles = self.euler_from_matrix(R)
            print(f"Euler angles: {euler_angles}")

            # Draw all detected markers
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            # Add text overlay with detection info
            y_offset = 30
            for i, marker_id in enumerate(ids.flatten()):
                color = (0, 255, 0) if int(marker_id) in self.cube_marker_ids else (255, 255, 255)
                cv2.putText(image, f"ID: {marker_id}", (10, y_offset + i*20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw cube axes on image
            cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, 
                            R_aligned, cube_center, self.cube_size)

            # Apply moving average filter
            if len(self.cube_positions) > 0:
                smoothed_position = self._moving_average_filter(
                    self.cube_positions + [tvec_plot], self.window_size)
            else:
                smoothed_position = tvec_plot
            
            self.cube_positions.append(smoothed_position)
            self.cube_orientations.append(rotation_matrix_plot)

            return smoothed_position, rotation_matrix_plot, cube_markers

        else:
            return None, None, None
