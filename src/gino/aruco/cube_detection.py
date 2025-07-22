import cv2
import numpy as np
import time
import subprocess
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import visualization_utils as viz

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

        # ArUco detector setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
    @staticmethod
    def _moving_average_filter(data, window_size=5):
        if len(data) < window_size:
            return data[-1] if data else np.array([0, 0, 0])
        arr = np.array(data[-window_size:])
        return np.mean(arr, axis=0)
    
    def _estimate_cube_pose(self, detected_markers, marker_corners, camera_matrix, dist_coeffs):
        """
        Estimate cube pose from multiple detected markers using PnP algorithm
        """
        if len(detected_markers) < 3:
            return None, None
        
        # Collect 3D-2D correspondences
        object_points = []
        image_points = []
        
        for i, marker_id in enumerate(detected_markers):
            if marker_id in self.cube_marker_positions:
                # Get 3D position of marker center on cube
                marker_3d_pos = self.cube_marker_positions[marker_id]
                object_points.append(marker_3d_pos)
                
                # Get 2D position of marker center in image
                corners = marker_corners[i]
                center_2d = np.mean(corners[0], axis=0)
                image_points.append(center_2d)
        
        if len(object_points) < 3:
            return None, None
        
        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)
        
        # Solve PnP to get cube pose
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_matrix, dist_coeffs
        )
        
        if success:
            return rvec, tvec
        return None, None

    def pose_estimation(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        parameters = cv2.aruco.DetectorParameters()
        
        # Detect markers
        corners, ids, _ = self.detector.detectMarkers(gray)
        
        if ids is not None:
            detected_ids = ids.flatten()
            cube_markers = [int(id) for id in detected_ids if int(id) in self.cube_marker_ids]
            
            print(f"Frame {frame_count}: Detected cube markers: {cube_markers}")
            
            if len(cube_markers) == 1:
                # Single marker pose estimation
                idx = np.where(detected_ids == cube_markers[0])[0][0]
                marker_corners_single = corners[idx]
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    marker_corners_single, self.marker_size, self.camera_matrix, self.dist_coeffs
                )
                rvec = rvecs[0][0]
                tvec = tvecs[0][0]
                marker_id = cube_markers[0]
                offset = self.cube_marker_positions[marker_id]
                R, _ = cv2.Rodrigues(rvec)
                cube_center = tvec + R @ offset
                # For visualization, transform coordinates
                tvec_plot = np.array([-cube_center[0], -cube_center[2], -cube_center[1]])
                rotation_matrix_plot = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]) @ R @ np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]).T

            elif len(cube_markers) == 2:
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
                    cube_center = tvec + R @ offset
                    centers.append(cube_center)
                    rotations.append(R)
                # Average the centers
                avg_center = np.mean(centers, axis=0)
                # Use the first marker's rotation for now
                R = rotations[0]
                # For visualization, transform coordinates
                tvec_plot = np.array([-avg_center[0], -avg_center[2], -avg_center[1]])
                rotation_matrix_plot = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]) @ R @ np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]).T

            elif len(cube_markers) >= 3:
                # Get indices of cube markers
                cube_indices = [np.where(detected_ids == id)[0][0] for id in cube_markers]
                cube_corners = [corners[i] for i in cube_indices]
                # Estimate cube pose
                rvec, tvec = None, None
                if len(cube_markers) == 3:
                    # Use SQPNP for 3 points
                    object_points = []
                    image_points = []
                    for i, marker_id in enumerate(cube_markers):
                        if marker_id in self.cube_marker_positions:
                            marker_3d_pos = self.cube_marker_positions[marker_id]
                            object_points.append(marker_3d_pos)
                            corners2d = cube_corners[i]
                            center_2d = np.mean(corners2d[0], axis=0)
                            image_points.append(center_2d)
                    object_points = np.array(object_points, dtype=np.float32)
                    image_points = np.array(image_points, dtype=np.float32)
                    if len(object_points) == 3:
                        success, rvec, tvec = cv2.solvePnP(
                            object_points, image_points, self.camera_matrix, self.dist_coeffs,
                            flags=cv2.SOLVEPNP_SQPNP
                        )
                        if not success:
                            rvec, tvec = None, None
                else:
                    # Use default for 4+ points
                    rvec, tvec = self._estimate_cube_pose(cube_markers, cube_corners, 
                                                self.camera_matrix, self.dist_coeffs)
                if rvec is not None and tvec is not None:
                    tvec_plot = np.array([-tvec[0][0], -tvec[2][0], -tvec[1][0]])
                    rotation_matrix, _ = cv2.Rodrigues(rvec)
                    transform_matrix = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
                    rotation_matrix_plot = transform_matrix @ rotation_matrix @ transform_matrix.T

            # Draw all detected markers
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            # Add text overlay with detection info
            y_offset = 30
            for i, marker_id in enumerate(ids.flatten()):
                color = (0, 255, 0) if int(marker_id) in self.cube_marker_ids else (255, 255, 255)
                cv2.putText(image, f"ID: {marker_id}", (10, y_offset + i*20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if rvec is not None and tvec is not None:
                # Draw cube axes on image
                cv2.drawFrameAxes(image, self.camera_matrix, self.dist_coeffs, 
                                rvec, tvec, self.cube_size)

                # Apply moving average filter
                if len(cube_positions) > 0:
                    smoothed_position = self._moving_average_filter(
                        cube_positions + [tvec_plot], window_size)
                else:
                    smoothed_position = tvec_plot

                return smoothed_position, rotation_matrix_plot, cube_markers
            else:
                return None, rotation_matrix_plot, cube_markers

        else:
            return None, None, None

if __name__ == "__main__":
    # Visualization toggle
    VISUALIZE = False

    # 3D plot setup
    if VISUALIZE:
        fig_pos, ax_pos, fig_pose, ax_pose = viz.setup_visualization(True, True)
    else:
        fig_pos = ax_pos = fig_pose = ax_pose = None

    aruco = ArucoCubeTracker()

    cube_positions = []
    cube_orientations = []
    window_size = 5

    # Main script logic for video file
    start_script_time = time.time()
    cap = cv2.VideoCapture('data/vid2.avi')

    frame_count = 0
    key = None
    while key != ord('q') and key != 27:
        start_time = time.time()

        ret, image = cap.read()
        if not ret:
            break
        
        frame_count += 1

        smoothed_position, rotation_matrix_plot, cube_markers = aruco.pose_estimation(image)
        
        if smoothed_position is not None:
                cube_positions.append(smoothed_position)
                cube_orientations.append(rotation_matrix_plot)

                # Update plots
                if VISUALIZE:
                    viz.update_visualization(
                        True, True,
                        ax_pos, ax_pose,
                        cube_positions, smoothed_position,
                        rotation_matrix_plot, cube_markers,
                        aruco.cube_marker_positions, aruco.cube_size
                    )
        cv2.imshow('ArUco Cube Tracking', image)
        key = cv2.waitKey(1) & 0xFF

        print(f"fps: {1 / (time.time() - start_time)}")

    cap.release()
    cv2.destroyAllWindows()
    plt.ioff()
    if VISUALIZE:
        viz.close_visualization()

    total_exec_time = time.time() - start_script_time
    print(f"Tracking completed. Total frames processed: {frame_count}")
    print(f"Cube positions recorded: {len(cube_positions)}")
    print(f"Total execution time: {total_exec_time:.2f} seconds")

    # Save trajectory to file
    np.savez('data/cube_trajectory1.npz', positions=np.array(cube_positions), orientations=np.array(cube_orientations))
    print("Cube trajectory saved to 'data/cube_trajectory1.npz'")