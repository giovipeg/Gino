import cv2
import numpy as np
import time
import subprocess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Visualization toggle
VISUALIZE = False

# Camera calibration parameters
calib = np.load('data/camera_calib.npz')
camera_matrix = calib['camera_matrix']
dist_coeffs = calib['dist_coeffs']

# Cube configuration
CUBE_SIZE = 0.04  # meters - side length of the cube
MARKER_SIZE = 0.03  # meters - size of individual markers
CUBE_MARKER_IDS = [0, 1, 2, 3, 4, 5]  # IDs for markers on each face

# Define 3D positions of markers on cube faces (in cube's local coordinate system -> as seen from camera)
# Assuming markers are centered on each face
cube_marker_positions = {
    2: np.array([CUBE_SIZE/2, 0, 0]),      # Right face (+X)
    4: np.array([-CUBE_SIZE/2, 0, 0]),     # Left face (-X)
    3: np.array([0, CUBE_SIZE/2, 0]),      # Top face (+Y)
    1: np.array([0, -CUBE_SIZE/2, 0]),     # Bottom face (-Y)
    0: np.array([0, 0, CUBE_SIZE/2]),      # Front face (+Z)
    5: np.array([0, 0, -CUBE_SIZE/2])      # Back face (-Z)
}

# 3D plot setup
if VISUALIZE:
    plt.ion()
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Cube Center Position')

    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title('Cube Orientation & Individual Markers')

cube_positions = []
cube_orientations = []
window_size = 5

def moving_average_filter(data, window_size=5):
    if len(data) < window_size:
        return data[-1] if data else np.array([0, 0, 0])
    arr = np.array(data[-window_size:])
    return np.mean(arr, axis=0)

def estimate_cube_pose(detected_markers, marker_corners, camera_matrix, dist_coeffs):
    """
    Estimate cube pose from multiple detected markers using PnP algorithm
    """
    if len(detected_markers) < 3:
        return None, None
    
    # Collect 3D-2D correspondences
    object_points = []
    image_points = []
    
    for i, marker_id in enumerate(detected_markers):
        if marker_id in cube_marker_positions:
            # Get 3D position of marker center on cube
            marker_3d_pos = cube_marker_positions[marker_id]
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

def draw_cube_wireframe(ax, center, rotation_matrix, size=CUBE_SIZE):
    """
    Draw a wireframe cube at the given center with rotation
    """
    # Define cube vertices relative to center
    vertices = np.array([
        [-size/2, -size/2, -size/2],
        [size/2, -size/2, -size/2],
        [size/2, size/2, -size/2],
        [-size/2, size/2, -size/2],
        [-size/2, -size/2, size/2],
        [size/2, -size/2, size/2],
        [size/2, size/2, size/2],
        [-size/2, size/2, size/2]
    ])
    
    # Rotate and translate vertices
    rotated_vertices = np.dot(vertices, rotation_matrix.T) + center
    
    # Define edges
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
    ]
    
    # Draw edges
    for edge in edges:
        points = rotated_vertices[edge]
        ax.plot3D(*points.T, 'b-', alpha=0.6)
    
    return rotated_vertices

def visualize_cube_markers(ax, center, rotation_matrix, detected_markers):
    """
    Visualize detected markers on the cube
    """
    for marker_id in detected_markers:
        if marker_id in cube_marker_positions:
            # Get marker position in cube local coordinates
            local_pos = cube_marker_positions[marker_id]
            # Transform to world coordinates
            world_pos = np.dot(local_pos, rotation_matrix.T) + center
            ax.scatter(*world_pos, s=100, c='red', alpha=0.8)
            ax.text(world_pos[0], world_pos[1], world_pos[2], 
                   f'ID:{marker_id}', fontsize=8)

# Main script logic for video file
start_script_time = time.time()
cap = cv2.VideoCapture('data/vid2.avi')

frame_count = 0
while True:
    start_time = time.time()

    ret, image = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    
    # Create detector
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(gray)
    
    if ids is not None:
        detected_ids = ids.flatten()
        cube_markers = [int(id) for id in detected_ids if int(id) in CUBE_MARKER_IDS]
        
        print(f"Frame {frame_count}: Detected cube markers: {cube_markers}")
        
        if len(cube_markers) == 1:
            # Single marker pose estimation
            idx = np.where(detected_ids == cube_markers[0])[0][0]
            marker_corners_single = corners[idx]
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                marker_corners_single, MARKER_SIZE, camera_matrix, dist_coeffs
            )
            rvec = rvecs[0][0]
            tvec = tvecs[0][0]
            marker_id = cube_markers[0]
            offset = cube_marker_positions[marker_id]
            R, _ = cv2.Rodrigues(rvec)
            cube_center = tvec + R @ offset
            # For visualization, transform coordinates
            tvec_plot = np.array([-cube_center[0], -cube_center[2], -cube_center[1]])
            rotation_matrix_plot = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]) @ R @ np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]).T
            # Apply moving average filter
            if len(cube_positions) > 0:
                smoothed_position = moving_average_filter(
                    cube_positions + [tvec_plot], window_size)
            else:
                smoothed_position = tvec_plot
            cube_positions.append(smoothed_position)
            cube_orientations.append(rotation_matrix_plot)
            # Update plots
            if VISUALIZE:
                ax1.clear()
                ax1.set_xlabel('X (m)')
                ax1.set_ylabel('Y (m)')
                ax1.set_zlabel('Z (m)')
                ax1.set_title('Cube Center Position Trajectory')
                if len(cube_positions) > 1:
                    pos_arr = np.array(cube_positions)
                    ax1.plot(pos_arr[:, 0], pos_arr[:, 1], pos_arr[:, 2], 'b.-', alpha=0.7)
                ax1.scatter(smoothed_position[0], smoothed_position[1], 
                           smoothed_position[2], c='r', s=100, label='Current Center')
                ax1.legend()
                ax2.clear()
                ax2.set_xlabel('X (m)')
                ax2.set_ylabel('Y (m)')
                ax2.set_zlabel('Z (m)')
                ax2.set_title('Cube Orientation & Detected Markers')
                draw_cube_wireframe(ax2, smoothed_position, rotation_matrix_plot)
                visualize_cube_markers(ax2, smoothed_position, 
                                     rotation_matrix_plot, cube_markers)
                center = smoothed_position
                limit = 0.1
                ax2.set_xlim([center[0]-limit, center[0]+limit])
                ax2.set_ylim([center[1]-limit, center[1]+limit])
                ax2.set_zlim([center[2]-limit, center[2]+limit])
                plt.draw()
                plt.pause(0.001)
            # Draw cube axes on image
            cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, 
                            rvec, tvec, CUBE_SIZE)
        elif len(cube_markers) == 2:
            # Estimate pose for each marker, compute cube center from each, average centers
            centers = []
            rotations = []
            for marker_id in cube_markers:
                idx = np.where(detected_ids == marker_id)[0][0]
                marker_corners_single = corners[idx]
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    marker_corners_single, MARKER_SIZE, camera_matrix, dist_coeffs
                )
                rvec = rvecs[0][0]
                tvec = tvecs[0][0]
                offset = cube_marker_positions[marker_id]
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
            # Apply moving average filter
            if len(cube_positions) > 0:
                smoothed_position = moving_average_filter(
                    cube_positions + [tvec_plot], window_size)
            else:
                smoothed_position = tvec_plot
            cube_positions.append(smoothed_position)
            cube_orientations.append(rotation_matrix_plot)
            # Update plots
            if VISUALIZE:
                ax1.clear()
                ax1.set_xlabel('X (m)')
                ax1.set_ylabel('Y (m)')
                ax1.set_zlabel('Z (m)')
                ax1.set_title('Cube Center Position Trajectory')
                if len(cube_positions) > 1:
                    pos_arr = np.array(cube_positions)
                    ax1.plot(pos_arr[:, 0], pos_arr[:, 1], pos_arr[:, 2], 'b.-', alpha=0.7)
                ax1.scatter(smoothed_position[0], smoothed_position[1], 
                           smoothed_position[2], c='r', s=100, label='Current Center')
                ax1.legend()
                ax2.clear()
                ax2.set_xlabel('X (m)')
                ax2.set_ylabel('Y (m)')
                ax2.set_zlabel('Z (m)')
                ax2.set_title('Cube Orientation & Detected Markers')
                draw_cube_wireframe(ax2, smoothed_position, rotation_matrix_plot)
                visualize_cube_markers(ax2, smoothed_position, 
                                     rotation_matrix_plot, cube_markers)
                center = smoothed_position
                limit = 0.1
                ax2.set_xlim([center[0]-limit, center[0]+limit])
                ax2.set_ylim([center[1]-limit, center[1]+limit])
                ax2.set_zlim([center[2]-limit, center[2]+limit])
                plt.draw()
                plt.pause(0.001)
            # Draw cube axes on image (use the first marker's pose)
            idx = np.where(detected_ids == cube_markers[0])[0][0]
            marker_corners_single = corners[idx]
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                marker_corners_single, MARKER_SIZE, camera_matrix, dist_coeffs
            )
            rvec = rvecs[0][0]
            tvec = tvecs[0][0]
            cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, 
                            rvec, tvec, CUBE_SIZE)
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
                    if marker_id in cube_marker_positions:
                        marker_3d_pos = cube_marker_positions[marker_id]
                        object_points.append(marker_3d_pos)
                        corners2d = cube_corners[i]
                        center_2d = np.mean(corners2d[0], axis=0)
                        image_points.append(center_2d)
                object_points = np.array(object_points, dtype=np.float32)
                image_points = np.array(image_points, dtype=np.float32)
                if len(object_points) == 3:
                    success, rvec, tvec = cv2.solvePnP(
                        object_points, image_points, camera_matrix, dist_coeffs,
                        flags=cv2.SOLVEPNP_SQPNP
                    )
                    if not success:
                        rvec, tvec = None, None
            else:
                # Use default for 4+ points
                rvec, tvec = estimate_cube_pose(cube_markers, cube_corners, 
                                              camera_matrix, dist_coeffs)
            if rvec is not None and tvec is not None:
                tvec_plot = np.array([-tvec[0][0], -tvec[2][0], -tvec[1][0]])
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                transform_matrix = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])
                rotation_matrix_plot = transform_matrix @ rotation_matrix @ transform_matrix.T
                if len(cube_positions) > 0:
                    smoothed_position = moving_average_filter(
                        cube_positions + [tvec_plot], window_size)
                else:
                    smoothed_position = tvec_plot
                cube_positions.append(smoothed_position)
                cube_orientations.append(rotation_matrix_plot)
                if VISUALIZE:
                    ax1.clear()
                    ax1.set_xlabel('X (m)')
                    ax1.set_ylabel('Y (m)')
                    ax1.set_zlabel('Z (m)')
                    ax1.set_title('Cube Center Position Trajectory')
                    if len(cube_positions) > 1:
                        pos_arr = np.array(cube_positions)
                        ax1.plot(pos_arr[:, 0], pos_arr[:, 1], pos_arr[:, 2], 'b.-', alpha=0.7)
                    ax1.scatter(smoothed_position[0], smoothed_position[1], 
                               smoothed_position[2], c='r', s=100, label='Current Center')
                    ax1.legend()
                    ax2.clear()
                    ax2.set_xlabel('X (m)')
                    ax2.set_ylabel('Y (m)')
                    ax2.set_zlabel('Z (m)')
                    ax2.set_title('Cube Orientation & Detected Markers')
                    draw_cube_wireframe(ax2, smoothed_position, rotation_matrix_plot)
                    visualize_cube_markers(ax2, smoothed_position, 
                                         rotation_matrix_plot, cube_markers)
                    center = smoothed_position
                    limit = 0.1
                    ax2.set_xlim([center[0]-limit, center[0]+limit])
                    ax2.set_ylim([center[1]-limit, center[1]+limit])
                    ax2.set_zlim([center[2]-limit, center[2]+limit])
                    plt.draw()
                    plt.pause(0.001)
                cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, 
                                rvec, tvec, CUBE_SIZE)
        # Draw all detected markers
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        # Add text overlay with detection info
        y_offset = 30
        for i, marker_id in enumerate(ids.flatten()):
            color = (0, 255, 0) if int(marker_id) in CUBE_MARKER_IDS else (255, 255, 255)
            cv2.putText(image, f"ID: {marker_id}", (10, y_offset + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imshow('ArUco Cube Tracking', image)
    key = cv2.waitKey(1) & 0xFF

    print(f"fps: {1 / (time.time() - start_time)}")

    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
if VISUALIZE:
    plt.show()

total_exec_time = time.time() - start_script_time
print(f"Tracking completed. Total frames processed: {frame_count}")
print(f"Cube positions recorded: {len(cube_positions)}")
print(f"Total execution time: {total_exec_time:.2f} seconds")

# Save trajectory to file
np.savez('data/cube_trajectory1.npz', positions=np.array(cube_positions), orientations=np.array(cube_orientations))
print("Cube trajectory saved to 'data/cube_trajectory1.npz'")