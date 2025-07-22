import matplotlib.pyplot as plt
import numpy as np

def setup_visualization(visualize_position, visualize_pose):
    """Sets up the 3D plots for visualization."""
    fig_pos, ax_pos = None, None
    fig_pose, ax_pose = None, None

    if visualize_position or visualize_pose:
        plt.ion()

    if visualize_position:
        fig_pos = plt.figure(figsize=(8, 6))
        ax_pos = fig_pos.add_subplot(111, projection='3d')
        ax_pos.set_xlabel('X (m)')
        ax_pos.set_ylabel('Y (m)')
        ax_pos.set_zlabel('Z (m)')
        ax_pos.set_title('Cube Center Position')

    if visualize_pose:
        fig_pose = plt.figure(figsize=(8, 6))
        ax_pose = fig_pose.add_subplot(111, projection='3d')
        ax_pose.set_xlabel('X (m)')
        ax_pose.set_ylabel('Y (m)')
        ax_pose.set_zlabel('Z (m)')
        ax_pose.set_title('Cube Orientation & Individual Markers')
        
    return fig_pos, ax_pos, fig_pose, ax_pose

def draw_cube_wireframe(ax, center, rotation_matrix, size):
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

def visualize_cube_markers(ax, center, rotation_matrix, detected_markers, cube_marker_positions):
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

def update_visualization(visualize_position, visualize_pose, 
                         ax_pos, ax_pose,
                         cube_positions, smoothed_position, 
                         rotation_matrix_plot, cube_markers,
                         cube_marker_positions, CUBE_SIZE):
    """Updates the visualization plots in the main loop."""

    if visualize_position:
        ax_pos.clear()
        ax_pos.set_xlabel('X (m)')
        ax_pos.set_ylabel('Y (m)')
        ax_pos.set_zlabel('Z (m)')
        ax_pos.set_title('Cube Center Position Trajectory')
        if len(cube_positions) > 1:
            pos_arr = np.array(cube_positions)
            ax_pos.plot(pos_arr[:, 0], pos_arr[:, 1], pos_arr[:, 2], 'b.-', alpha=0.7)
        ax_pos.scatter(smoothed_position[0], smoothed_position[1], 
                   smoothed_position[2], c='r', s=100, label='Current Center')
        ax_pos.legend()

    if visualize_pose:
        ax_pose.clear()
        ax_pose.set_xlabel('X (m)')
        ax_pose.set_ylabel('Y (m)')
        ax_pose.set_zlabel('Z (m)')
        ax_pose.set_title('Cube Orientation & Detected Markers')
        draw_cube_wireframe(ax_pose, smoothed_position, rotation_matrix_plot, CUBE_SIZE)
        visualize_cube_markers(ax_pose, smoothed_position, 
                             rotation_matrix_plot, cube_markers, cube_marker_positions)
        center = smoothed_position
        limit = 0.1
        ax_pose.set_xlim([center[0]-limit, center[0]+limit])
        ax_pose.set_ylim([center[1]-limit, center[1]+limit])
        ax_pose.set_zlim([center[2]-limit, center[2]+limit])

    if visualize_position or visualize_pose:
        plt.draw()
        plt.pause(0.001)

def close_visualization():
    """Closes the plots."""
    plt.ioff()
    plt.show() 