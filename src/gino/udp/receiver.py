import socket
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import time

class Receiver():
    def __init__(self, udp_port: int = 5005, udp_ip: str = "0.0.0.0") -> None:
        self.udp_port = udp_port
        self.udp_ip = udp_ip

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.udp_ip, self.udp_port))

        print(f"Listening on port {self.udp_port}...")

    def receive(self):
        data, _ = self.sock.recvfrom(1024)
        msg = data.decode('utf-8')
        
        # Split data
        parts = msg.split(',')
        if len(parts) != 13:
            print(f"Invalid message length: {len(parts)}")

        pos_x, pos_y, pos_z = map(float, parts[0:3])
        quat_x, quat_y, quat_z, quat_w = map(float, parts[3:7])
        button1 = bool(int(parts[7]))
        button2 = bool(int(parts[8]))
        slider = float(parts[9])

        position = [pos_x, pos_y, pos_z]
        quaternion = [quat_x, quat_y, quat_z, quat_w]
        controls = [button1, button2, slider]

        return position, quaternion, controls
    
    def disconnect(self):
        self.sock.close()

if __name__ == "__main__":
    # Global variables for visualization
    current_quat = np.array([0, 0, 0, 1])  # [x, y, z, w] - identity quaternion
    fig = None
    ax = None
    lines = None
    last_update_time = 0
    UPDATE_INTERVAL = 0.05  # Update every 50ms (20 FPS)

    # 3D quaternion visualization only

    def create_coordinate_system():
        """Create a 3D coordinate system representation matching Unity's convention"""
        # Unity coordinate system: X=right, Y=up, Z=forward
        # For visualization: X=right, Y=up, Z=out of screen
        axes = np.array([
            [1, 0, 0],   # X axis (red) - right
            [0, 1, 0],   # Y axis (green) - up  
            [0, 0, 1]    # Z axis (blue) - forward (out of screen)
        ])
        return axes

    def quaternion_to_rotation_matrix(quat):
        """Convert quaternion [x, y, z, w] to rotation matrix"""
        # Normalize quaternion
        quat = quat / np.linalg.norm(quat)
        x, y, z, w = quat
        
        # Convert to rotation matrix
        rotation = R.from_quat([x, y, z, w])
        return rotation.as_matrix()

    def update_visualization(quat):
        """Update the 3D visualization with new quaternion"""
        global ax, lines
        
        if ax is None:
            return
        
        # Convert quaternion to rotation matrix
        rotation_matrix = quaternion_to_rotation_matrix(quat)
        
        # Get coordinate system
        axes = create_coordinate_system()
        
        # Apply rotation to coordinate system
        rotated_axes = axes @ rotation_matrix.T
        
        # Apply coordinate system transformation to match matplotlib 3D view
        # Unity coordinates: X=right, Y=up, Z=forward
        # Matplotlib coordinates: X=right, Y=up, Z=out of screen
        transformed_axes = np.zeros_like(rotated_axes)
        transformed_axes[0] = rotated_axes[0]  # Unity X -> Matplotlib X (right)
        transformed_axes[1] = rotated_axes[1]  # Unity Y -> Matplotlib Y (up)
        transformed_axes[2] = rotated_axes[2]  # Unity Z -> Matplotlib Z (out)
        
        # Clear previous lines
        for line in lines:
            line.remove()
        lines.clear()
        
        # Draw rotated coordinate system (simplified)
        origin = np.array([0, 0, 0])
        colors = ['red', 'green', 'blue']
        labels = ['X', 'Y', 'Z']
        
        for axis, color, label in zip(transformed_axes, colors, labels):
            # Draw axis line only
            line = ax.plot([origin[0], axis[0]], 
                        [origin[1], axis[1]], 
                        [origin[2], axis[2]], 
                        color=color, linewidth=2)[0]
            lines.append(line)
        
        # Refresh the plot
        plt.draw()
        plt.pause(0.001)

    def setup_visualization():
        """Initialize the 3D quaternion visualization"""
        global fig, ax, lines
        
        # Initialize lines list first
        lines = []
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # Set up the plot
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Show initial coordinate system
        update_visualization(current_quat)
        
        plt.show(block=False)

    receiver = Receiver(udp_port=5005)

    print("Showing 3D quaternion visualization (coordinate system)")

    # Setup visualization
    setup_visualization()

    try:
        while True:
            position, quaternion, controls = receiver.receive()

            pos_x, pos_y, pos_z = position
            quat_x, quat_y, quat_z, quat_w = quaternion
            button1, button2, slider = controls

            # Update current quaternion
            current_quat = np.array([quat_x, quat_y, quat_z, quat_w])
            
            # Update visualization (throttled)
            current_time = time.time()
            if current_time - last_update_time >= UPDATE_INTERVAL:
                update_visualization(current_quat)
                last_update_time = current_time

            print(f"Pos: ({pos_x:.2f},{pos_y:.2f},{pos_z:.2f}) cm | "
                f"Btn1: {button1}, Btn2: {button2}, Slider: {slider:.2f} | "
                f"Quat: ({quat_x:.4f},{quat_y:.4f},{quat_z:.4f},{quat_w:.4f})")

    except KeyboardInterrupt:
        print("\nShutting down...")
        receiver.disconnect()
        plt.close()
