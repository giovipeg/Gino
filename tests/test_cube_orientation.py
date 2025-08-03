import cv2
import numpy as np
import sys
import os

# Add the src directory to the path to import gino modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gino.aruco.cube_detection import ArucoCubeTracker

def main():
    # Initialize the cube tracker
    aruco = ArucoCubeTracker()
    
    # Open camera (adjust camera index if needed)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Cube Orientation Test")
    print("Press 'q' to quit")
    print("The cube's roll, tilt, and jaw values will be displayed on screen")
    
    while True:
        ret, image = cap.read()
        if not ret:
            break
            
        # Get cube pose estimation
        smoothed_position, rotation_matrix_plot, cube_markers = aruco.pose_estimation(image)
        
        # Extract roll, tilt, jaw values
        if rotation_matrix_plot is not None:
            roll, tilt, jaw = aruco.get_cube_euler_angles(rotation_matrix_plot)
            
            # Display the values on screen
            cv2.putText(image, f"Roll: {roll:.1f}°", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Tilt: {tilt:.1f}°", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Jaw: {jaw:.1f}°", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Also print to console for debugging
            print(f"Roll: {roll:.1f}°, Tilt: {tilt:.1f}°, Jaw: {jaw:.1f}°")
        else:
            cv2.putText(image, "No cube detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show the image
        cv2.imshow('Cube Orientation Test', image)
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 