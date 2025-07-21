import cv2
import os

# === USER INSTRUCTIONS ===
# The calibration chessboard can be found at:
# resources/calib.io_checker_200x150_8x11_15.pdf
# If printed on a piece of paper it should be coupled to a rigid body (e.g. cardboard).

print("""
Camera Calibration Image Collector
==================================
- Hold your printed chessboard pattern in front of the camera.
- Move and tilt the chessboard to different positions and angles for each shot.
- Try to cover all parts of the camera's field of view (center, corners, edges).
- Press SPACE to capture an image.
- Press ESC when done.

Recommended: Take at least 20 images (preferably 30–40) for best calibration accuracy.\nVary the chessboard position and angle in each shot, and make sure it is fully visible and in focus.

Images will be saved in the 'calib_images' folder.
""") 

# === SETTINGS ===
output_dir = 'data/calib_images'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
img_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from camera.")
        break
    original_frame = frame.copy()  # Save a copy before drawing annotations
    cv2.putText(frame, f'Images saved: {img_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, 'Press SPACE to capture, ESC to exit', (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    cv2.imshow('Calibration Image Collector', frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        img_path = os.path.join(output_dir, f'calib_{img_count:02d}.jpg')
        cv2.imwrite(img_path, original_frame)  # Save the unannotated frame
        print(f"Saved {img_path}")
        img_count += 1

cap.release()
cv2.destroyAllWindows()
print(f"\nDone! {img_count} images saved in '{output_dir}'.")
