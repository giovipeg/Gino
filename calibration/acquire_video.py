import cv2
import time
import subprocess

output_file = "data/vid3.avi"

# Helper functions for camera settings
def get_current_setting(ctrl):
    result = subprocess.run(['v4l2-ctl', '--get-ctrl', ctrl], capture_output=True, text=True)
    value_str = result.stdout.strip().split(': ')[1]
    value = value_str.split()[0]
    return int(value)

def set_setting(ctrl, value):
    subprocess.run(['v4l2-ctl', '-c', f'{ctrl}={value}'])

# Store original settings
orig_auto_exposure = get_current_setting('auto_exposure')
orig_exposure_time = get_current_setting('exposure_time_absolute')

try:
    # Set manual exposure
    set_setting('auto_exposure', 1)
    set_setting('exposure_time_absolute', 30)  # Set to your desired value

    # --- Video acquisition logic ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: Could not open video capture.')
        exit(1)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # fallback if camera doesn't report FPS

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    print('Recording... Press q or ESC to stop.')
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Error: Failed to capture frame.')
            break
        # Create mirrored frame for visualization
        mirrored_frame = cv2.flip(frame, 1)
        out.write(frame)  # Save original frame
        cv2.imshow('Recording', mirrored_frame)  # Show mirrored frame
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f'Video saved as {output_file}')

finally:
    # Restore original settings
    set_setting('auto_exposure', orig_auto_exposure)
    set_setting('exposure_time_absolute', orig_exposure_time)
    print('Camera settings restored.') 