def get_current_setting(ctrl):
    result = subprocess.run(['v4l2-ctl', '--get-ctrl', ctrl], capture_output=True, text=True)
    # Output format: "ctrl: value (description)" or "ctrl: value"
    value_str = result.stdout.strip().split(': ')[1]
    value = value_str.split()[0]  # Take only the first part before any space/parenthesis
    return int(value)

def set_setting(ctrl, value):
    subprocess.run(['v4l2-ctl', '-c', f'{ctrl}={value}'])

# Store original settings
# Set auto-exposure command: v4l2-ctl -c auto_exposure=3
orig_auto_exposure = get_current_setting('auto_exposure')
orig_exposure_time = get_current_setting('exposure_time_absolute')

# TO-DO: disable auto-wb, auto-focus

# Camera calibration parameters
calib = np.load('aruco/camera_calib.npz')
camera_matrix = calib['camera_matrix']
dist_coeffs = calib['dist_coeffs']

