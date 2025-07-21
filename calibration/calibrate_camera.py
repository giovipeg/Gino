import cv2
import numpy as np
import glob
import os

# === USER SETTINGS ===
chessboard_size = (10, 7)  # (columns, rows) of inner corners (intersections between squares)
square_size = 0.015  # meters (set to your printed square size)
image_folder = 'data/calib_images'  # folder with calibration images

# === PREPARE OBJECT POINTS ===
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

# === LOAD IMAGES ===
images = glob.glob(os.path.join(image_folder, '*.jpg')) + glob.glob(os.path.join(image_folder, '*.png'))

if not images:
    print(f"No images found in {image_folder}. Please add calibration images and try again.")
    exit(1)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Current image', gray)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(100)
    else:
        print(f"Chessboard not found in {fname}")
        cv2.waitKey(1)
cv2.destroyAllWindows()

if not objpoints:
    print("No chessboard corners found in any image. Check your chessboard size and images.")
    exit(1)

# === CALIBRATE ===
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

print("\n=== Calibration Results ===")
print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs.ravel())

# === COMPUTE REPROJECTION ERROR ===
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error

mean_error = total_error / len(objpoints)
print(f"\nMean reprojection error: {mean_error} pxls")

# === SAVE RESULTS ===
np.savez('data/camera_calib.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
print("\nCalibration data saved to camera_calib.npz")