import numpy as np
import cv2
import glob

# Define the size of the calibration pattern
pattern_size = (9, 6)

# Define the termination criteria for the optimization
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare the object points, which are the same for all images
objp = np.zeros((np.prod(pattern_size), 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# Arrays to store the object points and image points from all the images
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

class SafeCapture:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.cap = None

    def __enter__(self):
        self.cap = cv2.VideoCapture(*self.args, **self.kwargs)
        return self.cap
    
    def __exit__(self, type, value, traceback):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("Capture closed")

# Loop over all images
images = glob.glob("calibration_images/*.png")
with SafeCapture(0) as cam:
    while cv2.waitKey(1) != ord('c'):
        ret, imgOrig = cam.read()
        if not ret:
            continue
        img = imgOrig.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            objpoints.append(objp)
            
            # Refine the corner positions
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            
            # Draw the corners on the image
            cv2.drawChessboardCorners(img, pattern_size, corners, ret)
        cv2.imshow("Calibration pattern", img)

# Calibrate the camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save the camera matrix and distortion coefficients to .npy files
np.save("camera_matrix.npy", camera_matrix)
np.save("dist_coeffs.npy", dist_coeffs)



# Undistort an example image to validate the calibration
with SafeCapture(0) as cap:
    # while True:
    #     ret, image = cap.read()
    #     if ret:
    #         cv2.imshow("Image", image)
    while True:
        ret, img = cap.read()
        if not ret:
            continue
        else:
            break
    h, w = img.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    while True:
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
        ret = cap.grab()
        if not ret:
            continue
        else:
            ret, img = cap.retrieve()
        
        undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
        cv2.imshow("Distorted image", img)
        cv2.imshow("Undistorted image", undistorted_img)
        
cv2.destroyAllWindows()
