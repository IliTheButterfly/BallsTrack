import cv2
import numpy as np
import math

# Load the camera calibration parameters
camera_matrix = np.load("camera_matrix.npy")
dist_coeffs = np.load("dist_coeffs.npy")

# Define the dictionary of markers and their parameters
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters =  cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

board = cv2.aruco.GridBoard((6,6), 0.06, 0.01,dictionary)

w, h = 6000, 4000

#---------------- ARUCO MARKER ---------------#
# Create vectors we'll be using for rotations and translations for postures
rvec, tvec = None, None
R_ct = 0
R_tc = 0

corners = 0

R_flip  = np.zeros((3,3), dtype=np.float32)
R_flip[0,0] = 1.0
R_flip[1,1] =-1.0
R_flip[2,2] =-1.0

#-- Font for the text in the image
font = cv2.FONT_HERSHEY_PLAIN

#------------------------------------------------------------------------------
#------- ROTATIONS https://www.learnopencv.com/rotation-matrix-to-euler-angles/
#------------------------------------------------------------------------------
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-2

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-2

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

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
with SafeCapture(1) as cap1:

    # Loop over each frame from the cameras
    while True:
        # Load the next frame from each camera
        ret, frame1 = cap1.read()
        if not ret:
            continue
        # frame2 = cv2.imread("camera2.jpg")
        
        # Detect markers in each frame
        corners1, ids1, rejected_img_points1 = cv2.aruco.detectMarkers(frame1, dictionary, parameters=parameters)
        # corners2, ids2, rejected_img_points2 = cv2.aruco.detectMarkers(frame2, dictionary, parameters=parameters)


        # Estimate the pose of each marker in each frame
        # rvecs1, tvecs1, _ = cv2.solvePnP(corners1, 0.05, camera_matrix, dist_coeffs)
        # rvecs2, tvecs2, _ = cv2.aruco.estimatePoseSingleMarkers(corners2, 0.05, camera_matrix, dist_coeffs)
        
        # Draw the markers and their poses in each frame
        frame1 = cv2.aruco.drawDetectedMarkers(frame1, corners1, ids1)
        if ids1 is not None and len(ids1) > 0:
            # Estimate the posture of the gridboard, which is a construction of 3D space based on the 2D video
            pose, rvec, tvec = cv2.aruco.estimatePoseBoard(corners1, ids1, board, camera_matrix, dist_coeffs, rvec, tvec)

            if pose:
                # Draw the camera posture calculated from the gridboard
                #tvec[2] = tvec[2] * (-1)
                #frame = aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.3)
                frame1 = cv2.drawFrameAxes(frame1, camera_matrix, dist_coeffs, rvec, tvec, 0.2)

                tvec[0] *= 95
                tvec[1] *= 95
                tvec[2] *= 95

                # -- Print the tag position in camera frame
                # str_position = "MARKER Position x=%4.0f  y=%4.0f  z=%4.0f"%(tvec[0], tvec[1], tvec[2])
                # cv2.putText(frame, str_position, (0, 360), font, 1.3, (0, 255, 0), 2, cv2.LINE_AA)

                # -- Obtain the rotation matrix tag->camera
                R_ct = np.matrix(cv2.Rodrigues(rvec)[0])
                R_tc = R_ct.T

                # -- Get the attitude in terms of euler 321 (Needs to be flipped first)
                roll_marker, pitch_marker, yaw_marker = rotationMatrixToEulerAngles(R_flip*R_tc)

                #-- Print the marker's attitude respect to camera frame
                # str_attitude = "MARKER Attitude r=%4.0f  p=%4.0f  y=%4.0f"%(math.degrees(roll_marker),math.degrees(pitch_marker),
                #                     math.degrees(yaw_marker))
                # cv2.putText(frame, str_attitude, (0, 380), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # -- Now get Position and attitude f the camera respect to the marker
                pos_camera = -R_tc * np.matrix(tvec)
                str_position = "CAMERA Position x=%4.0f  y=%4.0f"%(pos_camera[2], pos_camera[0])
                cv2.putText(frame1, str_position, (0, 430), font, 1.4, (0, 255, 0), 2, cv2.LINE_AA)

                # -- Get the attitude of the camera respect to the frame
                pitch_camera, yaw_camera, roll_camera = rotationMatrixToEulerAngles(R_flip * R_tc)
                str_attitude = "CAMERA Attitude roll=%4.1f  pitch=%4.1f  yaw=%4.1f"%(math.degrees(roll_camera),math.degrees(pitch_camera),
                                    math.degrees(yaw_camera))
                cv2.putText(frame1, str_attitude, (0, 460), font, 1.4, (0, 255, 0), 2, cv2.LINE_AA)

        # frame2 = cv2.aruco.drawDetectedMarkers(frame2, corners2, ids2)
        # frame2 = cv2.aruco.drawAxis(frame2, camera_matrix, dist_coeffs, rvecs2, tvecs2, 0.1)
        
        # Show the frames
        cv2.imshow("Camera 1", frame1)
        # cv2.imshow("Camera 2", frame2)
        
        # Check if the user pressed the 'q' key to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the window and destroy all windows
    cv2.destroyAllWindows()









if __name__ == "__main__":
    print(cv2.aruco)