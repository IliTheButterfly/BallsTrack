from typing import List
import numpy as np
import cv2
from camlib.camera import Camera

class SceneSolver:
    def __init__(self, cams:List[Camera], ):

        
        self.cams = cams

        # Define the extrinsic parameters of each camera
        self.Rs = []
        self.Ts = []
        self.Ps = []

    def updatePs(self):
        self.Ps = [np.concatenate((np.concatenate((R, T), axis=1), [0, 0, 0, 1]), axis=0) for R, T in zip(self.Rs, self.Ts)]

    # Triangulate the 3D position of the markers
    def triangulate(self, ptsList):
        # Normalize the 2D points
        normalizedPtsList = [cv2.undistortPoints(pts, self.cams[i].camera_matrix, self.cams[i].dist_coeffs) for i, pts in enumerate(ptsList)]
        
        # Convert the 2D points to homogeneous coordinates
        homogeneousPtsList = [np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1) for pts in normalizedPtsList]
        
        # Triangulate the 3D position of each marker
        points4D = cv2.triangulatePoints([P for P in self.Ps], [pts.T for pts in homogeneousPtsList])
        points3D = points4D[:3,:] / np.tile(points4D[3,:], (3, 1))
        
        return points3D.T

    

"""
# Example usage
ptsList = [np.array([[x1, y1], [x2, y2], ...]),
        np.array([[x1, y1], [x2, y2], ...]),
        ...
        ]
points3D = triangulate(ptsList)
"""