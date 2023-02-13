from multiprocessing.managers import BaseManager
from typing import List
import numpy as np
import cv2
from camlib.camera import Camera
import multiprocessing as mp

class TData:
    Threshold_info = (0, 255, 255)
    Area_info = (0, 1500, 1500)
    Convexity_info = (0, 1, 1000)
    InertiaRatio_info = (0, 1, 1000)
    def __init__(self, **kwargs):
        self.params = cv2.SimpleBlobDetector_Params()
        self.thresh = 200
        self.positions = []
        self.minArea = 20
        self.maxArea = 200
        self.minConvexity = 0.8
        self.maxConvexity = 1
        self.minInertiaRatio = 0
        self.maxInertiaRatio = 0.8


    # @property
    # def minThreshold(self):
    #     return self.params.minThreshold
    # @minThreshold.setter
    # def minThreshold(self, val):
    #     self.params.minThreshold = val
    
    # @property
    # def maxThreshold(self):
    #     return self.params.maxThreshold
    # @maxThreshold.setter
    # def maxThreshold(self, val):
    #     self.params.filterByThreshold = val <= self.Area_info[0]
    #     self.params.maxThreshold = val
    
    @property
    def minArea(self):
        return self.params.minArea
    @minArea.setter
    def minArea(self, val):
        self.params.minArea = val
    
    @property
    def maxArea(self):
        return self.params.maxArea
    @maxArea.setter
    def maxArea(self, val):
        self.params.filterByArea = val <= self.Area_info[0]
        self.params.maxArea = val
    
    @property
    def minConvexity(self):
        return self.params.minConvexity
    @minConvexity.setter
    def minConvexity(self, val):
        self.params.minConvexity = val
    
    @property
    def maxConvexity(self):
        return self.params.maxConvexity
    @maxConvexity.setter
    def maxConvexity(self, val):
        self.params.filterByConvexity = val <= self.Convexity_info[0]
        self.params.maxConvexity = val
    
    @property
    def minInertiaRatio(self):
        return self.params.minInertiaRatio
    @minInertiaRatio.setter
    def minInertiaRatio(self, val):
        self.params.minInertiaRatio = val
    
    @property
    def maxInertiaRatio(self):
        return self.params.maxInertiaRatio
    @maxInertiaRatio.setter
    def maxInertiaRatio(self, val):
        self.params.filterByInertiaRatio = val <= self.InertiaRatio_info[0]
        self.params.maxInertiaRatio = val
    
class TManager(BaseManager):
    pass
TManager.register('TData', TData)

class TDatas:
    def __init__(self, manager:TManager, cams:List[int]):
        self.cams:dict = manager.dict()
        for cam in cams:
            self.cams[cam] = manager.TData()

TManager.register('TDatas', TDatas)
    
def getPosition(camera_id:int, frame:cv2.Mat, data):
    global detector
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    t = 200
    try:
        t = data[camera_id]['thresh']
    except KeyError:
        return [None, gray]

    _, gray = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY_INV)
        
    # Detect blobs.
    try:
        keypoints = detector.detect(gray)
    except:
        return [None, gray]
    
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if len(keypoints) > 0:
        keypoints = cv2.KeyPoint_convert(keypoints)
    return [keypoints, im_with_keypoints]

class SceneCalibrator:
    def __init__(self, cams:List[Camera], ):
        self.cams = cams

        # Define the extrinsic parameters of each camera
        self.Rs = []
        self.Ts = []
        self.Ps = []
    
    def start(self):
        pass

    def stop(self):
        pass



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