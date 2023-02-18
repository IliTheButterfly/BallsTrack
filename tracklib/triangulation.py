from multiprocessing.managers import BaseManager
from time import time
from typing import List
import numpy as np
import cv2
from camlib.webCamera import WebCamera
import multiprocessing as mp
from tracklib.xrtrack import *

class TParams:
    Area_info = (0, 5000, 5000)
    Convexity_info = (0, 1, 1000)
    Circularity_info = (0, 1, 1000)
    InertiaRatio_info = (0, 1, 1000)
    def __init__(self, **kwargs):
        self.params = cv2.SimpleBlobDetector_Params()
        self.minArea = 200
        self.maxArea = 5000
        # self.minConvexity = 0.8
        # self.maxConvexity = 0
        self.minCircularity = 0.8
        self.maxCircularity = 1
        # self.minInertiaRatio = 0.0001
        # self.maxInertiaRatio = 0.8
        self.__dict__.update(**kwargs)
        self.update()
    def update(self):
        try:
            self.detector = cv2.SimpleBlobDetector_create(self.params)
        except:
            pass

    @property
    def minArea(self):
        return self.params.minArea
    @minArea.setter
    def minArea(self, val):
        self.params.minArea = val
        self.update()
    
    @property
    def maxArea(self):
        return self.params.maxArea
    @maxArea.setter
    def maxArea(self, val):
        self.params.filterByArea = val > self.Area_info[0]
        self.params.maxArea = val
        self.update()
    
    @property
    def minCircularity(self):
        return self.params.minCircularity
    @minCircularity.setter
    def minCircularity(self, val):
        self.params.minCircularity = val
        self.update()
    
    @property
    def maxCircularity(self):
        return self.params.maxCircularity
    @maxCircularity.setter
    def maxCircularity(self, val):
        self.params.filterByCircularity = val > self.Circularity_info[0]
        self.params.maxCircularity = val
        self.update()

    @property
    def minConvexity(self):
        return self.params.minConvexity
    @minConvexity.setter
    def minConvexity(self, val):
        self.params.minConvexity = val
        self.update()
    
    @property
    def maxConvexity(self):
        return self.params.maxConvexity
    @maxConvexity.setter
    def maxConvexity(self, val):
        self.params.filterByConvexity = val > self.Convexity_info[0]
        self.params.maxConvexity = val
        self.update()
    
    @property
    def minInertiaRatio(self):
        return self.params.minInertiaRatio
    @minInertiaRatio.setter
    def minInertiaRatio(self, val):
        self.params.minInertiaRatio = val
        self.update()
    
    @property
    def maxInertiaRatio(self):
        return self.params.maxInertiaRatio
    @maxInertiaRatio.setter
    def maxInertiaRatio(self, val):
        self.params.maxInertiaRatio = val
        self.update()

class TData:
    Threshold_info = (0, 255, 255)
    def __init__(self, camera:WebCamera, **kwargs):
        self.params = cv2.SimpleBlobDetector_Params()
        self.thresh = 200
        self.positions = []
        self.camera_matrix = camera.camera_matrix
        self.dist_coeffs = camera.dist_coeffs
        self.run = True
        self.__dict__.update(**kwargs)

class TManager(BaseManager):
    pass
TManager.register('TData', TData)

class TDatas:
    def __init__(self, cams:List[WebCamera], manager:TManager = None):
        if manager is None:
            self.cams:List[TData] = list()
        else:
            self.cams:List[TData] = manager.list()
        for cam in cams:
            if manager is None:
                self.cams.append(TData(cam))
            else:
                self.cams.append(manager.TData())
        self.params = TParams()
        
    def stop(self):
        for data in self.cams:
            data.run = False

TManager.register('TDatas', TDatas)
    
def getPosition(camera_id:int, frame:cv2.Mat, data:TData, params:TParams):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    _, gray = cv2.threshold(gray, data.thresh, 255, cv2.THRESH_BINARY_INV)
        
    # Detect blobs.
    try:
        keypoints = params.detector.detect(gray)
    except:
        return [None, gray]
    
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if len(keypoints) > 0:
        keypoints = cv2.KeyPoint_convert(keypoints)
    return [keypoints, im_with_keypoints]

def _grab_and_retrieve(camera_id:int, q:mp.Queue, data:TData, camera:WebCamera):
    # Start the capture
    with camera as cam:
        while data.run:
            # Grab a new frame
            grabbed = cam.grab()
            
            # Check if the frame was successfully grabbed
            if grabbed:
                # Retrieve the frame
                _, frame = cam.retrieve()

                frame = getPosition(camera_id, frame, data)
                
                # Put the frame into the queue
                q.put((camera_id, frame))

class ParallelSceneSolver:
    def __init__(self, cams:List[WebCamera]):
        self.cams = cams

        # Define the extrinsic parameters of each camera
        self.Rs = np.zeros((len(cams),3,3))
        self.Ts = np.zeros((len(cams),3,1))
        self.Ps = np.zeros((len(cams),3,4))
        self.vrPoints = []
        self.camPoints = [[] for _ in range(len(cams))]
        self.manager = TManager()
        self.datas = self.manager.TDatas()


    def updatePs(self):
        self.Ps = [np.concatenate((np.concatenate((R, T), axis=1), [0, 0, 0, 1]), axis=0) for R, T in zip(self.Rs, self.Ts)]

    def appendCalibration(self, vrPoint, camPoints):
        self.vrPoints.append(vrPoint)
        for i in range(len(self.cams)):
            self.camPoints[i].append(camPoints[i])

    def clearCalibration(self):
        self.vrPoints = []
        self.camPoints = [[] for _ in range(len(self.cams))]

    def calibrate(self):
        # Loop through each camera
        for i in range(len(self.cams)):
            # Compute the intrinsic parameters of the camera
            cam = self.cams[i]
            ret, K, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(self.vrPoints, self.camPoints[i], cam.size, cam.camera_matrix, cam.dist_coeffs)
            
            # Compute the extrinsic parameters of the camera
            ret, rvec, tvec = cv2.solvePnP(self.vrPoints, self.camPoints[i], K, distCoeffs)
            
            # Convert the rotation vector to a rotation matrix
            R, _ = cv2.Rodrigues(rvec)
            
            # Store the extrinsic parameters in a list
            self.Rs[i] = R
            self.Ts[i] = tvec
        self.updatePs()

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

    def run(self):
        with self.manager:
            q = mp.Queue()
            processes:List[mp.Process] = []
            for i, cam in enumerate(self.cams):
                processes.append(mp.Process(None, _grab_and_retrieve, args=(i, q, self.datas.cams[i], cam)))

            for p in processes:
                p.start()

            lastPoints = [[] for _ in range(len(self.cams))]
            nextCap = -1

            while True:
                (i,img) = q.get()
                lastPoints[i] = img[0]
                cv2.line(img[1], img[0]-[10,0], img[0]+[10,0], (255, 0, 0))
                cv2.line(img[1], img[0]-[0,10], img[0]+[0,10], (255, 0, 0))
                cv2.imshow(f"Camera {i}", img[1])

                k = cv2.waitKey(1)
                if k == 27:
                    break
                if k == ord('c'):
                    print("Appending point in 5 sec")
                    nextCap = time() + 5

                if nextCap != -1 and nextCap < time():
                    nextCap = -1
                    self.appendCalibration([0,0,0], lastPoints)
                    print("Appended calibration")
                
                if k == ord('d'):
                    print("Calibrating")
                    self.calibrate()
                    
            print("Stopping")
            self.datas.stop()
            for p in processes:
                p.join()
            cv2.destroyAllWindows()
        

class SceneSolver:
    def __init__(self, cams:List[WebCamera]):
        self.cams = cams

        # Define the extrinsic parameters of each camera
        self.Rs = np.zeros((len(cams),3,3))
        self.Ts = np.zeros((len(cams),3,1))
        self.Ps = np.zeros((len(cams),3,4))
        self.vrPoints = []
        self.camPoints = [[] for _ in range(len(cams))]
        self.datas = TDatas(cams)
        self.calibrated = False
        self.xr = xrutils()

    def updatePs(self):
        self.Ps = [np.concatenate((np.concatenate((R, T), axis=1), [0, 0, 0, 1]), axis=0) for R, T in zip(self.Rs, self.Ts)]

    def appendCalibration(self, vrPoint, camPoints):
        self.vrPoints.append(vrPoint)
        for i in range(len(self.cams)):
            self.camPoints[i].append(camPoints[i])

    def clearCalibration(self):
        self.vrPoints = []
        self.camPoints = [[] for _ in range(len(self.cams))]

    def calibrate(self):
        # Loop through each camera
        for i in range(len(self.cams)):
            # Compute the intrinsic parameters of the camera
            cam = self.cams[i]
            ret, K, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(self.vrPoints, self.camPoints[i], cam.size, cam.camera_matrix, cam.dist_coeffs)
            
            # Compute the extrinsic parameters of the camera
            ret, rvec, tvec = cv2.solvePnP(self.vrPoints, self.camPoints[i], K, distCoeffs)
            
            # Convert the rotation vector to a rotation matrix
            R, _ = cv2.Rodrigues(rvec)
            
            # Store the extrinsic parameters in a list
            self.Rs[i] = R
            self.Ts[i] = tvec
        self.updatePs()
        self.calibrated = True

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

    def getPosition(self, frame:cv2.Mat, data:TData):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, gray = cv2.threshold(gray, data.thresh, 255, cv2.THRESH_BINARY_INV)
            
        # Detect blobs.
        try:
            keypoints = self.datas.params.detector.detect(gray)
        except:
            return (None, gray)
        
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        if len(keypoints) > 0:
            keypoints = cv2.KeyPoint_convert(keypoints)
        return (keypoints, im_with_keypoints)

    def run(self):
        lastPoints = [[] for _ in range(len(self.cams))]
        nextCap = -1
        # with self.xr as xr:
        if True:
            # xr.update_hmd_pose()
            while True:
                for i in range(len(self.cams)):
                    ret, img = self.cams[i].read()
                    if not ret:
                        continue
                    points, img = self.getPosition(img, self.datas.cams[i])
                    lastPoints[i] = points
                    (h, w) = img.shape[:2]
                    if points is not None:
                        for p in points:
                            ll = 10
                            x = int(p[0])
                            y = int(p[1])
                            t = int(min(h, y + ll))
                            b = int(max(0, y - ll))
                            l = int(max(0, x + ll))
                            r = int(min(w, x - ll))
                            
                            cv2.line(img, (l, y), (r, y), (255, 0, 0))
                            cv2.line(img, (x, b), (x, t), (255, 0, 0))
                    cv2.imshow(f"Camera {i}", img)

                k = cv2.waitKey(1)
                if k == 27:
                    break
                if k == ord('c'):
                    print("Appending point in 5 sec")
                    nextCap = time() + 5

                if nextCap != -1 and nextCap < time():
                    nextCap = -1
                    self.appendCalibration([0,0,0], lastPoints)
                    print("Appended calibration")
                
                if k == ord('d'):
                    print("Calibrating")
                    self.calibrate()
                    
        print("Stopping")
        self.datas.stop()
        cv2.destroyAllWindows()





            
"""
# Example usage
ptsList = [np.array([[x1, y1], [x2, y2], ...]),
        np.array([[x1, y1], [x2, y2], ...]),
        ...
        ]
points3D = triangulate(ptsList)
"""