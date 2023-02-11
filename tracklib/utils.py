from ctypes.wintypes import RECT
from functools import cached_property
import math
from typing import Tuple
import cv2
from camlib.camera import Camera
from camlib.webCamera import WebCamera
from tracklib.aruco import ArucoParams
import numpy as np

def track3D(img:cv2.Mat, params:ArucoParams, camera_matrix, dist_coeffs):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected_img_points = params.detector.detectMarkers(gray)

    cv2.aruco.drawDetectedMarkers(img, corners, ids)

    for recs in rejected_img_points:
        for p in recs:
            
            cv2.polylines(img, np.int32([p]), True, (125,255,0))
    
    if np.all(ids is not None):  # If there are markers found by detector
        for i in range(0, len(ids)):  # Iterate in markers
            # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, camera_matrix,
                                                                        dist_coeffs)
            (rvec - tvec).any()  # get rid of that nasty numpy value array error
            cv2.aruco.drawDetectedMarkers(img, corners)  # Draw A square around the markers
            cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, 0.01)  # Draw Axis

def calcPSF(filterSize:Tuple[int,int], len:int, theta:float):
    h = np.zeros(filterSize, np.float32)
    point = (int(filterSize[0] / 2), int(filterSize[1] / 2))
    h = cv2.ellipse(h, point, (0, round(float(len) / 2.0)), 90.0 - theta, 0, 360, (255,255,255), cv2.FILLED)
    
    return h / h.sum()

def fftshift(inputImg:cv2.Mat):
    outputImg = inputImg.copy()
    cx = outputImg.shape[0] // 2
    cy = outputImg.shape[1] // 2
    q0 = outputImg[0:cx, 0:cy]
    q1 = outputImg[cx:cx, 0:cy]
    q2 = outputImg[0:cx, cy:cy]
    q3 = outputImg[cx:cx, cy:cy]
    tmp = cv2.Mat
    tmp = q0.copy()
    q0 = q3.copy()
    q3 = tmp.copy()
    tmp = q1.copy()
    q1 = q2.copy()
    q2 = tmp.copy()
    return outputImg

def filter2DFreq(inputImg:cv2.Mat, H:cv2.Mat):
    planes = np.array([ inputImg.copy(), np.zeros(inputImg.shape, np.float32) ])
    complexI = cv2.merge(planes)
    cv2.dft(complexI, complexI, cv2.DFT_SCALE)
    planesH = np.array([ H.copy(), np.zeros(H.shape, np.float32) ])
    complexH = cv2.merge(planesH[:, 0:480,0:480])
    complexIH = cv2.mulSpectrums(complexI, complexH, 0)
    cv2.idft(complexIH, complexIH)
    cv2.split(complexIH, planes)
    return planes[0]

def calcWnrFilter(input_h_PSF:cv2.Mat, nsr:float):
    h_PSF_shifted = fftshift(input_h_PSF)
    planes = np.array([h_PSF_shifted.copy() , np.zeros_like(h_PSF_shifted)])
    complexI = cv2.merge(planes)
    cv2.dft(complexI, complexI)
    cv2.split(complexI, planes)
    denom = pow(abs(planes[0]), 2)
    denom += nsr
    return planes[0] / denom

def edgetaper(inputImg:cv2.Mat, gamma:float, beta:float):
    Nx = inputImg.shape[0]
    Ny = inputImg.shape[1]
    w1 = np.zeros((1, Nx), np.float32)
    w2 = np.zeros((Ny, 1), np.float32)
    dx = float(2.0 * math.pi / Nx)
    x = float(-math.pi)
    for i in range(Nx):
        w1[0,i] = float(0.5 * (math.tanh((x + gamma / 2) / beta) - math.tanh((x - gamma / 2) / beta)))
        x += dx
    dy = float(2.0 * math.pi / Ny)
    y = float(-math.pi)
    for i in range(Ny):
        w2[i,0] = float(0.5 * (math.tanh((y + gamma / 2) / beta) - math.tanh((y - gamma / 2) / beta)))
        y += dy
    w = w2 * w1
    output = inputImg.copy()
    # output[:,:,0] = output[:,:,0] * w
    # output[:,:,1] = output[:,:,1] * w
    # output[:,:,2] = output[:,:,2] * w
    cv2.imshow("output", output)
    return np.array(output)[:,:,0] * np.matrix(w)
def deblur(len:int, theta:float, snr:int, img:cv2.Mat):
    """{image          |input.png    | input image name               }\n
    {LEN            |125          | length of a motion             }\n
    {THETA          |0            | angle of a motion in degrees   }\n
    {SNR            |700          | signal to noise ratio          }"""
    LEN = len
    THETA = theta
    snr = snr
    imgIn = img

    # it needs to process even image only
    roi = (imgIn.shape[0] - imgIn.shape[0] % 2, imgIn.shape[1] - imgIn.shape[1] % 2)
    # Hw calculation (start)
    h = calcPSF(roi, LEN, THETA)
    Hw = calcWnrFilter(h, 1.0 / float(snr))
    # Hw calculation (stop)
    imgIn.astype(np.float32)
    imgIn = edgetaper(imgIn, 5.0, 0.2)
    # filtering (start)
    imgOut = filter2DFreq(imgIn[0:imgIn.shape[0] & -2, 0:imgIn.shape[1] & -2], Hw)
    # filtering (stop)
    imgOut.astype(np.uint8)
    cv2.normalize(imgOut, imgOut, 0, 255, cv2.NORM_MINMAX)
    cv2.imshow("result.jpg", imgOut)



def blur_edge(img, d=31):
    h, w  = img.shape[:2]
    img_pad = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_WRAP)
    img_blur = cv2.GaussianBlur(img_pad, (2*d+1, 2*d+1), -1)[d:-d,d:-d]
    y, x = np.indices((h, w))
    dist = np.dstack([x, w-x-1, y, h-y-1]).min(-1)
    w = np.minimum(np.float32(dist)/d, 1.0)
    return img*w + img_blur*(1-w)

def motion_kernel(angle, d, sz=65):
    kern = np.ones((1, max(int(d), 1)), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    A[:,2] = (sz2, sz2) - np.dot(A[:,:2], ((d-1)*0.5, 0))
    kern = cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_CUBIC)
    return kern

def defocus_kernel(d, sz=65):
    kern = np.zeros((sz, sz), np.uint8)
    cv2.circle(kern, (sz, sz), d, 255, -1, cv2.LINE_AA, shift=1)
    kern = np.float32(kern) / 255.0
    return kern


class differenceFinder:
    def __init__(self, cam:Camera):
        self.cam = cam
        self.cleanPlate = None

    def getCleanPlate(self):
        while True:
            ch = cv2.waitKey(1)
            if ch == 27:
                break
            ret, img = self.cam.read()
            if ret:
                self.cleanPlate = np.float32(img)/255
                return

    def read(self):
        ret, img = self.cam.read()
        if not ret:
            return ret, None
        img = np.float32(img)/255
        

        if self.cleanPlate is None:
            return ret, img, None
        return ret, img, ((img - self.cleanPlate) + 1) / 2

class udeblurMarker:
    def __init__(self, cam:Camera, camera_matrix, dist_coeffs, params):
        self.cam = cam
        self.win = "deconvolution"
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.params = params

    def updateWin(self, _):
        self.ang = np.deg2rad( cv2.getTrackbarPos('angle', self.win) )
        # self.d = float(cv2.getTrackbarPos('d', self.win))/10
        self.d = cv2.getTrackbarPos('d', self.win)
        self.noise = 10**(-0.1*cv2.getTrackbarPos('SNR (db)', self.win))
        self.id = cv2.getTrackbarPos('Marker', self.win)
        self.roiSize = cv2.getTrackbarPos("ROI size", self.win)
        self.kernelSize = cv2.getTrackbarPos("Kernel size", self.win)

    def processSingleMarker(self, ang, speed, prev_roi:cv2.UMat, curr_roi:cv2.UMat):

        # psf = motion_kernel(ang + self.ang, max(10,int(self.d*speed*1000000.0)))
        # ppsf = motion_kernel(self.ang, int(self.d), self.kernelSize)
        # psf = cv2.copyTo()

        # psf /= psf.sum()
        # psf_pad = np.zeros_like(curr_roi)
        # kh, kw = psf.shape
        # psf_pad[:kh, :kw] = psf
        # PSF = cv2.dft(psf_pad, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows = kh)
        # PSF2 = (PSF**2).sum(-1)
        # iPSF = PSF / (PSF2 + self.noise)[...,np.newaxis]
        # RES = cv2.mulSpectrums(IMG_roi, iPSF, 0)
        # res = cv2.idft(RES, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
        # res = np.roll(res, -kh//2, 0)
        # res = np.roll(res, -kw//2, 1)
        
        _, res = cv2.threshold(curr_roi, self.d/1000, 1, cv2.THRESH_BINARY)
        # clahe = cv2.createCLAHE(clipLimit=self.d/100, tileGridSize=(self.kernelSize+1,self.kernelSize+1))
        # cl = clahe.apply(np.uint8(curr_roi*255))
        # res = np.float32(cl)/255

        return prev_roi, res

    def drawMarkers(self, img, corners, ids, rejected_img_points, camera_matrix, dist_coeffs):
        cv2.aruco.drawDetectedMarkers(img, corners, ids)


        for recs in rejected_img_points:
            for p in recs.get():
                
                cv2.polylines(img.get(), np.int32([p]), True, (125,255,0))
        if ids is None:
            return
        ids = ids.get()
        
        if np.all(ids is not None):  # If there are markers found by detector
            for i in range(0, len(ids)):  # Iterate in markers
                # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i].get(), 0.02, camera_matrix,
                                                                            dist_coeffs)
                (rvec - tvec).any()  # get rid of that nasty numpy value array error
                cv2.aruco.drawDetectedMarkers(img, corners)  # Draw A square around the markers
                cv2.drawFrameAxes(img.get(), camera_matrix, dist_coeffs, rvec, tvec, 0.01)  # Draw Axis

    def run(self):
        cv2.namedWindow(self.win)
        cv2.namedWindow('psf', 0)
        cv2.namedWindow("prev", 0)
        cv2.namedWindow("curr", 0)
        cv2.createTrackbar('angle', self.win, int(135), 360, self.updateWin)
        cv2.createTrackbar('d', self.win, int(22), 1000, self.updateWin)
        cv2.createTrackbar('SNR (db)', self.win, int(25), 50, self.updateWin)
        cv2.createTrackbar('Marker', self.win, int(0), 40, self.updateWin)
        cv2.createTrackbar('ROI size', self.win, int(65), 200, self.updateWin)
        cv2.createTrackbar('Kernel size', self.win, int(65), 200, self.updateWin)
        self.updateWin(None)


        uprev_gray = None
        while True:
            ch = cv2.waitKey(1)
            if ch == 27:
                break
            # Get the current frame from the video capture
            ret, img = self.cam.read()
            if not ret:
                continue
            shape = img.shape
            grayshape = img.shape[0:2]
            img = cv2.UMat(img)
            uraw_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


            mask = cv2.UMat(np.ones(grayshape, np.uint8))
            
            ucurr_gray = cv2.copyTo(uraw_gray, mask)
            cv2.divide(ucurr_gray, 255.0, ucurr_gray, dtype=cv2.CV_32F)

            if uprev_gray is None:
                ucurr_gray_cp = cv2.copyTo(ucurr_gray, mask)
                uprev_gray = cv2.copyTo(ucurr_gray, mask)
                corners, ids, rejected_img_points = self.params.detector.detectMarkers(uraw_gray)
            cv2.imshow('input', ucurr_gray)

            # ucurr_gray = blur_edge(ucurr_gray)
            # IMG = cv2.dft(ucurr_gray, flags=cv2.DFT_COMPLEX_OUTPUT)
            self.updateWin(None)
            # self.process(uprev_gray, ucurr_gray, corners, grayshape, ids)
            # cv2.multiply(ucurr_gray, 255, uraw_gray, dtype=cv2.CV_8U)
            corners, ids, rejected_img_points = self.params.detector.detectMarkers(uraw_gray)
            # imgg = cv2.cvtColor(ucurr_gray, cv2.COLOR_GRAY2BGR)
            self.drawMarkers(img, corners, ids, rejected_img_points, self.camera_matrix, self.dist_coeffs)
            cv2.imshow(self.win, img)

            cv2.copyTo(ucurr_gray_cp, mask, uprev_gray)

    def process(self, prev_gray, curr_gray, corners, grayShape, ids):
        if ids is None:
            return
        
        ids = ids.get()
        if ids is None:
            return

        _, res = cv2.threshold(curr_gray, self.d/1000, 1, cv2.THRESH_BINARY)
        # flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Loop through each detected marker
        mask = np.zeros(grayShape, np.uint8)
        corners = [c.get() for c in corners]
        for i in range(len(ids)):
            # Get the current position of the marker
            curr_position = (corners[i][0][0] + corners[i][0][2]) / 2

            


            # Define a region of interest (ROI) around the current position
            roi_size = self.roiSize
            mask[int(curr_position[1]-roi_size/2):int(curr_position[1]+roi_size/2), 
                int(curr_position[0]-roi_size/2):int(curr_position[0]+roi_size/2)] = 1

        cv2.copyTo(res,cv2.UMat(mask), curr_gray)


        for i in range(len(ids)):
            # Get the current position of the marker
            curr_position = (corners[i][0][0] + corners[i][0][2]) / 2

            


            # Define a region of interest (ROI) around the current position
            roi_size = self.roiSize
            prev_roi = prev_gray.get()[int(curr_position[1]-roi_size/2):int(curr_position[1]+roi_size/2), 
                        int(curr_position[0]-roi_size/2):int(curr_position[0]+roi_size/2)]
            # curr_roi = cv2.UMat(curr_gray.get()[int(curr_position[1]-roi_size/2):int(curr_position[1]+roi_size/2), 
            #             int(curr_position[0]-roi_size/2):int(curr_position[0]+roi_size/2)])

            # Use optical flow to estimate the motion of the ROI
            
            # flow_at_point = flow[int(curr_position[1]-roi_size/2):int(curr_position[1]+roi_size/2), 
            #                     int(curr_position[0]-roi_size/2):int(curr_position[0]+roi_size/2)]
            
            # Calculate the angle and speed of the motion
            # angle = np.arctan2(flow_at_point[0, 0, 1], flow_at_point[0, 0, 0])
            # speed = np.linalg.norm(flow_at_point[0, 0, :])
            angle = 0
            speed = 0
            
            # Print the angle and speed of the motion
            # print("Marker ID: {} - Angle: {:.2f} - Speed: {:.2f}".format(ids[i], angle, speed))
            # try:
            #     # prev_roi, res = self.processSingleMarker(angle, speed, prev_roi, curr_roi)
                
            #     curr_gray = curr_gray.get()
            #     curr_gray[int(curr_position[1]-roi_size/2):int(curr_position[1]+roi_size/2), 
            #         int(curr_position[0]-roi_size/2):int(curr_position[0]+roi_size/2)] = res
            # except ValueError as e:
            #     print(e)
            #     return

            if ids[i] == self.id:
                cv2.imshow("prev", prev_roi)
                cv2.imshow("curr", res)
                # cv2.imshow("psf", psf*psf.sum())

class deblurMarker:
    def __init__(self, cam:Camera, camera_matrix, dist_coeffs, params):
        self.cam = cam
        self.win = "deconvolution"
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.params = params

    def updateWin(self, _):
        self.ang = np.deg2rad( cv2.getTrackbarPos('angle', self.win) )
        # self.d = float(cv2.getTrackbarPos('d', self.win))/10
        self.d = cv2.getTrackbarPos('d', self.win)
        self.noise = 10**(-0.1*cv2.getTrackbarPos('SNR (db)', self.win))
        self.id = cv2.getTrackbarPos('Marker', self.win)
        self.roiSize = cv2.getTrackbarPos("ROI size", self.win)
        self.kernelSize = cv2.getTrackbarPos("Kernel size", self.win)

    def processSingleMarker(self, ang, speed, prev_roi, curr_roi, IMG_roi):

        # psf = motion_kernel(ang + self.ang, max(10,int(self.d*speed*1000000.0)))
        ppsf = motion_kernel(self.ang, int(self.d), self.kernelSize)
        psf = ppsf.copy()

        # psf /= psf.sum()
        # psf_pad = np.zeros_like(curr_roi)
        # kh, kw = psf.shape
        # psf_pad[:kh, :kw] = psf
        # PSF = cv2.dft(psf_pad, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows = kh)
        # PSF2 = (PSF**2).sum(-1)
        # iPSF = PSF / (PSF2 + self.noise)[...,np.newaxis]
        # RES = cv2.mulSpectrums(IMG_roi, iPSF, 0)
        # res = cv2.idft(RES, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
        # res = np.roll(res, -kh//2, 0)
        # res = np.roll(res, -kw//2, 1)
        
        _, res = cv2.threshold(curr_roi, self.d/1000, 1, cv2.THRESH_BINARY)
        # clahe = cv2.createCLAHE(clipLimit=self.d/100, tileGridSize=(self.kernelSize+1,self.kernelSize+1))
        # cl = clahe.apply(np.uint8(curr_roi*255))
        # res = np.float32(cl)/255

        return ppsf, prev_roi, res

    def drawMarkers(self, img, corners, ids, rejected_img_points, camera_matrix, dist_coeffs):
        cv2.aruco.drawDetectedMarkers(img, corners, ids)

        for recs in rejected_img_points:
            for p in recs:
                
                cv2.polylines(img, np.int32([p]), True, (125,255,0))
        
        if np.all(ids is not None):  # If there are markers found by detector
            for i in range(0, len(ids)):  # Iterate in markers
                # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, camera_matrix,
                                                                            dist_coeffs)
                (rvec - tvec).any()  # get rid of that nasty numpy value array error
                cv2.aruco.drawDetectedMarkers(img, corners)  # Draw A square around the markers
                cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, 0.01)  # Draw Axis

    def run(self):
        cv2.namedWindow(self.win)
        cv2.namedWindow('psf', 0)
        cv2.namedWindow("prev", 0)
        cv2.namedWindow("curr", 0)
        cv2.createTrackbar('angle', self.win, int(135), 360, self.updateWin)
        cv2.createTrackbar('d', self.win, int(22), 1000, self.updateWin)
        cv2.createTrackbar('SNR (db)', self.win, int(25), 50, self.updateWin)
        cv2.createTrackbar('Marker', self.win, int(0), 40, self.updateWin)
        cv2.createTrackbar('ROI size', self.win, int(65), 200, self.updateWin)
        cv2.createTrackbar('Kernel size', self.win, int(65), 200, self.updateWin)
        self.updateWin(None)

        prev_gray = None
        while True:
            ch = cv2.waitKey(1)
            if ch == 27:
                break
            # Get the current frame from the video capture
            ret, img = self.cam.read()
            if not ret:
                continue
            raw_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            curr_gray = np.float32(raw_gray)/255.0
            curr_gray_cp = curr_gray.copy()
            if prev_gray is None:
                prev_gray = curr_gray.copy()
                corners, ids, rejected_img_points = self.params.detector.detectMarkers(raw_gray)
            cv2.imshow('input', curr_gray)

            curr_gray = blur_edge(curr_gray)
            IMG = cv2.dft(curr_gray, flags=cv2.DFT_COMPLEX_OUTPUT)
            self.updateWin(None)
            self.process(prev_gray, curr_gray, IMG, corners, ids)
            raw_gray = np.uint8(curr_gray*255)
            corners, ids, rejected_img_points = self.params.detector.detectMarkers(raw_gray)
            imgg = cv2.cvtColor(curr_gray, cv2.COLOR_GRAY2BGR)
            self.drawMarkers(imgg, corners, ids, rejected_img_points, self.camera_matrix, self.dist_coeffs)
            cv2.imshow(self.win, imgg)
            prev_gray = curr_gray_cp

    def process(self, prev_gray, curr_gray, IMG, corners, ids):
        if ids is None:
            return
        
        # flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # Loop through each detected marker
        for i in range(len(ids)):
            # Get the current position of the marker
            curr_position = (corners[i][0][0] + corners[i][0][2]) / 2

            


            # Define a region of interest (ROI) around the current position
            roi_size = self.roiSize
            prev_roi = prev_gray[int(curr_position[1]-roi_size/2):int(curr_position[1]+roi_size/2), 
                        int(curr_position[0]-roi_size/2):int(curr_position[0]+roi_size/2)]
            curr_roi = curr_gray[int(curr_position[1]-roi_size/2):int(curr_position[1]+roi_size/2), 
                        int(curr_position[0]-roi_size/2):int(curr_position[0]+roi_size/2)]
            IMG_roi = IMG[int(curr_position[1]-roi_size/2):int(curr_position[1]+roi_size/2), 
                        int(curr_position[0]-roi_size/2):int(curr_position[0]+roi_size/2)]

            # Use optical flow to estimate the motion of the ROI
            
            # flow_at_point = flow[int(curr_position[1]-roi_size/2):int(curr_position[1]+roi_size/2), 
            #                     int(curr_position[0]-roi_size/2):int(curr_position[0]+roi_size/2)]
            
            # Calculate the angle and speed of the motion
            # angle = np.arctan2(flow_at_point[0, 0, 1], flow_at_point[0, 0, 0])
            # speed = np.linalg.norm(flow_at_point[0, 0, :])
            angle = 0
            speed = 0
            
            # Print the angle and speed of the motion
            # print("Marker ID: {} - Angle: {:.2f} - Speed: {:.2f}".format(ids[i], angle, speed))
            try:
                psf, prev_roi, res = self.processSingleMarker(angle, speed, prev_roi, curr_roi, IMG_roi)
                
                curr_gray[int(curr_position[1]-roi_size/2):int(curr_position[1]+roi_size/2), 
                    int(curr_position[0]-roi_size/2):int(curr_position[0]+roi_size/2)] = res
            except ValueError as e:
                print(e)
                return

            if ids[i] == self.id:
                cv2.imshow("prev", prev_roi)
                cv2.imshow("curr", res)
                cv2.imshow("psf", psf*psf.sum())

class propInfo:
    def __init__(self, max:float, min:float = 0, niceName:str=None, unit:str=None, scale:float=1, desc:str=None):
        self.name = niceName
        self.min = min
        self.max = max
        self.unit = unit
        self.scale = scale
        self.desc = desc
        if desc:
            self.__doc__ = desc

    def getPropName(self):
        words = self.name.lower().split(' ')
        prop = ""
        for i, word in enumerate(words):
            cword = word.capitalize() if len(word) <= 1 else word[0].capitalize() + word[1:]
            if i == 0:
                prop += word
            else:
                prop += cword
        return prop

    def __call__(self, func):
        if self.name is None:
            self.name = func.__name__
        if self.desc is None:
            self.desc = func.__doc__

        def predicate(slf, *args, **kwargs):
            n = func.__name__
            if func.__name__.startswith('_'):
                n = self.getPropName()
            setattr(slf, f'{n}_info', self)
            return func(slf,*args,**kwargs)
        return predicate
    
    def __str__(self):
        return f"{self.name} {self.unit if self.unit else ''}"

    def __repr__(self) -> str:
        unit = self.unit if self.unit else ""
        desc = self.desc if self.desc else ""
        return f"{self.name} range:({self.min*self.scale}{unit} - {self.max*self.scale}{unit}) {desc}"




class CamSettings:
    def __init__(self, cam:WebCamera):
        self.cam = cam
    
    def __str__(self):
        ret = ""
        ret += self.__class__.__name__
        for m in dir(self):
            if m.startswith('__') or m.endswith('_info') or m == 'cam':
                continue
            v = str(getattr(self, m))
            ret += f"\n\t{str(getattr(self,f'{m}_info'))}:{v}"
        return ret
    def __repr__(self):
        ret = ""
        ret += self.__class__.__name__
        for m in dir(self):
            if m.startswith('__'):
                continue
            v = str(getattr(self, m))
            ret += f"\n\t{repr(getattr(self,f'{m}_info'))}:{v}"
        return ret

    @classmethod
    def __add_rest__(cls):
        def exists(name:str):
            return name in dir(CamSettings)
        def neatName(name:str):
            name = name.removeprefix('CAP_PROP_').lower()
            words = name.split('_')
            prop = ""
            name = ""
            for i, word in enumerate(words):
                cword = word.capitalize() if len(word) <= 1 else word[0].capitalize() + word[1:]
                if i == 0:
                    prop += word
                    name += cword
                else:
                    prop += cword
                    name += ' ' + word
            return name, prop

        for name, member in cv2.__dict__.items():
            if not name.startswith('CAP_PROP_'):
                continue

            nName, prop = neatName(name)
            if exists(prop):
                continue

            def __get(n, nn, d, p):
                @propInfo(-100,100, nn, desc=d)
                def _get(self):
                    return self.cam.get(getattr(cv2,n))
                _get.__name__ = p
                return _get
            
            def __set(n, p):
                def _set(self, value:float):
                    self.cam.set(getattr(cv2,n), value)
                _set.__name__ = p
                return _set

            setattr(CamSettings, prop, property(__get(name, nName, member.__doc__, prop),__set(name, prop)))
            

    def __clean_props__(self):
        s = dir(self)
        
        for prop in s:
            if prop.startswith('__'):
                continue
            p = eval(f'self.{prop}')
            if p == 0:
                delattr(self,prop)
        

    @property
    @propInfo(32)
    def aperture(self):
        return self.cam.get(cv2.CAP_PROP_APERTURE)
    @aperture.setter
    def aperture(self, value):
        self.cam.set(cv2.CAP_PROP_APERTURE, value)

    @property
    @propInfo(1)
    def autoExposure(self):
        return self.cam.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    @autoExposure.setter
    def autoExposure(self, value):
        self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, value)

    @property
    @propInfo(1)
    def autoWB(self):
        return self.cam.get(cv2.CAP_PROP_AUTO_WB)
    @autoWB.setter
    def autoWB(self, value):
        self.cam.set(cv2.CAP_PROP_AUTO_WB, value)

    @property
    @propInfo(1)
    def autoFocus(self):
        return self.cam.get(cv2.CAP_PROP_AUTOFOCUS)
    @autoFocus.setter
    def autoFocus(self, value):
        self.cam.set(cv2.CAP_PROP_AUTOFOCUS, value)

    @property
    @propInfo(5000)
    def bitrate(self):
        return self.cam.get(cv2.CAP_PROP_BITRATE)
    @bitrate.setter
    def bitrate(self, value):
        self.cam.set(cv2.CAP_PROP_BITRATE, value)

    @property
    @propInfo(500)
    def zoom(self):
        return self.cam.get(cv2.CAP_PROP_ZOOM)
    @zoom.setter
    def zoom(self, value):
        self.cam.set(cv2.CAP_PROP_ZOOM, value)

    @property
    @propInfo(500)
    def brightness(self):
        return self.cam.get(cv2.CAP_PROP_BRIGHTNESS)
    @brightness.setter
    def brightness(self, value):
        self.cam.set(cv2.CAP_PROP_BRIGHTNESS, value)

    @property
    @propInfo(4096)
    def bufferSize(self):
        return self.cam.get(cv2.CAP_PROP_BUFFERSIZE)
    @bufferSize.setter
    def bufferSize(self, value):
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE, value)

    @property
    @propInfo(10)
    def codecPixelFormat(self):
        return self.cam.get(cv2.CAP_PROP_CODEC_PIXEL_FORMAT)
    @codecPixelFormat.setter
    def codecPixelFormat(self, value):
        self.cam.set(cv2.CAP_PROP_CODEC_PIXEL_FORMAT, value)

    @property
    @propInfo(3)
    def channel(self):
        return self.cam.get(cv2.CAP_PROP_CHANNEL)
    @channel.setter
    def channel(self, value):
        self.cam.set(cv2.CAP_PROP_CHANNEL, value)
    
    @property
    @propInfo(500)
    def contrast(self):
        return self.cam.get(cv2.CAP_PROP_CONTRAST)
    @contrast.setter
    def contrast(self, value):
        self.cam.set(cv2.CAP_PROP_CONTRAST, value)
    
    @property
    @propInfo(1)
    def convertRGB(self):
        return self.cam.get(cv2.CAP_PROP_CONVERT_RGB)
    @convertRGB.setter
    def convertRGB(self, value):
        self.cam.set(cv2.CAP_PROP_CONVERT_RGB, value)
    
    @property
    @propInfo(-1, -14)
    def exposure(self):
        return self.cam.get(cv2.CAP_PROP_EXPOSURE)
    @exposure.setter
    def exposure(self, value):
        self.cam.set(cv2.CAP_PROP_EXPOSURE, value)
    
    @property
    @propInfo(500)
    def focus(self):
        return self.cam.get(cv2.CAP_PROP_FOCUS)
    @focus.setter
    def focus(self, value):
        self.cam.set(cv2.CAP_PROP_FOCUS, value)
    
    @property
    @propInfo(10)
    def format(self):
        return self.cam.get(cv2.CAP_PROP_FORMAT)
    @format.setter
    def format(self, value):
        self.cam.set(cv2.CAP_PROP_FORMAT, value)
    
    @property
    @propInfo(10)
    def fourcc(self):
        return self.cam.get(cv2.CAP_PROP_FOURCC)
    @fourcc.setter
    def fourcc(self, value):
        self.cam.set(cv2.CAP_PROP_FOURCC, value)
    
    @property
    @propInfo(10000)
    def frameCount(self):
        return self.cam.get(cv2.CAP_PROP_FRAME_COUNT)
    @frameCount.setter
    def frameCount(self, value):
        self.cam.set(cv2.CAP_PROP_FRAME_COUNT, value)
    
    @property
    @propInfo(60)
    def fps(self):
        return self.cam.get(cv2.CAP_PROP_FPS)
    @fps.setter
    def fps(self, value):
        self.cam.set(cv2.CAP_PROP_FPS, value)
    
    @property
    @propInfo(1080)
    def frameHeight(self):
        return self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    @frameHeight.setter
    def frameHeight(self, value):
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, value)
    
    @property
    @propInfo(1920)
    def frameWidth(self):
        return self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    @frameWidth.setter
    def frameWidth(self, value):
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, value)
    
    @property
    @propInfo(500)
    def gain(self):
        return self.cam.get(cv2.CAP_PROP_GAIN)
    @gain.setter
    def gain(self, value):
        self.cam.set(cv2.CAP_PROP_GAIN, value)
    
    @property
    @propInfo(500)
    def gamma(self):
        return self.cam.get(cv2.CAP_PROP_GAMMA)
    @gamma.setter
    def gamma(self, value):
        self.cam.set(cv2.CAP_PROP_GAMMA, value)
    
    @property
    @propInfo(100)
    def iris(self):
        return self.cam.get(cv2.CAP_PROP_IRIS)
    @iris.setter
    def iris(self, value):
        self.cam.set(cv2.CAP_PROP_IRIS, value)
    
    @property
    @propInfo(1)
    def monochrome(self):
        return self.cam.get(cv2.CAP_PROP_MONOCHROME)
    @monochrome.setter
    def monochrome(self, value):
        self.cam.set(cv2.CAP_PROP_MONOCHROME, value)
    
    @property
    @propInfo(10)
    def mode(self):
        return self.cam.get(cv2.CAP_PROP_MODE)
    @mode.setter
    def mode(self, value):
        self.cam.set(cv2.CAP_PROP_MODE, value)
    
    @property
    @propInfo(1600)
    def isoSpeed(self):
        return self.cam.get(cv2.CAP_PROP_ISO_SPEED)
    @isoSpeed.setter
    def isoSpeed(self, value):
        self.cam.set(cv2.CAP_PROP_ISO_SPEED, value)
    
    @property
    @propInfo(100)
    def roll(self):
        return self.cam.get(cv2.CAP_PROP_ROLL)
    @roll.setter
    def roll(self, value):
        self.cam.set(cv2.CAP_PROP_ROLL, value)
    
    @property
    @propInfo(8000)
    def temperature(self):
        return self.cam.get(cv2.CAP_PROP_TEMPERATURE)
    @temperature.setter
    def temperature(self, value):
        self.cam.set(cv2.CAP_PROP_TEMPERATURE, value)
    
    @property
    @propInfo(100)
    def speed(self):
        return self.cam.get(cv2.CAP_PROP_SPEED)
    @speed.setter
    def speed(self, value):
        self.cam.set(cv2.CAP_PROP_SPEED, value)
    
    @property
    @propInfo(100)
    def hue(self):
        return self.cam.get(cv2.CAP_PROP_HUE)
    @hue.setter
    def hue(self, value):
        self.cam.set(cv2.CAP_PROP_HUE, value)
    
    @property
    @propInfo(100)
    def saturation(self):
        return self.cam.get(cv2.CAP_PROP_SATURATION)
    @saturation.setter
    def saturation(self, value):
        self.cam.set(cv2.CAP_PROP_SATURATION, value)

    @property
    @propInfo(100)
    def pan(self):
        return self.cam.get(cv2.CAP_PROP_PAN)
    @pan.setter
    def pan(self, value):
        self.cam.set(cv2.CAP_PROP_PAN, value)

    @property
    @propInfo(100)
    def tilt(self):
        return self.cam.get(cv2.CAP_PROP_TILT)
    @tilt.setter
    def tilt(self, value):
        self.cam.set(cv2.CAP_PROP_TILT, value)
    
    @property
    @propInfo(100)
    def sharpness(self):
        return self.cam.get(cv2.CAP_PROP_SHARPNESS)
    @sharpness.setter
    def sharpness(self, value):
        self.cam.set(cv2.CAP_PROP_SHARPNESS, value)
    
    @property
    @propInfo(8000)
    def wbTemperature(self):
        return self.cam.get(cv2.CAP_PROP_WB_TEMPERATURE)
    @wbTemperature.setter
    def wbTemperature(self, value):
        self.cam.set(cv2.CAP_PROP_WB_TEMPERATURE, value)
    
    @property
    @propInfo(100)
    def whiteBalanceRedV(self):
        return self.cam.get(cv2.CAP_PROP_WHITE_BALANCE_RED_V)
    @whiteBalanceRedV.setter
    def whiteBalanceRedV(self, value):
        self.cam.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, value)
    
    @property
    @propInfo(100)
    def whiteBalanceBlueU(self):
        return self.cam.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U)
    @whiteBalanceBlueU.setter
    def whiteBalanceBlueU(self, value):
        self.cam.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, value)
    
CamSettings.__add_rest__()