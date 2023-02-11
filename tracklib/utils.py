import math
from typing import Tuple
import cv2
from camlib.camera import Camera
from camlib.webCamera import WebCamera
import numpy as np

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

