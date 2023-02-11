import cv2


class ArucoParams:
    def __init__(self, dictSet):
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictSet)
        self.parameters =  cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)