from pathlib import Path
from typing import Tuple, Union, overload
from typing_extensions import override, Self
from camlib.camera import Camera
import cv2
import numpy as np

def init_videocapture(location, width=1920, height=1080):
    camera = cv2.VideoCapture(location, cv2.CAP_ANY)
    # camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    # camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return camera

class WebCamera(Camera):
    def __init__(self, location:Union[str, int], calibName:str=None):
        if calibName:
            path_camera_matrix = Path(f'cameraData\\{calibName}\\camera_matrix.npy')
            path_dist_coeffs = Path(f'cameraData\\{calibName}\\dist_coeffs.npy')
            if path_camera_matrix.exists() and path_dist_coeffs.exists():
                camera_matrix = np.load(path_camera_matrix.absolute())
                dist_coeffs = np.load(path_dist_coeffs.absolute())
                super().__init__(camera_matrix, dist_coeffs)
            else:
                super().__init__()
        else:
            super().__init__()

        self.location = location
        self._cam:cv2.VideoCapture = None

    @override
    def grab(self) -> bool:
        if self._cam is None:
            return False
        return self._cam.grab()

    @override
    def retrieve(self) -> Tuple[bool, Union[cv2.Mat, None]]:
        if self._cam is None:
            return False, None
        return self._cam.retrieve()

    @override
    def read(self) -> Tuple[bool, Union[cv2.Mat, None]]:
        if self._cam is None:
            return False, None
        return self._cam.read()

    @property
    def size(self) -> Tuple[int, int]:
        if self._cam is None:
            return (0,0)
        return self._cam.get_size()
    
    @property
    def isOpened(self) -> bool:
        if self._cam is None:
            return False
        return self._cam.isOpened()

    def get(self, propId):
        return self._cam.get(propId)

    def set(self, propId, value):
        return self._cam.set(propId, float(value))

    @override
    def __enter__(self) -> Self:
        self._cam = init_videocapture(self.location)
        return self


    @override
    def __exit__(self, type, value, traceback):
        self._cam.release()
