from abc import abstractmethod, abstractproperty
from typing import Tuple, Union
from typing_extensions import Self
import cv2

class Camera:
    def __init__(self, camera_matrix = None, dist_coeffs = None):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    @abstractproperty
    def size(self) -> Tuple[int,int]:
        return (0,0)

    @abstractproperty
    def isOpened(self) -> bool:
        return False

    @abstractmethod
    def grab(self) -> bool:
        return False

    @abstractmethod
    def retrieve(self) -> Tuple[bool, Union[cv2.Mat, None]]:
        return False, None

    @abstractmethod
    def read(self) -> Tuple[bool, Union[cv2.Mat, None]]:
        return False, None

    def get(self, propId):
        return self._cam.get(propId)

    def set(self, propId, value):
        return self._cam.set(propId, value)

    @abstractmethod
    def __enter__(self) -> Self:
        return self

    @abstractmethod
    def __exit__(self, type, value, traceback):
        pass
