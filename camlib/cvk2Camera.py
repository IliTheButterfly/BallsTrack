from typing import Tuple, Union, overload
import typing
from typing_extensions import override, Self
import typing_extensions
from camlib.camera import Camera
import cv2


def init_videocapture(location, width=1920, height=1080):
    camera = cv2.VideoCapture(location, cv2.CAP_ANY)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, 30)
    return camera

class CVK2Camera(Camera):
    def __init__(self, index=0, depth=False):
        super().__init__()
        self.location = index
        self._cam :cv2.VideoCapture = None
        self.openni = index in (cv2.CAP_OPENNI, cv2.CAP_OPENNI2)
        self.fps = 0
        self.depth = depth

    @override
    def grab(self) -> bool:
        if self._cam is None:
            return False
        return self._cam.grab()

    @override
    def retrieve(self) -> Tuple[bool, Union[cv2.Mat, None]]:
        if self._cam is None:
            return False, None
        if self.openni:
            if self.depth:
                return self._cam.retrieve(cv2.CAP_OPENNI_DEPTH_MAP)
            else:
                return self._cam.retrieve(cv2.CAP_OPENNI_BGR_IMAGE)
        else:
            return self._cam.retrieve()

    @override
    def read(self) -> Tuple[bool, Union[cv2.Mat, None]]:
        if self._cam is None:
            return False, None
        if self.openni:
            if not self._cam.grab():
                return False, None
            if self.depth:
                return self._cam.retrieve(cv2.CAP_OPENNI_DEPTH_MAP)
            else:
                return self._cam.retrieve(cv2.CAP_OPENNI_BGR_IMAGE)
        else:
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
        return self._cam.set(propId, value)

    @override
    def __enter__(self) -> Self:
        self._cam = init_videocapture(self.location)
        return self


    @override
    def __exit__(self, type, value, traceback):
        self._cam.release()
