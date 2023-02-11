from typing import Tuple, Union, overload
import typing
from typing_extensions import override, Self
import typing_extensions
from camlib.camera import Camera
import cv2
import pykinect2.PyKinectRuntime as k2r
import pykinect2.PyKinectV2 as k2


class K2Camera(Camera):
    def __init__(self):
        super().__init__()
        self._cam = None


    @override
    def grab(self) -> bool:
        if self._cam is None:
            return False
        return self._cam.has_new_color_frame()

    @override
    def retrieve(self) -> Tuple[bool, Union[cv2.Mat, None]]:
        if self._cam is None:
            return False, None
        f = self._cam.get_last_color_frame()
        return f is not None, f

    @override
    def read(self) -> Tuple[bool, Union[cv2.Mat, None]]:
        return self.retrieve()

    @property
    def size(self) -> Tuple[int, int]:
        if self._cam is None:
            return (0,0)
        return (1920,1080)
    
    @property
    def isOpened(self) -> bool:
        return self._cam is not None

    
        

    @override
    def __enter__(self) -> Self:
        self._cam = k2r.PyKinectRuntime(k2r.FrameSourceTypes_Color)
        return self


    @override
    def __exit__(self, type, value, traceback):
        self._cam.close()
