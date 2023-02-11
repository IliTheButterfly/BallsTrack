from typing import Tuple, Union, overload
import typing
from typing_extensions import override, Self
import typing_extensions
from camlib.camera import Camera
import cv2
from primesense import openni2#, nite2
from primesense import _openni2 as c_api
import numpy as np


# dist ='/home/carlos/Install/openni2/OpenNI-Linux-x64-2.2/Redist'



def init_videocapture(location, width=1920, height=1080):
    camera = cv2.VideoCapture(location, cv2.CAP_ANY)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, 30)
    return camera

class NIK2Camera(Camera):
    def __init__(self, depth=False):
        super().__init__()
        self._cam = None
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
        bgr = np.fromstring(self._stream.read_frame().get_buffer_as_uint8(),dtype=np.uint8).reshape(1080,1920,3)
        return True, bgr

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
        openni2.initialize("C:\\Program Files\\OpenNI2\\Redist")
        if (openni2.is_initialized()):
            print("openNI2 initialized")
        else:
            print("openNI2 not initialized")
        self._cam = openni2.Device.open_any()
        self._stream = self._cam.create_color_stream()
        self._stream.start()
        return self


    @override
    def __exit__(self, type, value, traceback):
        self._stream.stop()
        self._cam.close()
        openni2.unload()

