from enum import Enum
from typing import List
import openvr
import numpy as np

class Controller:
    Btn1 = 128
    Btn2 = 2
    Stick = 4294967296
    Grip = 17179869188
    Trig = 8589934592
    
    def __init__(self, id):
        self.id = id
        self.mat = None
        self.pos = [0,0,0]
        self.trigger = False
        self.grip = False
        self.btn1 = False
        self.btn2 = False
        self.stickbtn = False
        self.axies = None
        self.state = 0

    @property
    def joystick(self):
        if self.axies:
            return np.array([self.axies[0].x, self.axies[1].y], np.float32)
        return np.zeros((2), np.float32)
    
    @property
    def ftrigger(self) -> float:
        if self.axies:
            return self.axies[1].x
        return 0.0

    @property
    def fgrip(self) -> float:
        if self.axies:
            return self.axies[2].x
        return 0.0
    
        
    def _flag(self, val:int, test:int):
        return (val & test) == test

    def update(self, poses):

        self.mat = poses[self.id].mDeviceToAbsoluteTracking
        _, controller_state = openvr.VRSystem().getControllerState(self.id)

        # check the state of the trigger button
        self.pos = np.array([self.mat[0][3],self.mat[1][3],self.mat[2][3]])
        self.state = controller_state
        self.axies = [axis for axis in self.state.rAxis]
        self.state = controller_state.ulButtonPressed
        self.trigger = self._flag(self.state, int(Controller.Trig))
        self.grip = self._flag(self.state, int(Controller.Grip))
        self.stick = self._flag(self.state, int(Controller.Stick))
        self.btn1 = self._flag(self.state, int(Controller.Btn1))
        self.btn2 = self._flag(self.state, int(Controller.Btn2))

    def __str__(self):
        s = f"id: {self.id} | trig: {int(self.trigger)} | Grip: {int(self.grip)}\n  "
        for i, a in enumerate(self.axies):
            s += f"{i}: ({a.x},{a.y}) | "
        return s

class xrutils:
    def __init__(self):
        self.hmd = None
        self.leftController = None
        self.rightController = None
        self.headset = None
        self.controllers:List[Controller] = []
        self.poses = []

    def __enter__(self):
        for _ in range(10):
            try:
                self.hmd = openvr.init(openvr.VRApplication_Scene)
                print("Successfully connected to SteamVR")
                self.refreshDevices()
                break
            except openvr.OpenVRError:
                print("Failed to open SteamVR retrying")
        return self
        
    def __exit__(self, type, value, traceback):
        openvr.shutdown()

    def refreshDevices(self):
        # get device ids
        self.controllers = []
        for i in range(openvr.k_unMaxTrackedDeviceCount):
            device_class = openvr.VRSystem().getTrackedDeviceClass(i)
            if device_class != openvr.TrackedDeviceClass_Invalid:
                role = openvr.VRSystem().getControllerRoleForTrackedDeviceIndex(i)
                controller = Controller(i)
                self.controllers.append(controller)
                if role == openvr.TrackedControllerRole_LeftHand:
                    self.leftController = controller
                elif role == openvr.TrackedControllerRole_RightHand:
                    self.rightController = controller
                elif role == openvr.TrackedDeviceClass_HMD:
                    self.headset = controller
    
    def update(self):
        for c in self.controllers:
            c:Controller
            self.poses, _ = openvr.VRCompositor().waitGetPoses(self.poses, None)
            if c:
                c.update(self.poses)
