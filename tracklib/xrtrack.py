from typing import List
import openvr

class Controller:
    def __init__(self, id):
        self.id = id
        self.pos = None
        self.trigger = False
        self.grip = False
        self.axies = []
        self.dev = None
        
    def update(self):

        if self.dev:
            openvr.VRSystem().getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseStanding, 0.0, self.id, self.dev)
        else:
            self.dev = openvr.VRSystem().getDeviceToAbsoluteTrackingPose(openvr.TrackingUniverseStanding, 0.0, self.id, None)
        self.pos = self.dev.contents.mDeviceToAbsoluteTracking
        _, controller_state = openvr.VRSystem().getControllerState(self.id)

        # check the state of the trigger button
        self.state = controller_state
        self.axies = [axis for axis in self.state.rAxis]
        # self.trigger = controller_state.ulButtonPressed & openvr.Butto.ButtonMaskFromId(openvr.k_EButton_SteamVR_Trigger)
        # self.grip = controller_state.ulButtonPressed & openvr.ButtonMaskFromId(openvr.k_EButton_Grip)

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

    def __enter__(self):
        for _ in range(10):
            try:
                self.hmd = openvr.init(openvr.VRApplication_Other)
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
                self.controllers.append(Controller(i))
                if role == openvr.TrackedControllerRole_LeftHand:
                    self.leftController = Controller(i)
                elif role == openvr.TrackedControllerRole_RightHand:
                    self.rightController = Controller(i)
                elif role == openvr.TrackedDeviceClass_HMD:
                    self.headset = Controller(i)
    
    def update(self):
        for c in self.controllers:
            # c:Controller
            if c:
                c.update()
