from time import sleep
from numpy import matrix
import numpy as np
import openvr
import keyboard
# Initialize OpenVR
openvr.init(openvr.VRApplication_Scene)

class tracker:
    def __init__(self, name:str, index:int):
        self.name = name
        self.index = index

trackers = []
#updates scene tracker list
def get_trackers():
    vrsys = openvr.VRSystem()
    types = {
        str(openvr.TrackedDeviceClass_HMD): "HMD",
        str(openvr.TrackedDeviceClass_Controller): "Controller", 
        str(openvr.TrackedDeviceClass_GenericTracker): "GenericTracker"
    }
    for i in range(openvr.k_unMaxTrackedDeviceCount):
        if str(vrsys.getTrackedDeviceClass(i)) in types:
            #add any new trackers
            name = types[str(vrsys.getTrackedDeviceClass(i))]+'_%03d' %i
            if name not in trackers:
                trackers.append(tracker(name, i))

def printPose(pose, tracker:tracker):
    s = f"{tracker.index}:{tracker.name}: "
    for row in pose:
        for v in row:
            s += "{:.3f}|".format(v)
    print(s, flush=True)

def getPoses():
    poses = []
    poses, _ = openvr.VRCompositor().waitGetPoses(poses, None)
    for tracker in trackers:
    
        mat = poses[tracker.index].mDeviceToAbsoluteTracking
        printPose(mat, tracker)
        
if __name__ == "__main__":
    get_trackers()
    while True:
        getPoses()