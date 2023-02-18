import openvr

class xrutils:
    def __init__(self):
        self.hmd = None
        self.dev_class_char = dict()

    def __enter__(self):
        for _ in range(10):
            try:
                self.hmd = openvr.init(openvr.VRApplication_Scene)
                print("Successfully connected to SteamVR")
            except openvr.OpenVRError:
                print("Failed to open SteamVR retrying")
        return self
        
    def __exit__(self, type, value, traceback):
        openvr.shutdown()

    def process_vr_event(self, event):
        if event.eventType == openvr.VREvent_TrackedDeviceDeactivated:
            print(f'Device {event.trackedDeviceIndex} detached')
        elif event.eventType == openvr.VREvent_TrackedDeviceUpdated:
            print(f'Device {event.trackedDeviceIndex} updated')

    def handle_input(self):
        # Note: Key events are handled by glfw in key_callback
        # Process SteamVR events
        event = openvr.VREvent_t()
        has_events = True
        while has_events:
            has_events = self.hmd.pollNextEvent(event)
            self.process_vr_event(event)
        # Process SteamVR action state

    def update_hmd_pose(self):
        if not self.hmd:
            return
        self.poses = self.hmd.getDeviceToAbsoluteTrackingPose(
        openvr.TrackingUniverseStanding,
        0,
        openvr.k_unMaxTrackedDeviceCount)
        self.valid_pose_count = 0
        self.pose_classes = ''
        for nDevice, pose in enumerate(self.poses):
            if pose.bPoseIsValid:
                self.valid_pose_count += 1
                if nDevice not in self.dev_class_char:
                    c = self.hmd.getTrackedDeviceClass(nDevice)
                    if c == openvr.TrackedDeviceClass_Controller:
                        self.dev_class_char[nDevice] = 'C'
                        self.pose_classes += self.dev_class_char[nDevice]
                        print(self.dev_class_char[nDevice])
                    elif c == openvr.TrackedDeviceClass_HMD:
                        self.dev_class_char[nDevice] = 'H'
                        self.pose_classes += self.dev_class_char[nDevice]
                        print(self.dev_class_char[nDevice])
                    # elif c == openvr.TrackedDeviceClass_Invalid:
                    #     self.dev_class_char[nDevice] = 'I'
                    #     continue
                    # elif c == openvr.TrackedDeviceClass_GenericTracker:
                    #     self.dev_class_char[nDevice] = 'G'
                    # elif c == openvr.TrackedDeviceClass_TrackingReference:
                    #     self.dev_class_char[nDevice] = 'T'
                    # else:
                    #     self.dev_class_char[nDevice] = '?'
        print("Done updating poses")