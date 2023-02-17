from pythonosc import osc_message_builder
from pythonosc import osc_bundle_builder
from pythonosc import udp_client
from pythonosc.dispatcher import Dispatcher

class xrutils:
    def __init__(self):
        self.deviceCount = 0
        self.devices = {
            'tracker': [],
            'hmd': [],
            'controller': [],
            'tracking reference': []
        }
        self.client = None
        self.bundle = None
        self.dispatcher = None

    def __enter__(self):
        self.client = udp_client.SimpleUDPClient("127.0.0.1", 7000)
        self.bundle = osc_bundle_builder.OscBundleBuilder(osc_bundle_builder.IMMEDIATELY)
        self.dispatcher = Dispatcher()
        self.dispatcher.map('/*', self.readOsc)
        return self

    def __exit__(self, type, value, traceback):
        pass

    def refreshDevices(self):
        pass

    def readOsc(self, address:str, *args):
        print(f"{address}: {args}")


