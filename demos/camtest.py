import cv2

class SafeCapture:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.cap = None

    def __enter__(self):
        self.cap = cv2.VideoCapture(*self.args, **self.kwargs)
        return self.cap
    
    def __exit__(self, type, value, traceback):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("Capture closed")


with SafeCapture(0) as cap:
    while True:
        ret, image = cap.read()

        if ret:
            cv2.imshow("Image", image)
            