from typing import Any, Callable, Dict, Iterable, List, Tuple
import cv2
import numpy as np
import multiprocessing as mp
from camlib import Camera


camCB = Tuple[Camera, Callable[[np.ndarray],None]]

def grab_and_retrieve(camera_id, q):
    # Initialize the camera
    camera = cv2.VideoCapture(camera_id)
    
    # Start the capture
    
    while True:
        # Grab a new frame
        grabbed = camera.grab()
        
        # Check if the frame was successfully grabbed
        if grabbed:
            # Retrieve the frame
            _, frame = camera.retrieve()
            
            # Put the frame into the queue
            q.put((camera_id, frame))
            
            # Break the loop if the "q" key is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
    # Release the camera
    camera.release()

def display_frames(q, cams:List[int]):
    while True:
        # Get the frames from the queue
        frames = {camera_id: frame for camera_id, frame in [q.get() for _ in cams]}
        
        # Sort the frames by camera ID
        frames = [(camera_id, frames[camera_id]) for camera_id in frames.keys()]
        
        # Process the frames
        # ...
        
        # Show the processed frames
        for i, frame in frames:
            cv2.imshow("Camera {}".format(i), frame)
        
        # Break the loop if the "q" key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

def startCams(cams:List[int]):
    
    # Create a queue to share the frames between processes
    q = mp.Queue()
    
    # Create a process for each camera to grab and retrieve frames
    processes = [mp.Process(target=grab_and_retrieve, args=(camera_id, q)) for camera_id in cams]
    
    # Start the processes
    for process in processes:
        process.start()
    
    # Create a process to display the frames
    display_process = mp.Process(target=display_frames, args=(q, cams))
    
    # Start the display process
    display_process.start()
    
    # Join the processes
    for process in processes + [display_process]:
        process.join()
        
    # Destroy the windows
    cv2.destroyAllWindows()

def _grab_and_retrieve(camera_id:int, q:mp.Queue, proc, data, multiThread:bool = True, camera=None):
    # Initialize the camera
    if camera is None:
        camera = cv2.VideoCapture(camera_id)
    camera.set(cv2.CAP_PROP_EXPOSURE, -14)
    camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    
    # Start the capture
    while True:
        # Grab a new frame
        grabbed = camera.grab()
        
        # Check if the frame was successfully grabbed
        if grabbed:
            # Retrieve the frame
            _, frame = camera.retrieve()

            frame = proc(camera_id, frame, data)
            
            # Put the frame into the queue
            q.put((camera_id, frame))
        if not multiThread:
            return


    # Release the camera
    camera.release()

class MultiCam:
    def __init__(self,cams:List[int], processCB, readCB, datas, multiThread:bool = False):
        self.multiThread = multiThread
        self.cams = cams
        self.procCB = processCB
        self.readCB = readCB
        self._run = False
        self.datas = datas

    def _read(self, q:mp.Queue):
        # Get the frames from the queue
        frames = {camera_id: frame for camera_id, frame in [q.get() for _ in self.cams]}
        
        # Sort the frames by camera ID
        frames = [(camera_id, frames[camera_id]) for camera_id in frames.keys()]

        return self.readCB(frames, self.datas)

    def run(self):
        # Create a queue to share the frames between processes
        q = mp.Queue()
        if self.multiThread:
            # Create a process for each camera to grab and retrieve frames
            processes = [mp.Process(target=_grab_and_retrieve, args=(camera_id, q, self.procCB, self.datas)) for camera_id in self.cams]
            self._run = True
            
            # Start the processes
            for process in processes:
                process.start()
            
            while self._read(q):
                pass
                
            self._run = False
        else:
            cams = {camera_id:cv2.VideoCapture(camera_id) for camera_id in self.cams}
            while True:
                for camera_id in self.cams:
                    _grab_and_retrieve(camera_id, q, self.procCB, self.datas[camera_id], False, cams[camera_id])
                if not self._read(q):
                    break
            for cam in cams.values():
                cam.release()
        cv2.destroyAllWindows()

