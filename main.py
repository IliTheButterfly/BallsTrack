from os import mkdir, sep
from pathlib import Path
from time import sleep, time
from typing import Any, Callable, Iterable, List, Union

import openvr
from camlib import *
from camlib.cameraSettings import CamSettings
import cv2
import numpy as np
from cfglib import CamCFG, SceneCFG
from tracklib.xrtrack import xrutils
from tracklib.triangulation import SceneSolver
from benchmark import *

# List of commands to display
cmds:List[Callable[[], Any]] = []

class command:
    # Decorator to add the function to the list of commands. The function needs a *args param, the args will be strings
    def __init__(self, name:str, desc:str):
        """_summary_

        Args:
            name (str): Commands name used to call this function (spaces are not allowed). Not case-sensitive.
            desc (str): Brief description of the command. Will be displayed on the help command
        """
        self.name = name
        self.desc = desc

    def __call__(self, func):
        func.name = self.name
        func.desc = self.desc
        cmds.append(func)
        return func

@command("exit", "Quit app")
def exitApp(*args:str):
    """Exits the app
    """
    exit(0)

@command("help", "Show this")
def showHelp(*args:str):
    """Displays the list of commands
    """
    for cmd in cmds:
        print(f"{cmd.name} -> {cmd.desc}")

def getCalib(cam, objp, objpoints, imgpoints, pattern_size, criteria):
    """Prompts the user to calibrate the camera

    Args:
        cam (WebCamera): The WebCamera to calibrate
    Returns:
        (bool, Any, Any, cv2.Mat, cv2.Mat): (res, objpoints, imgpoints, gray, img)
        res (bool): Was the calibration successful
        gray (cv2.Mat): Grayscaled image
        img (cv2.Mat): Image with superimposed grid
    """
    count = 0
    print("Press 'v' to capture an image to calibrate.\nPress 'c' to validate and calibrate the camera.\nPress 'esc' to cancel return to main screen.")
    print("It is preferable to get at least 10 pictures to calibrate.")

    while True:
        # Read key
        k = cv2.waitKey(1)

        # Read camera
        ret, img = cam.read()
        
        # Skip if capture failed
        if not ret or img is None:
            continue

        # Grayscale image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            # Refine the corner positions
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            if k == ord('v'): # Capture
                objpoints.append(objp)
                imgpoints.append(corners)
                count += 1
            
            # Draw the corners on the image
            cv2.drawChessboardCorners(img, pattern_size, corners, ret)

        # Display the captured images count
        cv2.putText(img, f"Image count: {count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (127,255,0))
        cv2.imshow("Calibration pattern", img)
        if k == 27: # Cancel
            return False, objpoints, imgpoints, gray, img
        if k == ord('c'): # Calibrate
            return True, objpoints, imgpoints, gray, img
        

def saveCalib(img, objpoints, imgpoints, gray):

    # Calibrate the camera
    print("Calibrating... This may take a while")
    cv2.destroyAllWindows()
    start = time()
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    end = time()
    print(f"RMS = {ret}")
    print(f"Took {end-start}s")
    sleep(0.2)

    h = 0
    w = 0
    if img is not None:
        h, w = img.shape[:2] # Get camera resolution

    name = input("Enter a name for camera: ")

    # Create a directory to store the camera data
    mkdir(f"cameraData{sep}{name}")
    mFile = f"cameraData{sep}{name}{sep}camera_matrix.npy"
    dFile = f"cameraData{sep}{name}{sep}dist_coeffs.npy"

    # Save the camera matrix and distortion coefficients to .npy files
    np.save(mFile, camera_matrix)
    np.save(dFile, dist_coeffs)
    print("Saved!")
    return h, w, camera_matrix, dist_coeffs, name

def showUndistorted(cam, h, w, camera_matrix, dist_coeffs):
    """Display the distorted and undistored view of the camera for reference

    Args:
        cam (WebCamera): Camera to display
        h (int): Height
        w (int): Width
        camera_matrix (numpy.NDArray): Camera matrix parameters
        dist_coeffs (numpy.NDArray): Camera distortion coefficients

    Returns:
        (bool): Confirm or retry
    """
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    print("Press 'c' to confirm.\nPress 'r' to retry.")
    while True:
        k = cv2.waitKey(1)
        if k == ord('c'):
            return False
        if k == ord('r'):
            return True
            
        ret, img = cam.read()
        if not ret or img is None:
            continue
        
        undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
        cv2.imshow("Distorted image", img)
        cv2.imshow("Undistorted image", undistorted_img)

@command("calib", "calib <id> - Calibrate camera values")
def calibMain(*args:str) -> Union[None, str]:
    """Create a directory and files to store the calibration data of the camera with the corresponding id. The id is either an int or a str (str has not been tested yet).
    This will guide the user to calibrate the camera.

    Returns:
        Union[None, str]: None if id is invalid. Returns the name of the cameras calibration name.
    """
    
    if len(args) != 1:
        print("Usage: calib <id>")
        return None
    # Define the size of the calibration pattern
    pattern_size = (9, 6)

    print(f"Please print the image found in .{sep}calibration_images{sep}pattern.png. Then stick it to a rigid surface to be able to hold it out and keep it straight.")
    print("Then point it at the selected camera and follow the instructions.")

    # Define the termination criteria for the optimization
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare the object points, which are the same for all images
    objp = np.zeros((np.prod(pattern_size), 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

    # Arrays to store the object points and image points from all the images
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane

    try:
        device = int(args[0])
    except ValueError:
        print("Invalid argument")
        print("Usage: calib <id>")
        return None
    
    while True:
        res = True
        with WebCamera(device) as cam:
            res, objpoints, imgpoints, gray, img = getCalib(cam, objp, objpoints, imgpoints, pattern_size, criteria)
        if not res:
            cv2.destroyAllWindows()
            return None
        h, w, camera_matrix, dist_coeffs, name = saveCalib(img, objpoints, imgpoints, gray)
        with WebCamera(device) as cam:
            res &= showUndistorted(cam, h, w, camera_matrix, dist_coeffs)
        if not res:
            cv2.destroyAllWindows()
            return name

def run():
    """Prompts the user to run a command (functions decorated by @command) and executes the correct function
    """
    showHelp()
    while True:
        i = input("Enter command:")
        args = i.split(' ')
        res = False
        for cmd in cmds:
            if args[0].lower() == cmd.name.lower():
                res = True
                if len(args) > 1:
                    cmd(*args[1:])
                else:
                    cmd()


@command("point", "point <cam> - Point finder")
def pointMain(*args:str):
    """Test a single camera to detect a point

    Returns:
        _type_: _description_
    """
    if len(args) != 1:
        print("Usage: point <cam>")
        return
    device = 0
    try:
        device = int(args[0])
    except ValueError:
        print("Invalid argument")
        print("Usage: calib <id>")
        return
    dev = WebCamera(device)

    with dev as cam:
        i = 0
        deltas = np.zeros((30), np.float32)
        def update(_):
            pass

        def updateExp(v):
            cam._cam.set(cv2.CAP_PROP_EXPOSURE, v)

        def updateParams(_):
            global detector
            params = cv2.SimpleBlobDetector_Params()
  
            # Set Area filtering parameters
            params.filterByArea = True
            params.minArea = cv2.getTrackbarPos("Min area", "Params")

            # Set Circularity filtering parameters
            params.filterByCircularity = True 
            params.minCircularity = cv2.getTrackbarPos("Min circl", "Params")/100
            
            # Set Convexity filtering parameters
            params.filterByConvexity = True 
            params.minConvexity = cv2.getTrackbarPos("Min convex", "Params")/100
                
            # Set inertia filtering parameters
            params.filterByInertia = True
            params.minInertiaRatio = cv2.getTrackbarPos("Min inertia", "Params")/100
            detector = cv2.SimpleBlobDetector_create(params)

        cv2.namedWindow("Keypoints")
        cv2.namedWindow("Params")
        cv2.createTrackbar("Thresh", "Params", int(200), 255, update)
        cv2.createTrackbar("Exposure", "Params", int(-7), 12, updateExp)
        cv2.setTrackbarMin("Exposure", "Params", -13)
        cv2.setTrackbarMax("Exposure", "Params", -1)
        cv2.createTrackbar("Min area", "Params", int(100), 500, updateParams)
        cv2.createTrackbar("Min convex", "Params", int(20), 100, updateParams)
        cv2.createTrackbar("Min circl", "Params", int(90), 100, updateParams)
        cv2.createTrackbar("Min inertia", "Params", int(1), 100, updateParams)
        

        params = cv2.SimpleBlobDetector_Params()
  
        # Set Area filtering parameters
        params.filterByArea = True
        params.minArea = 100
        detector = cv2.SimpleBlobDetector_create(params)

        while True:
            start = time.time()
            k = cv2.waitKey(1)
            if k == 27:
                break
            ret, img = cam.read()
            if not ret:
                continue
            
            cv2.imshow("Image", img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, gray = cv2.threshold(gray, cv2.getTrackbarPos("Thresh", "Params"), 255, cv2.THRESH_BINARY_INV)
            
            # Detect blobs.
            try:
                keypoints = detector.detect(gray)
            except:
                continue
            
            # Draw detected blobs as red circles.
            # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
            im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            # Show keypoints
            cv2.imshow("Keypoints", im_with_keypoints)
            i += 1


            end = time.time()
            delta = end - start
            deltas[i%30] = 1/delta
            if i % 30 == 0:
                print(f"Max: {deltas.max()} | Min: {deltas.min()} | Mean: {deltas.mean()}")
        cv2.destroyAllWindows()

@command("cfg", "cfg <cam> - Camera config")
def diffMain(*args:str):
    """BROKEN Suposed to display all available camera properties and allow the user to change the parameter values, it correctly displays the values but changing the values seem to have no effect and the old values come back on the refresh.
    """
    if len(args) != 1:
        print("Usage: cfg <cam>")
        return
    try:
        device = int(args[0])
    except ValueError:
        print("Invalid argument")
        print("Usage: cfg <cam>")
        return
    
    threshType = 0

    cv2.namedWindow("Cam")
    cv2.namedWindow("Params", cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow("Params", 500, 500)

    with WebCamera(device) as cam:
        while not cam.isOpened:
            cam._cam.open(device)
        settings = CamSettings(cam)
        # cam._cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # cam._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        first = True

        def setupParams():
            for prop in dir(settings):
                if prop.startswith('__') or prop.endswith('_info') or prop == 'cam':
                    continue
                val = getattr(settings, prop)
                info = getattr(settings, f'{prop}_info')
                if val == -1:
                    continue

                def updateParam(value):
                    setattr(settings, prop, value)

                cv2.createTrackbar(prop, "Params", int(getattr(settings, prop)), int(info.max - info.min), updateParam)
                cv2.setTrackbarMin(prop, "Params", int(info.min))
                cv2.setTrackbarMax(prop, "Params", int(info.max))
        def update():
            for prop in dir(settings):
                if prop.startswith('__') or prop.endswith('_info') or prop == 'cam':
                    continue
                val = getattr(settings, prop)
                if val == -1:
                    continue

                vv = int(getattr(settings, prop))
                cv2.setTrackbarPos(prop, "Params", vv)

        i = 0
        while True:
            k = cv2.waitKey(1)
            if k == 27:
                break
            ret, img = cam.read()
            if not ret:
                continue
            if first:
                first = False
                a = settings.aperture
                settings.__add_rest__()
                print(str(settings))
                setupParams()

            if i % 30 == 0:
                update()
                
            cv2.imshow("Cam", img)
            i += 1

    cv2.destroyAllWindows()

def cinput(prompt:str, dtype, validate:Union[None,Callable[[Any],bool]]=None):
    """Custom input prompt

    Args:
        prompt (str): Prompt for the input
        dtype (type): Output type for parsing
        validate (Union[None,Callable[[Any],bool]], optional): Function that takes the parsed value and returns if the parsed value is valid. Defaults to None.

    Returns:
        dtype: Parsed and validated input value
    """
    while True:
        i = input(prompt)
        try:
            v = dtype(i)
            if validate is not None:
                if validate(v):
                    return v
                continue
            return v
            
        except ValueError:
            pass
        print('Invalid value (CTRL+D to exit)')

def yesnoValidate(val:str):
    """Validator for a y/n prompt

    Args:
        val (str): Value to validate

    Returns:
        bool: Is valid input
    """
    return val.lower() == 'y' or val.lower() == 'n'

def exitPrompt():
    """Prompt user to exit

    Returns:
        bool: True to exit
    """
    while True:
        try:
            r = cinput('Exit without save? y/n ', str, yesnoValidate).lower()
        except EOFError:
            pass
        if r == 'y':
            return True
        if r == 'n':
            return False

def choice(count:int):
    """Generates a validator for a int choice value from 0 to count-1

    Args:
        count (int): Number of valid inputs
    """
    def validate(val:int):
        if val >= 0 and val < count:
            return True
        else:
            print(f'Value should be between 0 and {count - 1}')
    return validate

def chooseVal(prompt:str, values:List[str]):
    """Multiple choice prompt

    Args:
        prompt (str): Prompt to display
        values (List[str]): Values to choose from

    Returns:
        Union[None,str]: None if canceled. Otherwise, returns the name of the choice
    """
    while True:
        print(prompt)
        for i, v in enumerate(values):
            print(f'  {i}: {v}')
        print(f'  {i+1}: Cancel')
        
        res = cinput("Choose input: ", int, choice(i+1))

        if res == i+1:
            return None
        return values[res]

@command('mkcfg', "Make a camera configuration")
def mkcfgMain(*args:str):
    """Creates or edits a config file containing the information of a group of camera, including their respective calibration data, for use with other commands.

    Returns:
        Union[None, str]: None if canceled. Otherwise, returns the name of the config created.
    """
    while True:
        try:
            name = cinput("Name: ", str)
        except EOFError:
            return

        path = Path(f'configs{sep}{name}.json')
        if path.exists():
            try:
                a = cinput('Config already exists, edit? y/n ', str, yesnoValidate)
            except EOFError:
                return
            if a == 'n':
                continue
            else:
                break
        else:
            break

    scfg = SceneCFG(name)
    with scfg as cfg: # SceneCFG will read or create an config file. Uppon exiting the context, it will automatically save the config unless scfg.dontsave() is called.
        while True:
            print('0: Add cam')
            print('1: Remove cam')
            print('2: Save and exit')
            print('3: Cancel')
            try:
                a = cinput('Action: ', int, choice(4))
            except:
                if exitPrompt():
                    return
            
            if a == 0: # Add cam
                try:
                    id = cinput('Cam ID: ', int)
                except EOFError:
                    continue

                print('0: Calibrate now')
                print('1: Load calibration')
                print('2: Cancel')

                try:
                    a = cinput('Action: ', int, choice(3))
                except EOFError:
                    continue
                if a == 0: # Calibrate now
                    res = calibMain(str(id))
                    if res is None:
                        print("Calib failed")
                        continue
                    cfg.addCamera(WebCamera(id, res), res)
                    continue
                if a == 1: # Load calibration
                    def pathValidate(val:str):
                        try:
                            camera_matrix = np.load(f"cameraData{sep}{val}{sep}camera_matrix.npy")
                            dist_coeffs = np.load(f"cameraData{sep}{val}{sep}dist_coeffs.npy")
                            return True
                        except Exception as e:
                            print("Invalid camera name")
                            return False
                    try:
                        res = cinput('Calib name: ', str, pathValidate)
                        cfg.addCamera(WebCamera(id, res), res)
                    except EOFError:
                        pass
                    continue

                if a == 2: # Cancel
                    continue

            if a == 1: # Remove cam
                if len(cfg.cameras) == 0:
                    print("No cameras")
                    continue
                i = 0
                for i, item in enumerate(cfg.cameras):
                    item:CamCFG
                    print(f'{i}: {item.name} > {item.location}')
                print(f'{i+1}: Cancel')
                a = cinput('Remove: ', int, choice(i+2))
                if a == i+1:
                    continue
                cfg.removeCameraIndex(a)
                continue

            if a == 2: # Save and exit
                return name # scfg will save cfg on exit

            if a == 3: # Cancel

                if exitPrompt():
                    scfg.dontsave() # Instruct not to save on exit
                    return

class Contexts:
    # Allows to open multiple contexts and close each of them on exit
    def __init__(self, contexts:Iterable):
        self.contexts = contexts
    
    def __enter__(self):
        l = list()
        for c in self.contexts:
            l.append(c.__enter__())
        return l

    def __exit__(self, type, value, traceback):
        for c in self.contexts:
            c.__exit__(type, value, traceback)

@command('vrtrack', "vrtrack - Track using specific cams, and allow calibration using OSC")
def vrtrackMain(*args:str):
    """Opens or creates a config and opens the cameras. Connects to openvr to gather controllers position to perform a 3d calibration (camera extrinsics)
    """
    cams = [p.name.removesuffix('.json') for p in Path("configs").iterdir()]
    cams.append('new')

    while True:
        try:
            res = chooseVal("Choose a config or make a new one", cams)
            if res == "new":
                res = mkcfgMain()
                if res is None:
                    continue
                break
            if res is None:
                return
            else:
                break
        except EOFError:
            return

    # Create the config manager
    scfg = SceneCFG(res)

    # Uppon __exit__ this will save the config
    with scfg as cfg:
        
        # Safely open all cameras loaded from config
        with Contexts([c.asWebCam() for c in cfg.cameras]) as cams:
            solver = SceneSolver(cams)

            # Load config
            cfg.params.updateParams(solver.datas.params)
            for data, cam in zip(solver.datas.cams, cfg.cameras):
                data.thresh = cam.threshold
            
            # Setup save callback
            def savecb():
                for cam, data in zip(cfg.cameras, solver.datas.cams):
                    cam.threshold = data.thresh
                cfg.params.update(solver.datas.params)
            scfg.savecb = savecb
            
            # Define callback for params trackbar
            def updateParam(paramName:str, scale:float):
                def up(val):
                    setattr(solver.datas.params, paramName, val*scale)
                return up
            
            # Create Params window
            cv2.namedWindow("Params")
            
            # Define callback for threshold updates
            def createFunc(id:int):
                def updateThresh(val):
                    solver.datas.cams[id].thresh = val
                return updateThresh
            
            # Create camera windows
            for i in range(len(solver.datas.cams)):
                w = f"Camera {i}"
                cv2.namedWindow(w)
                cv2.resizeWindow(w, 640, 360)
                f = createFunc(i)
                cv2.createTrackbar("Thresh", f"Camera {i}", int(solver.datas.cams[i].thresh), 255, f)

            # Define function to create trackbars
            def createTrackbar(name:str, window:str):
                min, max, count = getattr(solver.datas.params, f'{name.removeprefix("min").removeprefix("max")}_info')
                default = getattr(cfg.params, name)
                cv2.createTrackbar(name, window, int(default/((max-min)/count)), count, updateParam(name, (max-min)/count))
            
            # Create trackbars
            for k in dir(solver.datas.params):
                if k.startswith('min') or k.startswith('max'):
                    createTrackbar(k, 'Params')
            # Run the solver
            solver.run()

@command('testvr', "Test getting info for openvr")
def testvrMain(*args:str):
    with xrutils() as xr:
        cv2.namedWindow("Debug")
        devIndex = 0
        # count = openvr.k_unMaxTrackedDeviceCount
        xr.refreshDevices()
        count = len(xr.controllers)
        while True:
            k = cv2.waitKey(1)
            if k == 27:
                break
            if k == ord('s'): # Previous device
                devIndex = devIndex - 1 if devIndex > 0 else count - 1
            if k == ord('w'): # Next device
                devIndex = (devIndex + 1) % count
            xr.update()
            img = np.zeros((640,640), np.uint8) + 255
            dev = xr.controllers[devIndex]
            if dev:
                x = round(dev.pos[0], 3)
                y = round(dev.pos[1], 3)
                z = round(dev.pos[2], 3)
                values = {
                    'index':devIndex, 
                    'id':dev.id, 
                    'x':x,
                    'y':y,
                    'z':z, 
                    'trig':dev.trigger, 
                    'grip':dev.grip, 
                    'state':bin(dev.state), 
                    'state dec':dev.state
                    }
                for i, a in enumerate(dev.axies):
                    values[f'Axis{i}'] = f"({round(a.x, 3)},{round(a.y, 3)})"
                for i, v in enumerate(values.items()):
                    n,v = v
                    p = (10, 25 * i + 30)
                    cv2.putText(img, f"{n}: {v}", p, cv2.FONT_HERSHEY_SIMPLEX, 1, (0))
            cv2.imshow("Debug", img)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()