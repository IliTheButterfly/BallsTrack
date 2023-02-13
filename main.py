import json
from os import mkdir
from pathlib import Path
from time import sleep
import time
from typing import Any, Callable, Dict, Iterable, List, Tuple, Union
from camlib import *
from camlib.cameraSettings import CamSettings
from camlib.multiCamera import MultiCam, startCams
from camlib.nik2Camera import NIK2Camera
import cv2
import numpy as np
import multiprocessing as mp
from tracklib.utils import blur_edge, deblur, defocus_kernel, differenceFinder, motion_kernel


def stall():
    k = cv2.waitKey(1)
    if k == ord('q'):
        return False
    return True


def openCam(cam: str):
    if cam.startswith('k2'):
        try:
            dev = int(cam.removeprefix('k2'))
        except:
            return None
        return CVK2Camera(dev)
    if cam.startswith('ni'):
        return NIK2Camera()
    else:
        try:
            dev = int(cam)
        except:
            return None
        return WebCamera(dev)

cmds:List[Callable[[], Any]] = []


class command:
    def __init__(self, name:str, desc:str):
        self.name = name
        self.desc = desc

    def __call__(self, func):
        func.name = self.name
        func.desc = self.desc
        cmds.append(func)
        return func

@command("exit", "Quit app")
def exitApp(*args:str):
    exit(0)

@command("help", "Show this")
def showHelp(*args:str):
    for cmd in cmds:
        print(f"{cmd.name} -> {cmd.desc}")

def getCalib(cam, objp, objpoints, imgpoints, pattern_size, criteria):
    count = 0
    while True:
        k = cv2.waitKey(1)
        ret, img = cam.read()
        
        if not ret or img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            if k == ord('v'):
                objpoints.append(objp)
            
            # Refine the corner positions
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            if k == ord('v'):
                imgpoints.append(corners)
                count += 1
            
            # Draw the corners on the image
            cv2.drawChessboardCorners(img, pattern_size, corners, ret)

        cv2.putText(img, f"Image count: {count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (127,255,0))
        cv2.imshow("Calibration pattern", img)
        if k == ord('q'):
            return False, objpoints, imgpoints, gray, img
        if k == ord('c'):
            return True, objpoints, imgpoints, gray, img
        

def saveCalib(img, objpoints, imgpoints, gray):
    # Calibrate the camera
    print("Calibrating... This may take a while")
    cv2.destroyAllWindows()
    start = time.time()
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    end = time.time()
    print(f"RMS = {ret}")

    print(f"Took {end-start}s")
    sleep(0.2)
    cv2.imshow("Calibration pattern", img)
    h = 0
    w = 0
    if img is not None:
        h, w = img.shape[:2]
    name = input("Enter a name for camera: ")

    mkdir(f"cameraData\\{name}")
    mFile = f"cameraData\\{name}\\camera_matrix.npy"
    dFile = f"cameraData\\{name}\\dist_coeffs.npy"
    # pathlib.Path(mFile).touch(os.O_BINARY, True)
    # pathlib.Path(dFile).touch(os.O_BINARY, True)

    # Save the camera matrix and distortion coefficients to .npy files
    np.save(mFile, camera_matrix)
    np.save(dFile, dist_coeffs)
    print("Saved!")
    return h, w, camera_matrix, dist_coeffs, name
def showUndistorted(cam, h, w, camera_matrix, dist_coeffs):
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    while True:
        k = cv2.waitKey(1)
        if k == ord('q'):
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
def calibMain(*args:str):
    if len(args) != 1:
        print("Usage: calib <id>")
        return None
    # Define the size of the calibration pattern
    pattern_size = (9, 6)

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
        


@command("deblur", "Remove blur")
def deblurMain(*args:str):
    with WebCamera(0) as cam:
        while stall():
            ret, img = cam.read()
            if not ret or img is None:
                continue
            imgcp = img.copy()
            imgRes = deblur(125, 0, 700, imgcp)
            


def run():
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


@command("bdeblur", "Deblur motion")
def bdeblurMain(*args:str):
    import sys
    
    try:
        fn = args[0]
    except:
        fn = '.\\text_motion.jpg'

    win = 'deconvolution'
    
    img = cv2.imread(fn, 0)
    if img is None:
        print('Failed to load fn1:', fn)
        sys.exit(1)

    img = np.float32(img)/255.0
    cv2.imshow('input', img)

    img = blur_edge(img)
    IMG = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)

    defocus = 125

    def update(_):
        ang = np.deg2rad( cv2.getTrackbarPos('angle', win) )
        d = cv2.getTrackbarPos('d', win)
        noise = 10**(-0.1*cv2.getTrackbarPos('SNR (db)', win))

        if defocus:
            psf = defocus_kernel(d)
        else:
            psf = motion_kernel(ang, d)
        cv2.imshow('psf', psf)

        psf /= psf.sum()
        psf_pad = np.zeros_like(img)
        kh, kw = psf.shape
        psf_pad[:kh, :kw] = psf
        PSF = cv2.dft(psf_pad, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows = kh)
        PSF2 = (PSF**2).sum(-1)
        iPSF = PSF / (PSF2 + noise)[...,np.newaxis]
        RES = cv2.mulSpectrums(IMG, iPSF, 0)
        res = cv2.idft(RES, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
        res = np.roll(res, -kh//2, 0)
        res = np.roll(res, -kw//2, 1)
        cv2.imshow(win, res)

    cv2.namedWindow(win)
    cv2.namedWindow('psf', 0)
    cv2.createTrackbar('angle', win, int(135), 360, update)
    cv2.createTrackbar('d', win, int(22), 50, update)
    cv2.createTrackbar('SNR (db)', win, int(25), 50, update)
    update(None)

    while True:
        ch = cv2.waitKey()
        if ch == 27:
            break
        if ch == ord(' '):
            defocus = not defocus
            update(None)

@command("diff", "diff <cam> - Displays a difference image from cleanplate")
def diffMain(*args:str):
    if len(args) != 1:
        print("Usage: diff <cam>")
        return
    try:
        device = int(args[0])
    except ValueError:
        print("Invalid argument")
        print("Usage: diff <id>")
        return
    
    threshType = 0

    def update(_):
        pass
    
        

    cv2.namedWindow("Cam")
    cv2.namedWindow("Diff")
    cv2.createTrackbar("Thresh", "Diff", 0, 255, update)

    with WebCamera(device) as cam:
        while not cam.isOpened:
            cam._cam.open(device)
        diff = differenceFinder(cam)
        settings = CamSettings(cam)
        # settings.__clean_props__()
        while True:
            k = cv2.waitKey(1)
            if k == 27:
                break
            
            if k == ord('g'):
                for i in range(5):
                    print(f"Capturing clean plate in {5-i}")
                    sleep(1)
                diff.getCleanPlate()
                print("Captured!")
            
            if k == ord('t'):
                threshType = (threshType + 1) % 8
                if threshType == 5:
                    threshType += 2
            
            ret, img, dimg = diff.read()
            if not ret:
                continue

            if dimg is not None:
                thresh = cv2.getTrackbarPos("Thresh", "Diff")/255

                ret, t = cv2.threshold(dimg, thresh, 1, threshType if threshType != 5 else 16)
                cv2.imshow("Diff", t)
            cv2.imshow("Cam", img)

    cv2.destroyAllWindows()

@command("point", "point <cam> - Point finder")
def pointMain(*args:str):
    if len(args) != 1:
        print("Usage: point <cam>")
        return
    dev = openCam(args[0])
    if dev is None:
        print("Invalid argument")
        print("Usage: point <cam>")
        return


    with dev as cam:
        settings = CamSettings(cam)
        # cam._cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # cam._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
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
        diff = differenceFinder(cam)
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

        # settings.__clean_props__()
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


def mkCls(name:str, val):
    """Make a custom class"""

    class _myClass:
        """Custom class"""
        def __init__(self, n:str):
            self.name = n

        def __call__(self, func):
            """When instance is called (or as a decorator)"""

            # Override the base function
            def predicate(*args, **kwargs):
                # Do stuff
                print(self.name)

                # Return the base function call
                return func(*args, **kwargs)

            # Return the function
            return predicate

    # Add an attribute to the class with a custom name
    setattr(_myClass, name, val)

    # Return the generated class
    return _myClass

# Create the class definition
myclass = mkCls("Hi", "Foo")

# This instanciates the class and then uses __call__ with brainfuck as a parameter
@myclass("Bar")
def brainfuck():
    print("This is weird")


# > Foo
# > Bar
# > This is weird

def cinput(prompt:str, dtype, validate:Union[None,Callable[[Any],bool]]=None):
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

@command('mkcfg', "Make a camera configuration")
def mkcfgMain(*args:str):
    def yesnoValidate(val:str):
        return val.lower() == 'y' or val.lower() == 'n'
    def exitPrompt():
        
        while True:
            try:
                r = cinput('Exit without save? y/n ', str, yesnoValidate).lower()
            except EOFError:
                pass
            if r == 'y':
                return True
            if r == 'n':
                return False
    config = {
        'cams': {}
    }

    while True:
        try:
            name = cinput("Name: ", str)
        except EOFError:
            return

        def choice(count:int):
            def validate(val:int):
                if val >= 0 and val < count:
                    return True
                else:
                    print(f'Value should be between 0 and {count - 1}')
            return validate

        path = Path(f'configs\\{name}.json')
        if path.exists():
            try:
                a = cinput('Config already exists, edit? y/n ', str, yesnoValidate)
            except EOFError:
                return
            if a == 'n':
                continue
            else:
                with path.open('r') as r:
                    config = json.load(r)
                break
        else:
            break

    

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
                if res not in config["cams"].keys():
                    config["cams"][res] = id
                continue
            if a == 1: # Load calibration
                def pathValidate(val:str):
                    try:
                        camera_matrix = np.load(f"cameraData\\{val}\\camera_matrix.npy")
                        dist_coeffs = np.load(f"cameraData\\{val}\\dist_coeffs.npy")
                        return True
                    except Exception as e:
                        print("Invalid camera name")
                        return False
                try:
                    res = cinput('Calib name: ', str, pathValidate)
                    if res not in config["cams"].keys():
                        config["cams"][res] = id
                    continue
                except EOFError:
                    continue
                continue

            if a == 2: # Cancel
                continue

        if a == 1: # Remove cam
            if len(config) == 0:
                print("No cameras")
                continue
            i = 0
            for i, item in enumerate(config["cams"].items()):
                n, id = item
                print(f'{i}: {n} > {id}')
            print(f'{i+1}: Cancel')
            a = cinput('Remove: ', int, choice(i+2))
            if a == i+1:
                continue
            config["cams"].pop(list(config["cams"].keys())[a])
            continue

        if a == 2: # Save and exit
            if not path.exists():
                path.touch()
            with path.open('w') as w:
                json.dump(config, w)
            print(f"Saved to {path.absolute()}")
            return

        if a == 3: # Cancel
            if exitPrompt():
                return


@command('pcalib', "Calibrate camera extrinsics")
def pcalibMain(*args:str):
    pass

class Contexts:
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


def proccb(camera_id:int, frame:cv2.Mat, data):
    global detector
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    t = 200
    try:
        t = data[camera_id]['thresh']
    except KeyError:
        return [None, gray]

    _, gray = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY_INV)
        
    # Detect blobs.
    try:
        keypoints = detector.detect(gray)
    except:
        return [None, gray]
    
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if len(keypoints) > 0:
        keypoints = cv2.KeyPoint_convert(keypoints)
    return [keypoints, im_with_keypoints]

@command('testMulti', "Test mutiple cameras")
def testMulitMain(*args:str):
    procdata:Dict[int, Dict[str,Any]]
    procman = mp.Manager()
    procdata = procman.dict()
    camIDs = []
    for c in args:
        try:
            camIDs.append(int(c))
        except ValueError:
            print(f'Invalid {c}')
            continue
    global detector
    params = cv2.SimpleBlobDetector_Params()

    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea = 100
    params.filterByArea = True
    params.minArea = 100

    # Set Circularity filtering parameters
    params.filterByCircularity = True 
    params.minCircularity = 0.9
    
    # Set Convexity filtering parameters
    params.filterByConvexity = True 
    params.minConvexity = 0.2
        
    # Set inertia filtering parameters
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    detector = cv2.SimpleBlobDetector_create(params)
    

    def update(_):
        pass

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

    
    cv2.namedWindow("Params")
    
    for i in camIDs:
        cv2.namedWindow(f"Cam {i}")
        cv2.createTrackbar("Thresh", f"Cam {i}", int(200), 255, update)
        procdata[i] = procman.dict()
        procdata[i]['thresh'] = 200


    cv2.createTrackbar("Min area", "Params", int(10), 500, updateParams)
    cv2.createTrackbar("Min circl", "Params", int(40), 100, updateParams)
    cv2.createTrackbar("Min convex", "Params", int(20), 100, updateParams)
    cv2.createTrackbar("Min inertia", "Params", int(1), 100, updateParams)
    
    def readcb(frames, datas):
        k = cv2.waitKey(1)
        
        if k == 27:
            return False
        for i, frame in frames:
            try:
                t = cv2.getTrackbarPos("Thresh", f"Cam {i}")
                datas[i]['thresh'] = t
            except:
                datas[i]['thresh'] = 200
            cv2.imshow(f'Cam {i}', frame[1])

        return True

    mcam = MultiCam(camIDs, proccb, readcb, procdata, True)

    # startCams(camIDs)
    mcam.run()

if __name__ == "__main__":
    print(cv2.getBuildInformation())
    run()

