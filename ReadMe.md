# BallsTrack
Who doesn't like balls?
What if I told you, you could stick some balls on yourself for a small price and in return, you would get to E-R-...
I mean, get FBT (Full Body Tracking) for your VR setup.

This is yet another way to do so.

## Why BallsTrack
Well, currently, it is not in a working state. However the goal is to make FBT available to anyone that has a beefy computer and a 100$-200$ budget for 10 point FBT.

## How does it work?
Simple, get a bunch of cheap webcams and the means to connect them to your PC and get/make the trackers. At the moment, you would start the program, and be filled with disapointment when you realize the tracking algorithm is not even implemented yet.

But esentially, this is how the configuration would work:

1. For each camera you plan on using, you will need to calibrate it to account for lens distorsion. This is what we call the `camera intrinsics`. This helps the program get acurate readings on the 2D positions. Print out the calibration grid found under `calibration_images\pattern.png`. The calibration process will be explained later. **This is a one time step.**

2. Once that's done, there may be a need to account for different latencies between the cameras. For this calibration you will be required to switch on/of the lights in your room. Of couse, the minimmum latency of the tracking system will be dependent on the biggest latency of the cameras you are using. **This will most likely be a one time step.**

3. Now, to calibrate the position of each camera, you will need your VR headset, a VR Controller and a BallMarker. Stick the BallMarker to the Controller and put on your headset. By moving the Controller/BallMarker combo, you will need to move and hold positions around your tracking area in order for the cameras extrapolate their position and orientation in your room. This is what we call the `camera extrinsics`. **This step wiil have to be done evry time you move any camera or if you restart your headset.**

4. Put on the rest of your BallMarkers and have fun!



