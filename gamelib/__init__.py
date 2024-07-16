import pygame as pg
import numpy as np

class Scene:
    def __init__(self, w:int=800, h:int=600, fovV:float=np.pi/4):
        self.size = (w,h)
        self.fovV=fovV
        self.fovH = fovV*(w/h)
        self.clock = None
        self.surf = None
        self.cameraPos = np.array([13,.5,2,3.3,0])
        self.points = np.array([[1,1,1,1,1],[4,2,0,1,1],[1,.5,3,1,1]])
        self.triangles = np.array([[0,1,2]])
        
    def __enter__(self):
        pg.init()
        self.screen = pg.display.set_mode(self.size)
        self.running = True
        self.clock = pg.time.Clock()
        self.surf = pg.surface.Surface(self.size)
        return self


    def render(self):
        self.surf.fill([50,127,200])

        for event in pg.event.get():
            if event.type == pg.QUIT: self.running = False
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE: self.running = False

        self.projectPoints()
        for index in range(len(self.triangles)):
            triangle = [self.points[self.triangles[index][0]][3:],
                        self.points[self.triangles[index][1]][3:],
                        self.points[self.triangles[index][2]][3:]]
            color = [255,255,0]
            pg.draw.polygon(self.surf, color, triangle)

    def projectPoints(self):
        camera = self.cameraPos

        for point in self.points:
            hAngleCameraPoint = np.arctan((point[2]-camera[2])/(point[0]-camera[0] + 1e-16))

            if abs(camera[0]+np.cos(hAngleCameraPoint)-point[0]) > abs(camera[0]-point[0]):
                hAngleCameraPoint = (hAngleCameraPoint - np.pi) % (2*np.pi)

            hAngle = (hAngleCameraPoint-camera[3])%(2*np.pi)

            if hAngle > np.pi: hAngle = hAngle - 2*np.pi

            w, h = self.size
            point[3] = w*hAngle/self.fovH + w/2

            distance = np.sqrt((point[0]-camera[0])**2+(point[1]-camera[1])**2+(point[2]-camera[2])**2)

            vAngleCameraPoint = np.arcsin((camera[1]-point[1])/distance)

            vAngle = (vAngleCameraPoint - camera[4])%(2*np.pi)
            
            if vAngle > np.pi: vAngle = vAngle-2*np.pi

            point[4] = h*vAngle/self.fovV + h/2


    def __exit__(self, type, value, traceback):
        pg.quit()

if __name__ == "__main__":
    with Scene() as scene:
        while scene.running:
            scene.render()