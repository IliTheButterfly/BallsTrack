import json
import pathlib
from typing import Any, Dict, List, SupportsIndex, Tuple, Union
from camlib import *
from tracklib import *

class CamCFG(object):
    def __init__(self, location:Union[str, int], name:str=None, **kwargs):
        self.location = location
        self.name = name
        self.threshold = 200
        self.__dict__.update(**kwargs)

    def asWebCam(self):
        return WebCamera(self.location, self.name)

class ParamsCFG:
    def __init__(self, **kwargs):
        self.minArea = 200
        self.filterByArea = True
        self.maxArea = 5000
        self.minCircularity = 0.8
        self.filterByCircularity = True
        self.maxCircularity = 1
        self.minConvexity = 0.8
        self.filterByConvexity = False
        self.maxConvexity = 1
        self.minInertiaRatio = 0.0001
        self.maxInertiaRatio = 0.8
        self.__dict__.update(**kwargs)

    def update(self, params:TParams):
        for k, _ in self.__dict__.items():
            v = getattr(params, k)
            setattr(self, k, v)
    
    def updateParams(self, params:TParams):
        for k, _ in self.__dict__.items():
            v = getattr(self, k)
            setattr(params, k, v)

class CFG:
    def __init__(self, **kwargs):
        self.cameras:List[CamCFG] = []
        if 'cameras' in kwargs.keys():
            for cam in kwargs['cameras']:
                self.cameras.append(CamCFG(**cam))
        self.params = ParamsCFG()
        if 'params' in kwargs.keys():
            self.params = ParamsCFG(**kwargs['params'])
    
    def addCamera(self, cam:WebCamera, name:str):
        for c in self.cameras:
            if c.name == name:
                c.location = cam.location
                return
            if c.location == cam.location:
                c.name = name
                return
        self.cameras.append(CamCFG(cam.location, name))
    def removeCameraIndex(self, index:SupportsIndex):
        self.cameras.pop(index)

class SceneCFG:
    def __init__(self, name:str):
        self.path = pathlib.Path(f'configs\\{name}.json')
        self.cfg = CFG()
        self.save = True
        self.savecb = None

    def _encode(self, o:Union[Dict,List,Tuple,str,float,int,bool,Any]) -> Union[Dict,List,Tuple,str,float,int,bool]:
        if isinstance(o, (str,bool,int,float)):
            return o
        elif isinstance(o, (list,tuple)):
            r = []
            for i in o:
                r.append(self._encode(i))
            return r
        elif isinstance(o, dict):
            r = {}
            for k, v in o.items():
                r[k] = self._encode(v)
            return r
        else:
            return self._encode(o.__dict__)

    def dontsave(self):
        self.save = False

    def __enter__(self):
        if self.path.exists():
            with self.path.open('r') as r:
                d = json.load(r)
                self.cfg = CFG(**d)
        return self.cfg
        
    def __exit__(self, type, value, traceback):
        if not self.save:
            print(f"CFG not saved")
            return
        if self.savecb is not None:
            self.savecb()
        if not self.path.exists():
            self.path.touch()
        with self.path.open('w') as w:
            r = self._encode(self.cfg)
            json.dump(r, w, indent=4)
        print(f"CFG saved to {self.path.absolute()}")