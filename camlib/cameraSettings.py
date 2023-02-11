import cv2
from camlib.webCamera import WebCamera

class propInfo:
    def __init__(self, max:float, min:float = 0, niceName:str=None, unit:str=None, scale:float=1, desc:str=None):
        self.name = niceName
        self.min = min
        self.max = max
        self.unit = unit
        self.scale = scale
        self.desc = desc
        if desc:
            self.__doc__ = desc

    def getPropName(self):
        words = self.name.lower().split(' ')
        prop = ""
        for i, word in enumerate(words):
            cword = word.capitalize() if len(word) <= 1 else word[0].capitalize() + word[1:]
            if i == 0:
                prop += word
            else:
                prop += cword
        return prop

    def __call__(self, func):
        if self.name is None:
            self.name = func.__name__
        if self.desc is None:
            self.desc = func.__doc__

        def predicate(slf, *args, **kwargs):
            n = func.__name__
            if func.__name__.startswith('_'):
                n = self.getPropName()
            setattr(slf, f'{n}_info', self)
            return func(slf,*args,**kwargs)
        return predicate
    
    def __str__(self):
        return f"{self.name} {self.unit if self.unit else ''}"

    def __repr__(self) -> str:
        unit = self.unit if self.unit else ""
        desc = self.desc if self.desc else ""
        return f"{self.name} range:({self.min*self.scale}{unit} - {self.max*self.scale}{unit}) {desc}"




class CamSettings:
    def __init__(self, cam:WebCamera):
        self.cam = cam
    
    def __str__(self):
        ret = ""
        ret += self.__class__.__name__
        for m in dir(self):
            if m.startswith('__') or m.endswith('_info') or m == 'cam':
                continue
            v = str(getattr(self, m))
            ret += f"\n\t{str(getattr(self,f'{m}_info'))}:{v}"
        return ret
    def __repr__(self):
        ret = ""
        ret += self.__class__.__name__
        for m in dir(self):
            if m.startswith('__'):
                continue
            v = str(getattr(self, m))
            ret += f"\n\t{repr(getattr(self,f'{m}_info'))}:{v}"
        return ret

    @classmethod
    def __add_rest__(cls):
        def exists(name:str):
            return name in dir(CamSettings)
        def neatName(name:str):
            name = name.removeprefix('CAP_PROP_').lower()
            words = name.split('_')
            prop = ""
            name = ""
            for i, word in enumerate(words):
                cword = word.capitalize() if len(word) <= 1 else word[0].capitalize() + word[1:]
                if i == 0:
                    prop += word
                    name += cword
                else:
                    prop += cword
                    name += ' ' + word
            return name, prop

        for name, member in cv2.__dict__.items():
            if not name.startswith('CAP_PROP_'):
                continue

            nName, prop = neatName(name)
            if exists(prop):
                continue

            def __get(n, nn, d, p):
                @propInfo(-100,100, nn, desc=d)
                def _get(self):
                    return self.cam.get(getattr(cv2,n))
                _get.__name__ = p
                return _get
            
            def __set(n, p):
                def _set(self, value:float):
                    self.cam.set(getattr(cv2,n), value)
                _set.__name__ = p
                return _set

            setattr(CamSettings, prop, property(__get(name, nName, member.__doc__, prop),__set(name, prop)))
            

    def __clean_props__(self):
        s = dir(self)
        
        for prop in s:
            if prop.startswith('__'):
                continue
            p = eval(f'self.{prop}')
            if p == 0:
                delattr(self,prop)
        

    @property
    @propInfo(32)
    def aperture(self):
        return self.cam.get(cv2.CAP_PROP_APERTURE)
    @aperture.setter
    def aperture(self, value):
        self.cam.set(cv2.CAP_PROP_APERTURE, value)

    @property
    @propInfo(1)
    def autoExposure(self):
        return self.cam.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    @autoExposure.setter
    def autoExposure(self, value):
        self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, value)

    @property
    @propInfo(1)
    def autoWB(self):
        return self.cam.get(cv2.CAP_PROP_AUTO_WB)
    @autoWB.setter
    def autoWB(self, value):
        self.cam.set(cv2.CAP_PROP_AUTO_WB, value)

    @property
    @propInfo(1)
    def autoFocus(self):
        return self.cam.get(cv2.CAP_PROP_AUTOFOCUS)
    @autoFocus.setter
    def autoFocus(self, value):
        self.cam.set(cv2.CAP_PROP_AUTOFOCUS, value)

    @property
    @propInfo(5000)
    def bitrate(self):
        return self.cam.get(cv2.CAP_PROP_BITRATE)
    @bitrate.setter
    def bitrate(self, value):
        self.cam.set(cv2.CAP_PROP_BITRATE, value)

    @property
    @propInfo(500)
    def zoom(self):
        return self.cam.get(cv2.CAP_PROP_ZOOM)
    @zoom.setter
    def zoom(self, value):
        self.cam.set(cv2.CAP_PROP_ZOOM, value)

    @property
    @propInfo(500)
    def brightness(self):
        return self.cam.get(cv2.CAP_PROP_BRIGHTNESS)
    @brightness.setter
    def brightness(self, value):
        self.cam.set(cv2.CAP_PROP_BRIGHTNESS, value)

    @property
    @propInfo(4096)
    def bufferSize(self):
        return self.cam.get(cv2.CAP_PROP_BUFFERSIZE)
    @bufferSize.setter
    def bufferSize(self, value):
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE, value)

    @property
    @propInfo(10)
    def codecPixelFormat(self):
        return self.cam.get(cv2.CAP_PROP_CODEC_PIXEL_FORMAT)
    @codecPixelFormat.setter
    def codecPixelFormat(self, value):
        self.cam.set(cv2.CAP_PROP_CODEC_PIXEL_FORMAT, value)

    @property
    @propInfo(3)
    def channel(self):
        return self.cam.get(cv2.CAP_PROP_CHANNEL)
    @channel.setter
    def channel(self, value):
        self.cam.set(cv2.CAP_PROP_CHANNEL, value)
    
    @property
    @propInfo(500)
    def contrast(self):
        return self.cam.get(cv2.CAP_PROP_CONTRAST)
    @contrast.setter
    def contrast(self, value):
        self.cam.set(cv2.CAP_PROP_CONTRAST, value)
    
    @property
    @propInfo(1)
    def convertRGB(self):
        return self.cam.get(cv2.CAP_PROP_CONVERT_RGB)
    @convertRGB.setter
    def convertRGB(self, value):
        self.cam.set(cv2.CAP_PROP_CONVERT_RGB, value)
    
    @property
    @propInfo(-1, -14)
    def exposure(self):
        return self.cam.get(cv2.CAP_PROP_EXPOSURE)
    @exposure.setter
    def exposure(self, value):
        self.cam.set(cv2.CAP_PROP_EXPOSURE, value)
    
    @property
    @propInfo(500)
    def focus(self):
        return self.cam.get(cv2.CAP_PROP_FOCUS)
    @focus.setter
    def focus(self, value):
        self.cam.set(cv2.CAP_PROP_FOCUS, value)
    
    @property
    @propInfo(10)
    def format(self):
        return self.cam.get(cv2.CAP_PROP_FORMAT)
    @format.setter
    def format(self, value):
        self.cam.set(cv2.CAP_PROP_FORMAT, value)
    
    @property
    @propInfo(10)
    def fourcc(self):
        return self.cam.get(cv2.CAP_PROP_FOURCC)
    @fourcc.setter
    def fourcc(self, value):
        self.cam.set(cv2.CAP_PROP_FOURCC, value)
    
    @property
    @propInfo(10000)
    def frameCount(self):
        return self.cam.get(cv2.CAP_PROP_FRAME_COUNT)
    @frameCount.setter
    def frameCount(self, value):
        self.cam.set(cv2.CAP_PROP_FRAME_COUNT, value)
    
    @property
    @propInfo(60)
    def fps(self):
        return self.cam.get(cv2.CAP_PROP_FPS)
    @fps.setter
    def fps(self, value):
        self.cam.set(cv2.CAP_PROP_FPS, value)
    
    @property
    @propInfo(1080)
    def frameHeight(self):
        return self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    @frameHeight.setter
    def frameHeight(self, value):
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, value)
    
    @property
    @propInfo(1920)
    def frameWidth(self):
        return self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    @frameWidth.setter
    def frameWidth(self, value):
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, value)
    
    @property
    @propInfo(500)
    def gain(self):
        return self.cam.get(cv2.CAP_PROP_GAIN)
    @gain.setter
    def gain(self, value):
        self.cam.set(cv2.CAP_PROP_GAIN, value)
    
    @property
    @propInfo(500)
    def gamma(self):
        return self.cam.get(cv2.CAP_PROP_GAMMA)
    @gamma.setter
    def gamma(self, value):
        self.cam.set(cv2.CAP_PROP_GAMMA, value)
    
    @property
    @propInfo(100)
    def iris(self):
        return self.cam.get(cv2.CAP_PROP_IRIS)
    @iris.setter
    def iris(self, value):
        self.cam.set(cv2.CAP_PROP_IRIS, value)
    
    @property
    @propInfo(1)
    def monochrome(self):
        return self.cam.get(cv2.CAP_PROP_MONOCHROME)
    @monochrome.setter
    def monochrome(self, value):
        self.cam.set(cv2.CAP_PROP_MONOCHROME, value)
    
    @property
    @propInfo(10)
    def mode(self):
        return self.cam.get(cv2.CAP_PROP_MODE)
    @mode.setter
    def mode(self, value):
        self.cam.set(cv2.CAP_PROP_MODE, value)
    
    @property
    @propInfo(1600)
    def isoSpeed(self):
        return self.cam.get(cv2.CAP_PROP_ISO_SPEED)
    @isoSpeed.setter
    def isoSpeed(self, value):
        self.cam.set(cv2.CAP_PROP_ISO_SPEED, value)
    
    @property
    @propInfo(100)
    def roll(self):
        return self.cam.get(cv2.CAP_PROP_ROLL)
    @roll.setter
    def roll(self, value):
        self.cam.set(cv2.CAP_PROP_ROLL, value)
    
    @property
    @propInfo(8000)
    def temperature(self):
        return self.cam.get(cv2.CAP_PROP_TEMPERATURE)
    @temperature.setter
    def temperature(self, value):
        self.cam.set(cv2.CAP_PROP_TEMPERATURE, value)
    
    @property
    @propInfo(100)
    def speed(self):
        return self.cam.get(cv2.CAP_PROP_SPEED)
    @speed.setter
    def speed(self, value):
        self.cam.set(cv2.CAP_PROP_SPEED, value)
    
    @property
    @propInfo(100)
    def hue(self):
        return self.cam.get(cv2.CAP_PROP_HUE)
    @hue.setter
    def hue(self, value):
        self.cam.set(cv2.CAP_PROP_HUE, value)
    
    @property
    @propInfo(100)
    def saturation(self):
        return self.cam.get(cv2.CAP_PROP_SATURATION)
    @saturation.setter
    def saturation(self, value):
        self.cam.set(cv2.CAP_PROP_SATURATION, value)

    @property
    @propInfo(100)
    def pan(self):
        return self.cam.get(cv2.CAP_PROP_PAN)
    @pan.setter
    def pan(self, value):
        self.cam.set(cv2.CAP_PROP_PAN, value)

    @property
    @propInfo(100)
    def tilt(self):
        return self.cam.get(cv2.CAP_PROP_TILT)
    @tilt.setter
    def tilt(self, value):
        self.cam.set(cv2.CAP_PROP_TILT, value)
    
    @property
    @propInfo(100)
    def sharpness(self):
        return self.cam.get(cv2.CAP_PROP_SHARPNESS)
    @sharpness.setter
    def sharpness(self, value):
        self.cam.set(cv2.CAP_PROP_SHARPNESS, value)
    
    @property
    @propInfo(8000)
    def wbTemperature(self):
        return self.cam.get(cv2.CAP_PROP_WB_TEMPERATURE)
    @wbTemperature.setter
    def wbTemperature(self, value):
        self.cam.set(cv2.CAP_PROP_WB_TEMPERATURE, value)
    
    @property
    @propInfo(100)
    def whiteBalanceRedV(self):
        return self.cam.get(cv2.CAP_PROP_WHITE_BALANCE_RED_V)
    @whiteBalanceRedV.setter
    def whiteBalanceRedV(self, value):
        self.cam.set(cv2.CAP_PROP_WHITE_BALANCE_RED_V, value)
    
    @property
    @propInfo(100)
    def whiteBalanceBlueU(self):
        return self.cam.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U)
    @whiteBalanceBlueU.setter
    def whiteBalanceBlueU(self, value):
        self.cam.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, value)
    
CamSettings.__add_rest__()