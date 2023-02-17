import unittest
from cfglib import *

class Test_file(unittest.TestCase):
    def test_createCFG(self):
        scfg = SceneCFG("create_test")
        if scfg.path.exists():
            scfg.path.unlink()
        with scfg as _:
            pass

    def test_readCFG(self):
        scfg = SceneCFG("read_test")
        if scfg.path.exists():
            scfg.path.unlink()
        with scfg as cfg:
            cfg.params.maxArea = 0

        with scfg as cfg:
            self.assertEquals(cfg.params.maxArea, 0)

    def test_arraysCFG(self):
        scfg = SceneCFG("read_test")
        if scfg.path.exists():
            scfg.path.unlink()
        with scfg as cfg:
            cfg.addCamera(WebCamera(0, 'live'), 'live')
            cfg.addCamera(WebCamera(2, 'k2'), 'k2')

        with scfg as cfg:
            self.assertEquals(len(cfg.cameras), 2)

if __name__ == '__main__':
    unittest.main()

