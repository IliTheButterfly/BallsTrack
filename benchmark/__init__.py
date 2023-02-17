from time import time
import numpy as np


class tracker:
    def __init__(self, count:int):
        self.runtimes = np.zeros((count), np.float64)
        self.index = 0
        self.count = count
        self.length = 0

    def __enter__(self):
        self.start = time()
        return self

    def __exit__(self, type, value, traceback):
        self.end = time()
        self.runtimes[self.index] = self.end - self.start
        self.index = (self.index + 1) % self.count
        self.length = max(self.length, self.index)
    
    def reset(self):
        self.runtimes = np.zeros((self.count), np.float64)
        self.index = 0
        self.length = 0

    @property
    def mean(self):
        return self.runtimes[:self.length].mean()

    @property
    def min(self):
        return self.runtimes[:self.length].min()

    @property
    def max(self):
        return self.runtimes[:self.length].max()

    @property
    def std(self):
        return np.std(self.runtimes[:self.length])



class track:
    def __init__(self, count:int):
        self.tracker = tracker(count)

    @property
    def rate(self):
        return 1 / self.tracker.mean

    @property
    def mean(self):
        return self.tracker.mean

    def print(self):
        print(f"fps: {round(self.rate, 5)}")
    
    def reset(self):
        self.tracker.reset()
    





