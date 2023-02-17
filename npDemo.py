import math
from time import time
import numpy as np
from numba import njit, cuda


class benchmark:
    def __init__(self, count:int):
        self.count = count

    def __call__(self, func):
        def predicate(*args, **kwargs):
            runtimes = np.zeros((self.count), np.float64)
            for i in range(self.count):
                start = time()
                res = func(*args, **kwargs)
                end = time()
                runtimes[i] = end - start
            print(f"{func.__name__.ljust(20)} | count: {str(self.count).ljust(5)} | mean: {str(round(runtimes.mean(), 2)).ljust(5)} | std: {str(round(np.std(runtimes), 2)).ljust(5)} | min: {str(round(runtimes.min(),2)).ljust(5)} | max: {str(round(runtimes.max(),2)).ljust(5)} |")
            return res
        return predicate

@benchmark(5)
def fillArr():
    return np.random.randint(0,5000, (50000,50000))

@benchmark(50)
def operationArr(arr):
    return arr * 5

@cuda.jit
def gopreationArr(marr):
    x, y = cuda.grid(2)
    if x < marr.shape[0] and y < marr.shape[1]:
       marr[x, y] *= 5

@benchmark(50)
def invokeKernel(arr):
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(arr.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(arr.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    gopreationArr[blockspergrid, threadsperblock](arr)


arr = fillArr()
arr2 = operationArr(arr)

invokeKernel(arr)
