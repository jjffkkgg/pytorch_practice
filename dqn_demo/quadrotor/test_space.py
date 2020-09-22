from __future__ import annotations
import numpy as np

class Obstacle:
    
    def __init__(self, lim: list) -> None:
        self.xlim = lim[0]
        self.ylim = lim[1]
        self.zlim = lim[2]
        self.num_block = 0
        self.wallcenter = np.zeros([1,3])
        self.wallLen = np.zeros([1,3])

    def wall_coord(self, center: array, length: array) -> None:
        '''create obstacle with size'''
        err_msg = "wall location out of limit"
        assert (abs(center[0]) + xlen*0.5) <= xlim and \
                (abs(center[1]) + ylen*0.5) <= ylim and \
                (zlen <= zlim) and (zlen > 0), err_msg

        self.wallcenter = np.vstack((self.wallcenter, center))
        self.wallLen = np.vstack((self.wallLen, length))
        
        self.num_block += 1

    def rand_wall_sq(self, num: int) -> None:
        '''create random 1x1xrandZ obstacles'''
        for i in range(num):
            centerX = np.random.randint(-(self.xlim-1),(self.xlim-1))
            centerY = np.random.randint(-(self.ylim-1),(self.ylim-1))
            centerZ = np.random.randint(-(self.zlim-1),(self.zlim-1))
            randZ = np.random.uniform(0, self.zlim)
            center = np.array([centerX, centerY, centerZ])
            len = np.array([1, 1, randZ])
            self.wall_coord(center, len)


    def is_collide(self, center: array, radius: float) -> bool:
        '''check collision between sphere and obstacle'''
        sphcenter_np = center
        for i in range(len(self.wallcenter) - 1):
            sphcenter_np = np.vstack((sphcenter_np, center))

        minB = self.wallcenter - 0.5*self.wallLen
        maxB = self.wallcenter + 0.5*self.wallLen

        close = np.maximum(minB, np.minimum(sphcenter_np, maxB))
        distance = np.sqrt(np.sum((close - sphcenter_np)**2, axis = 1))

        for i in range(len(self.wallcenter)):
            if distance[i] < radius:
                return True