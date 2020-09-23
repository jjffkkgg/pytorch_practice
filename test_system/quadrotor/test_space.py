from __future__ import annotations
import numpy as np

class Obstacle:
    
    def __init__(self, lim: list) -> None:
        self.xlim = lim[0]
        self.ylim = lim[1]
        self.zlim = lim[2]
        self.num_block = 0
        self.wallcenter = np.array([10,10,10])      # hardcoded init condition. need fix
        self.wallLen = np.array([1,1,20])

    def wall_coord(self, center: np.array, length: np.array) -> None:
        '''create obstacle with size'''
        err_msg = "wall location out of limit"
        assert (abs(center[0]) + length[0]*0.5) <= self.xlim and \
                (abs(center[1]) + length[1]*0.5) <= self.ylim and \
                (length[2] <= self.zlim) and (length[2] >= 0), err_msg

        self.wallcenter = np.vstack((self.wallcenter, center))
        self.wallLen = np.vstack((self.wallLen, length))
        
        self.num_block += 1

    def rand_wall_sq(self, end: np.array, num: int = 1) -> None:
        '''create random 1x1xrandZ obstacles'''
        for i in range(num):
            while True:
                centerX = np.random.uniform(-(self.xlim-1),(self.xlim-1))
                centerY = np.random.uniform(-(self.ylim-1),(self.ylim-1))
                centerZ = np.random.uniform(0,(self.zlim-1)*0.5)
                if (abs(centerX) > 10 and abs(centerY) > 10 and
                    centerX-end[0] > 10 and centerY-end[1] > 10):           # avoid starting & endpoint obstacle
                    break
            #randZ = np.random.uniform(0, self.zlim)
            center = np.array([centerX, centerY, centerZ])
            len = np.array([1, 1, 2*centerZ])
            self.wall_coord(center, len)


    def is_collide(self, center: np.array, radius: float) -> bool:
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

        return False