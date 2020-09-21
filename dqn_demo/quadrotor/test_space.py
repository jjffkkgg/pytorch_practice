import numpy as np

class Obstacle:
    
    def __init__(self, xlim: int, ylim: int, zlim: int) -> None:
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.num_block = 0
        self.wallcenter = np.zeros([1,3])

    def wall_coord(self, center: array, xlen: float, ylen: float, zlen: float) -> array:
        '''create obstacle with size'''
        err_msg = "wall location out of limit"
        assert (abs(center[0]) + xlen*0.5) <= xlim and \
                (abs(center[1]) + ylen*0.5) <= ylim and \
                (zlen <= zlim) and (zlen > 0), err_msg

        self.wallcenter = np.vstack((self.wallcenter, center))
        self.xlen = xlen
        self.ylen = ylen
        self.zlen = zlen
        
        self.num_block += 1

    def rand_wall_sq(self, num: int) -> array:
        '''create random obstacles'''
        for i in range(num):
            #walls[]

    def getZone(self, sphcenter: array) -> int:
        '''get zone number of sphere'''
        circleX = sphcenter[0]
        circleY = sphcenter[1]
        circleZ = sphcenter[2]
        x = self.wall[:,0]
        y = self.wall[:,1]
        z = self.wall[:,2]
        xZone = 0 if circleX < np.min(x) else (2 if circleX > np.max(x) else 1)
        yZone = 0 if circleY < np.min(y) else (2 if circleY > np.max(y) else 1)
        zZone = 0 if circleZ < np.min(z) else (2 if circleZ > np.max(z) else 1)

        nZone = xZone + 3*yZone + 9*zZone

        return nZone

    def collision(self, center: array, radius: float) -> bool:
        '''check collision between sphere and obstacle'''
        collide = False
        d = []
        zone = np.zeros(self.wallcenter.shape[0])
        zone = self.getZone(center)
        diagxy = ((0.5*self.xlen)**2 + (0.5*self.ylen)**2)**0.5
        diagyz = ((0.5*self.ylen)**2 + (0.5*self.zlen)**2)**0.5
        diagxz = ((0.5*self.xlen)**2 + (0.5*self.zlen)**2)**0.5
        diag3d = ((diagxy**2) + (0.5*self.zlen)**2)**0.5
        for i in range(len(self.wallcenter)):
            d[i] = np.linalg.norm(self.wallcenter - center)

        if (zone == 0 or zone == 2 or zone == 6 or zone == 8 or zone == 18
            or zone == 20 or zone == 24 or zone == 26):
            if d < (radius + diag3d):
                collide = True
        elif (zone == 9 or zone == 11 or zone == 17 or zone == 15):
            if d < radius + diagxy