import numpy as np

class Obstacle:
    
    def __init__(self, xlim: int, ylim: int, zlim: int) -> None:
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim

    def wall_coord(self, center: array, x: int, y: int, z) -> array:
        err_msg = "wall location out of limit"
        assert (abs(center[0]) + x*0.5) <= xlim and \
                (abs(center[1]) + y*0.5) <= ylim and \
                (z <= zlim) and (z > 0), err_msg

        wall = np.zeros(8,3)
        wall[0] = np.array([center[0] - 0.5*x, center[1] - 0.5*y, 0])
        wall[1] = np.array([center[0] + 0.5*x, center[1] - 0.5*y, 0])
        wall[2] = np.array([center[0] - 0.5*x, center[1] + 0.5*y, 0])
        wall[3] = np.array([center[0] + 0.5*x, center[1] + 0.5*y, 0])
        wall[4] = np.array([center[0] - 0.5*x, center[1] - 0.5*y, z])
        wall[5] = np.array([center[0] + 0.5*x, center[1] - 0.5*y, z])
        wall[6] = np.array([center[0] - 0.5*x, center[1] + 0.5*y, z])
        wall[7] = np.array([center[0] + 0.5*x, center[1] + 0.5*y, z])

        return wall

    def rand_wall_sq(self, num: int) -> array:
        walls = np.zeros(num,8,3)
        for i in range(num):
            #walls[]


    def colision(self, center: array, walls: array, radius: float) -> bool:
        for i in walls:
            if np.linalg.norm(center - walls) < radius:
                return True
        