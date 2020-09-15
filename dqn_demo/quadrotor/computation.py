import casadi as ca
import numpy as np

class Computation:

    def __init__(self):
        pass

    @staticmethod
    def thrust(throttle, rho, r, V, kV, CT, Cm):
        """
        throttle: 0-1 controls voltage
        rho: density of air, [kg/m^3]
        r: radius of propeller, [m]
        V: voltage of battery [V]
        kV: motor kV constant, [rpm/V]
        """
        omega = throttle*V*kV*(2*ca.pi/60)
        q = 0.5*rho*(omega*r)**2
        s = ca.pi*r**2

        return CT*q*s,Cm*q*s

    @staticmethod
    def euler_kinematics(e, w):
        '''Derivative of Euler angles'''
        v = ca.SX.sym('v',3)
        v[0] = (w[1]*ca.sin(e[2])+w[2]*ca.cos(e[2]))/ca.cos(e[1])
        v[1] = w[1]*ca.cos(e[2]) - w[2]*ca.sin(e[2])
        v[2] = w[0] + (w[1]*ca.sin(e[2]+w[2]*ca.cos(e[2])))*ca.tan(e[1])
        return v

    @staticmethod
    def euler_to_dcm(e):
        ''' Transition from Euler angles to direction cosine matrix'''
        phi = e[0]
        theta = e[1]
        psi = e[2]
        R_z = np.array([
            [ca.cos(psi), -ca.sin(psi), 0],
            [ca.sin(psi), ca.cos(psi), 0],
            [0, 0, 1]
        ])
        R_y = np.array([
            [ca.cos(theta), 0, ca.sin(theta)],
            [0, 1, 0],
            [-ca.sin(theta), 0, ca.cos(theta)]
        ])
        R_x = np.array([
            [1, 0, 0],
            [0, ca.cos(phi), -ca.sin(phi)],
            [0, ca.sin(phi), ca.cos(phi)]
        ])

        R_zy = np.dot(R_z, R_y)
        R = np.dot(R_zy, R_x)
    
        return R

    @staticmethod
    def motor2mix(u_motor):
        '''from each motor duty to effect toward control angles'''
        map = np.array([
            [-1,1,1,-1],
            [1,-1,1,-1],
            [1,1,-1,-1],
            [1,1,1,1]])
        return ca.mtimes(map,u_motor)

        np.array([
                  [1,2,3,4]
        ])

    @staticmethod
    def mix2motor(u_mix):
        '''from input of control to each motor duty'''
        map = np.linalg.inv(np.array([
            [-1,1,1,-1],
            [1,-1,1,-1],
            [1,1,-1,-1],
            [1,1,1,1]]))
        return ca.mtimes(map,u_mix)

    @staticmethod
    def saturate(motor: ca.SX, len: int) -> ca.SX:
        ''' saturate the input motor voltage '''
        for i in range(4):
            temp = motor[i]
            temp = ca.if_else(temp > 1, 1,
                              ca.if_else(temp < 0, 0, temp))
            motor[i] = temp
        return motor