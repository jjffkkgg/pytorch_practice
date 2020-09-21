import numpy as np
import casadi as ca
import scipy.integrate
from computation import Computation as comp
import warnings

class QuadRotorEnv:

    def __init__(self):
        arm_angles_deg = [45, -135, -45, 135]
        arm_angles_rad = [
            (np.pi / 180) * i for i in arm_angles_deg
            ]
        motor_dirs = [1, 1, -1, -1]             # motor rotation direction

        self.action_roll = 0.0001               # [V]
        self.action_pitch = 0.0001
        self.action_yaw = 0.001
        self.action_thrust = 0.25
        self.steps_beyond_done = None

        self.done_threshold = [
            ca.pi/2, ca.pi/2, ca.pi,        # [rad/s]
            30,30,30,                       # [m/s]
            ca.pi/4, ca.pi/4, 4*ca.pi,      # [rad]
            500,500,500                    # [m]
            ]

        self.endpoint = np.array([15, 15, 5])   # [m]
        self.arrivetime = 0.0                   # [s]
        
        self.observation_space_size = 12    # size of state space
        self.action_space_size = 8          # size of action space

        # state (x)
        x = ca.SX.sym(
            'x',self.observation_space_size
            )
        omega_b = x[0:3]                      # Angular velocity (body)
        vel_b = x[3:6]                        # Velocity (body)
        euler = x[6:9]                        # Orientation (inertial) = r_nb
        pos_n = x[9:12]                       # Position (inertial)

        # input
        n_motors = len(arm_angles_deg)
        u_mix = ca.SX.sym('u_mix', 4)           # roll, pitch, yaw, throttle [V]

        # parameters
        p = ca.SX.sym('p', 12)
        m = p[0]                # mass of the body [kg]
        l_arm = p[1]            # length or the rotor arm [m]
        r = p[2]                # radius of propeller [m]
        rho = p[3]              # density of air [kg/m^3]
        V = p[4]                # voltage of battery [V]
        kV = p[5]               # motor kV constant, [rpm/V]
        CT = p[6]               # Thrust coeff
        Cm = p[7]               # moment coeff
        g = p[8]                # gravitational constant [m/s^2]
        Jx = p[9]
        Jy = p[10]
        Jz = p[11]

        J_b = ca.diag(ca.vertcat(Jx, Jy, Jz))                         # Moment of inertia of quadrotor

        # forces and moments
        C_bn = comp.euler_to_dcm(euler)                               # from euler to direction cosine matrix
        F_b = ca.vertcat(0, 0, 0)
        F_b = ca.mtimes(C_bn.T, ca.vertcat(0, 0, -m*g))               # Body Force Initialize  
        M_b = ca.SX.zeros(3)                                          # Body moment(torque) Initialize
        u_motor = comp.saturate(
            comp.mix2motor(u_mix), len(motor_dirs)
            )                                                         # convert u_mix(angle input) to motor duty

        for i in range(n_motors):
            ri_b = ca.vertcat(
                l_arm*ca.cos(arm_angles_rad[i]),
                l_arm*ca.sin(arm_angles_rad[i]), 0
            )                                                                           # vector to each motor from center
            Fi_b, Mi_b = comp.thrust(
                throttle = u_motor[i], rho = rho, r = r,
                V = V, kV = kV, CT = CT, Cm = Cm                                        # get scalar F and M of each rotor
            )                                     
            Fi_b_vec = ca.vertcat(0, 0, Fi_b)
            Mi_b_vec = ca.vertcat(0, 0, motor_dirs[i] * Mi_b)                           # get each rotor's F and M vector
            F_b += Fi_b_vec                                                             # sum up all rotor F vector
            M_b += Mi_b_vec + ca.cross(ri_b, Fi_b_vec)                                  # sum up all rotor M vector & M from each leg

        self.rhs = ca.Function('rhs',[x,u_mix,p],[ca.vertcat(
            ca.mtimes(ca.inv(J_b),
                      M_b - ca.cross(omega_b, ca.mtimes(J_b, omega_b))),                # omega dot (angular acceleration)
            F_b/m - ca.cross(omega_b,vel_b),                                            # v dot (acceleration)
            comp.euler_kinematics(euler,omega_b),                                       # omega (angular velocity) (inertial)
            ca.mtimes(C_bn, vel_b),                                                     # v (velocity) (inertial)
            )], ['x','u_mix','p'],['x_dot'])

    def step(self, action, step, dt=0.01):

        action_roll = np.array([self.action_roll,0,0,0])
        action_pitch = np.array([0,self.action_pitch,0,0])
        action_yaw = np.array([0,0,self.action_yaw,0])
        action_thrust = np.array([0,0,0,self.action_thrust])
        action_dic = {
            0: -action_roll,
            1: -action_pitch,
            2: -action_yaw,
            3: -action_thrust,
            4: action_roll,
            5: action_pitch,
            6: action_yaw,
            7: action_thrust
            }
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert (action in action_dic), err_msg                   # throw error if action not in bound

        reward = 0.0
        done = False
        arrive = False
        self.u += action_dic[action]

        res = scipy.integrate.solve_ivp(
            fun=lambda t, x: np.array(self.rhs(self.xi, self.u, self.p)).reshape(-1),
                t_span=[self.t, self.t+dt], t_eval=[self.t+dt], y0=self.xi
        )

        xi_new = res['y']
        if xi_new[11] < 0:         # ground constraint
            xi_new[11] = 0
            xi_new[0] = 0
            xi_new[1] = 0
            xi_new[2] = 0
        self.xi = np.array(xi_new).reshape(-1)
        self.t += dt

        for i in range(len(self.xi)):
            if self.xi[i] >= self.done_threshold[i] or\
                self.xi[i] <= -self.done_threshold[i]:
                done = True
                print('over the limit!')

        if self.xi[11] <= 0 and step >= 100:
            done = True
            print('crashed to ground')
        
        if np.linalg.norm(self.xi[9:12] - self.endpoint) <= 0.5:
            if np.linalg.norm(self.xi[3:6]) <= 0.01:
                arrive = True
                done = True
            else:
                done = True

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            warnings.warn(
                'You are calling step() even though this environment'
                'is already done. Please reset before running.'
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return self.xi, reward, done, arrive

    def reset(self, p):

        self.xi = [0] * 12
        self.steps_beyond_done = None
        self.p = p
        self.u = np.array([0.0,0.0,0.0,0.0])
        self.t = 0

        return self.xi
        