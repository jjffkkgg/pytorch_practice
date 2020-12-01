﻿import numpy as np
import casadi as ca
import scipy.integrate
from computation import Computation as comp
from test_space import Obstacle
import warnings
import params as par
import itertools

'''GLOBAL VARIABLE'''
DELTA_T = par.DELTA_T 

class QuadRotorEnv:

    def __init__(self, lim: list) -> None:
        # init of system config
        arm_angles_deg = par.arm_angles_deg
        arm_angles_rad = par.arm_angles_rad
        motor_dirs = par.motor_dirs            # motor rotation direction

        # control input dictionary(DELTA_T (s))
        self.steps_beyond_done = None

        action_nothing = np.zeros(4)
        action_roll = np.array([par.action_roll,0,0,0])
        action_pitch = np.array([0,par.action_pitch,0,0])
        action_yaw = np.array([0,0,par.action_yaw,0])
        action_thrust = np.array([0,0,0,par.action_thrust])

        # Action dictionary generation
        # actions = np.vstack((action_roll, 
        #                 action_pitch, 
        #                 action_yaw, 
        #                 action_thrust, 
        #                 -action_roll, 
        #                 -action_pitch, 
        #                 -action_yaw, 
        #                 -action_thrust
        #                 ))
        # action_array = np.zeros(4)
        # self.action_dic = {}
        # for i in range(actions.shape[0]):
        #     for j in range(len(list(itertools.combinations(actions, i+1)))):
        #         action_array = np.vstack((
        #             action_array,
        #             sum(list(itertools.combinations(actions, i+1))[j])
        #             ))
        # action_array = np.unique(action_array, axis=0)
        # 
        # for k in range(len(action_array)):
        #     self.action_dic[k] = action_array[k]

        self.action_dic = {
            0: -action_roll,
            1: -action_pitch,
            2: -action_yaw,
            3: -action_thrust,
            4: action_roll,
            5: action_pitch,
            6: action_yaw,
            7: action_thrust,
            8: action_nothing
            }

        # state limit of system
        self.done_threshold = [
            ca.pi/2, ca.pi/2, ca.pi,        # [rad/s]
            30,30,30,                       # [m/s]
            5*ca.pi/18, 5*ca.pi/18, 4*ca.pi,      # [rad]
            lim[0],lim[1],lim[2]                    # [m]
            ]

        # Random noise/force/initial state
        start_angle_deg = np.random.rand(3)* par.max_start_angle_deg
        self.start_angle_rad = start_angle_deg*(np.pi/180)


        # start - end
        # self.endpoint = par.endpoint   # [m]
        self.observation_space_size = 12                      # size of state space
        self.action_space_size = len(self.action_dic)         # size of action space

        # defining test space
        # self.test_space = Obstacle(lim)
        # self.test_space.rand_wall_sq(self.endpoint, num=1)

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
        C_nb = comp.euler_to_dcm(euler)                               # from euler to direction cosine matrix
        F_b = ca.vertcat(0, 0, 0)
        F_b = ca.mtimes(C_nb.T, ca.vertcat(0, 0, -m*g))               # Body Force Initialize  
        M_b = ca.SX.zeros(3)                                          # Body moment(torque) Initialize
        u_motor = comp.saturate(
            comp.mix2motor(u_mix), len(motor_dirs)
            )                                                         # convert u_mix(angle input) to motor duty

        # sum up forces and moments
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
            Mi_b_vec = ca.vertcat(0, 0, -motor_dirs[i] * Mi_b)                           # get each rotor's F and M vector
            F_b += Fi_b_vec                                                             # sum up all rotor F vector
            M_b += Mi_b_vec + ca.cross(ri_b, Fi_b_vec)                                  # sum up all rotor M vector & M from each leg

        # Equation of Motion of system
        self.rhs = ca.Function('rhs',[x,u_mix,p],[ca.vertcat(
            ca.mtimes(ca.inv(J_b),
                      M_b - ca.cross(omega_b, ca.mtimes(J_b, omega_b))),                # omega dot (body angular acceleration)
            F_b/m - ca.cross(omega_b,vel_b),                                            # v dot (body acceleration)
            comp.euler_kinematics(euler,omega_b),                                       # omega (angular velocity) (inertial)
            ca.mtimes(C_nb, vel_b),                                                     # v (velocity) (inertial)
            )], ['x','u_mix','p'],['x_dot'])

    def step(self, action: int, step, dt=DELTA_T):
        '''calculate state after step input'''

        err_msg = "%r (%s) invalid" % (action, type(action))
        assert (action in self.action_dic), err_msg                   # throw error if action not in bound

        # init vars
        reward = 0.0
        done = False
        hover = False
        hover_turn = False
        self.u += self.action_dic[action]

        # calculate next step state
        res = scipy.integrate.solve_ivp(
            fun=lambda t, x: np.array(self.rhs(self.xi, self.u, self.p)).reshape(-1),
                t_span=[self.t, self.t+dt], t_eval=[self.t+dt], y0=self.xi
        )

        # ground constraint
        xi_new = res['y']
        if xi_new[11] < 0:
            xi_new[11] = 0
            xi_new[0] = 0
            xi_new[1] = 0
            xi_new[2] = 0
        self.xi = np.array(xi_new).reshape(-1)
        self.t += dt

        # calculate distance to endpoint
        distance_vect = par.ref_point - self.xi[9:12]
        distance = np.linalg.norm(distance_vect)

        omega_b = self.xi[0:3]                      # Angular velocity (body)
        vel_b = self.xi[3:6]                        # Velocity (body)
        euler = self.xi[6:9]                        # Orientation (inertial) = r_nb
        pos_n = self.xi[9:12]                       # Position (inertial)
        C_nb = comp.euler_to_dcm(euler)
        vel_n = ca.mtimes(C_nb, self.xi[3:6])
        # vel_n_size = np.linalg.norm(vel_n)
        # vel_dot = np.dot(distance_vect/distance, vel_n/vel_n_size)           # minimum = -1, maximum = +1
        # vel_diff_vect = distance_vect - vel_n
        # vel_diff = np.linalg.norm(vel_diff_vect)
        
        # done by ground crash
        if self.xi[11] <= 0 and step >= 0.1*(1/DELTA_T):
            done = True
            print('------crashed to ground------')
        
        # done by off trajectory
        # off_dist = par.off_dist
        # if distance >= off_dist:
        #     print(f'------{off_dist}m apart from trajectory------')
        #     done = True
            
        # done by obstacle crash
        # if self.test_space.is_collide(self.xi[9:12], self.radius):
        #     done = True
        #     print('crashed to obstacle')

        # done by exceeding state limit
        for i in range(len(self.xi)):
            if self.xi[i] >= self.done_threshold[i] or\
                self.xi[i] <= -self.done_threshold[i]:
                done = True
                print('------over the limit------')
        
        # success cases
        if step >= (self.time)*(1/dt) - 10:
            if np.linalg.norm(vel_b) <= 0.1:
                print('------hovering------')
                done = True
                hover = True
            else:
                print('------Overtime!------')
                done = True
        
        # exception management
        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # just done with sim!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            warnings.warn(
                'You are calling step() even though this environment'
                'is already done. Please reset before running.'
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return self.xi, self.u, reward, done, np.array([hover, hover_turn]), distance_vect, vel_n.T

    def reset(self, p, hover_time: int):
        '''reset the environment. (init state)'''
        self.xi = [0] * 12
        self.xi[6:9] = self.start_angle_rad
        self.xi[9:12] = par.ref_point
        self.steps_beyond_done = None
        self.p = p
        self.hover_thrust = comp.throttle(p[0]*p[8],p[3],p[2],p[4],p[5],p[6],p[7])
        self.u = np.array([0.0,0.0,0.0,self.hover_thrust * 4])
        self.t = 0
        # self.radius = 2 * self.p[1]
        self.time = hover_time
        # self.trajectory = par.ref_trajectory

        return self.xi
        