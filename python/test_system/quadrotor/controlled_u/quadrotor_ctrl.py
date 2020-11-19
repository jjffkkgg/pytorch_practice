import numpy as np
import casadi as ca
import scipy.integrate
from computation import Computation as comp
from test_space import Obstacle
import warnings
import params as par
import itertools
import control

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

        # start - end
        self.endpoint = par.endpoint   # [m]
        
        self.observation_space_size = 12                      # size of state space
        self.action_space_size = len(self.action_dic)         # size of action space
        
        # reference position
        pos_n_ref = ca.SX.sym('pos_n_ref',3)

        # defining test space
        # self.test_space = Obstacle(lim)
        # self.test_space.rand_wall_sq(self.endpoint, num=1)

        # state (x)
        self.x = ca.SX.sym(
            'x',self.observation_space_size
            )
        omega_b = self.x[0:3]                      # Angular velocity (body)
        vel_b = self.x[3:6]                        # Velocity (body)
        euler = self.x[6:9]                        # Orientation (inertial) = r_nb
        pos_n = self.x[9:12]                       # Position (inertial)

        # input
        n_motors = len(arm_angles_deg)
        self.u_mix = ca.SX.sym('u_mix', 4)           # roll, pitch, yaw, throttle [V]

        # parameters
        self.p = ca.SX.sym('p', 12)
        m = self.p[0]                # mass of the body [kg]
        l_arm = self.p[1]            # length or the rotor arm [m]
        r = self.p[2]                # radius of propeller [m]
        rho = self.p[3]              # density of air [kg/m^3]
        V = self.p[4]                # voltage of battery [V]
        kV = self.p[5]               # motor kV constant, [rpm/V]
        CT = self.p[6]               # Thrust coeff
        Cm = self.p[7]               # moment coeff
        g = self.p[8]                # gravitational constant [m/s^2]
        Jx = self.p[9]
        Jy = self.p[10]
        Jz = self.p[11]

        J_b = ca.diag(ca.vertcat(Jx, Jy, Jz))                         # Moment of inertia of quadrotor

        # forces and moments
        C_nb = comp.euler_to_dcm(euler)                               # from euler to direction cosine matrix
        F_b = ca.vertcat(0, 0, 0)                                     # Body Force Initialize 
        g_b = ca.mtimes(C_nb.T, ca.vertcat(0, 0, -m*g))                
        F_b += g_b   
        M_b = ca.SX.zeros(3)                                          # Body moment(torque) Initialize
        u_motor = comp.saturate(
            comp.mix2motor(self.u_mix), len(motor_dirs)
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
            Mi_b_vec = ca.vertcat(0, 0, motor_dirs[i] * Mi_b)                           # get each rotor's F and M vector
            F_b += Fi_b_vec                                                             # sum up all rotor F vector
            M_b += Mi_b_vec + ca.cross(ri_b, Fi_b_vec)                                  # sum up all rotor M vector & M from each leg

        # Equation of Motion of system
        self.rhs = ca.Function('rhs',[self.x,self.u_mix,self.p],[ca.vertcat(
            ca.mtimes(ca.inv(J_b),
                      M_b - ca.cross(omega_b, ca.mtimes(J_b, omega_b))),                # omega dot (body angular acceleration)
            F_b/m - ca.cross(omega_b,vel_b),                                            # v dot (body acceleration)
            comp.euler_kinematics(euler,omega_b),                                       # omega (angular velocity) (inertial)
            ca.mtimes(C_nb, vel_b),                                                     # v (velocity) (inertial)
            )], ['x','u_mix','p'],['x_dot'])

        vel_ref = pos_n_ref - pos_n
        vel_ref_b = ca.mtimes(C_nb.T, vel_ref)
        acc_ref_b = vel_ref_b - vel_b
        F_ref_b = acc_ref_b * par.m
        F_ref_T_b = F_ref_b - g_b
        vel_err_z_b = F_ref_T_b[2] / par.m

        F_ref_b[2] = 0
        F_Tf_b = F_ref_b - g_b
        F_b_xy = F_b
        F_b_xy[2] = 0
        F_Ti_b = F_b_xy - g_b
        theta_err_b_mag = ca.acos(ca.dot(F_Tf_b, F_Ti_b) / (ca.norm_1(F_Tf_b)*ca.norm_1(F_Ti_b)))
        theta_err_b_hat = ca.cross(F_Ti_b, F_Tf_b) / ca.norm_1(ca.cross(F_Ti_b, F_Tf_b))
        omega_b_ref = theta_err_b_hat * theta_err_b_mag
        omega_b_err = omega_b_ref - omega_b
        
        self.f_control = ca.Function('f_control', [self.x,self.u_mix,self.p,pos_n_ref], [vel_err_z_b, omega_b_err, vel_ref_b, omega_b_ref],
        ['x','u_mix','p','pos_n_ref'],['vel_err_z_b','omega_b_err','vel_ref_b','omega_b_ref'])

    def step(self, action: int, step, dt=DELTA_T):
        '''calculate state after step input'''

        err_msg = "%r (%s) invalid" % (action, type(action))
        assert (action in self.action_dic), err_msg                   # throw error if action not in bound

        # init vars
        reward = 0.0
        done = False
        arrive = False
        arrive_turn = False

        [err_vbz, err_wb, ref_vbz, ref_wb] = self.f_control(self.xi, self.u, self.p_in, self.trajectory[step]+1e-6)
        self.k += self.action_dic[action]
        err_vbz *= self.k[3]
        err_wb *= self.k[0:3]


        # Input calc
        rx = np.dot(self.rollr_sys.A, self.rx_0) + np.dot(self.rollr_sys.B, err_wb[0])
        px = np.dot(self.pitchr_sys.A, self.px_0) + np.dot(self.pitchr_sys.B, err_wb[1])
        yx = np.dot(self.yawr_sys.A, self.yx_0) + np.dot(self.yawr_sys.B, err_wb[2])
        tx = np.dot(self.thrust_sys.A, self.tx_0) + np.dot(self.thrust_sys.B, err_vbz)

        self.u[0] = np.dot(self.rollr_sys.C, rx) + np.dot(self.rollr_sys.D, err_wb[0])
        self.u[1] = np.dot(self.pitchr_sys.C, px) + np.dot(self.pitchr_sys.D, err_wb[1])
        self.u[2] = np.dot(self.yawr_sys.C, yx) + np.dot(self.yawr_sys.D, err_wb[2])
        self.u[3] = np.dot(self.thrust_sys.C, tx) + np.dot(self.thrust_sys.D, err_vbz)

        # calculate next step state
        res = scipy.integrate.solve_ivp(
            fun=lambda t, x: np.array(self.rhs(self.xi, self.u, self.p_in)).reshape(-1),
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
        distance_vect = self.trajectory[step + 1] - self.xi[9:12]
        distance = np.linalg.norm(distance_vect)
        euler = self.xi[6:9]
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
        off_dist = par.off_dist
        if distance >= off_dist:
            print(f'------{off_dist}m apart from trajectory------')
            done = True
            
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
        
        # arrival cases
        if step >= (self.time + self.arr_hover_t)*(1/dt) - 5:
            if distance <= 0.1:
                if np.linalg.norm(vel_n) <= 0.05:
                    if np.linalg.norm(self.xi[0:3]) <= ca.pi/18:        # arrive with stop(hover)
                        print('------arrived!------')
                        arrive = True
                        done = True
                    else:
                        print('------hover with turning------')
                        arrive_turn = True
                        done = True
                else:
                    print('------arrive but not hover------')
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

        return self.xi, self.u, reward, done, np.array([arrive, arrive_turn]), distance_vect, vel_n.T

    def reset(self, p, arrive_time: int, hover_time: int):
        '''reset the environment. (init state)'''
        self.xi = [0] * 12
        self.xi[9:12] = par.startpoint
        self.steps_beyond_done = None
        self.p_in = p
        self.hover_thrust = comp.throttle(p[0]*p[8],p[3],p[2],p[4],p[5],p[6],p[7])
        self.u = np.array([0.0,0.0,0.0,self.hover_thrust * 4])
        self.k = np.array([1.0,1.0,1.0,1.0])
        self.t = 0
        # self.radius = 2 * self.p[1]
        self.time = arrive_time
        self.arr_hover_t = hover_time
        self.trajectory = par.ref_trajectory

        self.eqs = {
            'rhs': self.rhs,
            'x': self.x,
            'u_mix': self.u_mix,
            'p': self.p
        }

        lin_state_func = comp.linearize(self.eqs)
        [A,B,C,D] = lin_state_func(self.xi,self.u,self.p_in)
        sys = control.ss(A,B,C,D)
        tf_sys = control.ss2tf(sys).minreal(tol=1e-6)
        rollr_tf = tf_sys[0,0]
        pitchr_tf = tf_sys[1,1]
        yawr_tf = tf_sys[2,2]
        thrust_tf = tf_sys[5,3]

        TF = [rollr_tf, pitchr_tf, yawr_tf, thrust_tf]
    
        s = control.tf([1,0],[0,1])

        H_rollr = 1
        H_pitchr = 1
        H_yawr = 1
        H_thrustr = 1

        H = [H_rollr, H_pitchr, H_yawr, H_thrustr]

        for i in range(len(TF)):
            TF[i] *= H[i]
            TF[i] = TF[i].feedback()
        
        self.rollr_sys = control.tf2ss(control.c2d(TF[0], DELTA_T))
        self.pitchr_sys = control.tf2ss(control.c2d(TF[1], DELTA_T))
        self.yawr_sys = control.tf2ss(control.c2d(TF[2], DELTA_T))
        self.thrust_sys = control.tf2ss(control.c2d(TF[3], DELTA_T))

        self.rx_0 = np.zeros((self.rollr_sys.A.shape[0],1))
        self.px_0 = np.zeros((self.pitchr_sys.A.shape[0],1))
        self.yx_0 = np.zeros((self.yawr_sys.A.shape[0],1))
        self.tx_0 = np.zeros((self.thrust_sys.A.shape[0],1))

        return self.xi
        