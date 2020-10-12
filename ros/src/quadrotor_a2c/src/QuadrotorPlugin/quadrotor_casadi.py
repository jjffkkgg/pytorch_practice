import casadi as ca
import numpy as np
import sys
sys.path.insert(0,'../../../../../python/test_system/quadrotor')
from computation import Computation as comp

def quadrotor_eqation(jit=True):

    # init of system config
    arm_angles_deg = [45, -135, -45, 135]
    arm_angles_rad = [
        (np.pi / 180) * i for i in arm_angles_deg
        ]
    motor_dirs = [1, 1, -1, -1]             # motor rotation direction

    observation_space_size = 12    # size of state space
    action_space_size = 8          # size of action space

    # state (x)
    x = ca.SX.sym(
        'x',observation_space_size
        )
    omega_b = x[0:3]                      # Angular velocity (body)
    vel_b = x[3:6]                        # Velocity (body)
    euler = x[6:9]                        # Orientation (inertial) = r_nb
    pos_n = x[9:12]                       # Position (inertial)

    # input
    n_motors = len(arm_angles_deg)
    u_mix = ca.SX.sym('u_mix', 4)           # roll, pitch, yaw, throttle [V]

    # parameters
    p = ca.SX.sym('p', 13)
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
    CD0 = p[12]

    J_b = ca.diag(ca.vertcat(Jx, Jy, Jz))                         # Moment of inertia of quadrotor

    # forces and moments
    C_nb = comp.euler_to_dcm(euler)                               # from euler to direction cosine matrix
    F_b = ca.vertcat(0, 0, 0)
    F_b = ca.mtimes(C_nb, ca.vertcat(0, 0, -m*g))               # Body Force Initialize  
    # F_drag = Cd * 
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
        Mi_b_vec = ca.vertcat(0, 0, motor_dirs[i] * Mi_b)                           # get each rotor's F and M vector
        F_b += Fi_b_vec                                                             # sum up all rotor F vector
        M_b += Mi_b_vec + ca.cross(ri_b, Fi_b_vec)                                  # sum up all rotor M vector & M from each leg

    # Force moment Funciton
    force_moment = ca.Function(
        'force_moment',[x,u_mix,p],[F_b,M_b],['x','u_mix','p'],['F_b','M_b'])

    # Equation of Motion of system
    rhs = ca.Function('rhs',[x,u_mix,p],[ca.vertcat(
        ca.mtimes(ca.inv(J_b),
                    M_b - ca.cross(omega_b, ca.mtimes(J_b, omega_b))),                # omega dot (angular acceleration)
        F_b/m - ca.cross(omega_b,vel_b),                                            # v dot (acceleration)
        comp.euler_kinematics(euler,omega_b),                                       # omega (angular velocity) (inertial)
        ca.mtimes(C_nb.T, vel_b),                                                     # v (velocity) (inertial)
        )], ['x','u_mix','p'],['x_dot'])

    return {
        'rhs': rhs,
        'force_moment': force_moment,
        'x': x,
        'u_mix': u_mix,
        'p': p
        }

    
def gazebo_equations():
    omega_INE = ca.SX.sym('omega_INE', 3)
    vel_INE = ca.SX.sym('vel_INE', 3)
    euler_INE = ca.SX.sym('euler_INE', 3)
    pos_INE = ca.SX.sym('pos_INE', 3)
    x_gz = ca.vertcat(omega_INE, vel_INE, euler_INE, pos_INE)
    
    C_nb = comp.euler_to_dcm(euler_INE)
    vel_B = ca.mtimes(C_nb, vel_INE)
    omega_B = ca.mtimes(C_nb, omega_INE)

    x = ca.vertcat(omega_B, vel_B, euler_INE, pos_INE)

    state_from_gz = ca.Function('state_from_gz',[x_gz],[x],['x_gz'],['x'])

    return {
        'state_from_gz': state_from_gz
    }



def code_generation():
    x = ca.SX.sym('x', 12)
    x_gz = ca.SX.sym('x_gz', 12)
    p = ca.SX.sym('p', 13)
    u_mix = ca.SX.sym('u_mix', 4)
    t = ca.SX.sym('t')
    dt = ca.SX.sym('dt')

    eqs = quadrotor_eqation()
    gz_eqs = gazebo_equations()

    f_state = gz_eqs['state_from_gz']

    F_b,M_b = eqs['force_moment'](x,u_mix,p)
    
    f_force_moment = ca.Function('quad_force_moment',
        [x,u_mix,p],[F_b,M_b],['x','u_mix','p'],['F_b','M_b'])

    gen = ca.CodeGenerator(
        'casadi_gen.c',
        {'main': False, 'mex': False, 'with_header': True, 'with_mem': True})
    gen.add(f_state)
    gen.add(f_force_moment)
    gen.generate()

if __name__ == "__main__":
    code_generation()