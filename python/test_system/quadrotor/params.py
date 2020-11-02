import numpy as np
# import json

''' system parameters '''
l_arm = 0.4                             # length or the rotor arm [m]
m = 1                                # mass of vehicle [kg]
rho = 1.225                             # density of air [kg/m^3]
r = 0.1                                 # radius of propeller
V = 11.1                                # voltage of battery [V]
kV = 1550                               # motor kV constant, [rpm/V]
CT = 1.0e-2                             # Thrust coeff
Cm = 1e-4                               # moment coeff
g = 9.81                                # gravitational constant [m/s^2]
Jx = 0.021                              # Moment of inertia Ixx [kg*m^2]
Jy = 0.021                              # Moment of inertia Iyy [kg*m^2]
Jz = 0.042                              # Moment of inertia Izz [kg*m^2]
p = [m, l_arm, r, rho, V, kV, CT, Cm, g, Jx, Jy, Jz]

arm_angles_deg = [45, -135, -45, 135]
arm_angles_rad = [(np.pi / 180) * i for i in arm_angles_deg]
motor_dirs = [1, 1, -1, -1] 

'''Learning Variables'''
GAMMA = 0.999                # 시간할인율
NUM_EPISODES = 45000         # 최대 에피소드 수
is_resume = True

NUM_PROCESSES = 32          # 동시 실행 환경 수
NUM_ADVANCED_STEP = 20      # 총 보상을 계산할 때 Advantage 학습(action actor)을 할 단계 수

VALUE_LOSS_COEFF = 0.5
ENTROPY_COEFF = 0.1         # Local min 에서 벗어나기 위한 엔트로피 상수
MAX_GRAD_NORM = 0.5
DELTA_T = 0.01
learning_rate = 0.001

''' Trajectory '''
startpoint = np.array([0,0,5])
endpoint = np.array([5,5,30])
arrive_time = 3
hover_time = 2
time = np.arange(0, arrive_time + hover_time, DELTA_T)
ref_trajectory = np.linspace(startpoint, endpoint, int(arrive_time*(1/DELTA_T)))
for _ in range(int(hover_time * (1/DELTA_T))):
    ref_trajectory = np.vstack((ref_trajectory, endpoint))

'''Action control'''
off_dist = 1    # [m]
action_roll = 0.001               # [V]
action_pitch = 0.001
action_yaw = 0.001
action_thrust = 0.1