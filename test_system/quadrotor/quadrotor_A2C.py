#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
import matplotlib.animation as animation
#get_ipython().run_line_magic('matplotlib', 'inline')
#import gym
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import copy
from quadrotor import QuadRotorEnv 


# In[2]:
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)



# In[3]:

'''Global Variables'''
GAMMA = 0.99                # 시간할인율
MAX_STEPS = 2000             # 1에피소드 당 최대 단계 수 (0.01 second per step)
NUM_EPISODES = 10000         # 최대 에피소드 수

NUM_PROCESSES = 32          # 동시 실행 환경 수
NUM_ADVANCED_STEP = 10       # 총 보상을 계산할 때 Advantage 학습(action actor)을 할 단계 수

VALUE_LOSS_COEFF = 0.5
ENTROPY_COEFF = 0.01        # Local min 에서 벗어나기 위한 엔트로피 상수
MAX_GRAD_NORM = 0.5


# In[4]:


class RolloutStorage(object):
    '''Advantage 학습에 사용할 메모리 클래스'''
    def __init__(self, num_steps: int, num_processes: int, obs_shape: int) -> None:
        '''initialize tensors for work'''
        self.observations = torch.zeros(num_steps + 1, num_processes, obs_shape).to(device)        # obs tensor init
        self.masks = torch.ones(num_steps + 1, num_processes, 1).to(device)                        # mask -> is end of episode ? 0 | 1
        self.rewards = torch.zeros(num_steps, num_processes, 1).to(device)  
        self.actions = torch.zeros(num_steps, num_processes, 1).long().to(device)

        # 할인 총 보상(J(theta,s_t))저장하는 메모리 (Actor)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1).to(device)
        self.index = 0              # insert 하는 index          

    def insert(self, current_obs: tensor, action: tensor, reward: tensor, mask: FloatTensor) -> None:
        '''현재 인덱스 위치에 transition을 저장'''
        self.observations[self.index + 1].copy_(current_obs)        # torch.size([6, 32, 12])
        self.masks[self.index + 1].copy_(mask)
        self.rewards[self.index].copy_(reward)
        self.actions[self.index].copy_(action)

        self.index = (self.index + 1) % NUM_ADVANCED_STEP  # 인덱스 값 업데이트

    def after_update(self):
        '''Advantage학습 단계만큼 단계가 진행되면 가장 새로운 transition을 index0에 저장'''
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value):
        '''Advantage학습 범위 안의 각 단계에 대해 할인 총보상(J)을 계산'''

        # 주의 : 5번째 단계부터 거슬러 올라오며 계산
        # 주의 : 5번째 단계가 Advantage1, 4번째 단계는 Advantage2가 됨
        self.returns[-1] = next_value
        for ad_step in reversed(range(self.rewards.size(0))):
            # advantage 고려 Q함수의 수정
            self.returns[ad_step] = self.returns[ad_step + 1] * GAMMA * self.masks[ad_step + 1] + self.rewards[ad_step]


# In[5]:


class Net(nn.Module):
    def __init__(self, n_in: int, n_mid: int, n_out: int) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in,n_mid)
        self.fc2 = nn.Linear(n_mid,n_mid)
        self.actor = nn.Linear(n_mid, n_out)        # Actor net(8 action output)
        self.critic = nn.Linear(n_mid, 1)           # critic net (state value -> 1)
    
    def forward(self, x) -> list:
        '''Forward wave 계산을 정의'''
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        critic_output = self.critic(h2)
        actor_output = self.actor(h2)

        return critic_output, actor_output
    
    def act(self, x) -> Tensor:
        '''상태 x로부터 행동을 확률적으로 결정'''
        value, actor_output = self(x)
        action_probs = F.softmax(actor_output, dim=1)       # softmax 를 통한 행동의 확률 계산
        action = action_probs.multinomial(num_samples=1)
        return action

    def get_value(self, x):
        '''상태 x로부터 상태가치를 계산'''
        value, actor_output = self(x)

        return value
    
    def evaluate_actions(self, x: tensor, actions) -> list:
        '''상태 x로부터 상태가치, 실제 행동 actions의 로그 확률, 엔트로피를 계산'''
        value, actor_output = self(x)

        log_probs = F.log_softmax(actor_output, dim=1)      # dim=1이므로 행동의 종류에 대해 확률을 계산
        action_log_probs = log_probs.gather(1, actions)    # 실제 행동의 로그 확률(log_probs)을 구함

        probs = F.softmax(actor_output, dim=1)
        entropy = -(log_probs * probs).sum(-1).mean()

        return value, action_log_probs, entropy


# In[6]:


class Brain(object):
    def __init__(self, actor_critic: Net) -> None:
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.01)    # learning rate = 0.01 (faster)
        
    def update(self, rollouts: RolloutStorage) -> None:
        ''''Advantage학습의 대상이 되는 5단계 모두를 사용하여 수정'''
        obs_shape = rollouts.observations.size()[2:]    # torch.Size([12])
        num_steps = NUM_ADVANCED_STEP
        num_processes = NUM_PROCESSES

        values, action_log_probs, entropy = self.actor_critic.evaluate_actions(
            rollouts.observations[:-1].view(-1,12),
            rollouts.actions.view(-1,1)
        )

        # 주의 : 각 변수의 크기
        # rollouts.observations[:-1].view(-1, 4) torch.Size([192, 12])
        # rollouts.actions.view(-1, 1) torch.Size([192, 8])
        # values torch.Size([80, 1])
        # action_log_probs torch.Size([80, 1])
        # entropy torch.Size([])

        values = values.view(num_steps, num_processes, 1)   # torch.Size([5, 32, 1])
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        # advantage(행동가치-상태가치) 계산
        advantages = rollouts.returns[:-1] - values          # torch.Size([6, 32, 1])

        # Critic loss 계산
        value_loss = advantages.pow(2).mean()

        # Actor의 gain 계산, 나중에 -1을 곱하면 loss가 된다
        action_gain = (action_log_probs * advantages.detach()).mean()
        # detach 메서드를 호출하여 advantages를 상수로 취급

        # 오차함수의 총합
        total_loss = (value_loss * VALUE_LOSS_COEFF - action_gain - entropy * ENTROPY_COEFF)
        
        # 가중치 수정
        self.actor_critic.train()
        self.optimizer.zero_grad()
        total_loss.backward()
        # 결합 가중치가 한번에 너무 크게 변화하지 않도록, 경사를 0.5 이하로 제한함(클리핑)
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), MAX_GRAD_NORM)

        self.optimizer.step()


# In[7]:


class Environment:
    def run(self) -> None:
        '''running entry point'''
        # 동시 실행할 환경 수 만큼 env를 생성
        space_lim = [500, 500, 500]
        envs = [QuadRotorEnv(space_lim) for i in range(NUM_PROCESSES)]

        # 모든 에이전트가 공유하는 Brain 객체를 생성
        n_in = envs[0].observation_space_size           # state inputs
        n_out = envs[0].action_space_size               # action outpus
        n_mid = 192                                      # 384 mid junction
        actor_critic = Net(n_in, n_mid, n_out).to(device)          # Net init
        glob_brain = Brain(actor_critic)                # Brain init

        # 각종 정보를 저장하는 변수
        l_arm = 0.3                             # length or the rotor arm [m]
        m = 1.3                                 # mass of vehicle [kg]
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
        obs_shape = n_in
        current_obs = torch.zeros(NUM_PROCESSES, obs_shape).to(device)                         # (16,4) 의 tensor
        rollouts = RolloutStorage(NUM_ADVANCED_STEP, NUM_PROCESSES,obs_shape)       # RolloutStorage init
        episode_rewards = torch.zeros(NUM_PROCESSES, 1)                             # 현재 episode 의 reward
        final_rewards = torch.zeros(NUM_PROCESSES, 1)                               # 마지막 episode 의 reward
        obs_np = np.zeros([NUM_PROCESSES, obs_shape])                               # state 배열
        reward_np = np.zeros([NUM_PROCESSES, 1])                                    # 보상의 배열
        done_np = np.zeros([NUM_PROCESSES, 1])                                      # Done 여부의 배열
        done_info_np = np.zeros([NUM_PROCESSES, 3])                                    # Arrive 여부의 배열
        arrive_time = []                                                            # Arrive time 의 저장 버퍼
        distance_np = np.zeros([NUM_PROCESSES, 1])                                  # check distance array
        each_step = np.zeros(NUM_PROCESSES)                                         # 각 env 의 step record
        obs_replay_buffer = np.zeros([NUM_PROCESSES, 12000, 12])
        obs_step = np.zeros([NUM_PROCESSES, 12])
        episode = 0

        # 초기 state...
        obs = [envs[i].reset(p) for i in range(NUM_PROCESSES)]
        obs = np.array(obs)
        obs = torch.from_numpy(obs).float()                     # (32,12) 의 tensor
        current_obs = obs                                       # current obs 의 업데이트

        # advanced 학습(action actor)에 사용되는 객체 rollouts 첫번째 상태에 현재 상태를 저장
        rollouts.observations[0].copy_(current_obs)

        # 에피소드 반복문
        for episode in range(NUM_PROCESSES * NUM_EPISODES):
            # advanced 학습(action actor) 대상이 되는 각 단계에 대해 계산 (step 반복)
            for step in range(NUM_ADVANCED_STEP):
                # action 을 fetch
                with torch.no_grad():
                    action = actor_critic.act(rollouts.observations[step])
                # (16,1)→(16,) -> tensor를 NumPy변수로
                actions = action.squeeze(1).cpu().numpy()

                # process 반복
                for i in range(NUM_PROCESSES):
                    obs_np[i], reward_np[i], done_np[i], done_info_np[i]\
                       = envs[i].step(actions[i], each_step[i])

                    # state 의 저장 -> replay 위함
                    obs_step[i] = obs_np[i]

                    # episode의 종료가치, state_next를 설정
                    if done_np[i]:          # success or fail
                        print(f'{episode} set, {i} slot: {(each_step[i] + 1)/100} [s]')
                        episode += 1
                        if done_info_np[i,0]:    # done with arrival
                            reward_np[i] = 30
                            #if len(arrive_time) == 0:   # First arrival
                            #    reward_np[i] = 20.0
                            #    arrive_time.append(each_step[i] / 100)
                            #else:
                            #    for j in arrive_time:
                            #        if j > (each_step[i] / 100):    # better record of success
                            #            arrive_time.append(each_step[i] / 100)
                            #            reward_np[i] = 20.0
                            #        else:                           # poor or same performace to last success (probably not needed because of gamma)
                            #            reward_np[i] = 0.0
                        elif done_info_np[i,1]:
                            reward_np[i] = 20
                        elif done_info_np[i,2]:
                            reward_np[i] = 10
                        else:
                            reward_np[i] = -10
                        each_step[i] = 0.0            # step 초기화
                        obs_np[i] = envs[i].reset(p) # 환경 초기화
                    else:                           # 무너지거나 성공도 아님
                        reward_np[i] = 0.0
                        each_step[i] += 1           # 그대로 진행

                # 보상을 tensor로 변환하고, 에피소드의 총보상에 더해줌
                reward = torch.from_numpy(reward_np).float()
                episode_rewards += reward

                # 각 실행 환경을 확인하여 done이 true이면 mask를 0으로, false이면 mask를 1로
                masks = torch.FloatTensor([[0.0] if done_i else [1.0] for done_i in done_np])
                masks_arrive = torch.FloatTensor(
                    [[0.0] if arrive_i else [1.0] for arrive_i in done_info_np[:,0]]
                    )

                # 마지막 에피소드의 총 보상을 업데이트
                final_rewards *= masks      # done이 false이면 1을 곱하고, true이면 0을 곱해 초기화
                # done이 false이면 0을 더하고, true이면 episode_rewards를 더해줌
                final_rewards += (1 - masks)*episode_rewards

                # 에피소드의 총보상을 업데이트
                episode_rewards *= masks

                # 현재 done이 true이면 모두 0으로 
                current_obs *= masks

                # current_obs를 업데이트
                obs = torch.from_numpy(obs_np).float()  # torch.Size([32, 12])
                current_obs = obs  # 최신 상태의 obs를 저장

                # 메모리 객체에 현 단계의 transition을 저장
                rollouts.insert(current_obs, action.data, reward, masks)

                # state record 의 저장 및 초기화
                for i in range(NUM_PROCESSES):
                    if (1-masks[i]):
                        if (masks_arrive[i]):
                            obs_replay_buffer[i] = 0
                    else:
                        obs_replay_buffer[i,int(each_step[i])] = obs_step[i]

            # advanced 학습 for문 끝

            # advanced 학습 대상 중 마지막 단계의 상태로 예측하는 상태가치를 계산

            with torch.no_grad():
                next_value = actor_critic.get_value(rollouts.observations[-1]).detach()
                # next_value는 다음 state의 상태가치
                # rollouts.observations의 크기는 torch.Size([6, 16, 4])
            
            # 모든 단계의 할인총보상을 계산하고, rollouts의 변수 returns를 업데이트
            rollouts.compute_returns(next_value)

            # 신경망 및 rollout 업데이트
            glob_brain.update(rollouts)
            rollouts.after_update()

            # 모든 환경이 성공(도착)
            if final_rewards.sum().numpy() >= NUM_PROCESSES*20:
                print('모든 환경 성공')
                savepath = "./test_system/quadrotor/trained_net/quadrotor.pth"
                torch.save(actor_critic.state_dict(), savepath)

                return obs_replay_buffer
                break
            
        print('MAX Episode에 도달하여 학습이 종료되었습니다. (학습실패)')
        return obs_replay_buffer

class AnimationPlot(animation.TimedAnimation):
    def __init__(self, data):
        self.data = data
        self.t = np.arange(0,120,0.01)

        fig1 = plt.figure(num = 1, figsize=plt.figaspect(0.5))
        for i in range(NUM_PROCESSES):
            ax = fig1.add_subplot(4, 8, i+1, projection='3d')

            self.x = self.data[i,:,9]
            self.y = self.data[i,:,10]
            self.z = self.data[i,:,11]

            ax.set_xlabel('X axis[m]')
            ax.set_ylabel('Y axis[m]')
            ax.set_zlabel('Z axis[m]')
            plt.legend(['Trajectory'])
            plt.grid()

            self.line1 = Line3D([],[],[])
            ax.add_line(self.line1)
            #ax.set_xlim3d(-600,600)
            #ax.set_ylim3d(-600,600)
            #ax.set_zlim3d(0,1000)

            animation.TimedAnimation.__init__(self, fig1, interval=50, blit=True)

    def _draw_frame(self, framedata):
        i = framedata
        head = i - 1
        head_slice = (self.t > self.t[i] - 1.0) & (self.t < self.t[i])

        self.line1.set_data(self.x[:i], self.y[:i])
        self.line1a.set_data(self.x[head_slice], self.y[head_slice])
        self.line1e.set_data(self.x[head], self.y[head])

        self.line2.set_data(self.y[:i], self.z[:i])
        self.line2a.set_data(self.y[head_slice], self.z[head_slice])
        self.line2e.set_data(self.y[head], self.z[head])

        self.line3.set_data(self.x[:i], self.z[:i])
        self.line3a.set_data(self.x[head_slice], self.z[head_slice])
        self.line3e.set_data(self.x[head], self.z[head])

        self._drawn_artists = [self.line1, self.line1a, self.line1e,
                               self.line2, self.line2a, self.line2e,
                               self.line3, self.line3a, self.line3e]
# In[8]:


if __name__ == '__main__':
    quadrotor_env = Environment()
    data = quadrotor_env.run()
    np.save('./test_system/quadrotor/trained_net/flight_data.npy', data)

    t = np.arange(0,120,0.01)

    fig1 = plt.figure(num = 1, figsize=plt.figaspect(0.5))
    for i in range(NUM_PROCESSES):
        ax = fig1.add_subplot(4, 8, i+1, projection='3d')
        #ax.set_xlim3d(-600,600)
        #ax.set_ylim3d(-600,600)
        #ax.set_zlim3d(0,1000)
        plt.plot(data[i,:,9],
                 data[i,:,10],
                 data[i,:,11])
        plt.title(f'3D Trajectory {i} slot')
        plt.xlabel('X axis[m]')
        plt.ylabel('Y axis[m]')
        plt.legend(['Trajectory'])
        plt.grid()

    fig2 = plt.figure(num = 2, figsize=plt.figaspect(0.5))
    for i in range(NUM_PROCESSES):
        ax = fig2.add_subplot(4, 8, i+1)
        plt.plot(data[i,:,9],
                  data[i,:,10])
        plt.title('x-y Postition')
        plt.xlabel('X axis [m]')
        plt.ylabel('Y axis [m]')
        plt.grid()
    
    fig3 = plt.figure(num = 3, figsize=plt.figaspect(0.5))
    for i in range(NUM_PROCESSES):
        ax = fig3.add_subplot(4, 8, i+1)
        plt.plot(t,data[i,:,11])
        plt.title('Height')
        plt.xlabel('Time [s]')
        plt.ylabel('Height [m]')
        plt.grid()
    
    fig4 = plt.figure(num = 4, figsize=plt.figaspect(0.5))
    for i in range(NUM_PROCESSES):
        ax = fig4.add_subplot(4, 8, i+1)
        plt.plot(t,data[i,:,6]*180/np.pi,
                 t,data[i,:,7]*180/np.pi,
                 t,data[i,:,8]*180/np.pi)
        plt.title('Angle')
        plt.xlabel('Time [s]')
        plt.ylabel('Angle [deg]')
        plt.legend(['x(phi)','y(theta)','z(psi)'])
        plt.grid()
    
    fig5 = plt.figure(num = 5, figsize=plt.figaspect(0.5))
    for i in range(NUM_PROCESSES):
        ax = fig5.add_subplot(4, 8, i+1)
        plt.plot(t, data[i,:,0]*180/np.pi,
                 t, data[i,:,1]*180/np.pi,
                 t, data[i,:,2]*180/np.pi)
        plt.title('Angular Velocity(body)')
        plt.xlabel('Time [s]')
        plt.ylabel('Angular Velocity(body) [deg/s]')
        plt.legend(['x(phi)','y(theta)','z(psi)'])
        plt.grid()




