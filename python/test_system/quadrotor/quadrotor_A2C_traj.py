#!/usr/bin/env python
# coding: utf-8

# In[1]:

from __future__ import annotations
import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from quadrotor_traj import QuadRotorEnv 
import params as par


# In[2]:

'''define device'''
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# In[3]:

'''Global Variables'''
GAMMA = par.GAMMA                               # 시간할인율
NUM_EPISODES = par.NUM_EPISODES                 # 최대 에피소드 수

NUM_PROCESSES = par.NUM_PROCESSES               # 동시 실행 환경 수
NUM_ADVANCED_STEP = par.NUM_ADVANCED_STEP       # 총 보상을 계산할 때 Advantage 학습(action actor)을 할 단계 수

VALUE_LOSS_COEFF = par.VALUE_LOSS_COEFF
ENTROPY_COEFF = par.ENTROPY_COEFF               # Local min 에서 벗어나기 위한 엔트로피 상수
MAX_GRAD_NORM = par.MAX_GRAD_NORM
DELTA_T = par.DELTA_T

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

    def insert(self, current_obs: torch.Tensor, action: torch.Tensor,
                 reward: torch.Tensor, mask: torch.FloatTensor) -> None:
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
        self.actor = nn.Linear(n_mid, n_out)        # Actor net(81 action output)
        self.critic = nn.Linear(n_mid, 1)           # critic net (state value -> 1)
    
    def forward(self, x) -> list:
        '''Forward wave 계산을 정의'''
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        critic_output = self.critic(h2)
        actor_output = self.actor(h2)

        return critic_output, actor_output
    
    def act(self, x) -> torch.Tensor:
        '''상태 x로부터 행동을 확률적으로 결정'''
        value, actor_output = self(x)
        action_probs = F.softmax(actor_output, dim=1)       # softmax 를 통한 행동의 확률 계산
        action = action_probs.multinomial(num_samples=1)
        return action

    def get_value(self, x):
        '''상태 x로부터 상태가치를 계산'''
        value, actor_output = self(x)

        return value
    
    def evaluate_actions(self, x: torch.Tensor, actions) -> list:
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
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=par.learning_rate)    # learning rate -> local minima control
        
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
    def load_ckp(self, checkpoint_fpath, model, optimizer):
        checkpoint = torch.load(checkpoint_fpath)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        return model, optimizer, checkpoint['episode']

    def run(self, arrive_time: int, hover_time: int) -> None:
        '''running entry point'''
        # print device
        print(device)
        
        # 동시 실행할 환경 수 만큼 env를 생성
        space_lim = [500, 500, 500]         # [m]
        envs = [QuadRotorEnv(space_lim) for i in range(NUM_PROCESSES)]

        # 모든 에이전트가 공유하는 Brain 객체를 생성
        n_in = envs[0].observation_space_size           # state inputs
        n_out = envs[0].action_space_size               # action outpus
        n_mid = n_in * n_out                            # 972(12*81) mid junction
        episode = 0
        actor_critic = Net(n_in, n_mid, n_out).to(device)          # Net init
        glob_brain = Brain(actor_critic)                           # Brain init

        # Load saved model to resume learning (comment out to start new training)
        if par.is_resume:
            ckp_path = "./python/test_system/quadrotor/trained_net/A2C_quadrotor.pth"
            actor_critic, glob_brain.optimizer, episode = self.load_ckp(
                ckp_path, actor_critic, glob_brain.optimizer)

        msg = 'Please change NUM_EPISODES to bigger than previous: %i' % episode
        assert (episode < NUM_EPISODES), msg

        # Initialization of variables
        p = par.p
        travel_time = arrive_time + hover_time
        obs_shape = n_in
        current_obs = torch.zeros(NUM_PROCESSES, obs_shape).to(device)                         # (16,4) 의 tensor
        rollouts = RolloutStorage(NUM_ADVANCED_STEP, NUM_PROCESSES,obs_shape)       # RolloutStorage init
        episode_rewards = torch.zeros(NUM_PROCESSES, 1)                             # 현재 episode 의 reward
        final_rewards = torch.zeros(NUM_PROCESSES, 1)                               # 마지막 episode 의 reward
        obs_np = np.zeros([NUM_PROCESSES, obs_shape])                               # state 배열
        reward_np = np.zeros([NUM_PROCESSES, 1])                                    # 보상의 배열
        reward_past_32 = np.zeros(32)                                               # 32개 보상의 평균
        done_np = np.zeros([NUM_PROCESSES, 1])                                      # Done 여부의 배열
        done_info_np = np.zeros([NUM_PROCESSES, 2])                                 # Arrive 여부의 배열
        distance_vect_np = np.zeros([NUM_PROCESSES, 3])                                  # check distance array
        vel_vect_np = np.zeros([NUM_PROCESSES, 3])
        input_np = np.zeros([NUM_PROCESSES, 4])
        each_step = np.zeros(NUM_PROCESSES, dtype=int)                                         # 각 env 의 step record
        step_replay_buffer = np.zeros(NUM_PROCESSES)
        obs_replay_buffer = np.zeros([NUM_PROCESSES, int(travel_time*(1/DELTA_T)), obs_shape])          # state 저장 버퍼
        distance_replay_buffer = np.zeros([NUM_PROCESSES, int(travel_time*(1/DELTA_T))])         # 거리 저장 버퍼
        distance_vect_replay_buffer = np.zeros([NUM_PROCESSES, 3, int(travel_time*(1/DELTA_T))])
        vel_vect_replay_buffer = np.zeros([NUM_PROCESSES, 3, int(travel_time*(1/DELTA_T))])
        reward_replay_buffer = np.zeros([NUM_PROCESSES, int(travel_time*(1/DELTA_T))])           # 보상 저장 버퍼
        input_replay_buffer = np.zeros([NUM_PROCESSES, int(travel_time*(1/DELTA_T)), 4])         # input save buffer
        obs_step = np.zeros([NUM_PROCESSES, 12])
        distance_step = np.zeros([NUM_PROCESSES])
        vel_n_step = np.zeros([NUM_PROCESSES])

        # 초기 state...
        obs = [envs[i].reset(p, arrive_time, hover_time) for i in range(NUM_PROCESSES)]
        obs = np.array(obs)
        obs = torch.from_numpy(obs).float()                     # (32,12) 의 tensor
        current_obs = obs                                       # current obs 의 업데이트

        # Reference trajectory
        ref_trajectory = par.ref_trajectory

        # advanced 학습(action actor)에 사용되는 객체 rollouts 첫번째 상태에 현재 상태를 저장
        rollouts.observations[0].copy_(current_obs)

        # 에피소드 반복문
        while episode <= NUM_EPISODES:
            # advanced 학습(action actor) 대상이 되는 각 단계에 대해 계산 (step 반복)
            for step in range(NUM_ADVANCED_STEP):
                # action 을 fetch
                with torch.no_grad():
                    action = actor_critic.act(rollouts.observations[step])
                # (16,1)→(16,) -> tensor를 NumPy변수로
                actions = action.squeeze(1).cpu().numpy()

                # done_T-F mask init
                masks = torch.FloatTensor()
                masks_arrive = torch.FloatTensor()
                
                # process 반복
                for i in range(NUM_PROCESSES):
                    obs_np[i], input_np[i], reward_np[i], done_np[i],\
                    done_info_np[i], distance_vect_np[i], vel_vect_np[i]\
                       = envs[i].step(actions[i], each_step[i])

                    # train data 의 저장 -> reward & replay 위함
                    obs_step[i] = obs_np[i]
                    distance_step[i] = np.linalg.norm(distance_vect_np[i])
                    vel_n_step[i] = np.linalg.norm(vel_vect_np[i])
                    input_replay_buffer[i, int(each_step[i])] = input_np[i]
                    obs_replay_buffer[i,int(each_step[i])] = obs_np[i]
                    distance_replay_buffer[i,int(each_step[i])] = distance_step[i]
                    distance_vect_replay_buffer[i, :, int(each_step[i])] = distance_vect_np[i]
                    vel_vect_replay_buffer[i,:,int(each_step[i])] = vel_vect_np[i]
                    step_replay_buffer[i] = each_step[i]

                    # episode의 종료가치, state_next를 설정
                    if done_np[i]:          # success or fail
                        mask_step = torch.FloatTensor([[1.0]])
                        print(f'{episode+1} episode, {i+1} slot: {(each_step[i] + 1)/(1/DELTA_T)} [s]')
                        if done_info_np[i,0]:                               # done with arrival
                            masks_arrive_step= torch.FloatTensor([[1.0]])
                            reward_np[i] = 10000.0
                        elif done_info_np[i,1]:                             # done with turn
                            reward_np[i] = 5000.0
                        elif each_step[i] <= (1/DELTA_T)*0.15:              # not lifted up
                            reward_np[i] = -100000
                        elif np.linalg.norm(obs_np[i,9:12] - par.startpoint) <= 0.5:
                            reward_np[i] = -10000000
                        else:
                            reward_replay_buffer[i, each_step[i]] = -10
                            reward_np[i] = reward_replay_buffer[i,:each_step[i]+1].mean() + each_step[i] * DELTA_T
                            masks_arrive_step = torch.FloatTensor([[0.0]])

                        # reward_replay_buffer[i, int(each_step[i])] = reward_np[i]
                        reward_past_32 = np.hstack((reward_past_32[1:],
                                                    reward_replay_buffer[i,:each_step[i]+1].mean()))
                        print(f'slot_reward:    {round(reward_np[i,0],4)}\n'
                              # f'slot_reward:    {round(reward_replay_buffer[i,:each_step[i]+1].mean(),4)}\n'
                              f'done_point:     {np.round(obs_np[i,9:12],3)} m\n'
                              f'done_ref_point: {np.round(ref_trajectory[each_step[i],:],3)} m\n'
                              f'distance:       {round(distance_step[i],4)}m\n' 
                              f'reward_mean:    {round(reward_past_32.mean(),2)}\n'
                              f'reward_max:     {round(reward_past_32.max(),2)}'
                            #   f'reward_min: {round(reward_past_32.min(),2)}'
                              ) 
                        each_step[i] = 0                                        # step 초기화
                        obs_np[i] = envs[i].reset(p, arrive_time, hover_time)   # 환경 초기화
                        reward_replay_buffer[i] = 0
                        obs_replay_buffer[i] = 0
                        distance_replay_buffer[i] = 0
                        distance_vect_replay_buffer[i] = 0
                        vel_vect_replay_buffer[i] = 0
                    else:                           # 비행중
                        mask_step = torch.FloatTensor([[0.0]])
                        masks_arrive_step = torch.FloatTensor([[0.0]])

                        # original
                        # reward_np[i] = 0
                        
                        # vel_vector diff acceleration model
                        # vel_diff_current = np.linalg.norm(distance_vect_replay_buffer[i,:,each_step[i]] - 
                        #                         vel_vect_replay_buffer[i,:,each_step[i]])
                        # vel_diff_past = np.linalg.norm(distance_vect_replay_buffer[i,:,each_step[i]-1] - 
                        #                         vel_vect_replay_buffer[i,:,each_step[i]-1])
                        # vel_diff_dot = (vel_diff_current - vel_diff_past) / DELTA_T
                        # reward_np[i] = -vel_diff_dot

                        # vel_diff minimize model
                        # reward_np[i] = 10 - np.linalg.norm(distance_vect_np[i] - vel_vect_np[i])
                        # reward_np[i] = np.clip(1/(np.linalg.norm(distance_vect_np[i] - vel_vect_np[i])),0,1000)

                        # vel_diff dot product model
                        # vel_n_hat = vel_vect_np[i] / vel_n_step[i]
                        # distance_hat = distance_vect_np[i] / distance_step[i]
                        # reward_np[i] = np.dot(distance_hat, vel_n_hat) *\
                        #      np.clip(abs(1/((distance_step[i] / vel_n_step[i])-1)),0,10000)      # |1/(x-1)| -> argmax=1, saturate over than 100

                        # distance model
                        reward_np[i] = -distance_step[i]
                        # reward_np[i] = np.clip(1/distance_step[i],0,10000)

                        reward_replay_buffer[i, int(each_step[i])] = reward_np[i]
                        reward_np[i] = 0
                        each_step[i] += 1           # 그대로 진행

                    masks = torch.cat((masks, mask_step), dim=0)   
                    masks_arrive = torch.cat((masks_arrive, masks_arrive_step), dim=0)

                # 보상을 tensor로 변환하고, 에피소드의 총보상에 더해줌
                reward = torch.from_numpy(reward_np).float()
                episode_rewards += reward
                
                # 마지막 에피소드의 총 보상을 업데이트
                final_rewards *= (1 - masks)      # done이 false이면 1을 곱하고, true이면 0을 곱해 초기화
                # done이 false이면 0을 더하고, true이면 episode_rewards를 더해줌
                final_rewards += masks*episode_rewards

                # 에피소드의 총보상을 업데이트
                episode_rewards *= (1 - masks)

                # 현재 done이 true이면 모두 0으로 
                current_obs *= (1 - masks)

                # current_obs를 업데이트
                obs = torch.from_numpy(obs_np).float()  # torch.Size([32, 12])
                current_obs = obs  # 최신 상태의 obs를 저장

                # 메모리 객체에 현 단계의 transition을 저장
                rollouts.insert(current_obs, action.data, reward, (1 - masks))

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
            if torch.sum(masks_arrive) == NUM_PROCESSES:
                print('모든 환경 성공')
                savepath = "./python/test_system/quadrotor/trained_net/A2C_quadrotor.pth"
                checkpoint = {
                    'episode': episode,
                    'state_dict': actor_critic.state_dict(),
                    'optimizer': glob_brain.optimizer.state_dict()
                }
                torch.save(checkpoint, savepath)

                return {
                    'obs': obs_replay_buffer,
                    'dist': distance_replay_buffer,
                    'input': input_replay_buffer,
                    'step': each_step,
                    'reward': reward_replay_buffer
                }
            
            episode += 1
            
        print('MAX Episode에 도달하여 학습이 종료되었습니다. (학습실패)')
        savepath = "./python/test_system/quadrotor/trained_net/A2C_quadrotor.pth"
        checkpoint = {
                    'episode': episode,
                    'state_dict': actor_critic.state_dict(),
                    'optimizer': glob_brain.optimizer.state_dict()
                }
        torch.save(checkpoint, savepath)

        return {
                'obs': obs_replay_buffer,
                'dist': distance_replay_buffer,
                'input': input_replay_buffer,
                'step': each_step,
                'reward': reward_replay_buffer
            }