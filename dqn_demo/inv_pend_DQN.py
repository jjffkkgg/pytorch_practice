#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
env = gym.make('CartPole-v0')
env.reset()
random_episodes = 0
reward_sum = 0
while random_episodes < 10:
    env.render()
    action = env.action_space.sample()
    observation, reward, done, _ = env.step(action)
    print(observation, reward, done)
    reward_sum += reward
    if done:
        random_episodes += 1
        print(f"Reward for this episode was: {reward_sum}")
        reward_sum = 0
        env.reset()
env.close()
'''


# In[10]:


from __future__ import annotations
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import gym


# In[11]:


Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))

# set env with setting (especially reward and max episode)
gym.envs.register(
    id='CartPole_prefer-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=1000,      # CartPole-v0 uses 200
    reward_threshold=-110.0,
)

# 상수 정의
ENV = 'CartPole_prefer-v0'     # 태스크 이름
GAMMA = 0.99            # 시간할인율
MAX_STEPS = 500         # 1에피소드 당 최대 단계 수
NUM_EPISODES = 1000      # 최대 에피소드 수
batch_size = 32
capacity = 10000        # Memory capacity


# In[12]:


# 애니메이션을 만드는 함수
# 참고 URL http://nbviewer.jupyter.org/github/patrickmineault
# /xcorr-notebooks/blob/master/Render%20OpenAI%20gym%20as%20GIF.ipynb
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
from IPython.display import display


def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0),
               dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames),
                                   interval=50)

    anim.save('movie_cartpole_DQN.mp4')  # 애니메이션을 저장하는 부분
    display(display_animation(anim, default_mode='loop'))
    


# In[13]:


class ReplayMemory:
    ''' Memory for random selection of trials '''
    
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity    # Memory capacity
        self.memory = []            # Transition memory
        self.index = 0              # indicate saving location
        
    def push(self, state: torch.FloatTensor, action: torch.LongTensor,
             state_next: torch.FloatTensor, reward: torch.FloatTensor) -> None:
        '''Transition~(s,a,s_n,r) 메모리 저장'''
        if len(self.memory) < self.capacity:                                        # case memory not full
            self.memory.append(None)                                                # increase size of list to avoid index error
        
        self.memory[self.index] = Transition(state, action, state_next, reward)     # add namedtuple Transition
        self.index = (self.index + 1) % self.capacity                               # increase index (keep update over episode)
        
    def sample(self, batch_size: int) -> list:
        '''replay 메모리에서 batch size 만큼 Transition 랜덤 뽑기'''
        return random.sample(self.memory, batch_size)
    
    def __len__(self) -> int:
        '''return saved Transitions'''
        return len(self.memory)
        


# In[14]:


class Network:
    ''' Where deep network formed and optimized'''
    def __init__(self,num_states: int, num_actions: int) -> None:
        '''Initialize network models'''
        self.num_states = num_states
        self.num_actions = num_actions
        
        self.mem = ReplayMemory(capacity)                           # Initialize ReplayMem

        self.model = nn.Sequential()                                # Sequential container of modules(networks) 
        self.model.add_module('fc1', nn.Linear(num_states, 32))     # Fully-connected(FC) model with input of state output of 32
        self.model.add_module('relu1', nn.ReLU())                   # Backward diff wave alg (RELU typically)
        self.model.add_module('fc2', nn.Linear(32, 32))             # Second module, modules can also have CNN, RNN, etc..
        self.model.add_module('relu2', nn.ReLU())
        self.model.add_module('fc3', nn.Linear(32, num_actions))    # Goes through network with output of action

        print(self.model)                                           # print out the model
        
        # Selection of gradient descent model.(i.e. SGD, RMSprop, Adagrad,...)
        # change weight coeffs of network with gradient descent of params
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.0001)
        
    def replay(self) -> None:
        '''REPLAY OF MEMORY & TRAIN OF NETWORK'''
    # -----------------------------------------
    # 1. 저장된 transition 수 확인
    # -----------------------------------------
        # 1.1 저장된 transition의 수가 미니배치 크기보다 작으면 아무 것도 하지 않음
        if len(self.mem) < batch_size:
            return None

    # -----------------------------------------
    # 2. 미니배치 생성
    # -----------------------------------------
        # 2.1 메모리 객체에서 미니배치를 추출
        trans_sample = self.mem.sample(batch_size)

        # 2.2 각 변수를 미니배치에 맞는 형태로 변형
        # trans_sample 은 각 단계 별로 (state, action, state_next, reward) 형태로 BATCH_SIZE 갯수만큼 저장됨
        # 다시 말해, (state, action, state_next, reward) * BATCH_SIZE 형태가 된다
        # 이것을 미니배치로 만들기 위해
        # (state*BATCH_SIZE, action*BATCH_SIZE, state_next*BATCH_SIZE, reward*BATCH_SIZE) 형태로 변환한다
        batch = Transition(*zip(*trans_sample))
    
        # 2.3 각 변수의 요소를 미니배치에 맞게 변형하고, 신경망으로 다룰 수 있도록 Variable로 만든다
        # state를 예로 들면, [torch.FloatTensor of size 1*4] 형태의 요소가 BATCH_SIZE 갯수만큼 있는 형태이다
        # 이를 torch.FloatTensor of size BATCH_SIZE*4 형태로 변형한다
        # 상태, 행동, 보상, non_final 상태로 된 미니배치를 나타내는 Variable을 생성
        # cat은 Concatenates을 의미한다
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_state = torch.cat([s for s in batch.next_state if s is not None])

    # -----------------------------------------
    # 3. 정답신호로 사용할 Q(s_t, a_t)를 계산
    # -----------------------------------------
        # 3.1 신경망을 추론 모드로 전환
        self.model.eval()

        # 3.2 신경망으로 Q(s_t, a_t)를 계산
        # self.model(state_batch)은 왼쪽, 오른쪽에 대한 Q값을 출력하며
        # [torch.FloatTensor of size BATCH_SIZEx2] 형태이다
        # 여기서부터는 실행한 행동 a_t에 대한 Q값을 계산하므로 action_batch에서 취한 행동 a_t가 
        # 왼쪽이냐 오른쪽이냐에 대한 인덱스를 구하고, 이에 대한 Q값을 gather 메서드로 모아온다
        Q = self.model(state_batch).gather(1, action_batch)

        # 3.3 max{Q(s_t+1, a)}값을 계산한다 이때 다음 상태가 존재하는지에 주의해야 한다
        # cartpole이 done 상태가 아니고, next_state가 존재하는지 확인하는 인덱스 마스크를 만듬
        # lamba & map -> https://offbyone.tistory.com/73
        non_final_mask = torch.ByteTensor(tuple(map(lambda s: s is not None, batch.next_state)))

        # 먼저 전체를 0으로 초기화
        next_Q = torch.zeros(batch_size)

        # 다음 상태가 있는 인덱스에 대한 최대 Q값을 구한다
        # 출력에 접근하여 열방향 최대값(max(1))이 되는 [값, 인덱스]를 구한다
        # 그리고 이 Q값(인덱스=0)을 출력한 다음
        # detach 메서드로 이 값을 꺼내온다
        next_Q[non_final_mask] = self.model(non_final_next_state).max(1)[0].detach()
        # 3.4 정답신호로 사용할 Q(s_t, a_t)값을 Q러닝 식으로 계산한다
        expected_Q = reward_batch + GAMMA * next_Q

    # -----------------------------------------
    # 4. 결합 가중치 수정
    # -----------------------------------------
        # 4.1 신경망을 학습 모드로 전환
        self.model.train()

        # 4.2 손실함수를 계산 (smooth_l1_loss는 Huber 함수)
        # expected_Q는 size가 [minibatch]이므로 unsqueeze하여 [minibatch*1]로 만든다
        loss = F.smooth_l1_loss(Q, expected_Q.unsqueeze(1))

        # 4.3 결합 가중치를 수정한다
        self.optimizer.zero_grad()      # Initialize gradient(경사)
        loss.backward()                 # Calculate backward(역전파 계산)
        self.optimizer.step()           # perform single optimize step (가중치 수정)

    def decide_action(self, state: torch.FloatTensor, episode: int) -> torch.LongTensor:
        '''Decision of action with \epsilon greedy'''
        epsilon = 0.5 * (1 / (episode + 1))                         # epsilon decay
        if epsilon <= np.random.uniform(0,1):                           # weighted action
            self.model.eval()                                           # change network mode to inference
            with torch.no_grad():
                action = self.model(state).max(1)[1].view(1,1)
                # 신경망 출력의 최댓값에 대한 인덱스(action) = max(1)[1]
                # .view(1,1)은 [torch.LongTensor of size 1] 을 size 1*1로 변환하는 역할을 한다

        else:                                                           # random action
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]])                 # return 0 or 1 (LongTensor of size 1*1)

        return action


# In[15]:


class Environment:
    ''' Initialize and run the environment '''
    
    def __init__(self) -> None:
        self.env = gym.make(ENV)                                # env set
        num_states = self.env.observation_space.shape[0]        # Get State shape (4)
        num_actions = self.env.action_space.n                   # Get action numbers (2)
        self.network = Network(num_states,num_actions)          # Initialize Network Class

        
    def run(self) -> None:
        '''Run and update iteration'''
        episode_10_list = np.zeros(10)                          # Save steps succeeded for last 10 episodes
        complete_episodes = 0                                   # Episodes number that reached goal
        is_episode_final = False                                # did it succeeded for 10 episodes? (terminate)
        frames = []                                             # frame for animation

        for episode in range(NUM_EPISODES):                         # Episode iteration (single train per episode)
            observation = self.env.reset()                          # reset env (=initialize)
            state = observation                                     # state <- initialized env
            state = torch.from_numpy(state).type(
                torch.FloatTensor)                                  # convert np.array -> FloatTensor
            state = torch.unsqueeze(state, 0)                       # size 4 -> size 1*4 convert

            for step in range(MAX_STEPS):                           # Iterate through max action(or state) per episode
                
                if is_episode_final is True:                              # If the goal reached
                    frames.append(self.env.render(
                        mode = 'rgb_array'))                              # Save result of final episode with frame
                
                action = self.network.decide_action(state, episode) # determine action through eps-greedy

                new_observation, _, done, _ = self.env.step(
                    action.item())                                  # get new state from decided action (reward & info -> blank)

                if done:                                                # if pole lean down || iteration > max_episode_steps 
                    new_state = None                                    # No new state
                    episode_10_list = np.hstack(
                        (episode_10_list[1:], step + 1))                # save succeeded steps by every 'episode'
                    
                    if step < (MAX_STEPS - 5):                              # if pole lean down( < max_episode_steps)
                        reward = torch.FloatTensor([-1.0])                  # reward -> -1
                        complete_episodes = 0                               # reset complete_episode
                    else:                                                   # if pole does not lean down
                        reward = torch.FloatTensor([1.0])                   # reward -> 1
                        complete_episodes += 1                              # update succeeded number
                else:                                                   # not done(still in interation)
                    reward = torch.FloatTensor([0.0])                   # reward = 0
                    new_state = new_observation                         # update new state
                    new_state = torch.from_numpy(new_state).type(           # change type to FloatTensor
                        torch.FloatTensor)
                    new_state = torch.unsqueeze(new_state, 0)           # size 4 -> size 1*4 convert

                self.network.mem.push(
                    state, action, new_state, reward)               # store iteration information(Transition) into memory
                self.network.replay()                               # update Q with Experience Replay
                state = new_state                                   # update state

                if done:
                    print(f'{episode} Episode: Finished after {step + 1} steps：최근 10 에피소드의 평균 단계 수 = {episode_10_list.mean()}')
                    break

            if complete_episodes >= 10:                         # (iteration > max_episode_steps) && (pole does not lean down) for 10 episodes
                print('10 에피소드 연속 성공')
                is_episode_final = True                        # save result and break
            
            if is_episode_final is True:
                display_frames_as_gif(frames)                   # save result as gif
                break
            


# In[16]:

if __name__ == "__main__":
    cartpole_env = Environment()
    cartpole_env.run()
