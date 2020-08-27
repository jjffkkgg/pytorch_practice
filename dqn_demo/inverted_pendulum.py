#%%
from typing import MutableSequence
import gym                              # RL simulation 전용 라이브러리
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple      # factory function for creating tuple subclasses with named fields
from itertools import count             # 효율적인 루핑을 위한 이터레이터를 만드는 함수 	count(10) --> 10 11 12 13 14 ...
from PIL import Image                   # Python Image Library

#%%
import torch                            # pytorch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T      # torch visioning 라이브러리

#%%
env = gym.make('CartPole-v0').unwrapped     # gym environment (cartpole) 설정

# matplotlib 설정
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
""" Initial structure setting """

Transition = namedtuple('Transition',('state','action','next_state','reward'))          # tuple representing a single transition in our environment.

class ReplayMemory:                                             # cyclic buffer of bounded size that holds the transitions observed recently

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args: namedtuple):
        """Save transistion"""
        if len(self.memory) < self.capacity:                    # ??
            self.memory.append(None)
        self.memory[self.position] = Transition                 # Transition 의 info 를 받아 memory list 에 저장
        self.position = (self.position + 1) % self.capacity     # position index ++ (capacity 범위 내)

    def sample(self, batch_size: int) -> MutableSequence:
        return random.sample(self.memory, batch_size)           # Choose random sample from mem w/ size

    def __len__(self) -> int:                                   # len(class)
        return len(self.memory)

#%%
""" Q-Network """
class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()                                 # class inheritance from nn.module
        self.conv1 = nn.Conv2d(3,16,kernel_size = 5, stride = 2)    # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html 
        self.bn = nn.BatchNorm2d(16)                                # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 5, stride = 2) # 2d 블럭들에 대한 convolution 집합정보 2
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,32,kernel_size = 5, stride = 2)
        self.bn3 = nn.BatchNorm2d(32) 

        # conv2d layer output --(affect)--> Linear input connection | compute input image(conv2d) size
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))        # width sizeout
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))        # height sizeout
        linear_input_size = convw * convh * 32                              # w * h * batch size
        self.head = nn.Linear(linear_input_size, outputs)                   # Applies a linear transformation to the incoming data: y=xA^T+b
    
    def foward(self, x):
        x = F.relu(self.bn(self.conv1(x)))         # 2d block -> batchnorm -> https://ko.wikipedia.org/wiki/ReLU 
        x = F.relu(self.bn2(self.conv2(x)))        # 1->2->3 conv 지나가며 연산
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0),-1))

#%%
""" Image processing for gym"""
resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_cart_location(screen_width: float) -> int:
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    # gym이 요청한 화면은 400x600x3 이지만, 가끔 800x1200x3 처럼 큰 경우가 있습니다.
    # 이것을 Torch order (CHW)로 변환.
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # 카트는 아래쪽에 있으므로 화면의 상단과 하단을 제거.
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)

    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # 카트를 중심으로 정사각형 이미지가 되도록 가장자리를 제거.
    screen = screen[:, :, slice_range]
    # float 으로 변환하고,  rescale 하고, torch tensor 로 변환.
    # (이것은 복사를 필요로하지 않습니다)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # 크기를 수정하고 배치 차원(BCHW)을 추가.
    return resize(screen).unsqueeze(0).to(device)


env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
plt.show()

#%%
""" Parameter modelling & Utility definition """
BATCH_SIZE = 128        
GAMMA = 0.999           # optimization coeff
EPS_START = 0.9         # 임의의 action 선택할 starting probability
EPS_END = 0.05          # ``    ``      ``    end probability
EPS_DECAY = 200         # 확률의 지수적 감소 속도 제어
TARGET_UPDATE = 10

# AI gym에서 반환된 형태를 기반으로 계층을 초기화 하도록 화면의 크기를
# 가져옵니다. 이 시점에 일반적으로 3x40x90 에 가깝습니다.
# 이 크기는 get_screen()에서 고정, 축소된 렌더 버퍼의 결과입니다.
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# gym 행동 공간에서 행동의 숫자를 얻습니다.
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)     # 정책망. n_action: output, 연산 주체 설정(G/Cpu)
target_net = DQN(screen_height, screen_width, n_actions).to(device)     # 목표망. 같은 형식의 batch
target_net.load_state_dict(policy_net.state_dict())                     # 정책망의 param & buffer 들 로드
target_net.eval()                                                       # set module in evaluation mode

optimizer = optim.RMSprop(policy_net.parameters())                      # optimizer 설정
memory = ReplayMemory(10000)                                            # 리플레이백 메모리 설정

steps_done = 0                                                          # iterator

def select_action(state):
    """Epsilon Greedy 정책에 따라 행동을 선택합니다."""     # 모델을 사용하여 행동을 선택하거나 또는 모델을 균일하게 샘플링.
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END)*math.exp(-1.0 * steps_done / EPS_DECAY)     # Eps decay 모델 설정
    steps_done += 1
    if sample > eps_threshold:                   # sample 이 eps 보다 클 경우 ()
        with torch.no_grad():                   # gradient calc stop
            # t.max (1)은 각 행의 가장 큰 열 값을 반환합니다.
            # 최대 결과의 두번째 열은 최대 요소의 주소값이므로,
            # 기대 보상이 더 큰 행동을 선택할 수 있습니다.
            return policy_net(state).max(1)[1].view(1,1)                # 정책망 모델을 통한 행동
    else:
        return torch.tensor([[random.randrange(n_actions)]], device = device, dtype = torch.long)       # 균일한 샘플링

episode_durations=[]

def plot_durations():        # plotting duration of recent 100 episodes
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # 100개의 에피소드 평균을 가져 와서 도표 그리기
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # 도표가 업데이트되도록 잠시 멈춤
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

#%%
"""Training Loop"""
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). 이것은 batch-array의 Transitions을 Transition의 batch-arrays로
    # 전환합니다.
    batch = Transition(*zip(*transitions))

    # 최종이 아닌 상태의 마스크를 계산하고 배치 요소를 연결합니다
    # (최종 상태는 시뮬레이션이 종료 된 이후의 상태)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)            #상태 배치
    action_batch = torch.cat(batch.action)          # 행동 배치
    reward_batch = torch.cat(batch.reward)          # 보상 배치

    # Q(s_t, a) 계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 열을 선택합니다.
    # 이들은 policy_net에 따라 각 배치 상태에 대해 선택된 행동입니다.
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 모든 다음 상태를 위한 V(s_{t+1}) 계산
    # non_final_next_states의 행동들에 대한 기대값은 "이전" target_net을 기반으로 계산됩니다.
    # max(1)[0]으로 최고의 보상을 선택합니다.
    # 이것은 마스크를 기반으로 병합되어 기대 상태 값을 갖거나 상태가 최종인 경우 0을 갖습니다.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # 기대 Q 값 계산
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber 손실 계산
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # 모델 최적화
    optimizer.zero_grad()
    loss.backward()                         # backward 미분을 통한 loss 계산
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

#%%
num_episodes = 50
for i_episode in range(num_episodes):
    # 환경과 상태 초기화
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    for t in count():
        # 행동 선택과 수행
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        # 새로운 상태 관찰
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # 메모리에 변이 저장
        memory.push(state, action, next_state, reward)

        # 다음 상태로 이동
        state = next_state

        # 최적화 한단계 수행(목표 네트워크에서)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
    #목표 네트워크 업데이트, 모든 웨이트와 바이어스 복사
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()
# %%
