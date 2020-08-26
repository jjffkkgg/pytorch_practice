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