#%%
from typing import MutableSequence
import gym                              # RL simulation 전용 라이브러리
import math
import random
import numpy as np
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