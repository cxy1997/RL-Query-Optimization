import random
from collections import namedtuple, deque


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args, **kwargs):
        """Save a transition"""
        self.memory.append(Transition(*args, **kwargs))

    def sample(self, batch_size=1):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
