# Experience Replay
# to make and store this we use a python data structure DEQUE
# it is a double ended queue (add/remove from both ends)
# we will use it in FIFO mode (first in first out)
# we will store curr_state, action, next_state, reward and terminated in the experience replay

from collections import deque
import random


class ReplayMemory:

    # create FIFO queue
    # max_len is the size of the replay memory
    # seed value is meant for the random replays we take out for training the dqn (not used yet)
    def __init__(self, max_len, seed = None):
        self.memory = deque([], maxlen = max_len)

    # add one new experience, the oldest one is dropped automatically when the memory is full
    def push(self, new_exp):
        self.memory.append(new_exp)

    # sample_size is how many replays we want to randomly take out
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    # number of experiences currently stored
    def buffer_size(self):
        return len(self.memory)
