import gym
import numpy as np
import torch
from torchvision import transforms as T
from gym.spaces import Box

# Converts a (3, H, W) colour tensor into a (1, H, W) grey tensor.
# It is created once here so it is not rebuilt on every frame.
transform = T.Grayscale()


class GrayScale(gym.ObservationWrapper):
    """
    Turns the colour screen into a grey screen.

    Input from the game : (240, 256, 3) uint8 numpy array (height, width, colour)
    Output from here    : (240, 256) float32 torch tensor (height, width)

    Why this is done:
    Colour is not needed to tell Mario, pipes and enemies apart, and dropping
    it makes the data 3 times smaller for the network.
    """

    def __init__(self, env):
        super().__init__(env)

        # The old screen shape is (height, width, channels), for example (240, 256, 3)
        height, width, channels = self.observation_space.shape

        # The new screen has no colour channels, so only height and width remain
        self.observation_space = Box(low=0, high=255, shape=(height, width), dtype=np.uint8)

    def observation(self, obs):
        # 1. Reorder to (channels, height, width), which PyTorch image tools expect
        transposed_obs = np.transpose(obs, (2, 0, 1))

        # 2. Convert to a float tensor. copy() is needed because the
        # transposed array shares memory with the original one.
        tensor_obs = torch.tensor(transposed_obs.copy(), dtype=torch.float32)

        # 3. Convert colour to grey, giving shape (1, height, width)
        gray_obs = transform(tensor_obs)

        # 4. Remove the extra dimension of size 1, giving (height, width)
        return gray_obs.squeeze(0)
