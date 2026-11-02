import gym
import numpy as np
from torchvision import transforms as T
from gym.spaces import Box


class ResizeImage(gym.ObservationWrapper):
    """
    Shrinks the grey screen and scales the pixel values.

    Input  : (240, 256) float32 tensor with values from 0 to 255
    Output : (84, 84) float32 tensor with values from 0 to 1

    Why this is done:
    A small square image is much cheaper for the network, and values between
    0 and 1 keep the numbers inside the network in a comfortable range.
    """

    def __init__(self, env, shape):
        super().__init__(env)

        # shape can be one number (84) or a pair like (84, 84).
        # One number means a square, so it becomes (84, 84).
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        # The new screen size, so other code knows what to expect
        self.observation_space = Box(low=0, high=255, shape=self.shape, dtype=np.uint8)

        # Two steps in a row: resize the image, then divide every pixel by 255.
        # Normalize(mean, std) computes (x - mean) / std, so (x - 0) / 255.
        self.transforms = T.Compose([
            T.Resize(self.shape, antialias=True),
            T.Normalize(0, 255),
        ])

    def observation(self, obs):
        # Resize and Normalize expect a channel dimension, so (240, 256) becomes (1, 240, 256)
        obs = obs.unsqueeze(0)

        # Apply both steps, then remove the channel dimension again, giving (84, 84)
        return self.transforms(obs).squeeze(0)
