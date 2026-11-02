import gym
# Importing this package registers the Mario games inside gym, so gym.make can find them
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack

from environment.skip_frames import SkipFrames
from environment.grayscale import GrayScale
from environment.resize import ResizeImage


def make_env(render_mode="rgb"):
    """
    Returns the fully processed Mario environment (World 1, Stage 1) that the agent plays on.
    Every state it gives back has the shape (4, 84, 84).

    render_mode "rgb" is the normal setting for training and evaluation, nothing is drawn.
    render_mode "human" opens a window and draws every frame, used only for watching.
    """

    # Create the game and convert it to the 5 value step format
    env = gym.make("SuperMarioBros-1-1-v0", render_mode=render_mode, apply_api_compatibility=True)

    # The order of the wrappers matters. Each one wraps the one before it.
    # The result must be saved back into env, otherwise the wrapper is lost.

    # 1. Keep only two actions: "right" (0) and "right + jump" (1)
    env = JoypadSpace(env, [["right"], ["right", "A"]])

    # 2. Repeat every action for 4 frames. Drawing every frame is switched on only for "human".
    env = SkipFrames(env, skip=4, render=(render_mode == "human"))

    # 3. Convert the screen to grey
    env = GrayScale(env)

    # 4. Resize the screen to 84 by 84 and scale the pixels to 0 to 1
    env = ResizeImage(env, shape=84)

    # 5. Keep the last 4 frames and hand them over together, so Mario's movement is visible
    env = FrameStack(env, num_stack=4)

    return env
