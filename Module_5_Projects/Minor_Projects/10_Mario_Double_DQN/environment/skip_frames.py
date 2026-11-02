import time

import gym


class SkipFrames(gym.Wrapper):
    """
    Repeats every action for `skip` game frames and returns the total reward.

    Why this is done:
    Neighbouring frames in Mario look almost identical, so deciding on every
    single frame wastes effort. Repeating an action for 4 frames means the
    agent decides 4 times less often and each decision moves the game further.
    """

    def __init__(self, env, skip, render=False):
        # Connect this wrapper to the environment underneath it
        super().__init__(env)

        # How many frames each action is repeated for
        self.skip = skip

        # True only when watching the agent play, so every frame is drawn on screen
        self.render_frames = render

    def step(self, action):
        # Rewards from every repeated frame are added up into one number
        total_reward = 0.0

        for _ in range(self.skip):

            next_state, reward, terminated, truncated, info = self.env.step(action)

            # Draw the frame that was just played. This sits after the step, so the
            # last frame of an episode (the death or the flag) is also shown.
            if self.render_frames:

                # A short pause, so the game runs at a pace that looks normal to the human eye
                time.sleep(0.008)

                # Draws the window. It is the same as render mode "human" in the older gym version.
                self.env.render()

            total_reward = total_reward + reward

            # If Mario dies or the episode is cut off, stop repeating.
            # Stepping a finished game would give wrong rewards or errors.
            if terminated or truncated:
                break

        # This return sits outside the loop, so it runs after all repeats.
        # next_state is the last frame, which is what the world looks like
        # after the whole action has been played out.
        return next_state, total_reward, terminated, truncated, info
