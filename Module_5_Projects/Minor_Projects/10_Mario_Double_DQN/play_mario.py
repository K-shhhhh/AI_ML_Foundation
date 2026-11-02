# play_mario.py
# Opens World 1-1 to be played with the keyboard, to see the game the agent learns.
# Run from the project folder with:  python play_mario.py

from gym_super_mario_bros import SuperMarioBrosEnv
from nes_py.app.play_human import play_human

# Build the raw game directly (world 1, stage 1).
# gym.make is skipped here because the package's own command line player
# is broken with the gym version used in this project.
env = SuperMarioBrosEnv(target=(1, 1))

# Opens a window controlled with the keyboard:
# D right, A left, W up, S down, O jump, P run
play_human(env)
