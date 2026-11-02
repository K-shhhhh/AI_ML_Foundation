# play flappy bird yourself, press SPACE to flap

import gymnasium as gym
import flappy_bird_gymnasium
import pygame

# create the env. in human mode so the game window opens
# human mode already renders on every env.step, so we do not call env.render() ourselves
env = gym.make("FlappyBird-v0", render_mode = "human", use_lidar = False)
state, info = env.reset()

# flags that control the game loop
terminated = False
truncated = False
quit_game = False

# initialize the pygame keyboard
pygame.init()

while not (terminated or truncated or quit_game):
    action = 0  # 0 is no flap and 1 is flap

    # check the keyboard and the window close button
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit_game = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                action = 1  # flap

    state, reward, terminated, truncated, info = env.step(action)

# env.close() also shuts pygame down
env.close()
