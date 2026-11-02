# watch_mario.py
# Opens a window and shows the trained agent playing World 1-1.
# Run from the project folder with:  python watch_mario.py

from environment.mario_env import make_env
from agent.mario_agent import Mario


# Which checkpoint to watch, and how many episodes
CHECKPOINT = "checkpoints/full/mario_latest.pt"
NUM_EPISODES = 3

# 1. Make the game with a window, and the agent
env = make_env(render_mode="human")
mario = Mario((4, 84, 84), 2)

# 2. Load the trained weights
mario.load(CHECKPOINT)

# 3. Only exploit what was learned, no random moves.
# The minimum is set too, because act never lets epsilon fall below it.
mario.exploration_rate = 0
mario.exploration_rate_min = 0

# 4. Play the episodes
for episode in range(1, NUM_EPISODES + 1):

    terminated = False
    truncated = False
    total_rewards = 0
    total_steps = 0
    level_cleared = False

    # Start a new game
    current_state, extra = env.reset()

    # Keep playing until the game ends by death, time out or the flag
    while not (terminated or truncated):

        action = mario.act(current_state)
        next_state, reward, terminated, truncated, extra = env.step(action)

        total_rewards = total_rewards + reward
        total_steps = total_steps + 1

        # In this single level game, the episode ends when Mario touches the flag,
        # so one True is enough to know the level was cleared
        if extra["flag_get"]:
            level_cleared = True

        current_state = next_state

    print("episode:", episode)
    print("  total reward:", total_rewards)
    print("  total steps:", total_steps)
    print("  level cleared:", level_cleared)

# 5. Close the window
env.close()
