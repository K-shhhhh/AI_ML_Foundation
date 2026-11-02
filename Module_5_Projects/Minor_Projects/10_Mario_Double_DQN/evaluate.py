# evaluate.py
# Plays the saved Mario checkpoints and prints how well they do.
# No training happens here: no learning and no storing of experiences.
# Run from the project folder with:  python evaluate.py

import os
import csv

from environment.mario_env import make_env
from agent.mario_agent import Mario


# The checkpoints to test: a name for printing, and the file path
CHECKPOINTS = [
    ("best", os.path.join("checkpoints", "full", "mario_best.pt")),
    ("latest", os.path.join("checkpoints", "full", "mario_latest.pt")),
]

# Each checkpoint is played two ways, written as (epsilon, number of episodes).
# epsilon 0 means no random moves. The game is the same every time, so 3 episodes are enough.
# epsilon 0.05 means 5 percent random moves, so the episodes differ and 30 give a fair average.
RUNS = [
    (0.0, 3),
    (0.05, 30),
]

# If Mario stands still forever, the episode ends after this many steps
MAX_STEPS = 5000

# One summary row per run is saved here, for the README
RESULTS_PATH = os.path.join("results", "eval_results.csv")


def play(path, label, epsilon, num_episodes):

    # 1. Make the game and the agent, the same way as in train.py
    env = make_env()
    mario = Mario((4, 84, 84), 2)

    # 2. Load the weights that training saved
    mario.load(path)

    # 3. load put back the epsilon from training (0.1), so it is overwritten.
    # The minimum is overwritten too, because act never lets epsilon fall below
    # the minimum. With the minimum left at 0.1, an epsilon of 0 would turn into
    # 0.1 after the first action.
    mario.exploration_rate = epsilon
    mario.exploration_rate_min = epsilon

    print("checkpoint:", label, "| epsilon:", mario.exploration_rate, "| episodes:", num_episodes)

    # One number per episode is kept in these lists
    all_rewards = []
    all_lengths = []
    clears = 0

    # 4. Play the episodes
    for episode in range(1, num_episodes + 1):

        # Start a new game: reset gives the first state and an info dictionary
        state, info = env.reset()

        total_reward = 0.0
        steps = 0
        reached_flag = False

        # Keep playing until the episode is over
        while True:

            # Ask the agent for an action and do it in the game
            action = mario.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Only add up the numbers here, nothing is learned from them
            total_reward = total_reward + reward
            steps = steps + 1

            # The info dictionary says if Mario touched the flag
            if info["flag_get"]:
                reached_flag = True

            # The next state becomes the current state
            state = next_state

            # Stop when the episode is over or goes on for too long
            if done or steps >= MAX_STEPS:
                break

        # Save the results of this episode
        all_rewards.append(total_reward)
        all_lengths.append(steps)
        if reached_flag:
            clears = clears + 1

        print("  episode:", episode, "| reward:", total_reward, "| length:", steps, "| reached flag:", reached_flag)

    env.close()

    # 5. Check that epsilon is still the value that was set. If not, the results are wrong.
    print("epsilon after the run, it must be", epsilon, ":", mario.exploration_rate)

    # 6. Work out the summary with the built in sum, max, min and len
    mean_reward = round(sum(all_rewards) / len(all_rewards), 1)
    mean_length = round(sum(all_lengths) / len(all_lengths), 1)

    print("summary:", label, "| epsilon:", epsilon)
    print("  mean reward:", mean_reward)
    print("  max reward:", max(all_rewards))
    print("  min reward:", min(all_rewards))
    print("  mean episode length:", mean_length)
    print("  level clears:", clears, "out of", num_episodes)
    print()

    # 7. Add one row to the results file. The header is written only once.
    os.makedirs("results", exist_ok=True)
    write_header = not os.path.exists(RESULTS_PATH)
    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["checkpoint", "epsilon", "episodes", "mean_reward", "max_reward",
                             "min_reward", "mean_length", "clears"])
        writer.writerow([label, epsilon, num_episodes, mean_reward, max(all_rewards),
                         min(all_rewards), mean_length, clears])


# This part runs when the file is started: every checkpoint with every run setting
if __name__ == "__main__":
    for label, path in CHECKPOINTS:
        for epsilon, num_episodes in RUNS:
            play(path, label, epsilon, num_episodes)
    print("results saved to:", RESULTS_PATH)
