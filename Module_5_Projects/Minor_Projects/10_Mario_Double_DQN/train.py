"""
Trains the Mario agent with Double DQN.

How to run, from the project folder:
    python train.py smoke    a very short run to check that everything works
    python train.py full     the real long run

All the settings live in config/mario_parameters.yaml, one named set for each run.
If a checkpoint already exists in the run's folder, training resumes from it.
"""

import sys
import os
import time
import yaml

from environment.mario_env import make_env
from agent.mario_agent import Mario
from agent.logger import MetricLogger


# The settings file, relative to the project folder
PARAMS_PATH = os.path.join("config", "mario_parameters.yaml")


class Trainer:

    def __init__(self, param_set):

        # The name of the parameter set, for example "smoke" or "full"
        self.param_set = param_set

        # Open the YAML file that holds the settings of every run
        with open(PARAMS_PATH, "r") as f:
            all_params_set = yaml.safe_load(f)

        # Keep only the settings needed for this run
        params = all_params_set[param_set]

        # Store the needed settings, each under a clear name
        self.num_episodes = params["num_episodes"]
        self.print_every = params["print_every"]
        self.save_every = params["save_every"]
        self.burnin = params["burnin"]
        self.sync_every = params["sync_every"]
        self.exploration_decay = params["exploration_decay"]
        self.checkpoint_dir = params["checkpoint_dir"]
        self.log_path = params["log_path"]

        # Create the checkpoint folder and the log folder if they do not exist yet.
        # Each parameter set has its own folder, so runs never overwrite each other.
        # exist_ok=True means "if the folder is already there, do nothing and do not raise an error".
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        # The file paths used for saving, all inside the checkpoint folder
        # 1. self.latest_path is used to access mario_latest.pt
        self.latest_path = os.path.join(self.checkpoint_dir, "mario_latest.pt")

        # 2. self.best_path is used to access mario_best.pt
        self.best_path = os.path.join(self.checkpoint_dir, "mario_best.pt")

        # 3. self.best_avg_path is used to access best_average.txt
        self.best_avg_path = os.path.join(self.checkpoint_dir, "best_average.txt")

        # The processed game environment (all the wrappers are applied inside)
        self.env = make_env()

        # The agent. Inputs are (state_dim, action_dim)
        self.mario = Mario((4, 84, 84), 2)

        # The logger writes one row per episode into the CSV file
        self.logger = MetricLogger(self.log_path)

        # Apply this run's settings to the agent.
        # The values inside mario_agent.py are only defaults, so they are overwritten
        # with the values from the YAML file for this particular run.
        self.mario.burnin = self.burnin
        self.mario.sync_every = self.sync_every
        self.mario.exploration_rate_decay = self.exploration_decay

        # Resume from the latest checkpoint if there is one.
        # load restores the weights, the optimizer, the exploration rate and the step
        # counter. It leaves the settings above untouched.
        if os.path.exists(self.latest_path):
            self.mario.load(self.latest_path)
            print("resumed from checkpoint at step:", self.mario.curr_step)
        else:
            print("no checkpoint found, starting a fresh run")

        # The best rolling average reward seen so far.
        # Reading it from a file stops a resumed run from replacing a good best
        # model with a worse one. Minus infinity means any first average is a new best.
        if os.path.exists(self.best_avg_path):
            with open(self.best_avg_path, "r") as f:
                self.best_average = float(f.read())
        else:
            self.best_average = float("-inf")
        print("best rolling average so far:", self.best_average)

        # Remember where this session began, so steps per second only counts
        # the steps taken in this session
        self.start_step = self.mario.curr_step

        # Start the timer
        self.start_time = time.perf_counter()

    def run(self):
        """The training loop. Plays episodes, learns, logs and saves."""

        # try / finally: the finally block at the bottom always runs, even when training
        # is stopped with Ctrl+C or the program crashes, so the last progress is saved
        try:

            # One pass of this loop is one episode. episode counts 1, 2, 3 ...
            for episode in range(1, self.num_episodes + 1):

                current_state, extra = self.env.reset()

                # The counters start at zero for every episode
                total_rewards = 0
                experience_length = 0
                total_loss = 0.0
                q_sum = 0.0
                learn_count = 0

                while True:

                    # Choose an action, random or the best one depending on epsilon
                    action = self.mario.act(current_state)

                    # Take that action
                    next_state, reward, terminated, truncated, extra = self.env.step(action)

                    # The episode is over if either of the two is True
                    done = terminated or truncated

                    # Store the experience. The states stay in LazyFrames format to save memory.
                    self.mario.cache(current_state, next_state, action, reward, done)

                    # Most of the time learn returns (None, None).
                    # On a learning step it returns (mean_q, loss_number).
                    # q_value is the mean predicted Q value of the batch, not an episode average.
                    q_value, loss = self.mario.learn()

                    # Add reward and length to the totals
                    total_rewards = total_rewards + reward
                    experience_length = experience_length + 1

                    # If something was learned, add the Q value and loss to the totals
                    # (adding None to a number would raise an error, so this check is needed)
                    if loss is not None:
                        total_loss = total_loss + loss
                        q_sum = q_sum + q_value
                        learn_count = learn_count + 1

                    # The next state becomes the current state
                    current_state = next_state

                    # Stop when the episode is over
                    if done:
                        break

                # Finished level check.
                # env.step returns a dictionary as the fifth output, saved here as extra.
                # extra holds "flag_get", which is True when Mario touched the flag.
                # extra holds the info of the last step, which is the right one to ask.
                if extra["flag_get"]:
                    print("Level Finished in episode:", episode)

                # Average loss and average Q of this episode, only over the steps that learned.
                # During burn in nothing was learned, so both are None (an empty cell in the CSV).
                if learn_count > 0:
                    avg_loss = total_loss / learn_count
                    avg_q = q_sum / learn_count
                else:
                    avg_loss = None
                    avg_q = None

                # Add the values of the episode to the log file, in this order:
                # episode, reward, length, avg_loss, avg_q, epsilon, step.
                # epsilon and step come from the Mario object.
                self.logger.log_episode(episode, total_rewards, experience_length, avg_loss, avg_q,
                                        self.mario.exploration_rate, self.mario.curr_step)

                # Best model
                # Check 1: the deque is full, it holds the rewards of the last 50 episodes
                # Check 2: the average of those 50 rewards is higher than the best average so far
                if len(self.logger.recent_ep_rewards) == 50 and self.logger.rolling_average() > self.best_average:
                    self.best_average = self.logger.rolling_average()

                    # Save the model itself in the best model file
                    self.mario.save(self.best_path)

                    # Save the best average in its own text file, so a resumed run knows it
                    with open(self.best_avg_path, "w") as f:
                        f.write(str(self.best_average))

                    print("new best rolling average saved:", self.best_average)

                # After every "save_every" episodes, save the latest model
                if episode % self.save_every == 0:
                    self.mario.save(self.latest_path)

                # After every "print_every" episodes, print the progress
                if episode % self.print_every == 0:

                    seconds_passed = time.perf_counter() - self.start_time
                    steps_per_second = (self.mario.curr_step - self.start_step) / seconds_passed

                    print("episode:", episode)
                    print("total steps:", self.mario.curr_step)
                    print("exploration rate:", self.mario.exploration_rate)
                    print("rolling average reward (last 50 episodes):", self.logger.rolling_average())
                    print("episode length:", experience_length)
                    print("average loss this episode:", avg_loss)
                    print("average Q value this episode:", avg_q)
                    print("steps per second:", steps_per_second)

        finally:
            # Always runs, so the latest progress is never lost
            self.mario.save(self.latest_path)
            print("final checkpoint saved at step:", self.mario.curr_step)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("\nusage: python train.py smoke  or  python train.py full")

        # sys.exit(0) means the program ended successfully
        # sys.exit(1) means the program ended with a problem
        sys.exit(1)

    # Store the name of the run, for example "smoke" from:  python train.py smoke
    set_name = sys.argv[1]

    # Load the YAML file to see which parameter sets exist
    with open(PARAMS_PATH, "r") as f:
        all_param_set = yaml.safe_load(f)

    # Check that set_name is one of the parameter sets in the YAML file and not a typo
    if set_name in all_param_set:

        # Create the trainer with set_name
        trainer = Trainer(set_name)

        # Start training
        trainer.run()

    else:
        print("\nunknown parameter set:", set_name)
        print("valid sets are:", list(all_param_set.keys()))
        sys.exit(1)
