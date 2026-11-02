import csv
import os
from collections import deque


class MetricLogger():
    """
    Records how training is going, one row per episode, in a CSV file.

    It also keeps the rewards of the last 50 episodes, so a rolling average can
    be asked for. That average is what decides which model is the
    best so far, because one lucky episode says very little.
    """

    def __init__(self, path):

        # The CSV file path, for example "logs/training_log_full.csv"
        self.path = path

        # Write the header row only if the file does not exist yet.
        # Otherwise a resumed run would add a second header in the middle of the data.
        if not os.path.exists(self.path):
            # "a" means append, newline="" avoids blank lines between rows
            with open(self.path, "a", newline="") as f:
                writer = csv.writer(f)
                # The column names, in the same order as the rows in log_episode
                writer.writerow(["episode", "reward", "length", "avg_loss", "avg_q", "epsilon", "step"])

        # Rewards of the last 50 episodes, the oldest drops out automatically
        self.recent_ep_rewards = deque(maxlen=50)

    def log_episode(self, episode, reward, length, avg_loss, avg_q, epsilon, step):
        """
        Write one row for a finished episode.
        avg_loss and avg_q can be None (no learning happened yet),
        csv writes None as an empty cell.
        """

        # Open, write one row and close again, so the data is safe on disk
        # even if the training is stopped at any moment
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, reward, length, avg_loss, avg_q, epsilon, step])

        # Remember this reward for the rolling average
        self.recent_ep_rewards.append(reward)

    def rolling_average(self):
        """Average reward of the last 50 episodes (or fewer at the start)."""

        # No episodes yet, so there is nothing to average
        if len(self.recent_ep_rewards) == 0:
            return 0.0
        else:
            return sum(self.recent_ep_rewards) / len(self.recent_ep_rewards)


if __name__ == "__main__":
    # Quick test, run from the project folder with:  python agent/logger.py
    # It writes test_logger.csv. Delete that file before running the test again.
    test_logger = MetricLogger("test_logger.csv")

    # Keyword arguments make it clear which number is which.
    # The first episode has no learning yet, so loss and q are None.
    test_logger.log_episode(episode=1, reward=10, length=100, avg_loss=None, avg_q=None, epsilon=1.0, step=100)
    test_logger.log_episode(episode=2, reward=20, length=150, avg_loss=2.5, avg_q=0.1, epsilon=0.9, step=250)
    test_logger.log_episode(episode=3, reward=30, length=200, avg_loss=1.8, avg_q=0.4, epsilon=0.8, step=450)

    # Rewards 10, 20 and 30 should average to 20.0
    # rolling_average() needs the brackets, otherwise Python prints the method itself
    print("rolling average (should be 20.0):", test_logger.rolling_average())
