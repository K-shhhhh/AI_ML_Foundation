# plot_results.py
# Reads the training log and draws four plots against the training steps:
# reward, Q value, loss, and how often Mario cleared the level.
# Run from the project folder with:  python plot_results.py

import os
import pandas as pd
import matplotlib.pyplot as plt


# The log file written by train.py, and where the picture is saved
LOG_FILE = os.path.join("logs", "training_log_full.csv")
OUTPUT_FILE = os.path.join("results", "training_curves.png")

# Average over 200 episodes so the lines are smooth and not jumpy
WINDOW = 200

# An episode with at least this reward counts as a level clear.
# The log has no flag column, so this is only an estimate.
CLEAR_REWARD = 2900

# Clears are counted in blocks of this many steps
BLOCK_STEPS = 200000


# 1. Read the file

# read_csv gives a table. The columns are named like the csv header.
df = pd.read_csv(LOG_FILE)
print("number of episodes in the log:", len(df))
print("last step in the log:", df["step"].iloc[-1])

# Steps in millions go on the x axis. Steps are used and not episodes,
# because the episode counter starts again every time a run is resumed.
df["step_m"] = df["step"] / 1000000


# 2. Smooth the numbers

# rolling(200).mean() gives, for every episode, the average of the last 200 episodes.
# Empty cells (the loss and Q during burn in) are ignored. min_periods=20 lets the
# line start after 20 episodes.
df["reward_avg"] = df["reward"].rolling(WINDOW, min_periods=20).mean()
df["q_avg"] = df["avg_q"].rolling(WINDOW, min_periods=20).mean()
df["loss_avg"] = df["avg_loss"].rolling(WINDOW, min_periods=20).mean()


# 3. Count the level clears

# True for every episode that reached the clear reward
df["clear"] = df["reward"] >= CLEAR_REWARD

# The block number of every episode: steps 0 to 199999 is block 0, and so on
df["block"] = df["step"] // BLOCK_STEPS

# For each block there are two numbers: how many episodes, and how many clears
table = df.groupby("block")["clear"].agg(["count", "sum"])

print("estimated level clears in the whole run:", int(df["clear"].sum()), "out of", len(df), "episodes")

# Keep the blocks that have at least 50 episodes, so the rates are stable
block_x = []
block_rate = []
block_clears = []
for block, row in table.iterrows():
    print("steps", block * BLOCK_STEPS, "to", (block + 1) * BLOCK_STEPS,
          "| episodes:", int(row["count"]), "| estimated clears:", int(row["sum"]))
    if row["count"] >= 50:
        block_x.append(block * BLOCK_STEPS / 1000000)
        block_rate.append(100 * row["sum"] / row["count"])
        block_clears.append(int(row["sum"]))


# 4. Draw

plt.figure(figsize=(12, 8))

# Plot 1: reward. The grey dots are single episodes, the blue line is the average.
plt.subplot(2, 2, 1)
plt.scatter(df["step_m"], df["reward"], s=2, color="lightgrey")
plt.plot(df["step_m"], df["reward_avg"], color="blue")
plt.title("Reward (blue: average of 200 episodes)")
plt.xlabel("Training steps (millions)")
plt.ylabel("Reward")
plt.grid(True)

# Plot 2: Q value
plt.subplot(2, 2, 2)
plt.plot(df["step_m"], df["q_avg"], color="blue")
plt.title("Q value (average of 200 episodes)")
plt.xlabel("Training steps (millions)")
plt.ylabel("Q value")
plt.grid(True)

# Plot 3: loss
plt.subplot(2, 2, 3)
plt.plot(df["step_m"], df["loss_avg"], color="blue")
plt.title("Loss (average of 200 episodes)")
plt.xlabel("Training steps (millions)")
plt.ylabel("Loss")
plt.grid(True)

# Plot 4: estimated level clears per 100 episodes, with the number of clears above each bar
plt.subplot(2, 2, 4)
plt.bar(block_x, block_rate, width=0.17, align="edge", color="blue")
for i in range(len(block_x)):
    plt.text(block_x[i] + 0.085, block_rate[i], str(block_clears[i]), ha="center", va="bottom")
plt.title("Estimated level clears per 100 episodes (count above bar)")
plt.xlabel("Training steps (millions)")
plt.ylabel("Clears per 100 episodes")
plt.grid(True, axis="y")


# 5. Save the picture

os.makedirs("results", exist_ok=True)
plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=150)
print("picture saved as:", OUTPUT_FILE)
