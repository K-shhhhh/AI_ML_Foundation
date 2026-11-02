# Taxi with SARSA and Q Learning

## Problem Statement

A taxi on a 5 by 5 grid has to drive to a passenger, pick them up, drive to their destination and drop them off. Compared with Cliff Walking, the task has more states, more actions and more structure: the agent has to learn a sequence (go to the passenger, pick up, go to the destination, drop off), and illegal pickups or dropoffs are punished.

This project applies the same two **TD control** algorithms built for Cliff Walking, **SARSA** and **Q learning**, to this larger problem, with the code unchanged apart from the table size and the environment.

## Environment and Assumptions

* **Environment:** `Taxi-v4` from Gymnasium
* **State:** one integer encoding `(taxi_row, taxi_col, passenger_location, destination)`: 25 taxi positions × 5 passenger locations (4 pickup points or inside the taxi) × 4 destinations = **500 states**
* **Actions:** 6, move down, up, right or left (0 to 3), pickup (4) and dropoff (5)
* **Rewards:** -1 per step, +20 for a correct dropoff, -10 for an illegal pickup or dropoff
* **Episode cap:** Gymnasium's TimeLimit wrapper ends an episode after 200 steps (`truncated = True`)
* **Random start:** every reset places the taxi, the passenger and the destination at random
* **Q table:** 500 by 6, initialised to zero
* **Hyperparameters (both algorithms):** `gamma = 0.99`, `alpha = 0.5`, `epsilon = 0.1`, 500 episodes

## Approach

1. **Explore the environment:** state and action counts, the meaning of the encoded state, the three reward values, and the 200 step cap.
2. **SARSA training:** the on policy update `Q(s,a) += alpha * [R + gamma * Q(s',a') - Q(s,a)]`, carrying the next action forward between steps.
3. **Q learning training:** the off policy update `Q(s,a) += alpha * [R + gamma * max_a' Q(s',a') - Q(s,a)]`, choosing the action fresh at every step.
4. **Rendering:** every 50th training episode is drawn in a window to watch the progress, the rest run without one.
5. **Testing:** each trained Q table plays one greedy episode (no exploration) in a rendered window.

## Results

| Algorithm | Test reward | Test steps |
|---|---|---|
| SARSA | 7 | 14 |
| Q learning | 4 | 17 |

Both tests are successful deliveries: the dropoff step pays +20 instead of -1, so 13 steps at -1 plus 20 gives 7, and 16 steps at -1 plus 20 gives 4.

**Q learning training progression (printed every 50 episodes):**

| Episode | Reward | Steps |
|---|---|---|
| 1 | -551 | 200 (cut off) |
| 51 | -127 | 103 |
| 101 | -106 | 100 |
| 151 | -7 | 19 |
| 251 | 12 | 9 |
| 451 | 5 | 16 |

SARSA showed the same pattern: early episodes mostly hit the 200 step cap with large negative rewards, and from about episode 200 on the episodes became short deliveries with small positive rewards.

## Key Findings

* **The same code scales from 48 to 500 states.** Only the table shape and the environment name changed, which shows the algorithms are general, not tuned to one grid.
* **Early training is dominated by illegal actions.** A reward of -551 in 200 steps means many -10 pickup and dropoff attempts in the wrong place. The agent first learns *not* to do those, then learns the route.
* **Both algorithms solve Taxi.** Unlike Cliff Walking, Taxi has no cliff where one exploratory step is very costly, so the safe against short path difference between SARSA and Q learning does not show up here.
* **Single test runs are not a fair comparison here.** Every reset starts from a different random situation, so the 7 against 4 difference says more about the two starting positions than about the algorithms.

## Limitations

* Each algorithm was tested with one greedy episode from a random start. Averaging many test episodes would be needed for a real comparison.
* 500 episodes is a short training run. A longer run, or an epsilon that decays, would give a more stable final policy.
* Only one set of hyperparameters was tried.

## Files

* `SARSA_Implementation.ipynb`: environment exploration, SARSA training, testing and findings
* `Q_learning_Implementation.ipynb`: Q learning training, testing and the comparison with SARSA
* `README.md`: this file

## How to Run

```bash
pip install "gymnasium[toy-text]" numpy
jupyter notebook SARSA_Implementation.ipynb
```

Every 50th training episode and the test episode open a game window through `pygame`.
