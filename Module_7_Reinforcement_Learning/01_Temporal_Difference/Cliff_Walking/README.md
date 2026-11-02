# Cliff Walking with SARSA and Q Learning

## Problem Statement

Cliff Walking is a small grid world where an agent has to walk from a start cell to a goal cell along the edge of a cliff. Every step costs a little, and stepping off the cliff costs a lot. The shortest route runs right along the edge, so the task is a clean test of how an agent trades a short path against a risky one.

This project implements the two classic **temporal difference (TD) control** algorithms from scratch, **SARSA** (on policy) and **Q learning** (off policy), trains both with the same settings, and compares the routes they learn.

## Environment and Assumptions

* **Environment:** `CliffWalking-v1` from Gymnasium
* **Grid:** 4 rows by 12 columns, so 48 states. The agent starts at state 36 (bottom left) and the goal is at the bottom right.
* **Actions:** 4, up, right, down and left
* **Rewards:** -1 for every step, -100 for stepping into the cliff, which also sends the agent back to the start
* **Q table:** 48 by 4, one value per (state, action) pair, initialised to zero
* **Hyperparameters (both algorithms):** `gamma = 0.99`, `alpha = 0.5`, `epsilon = 0.1`, 500 episodes

## Approach

1. **Explore the environment:** the state and action counts, what `reset()` returns, and how a cell maps to a state index (`row * 12 + col`).
2. **Epsilon greedy policy:** with probability 0.1 take a random action, otherwise take the action with the highest Q value.
3. **SARSA:** the update uses the action the policy will actually take next, exploration included:
   ```
   Q(s,a) = Q(s,a) + alpha * [R + gamma * Q(s',a') - Q(s,a)]
   ```
   The next action is chosen before the update and carried forward into the next step.
4. **Q learning:** the update uses the best possible next action, whatever the policy does next:
   ```
   Q(s,a) = Q(s,a) + alpha * [R + gamma * max_a' Q(s',a') - Q(s,a)]
   ```
   The action is simply chosen fresh at the top of every step.
5. **Rendering against speed:** SARSA is trained once without any window and once with every 50th episode rendered, to see how much drawing the game slows training down.
6. **Testing:** each trained Q table is played greedily (`np.argmax`, no exploration) in a rendered window.

## Results

| Algorithm | Test reward | Test steps | Route |
|---|---|---|---|
| SARSA | -17 | 17 | safer, a row or two away from the cliff |
| **Q learning** | **-13** | **13** | the shortest possible route, along the cliff edge |

13 steps is the best any policy can do on this grid (11 across, 1 up and 1 down), so Q learning found the optimal path.

| SARSA training run | Time |
|---|---|
| no rendering | 0.15 seconds |
| every 50th episode rendered | 72.38 seconds |

## Key Findings

* **The classic result was reproduced from scratch.** Q learning converges to the optimal but risky edge path, SARSA to a slightly longer but safer one. This is the same outcome as Sutton and Barto's original Cliff Walking example.
* **The reason is one term in the update.** Q learning's target assumes the next move is always the best one, so it never prices in the chance that its own random exploration steps off the cliff. SARSA's target uses the move the policy really makes, exploration included, so it learns that walking the edge is more dangerous than it looks.
* **On policy and off policy show up in practice, not just in theory.** During training, Q learning still occasionally falls off the cliff while walking the edge (for example a reward of -237 in a 39 step episode), because the exploration noise is real even though its Q values ignore it.
* **Rendering is expensive.** Drawing only every 50th episode made the same training run about 480 times slower, so training should run without a window and rendering should be kept for testing.

## Limitations

* Each algorithm was tested with a single greedy run. On this deterministic grid that is enough to show the learned route, but it is not a statistical comparison.
* Epsilon stays fixed at 0.1. With epsilon decaying towards zero, SARSA would also move towards the optimal edge path.
* Only one set of hyperparameters was tried.

## Files

* `SARSA_Implementation.ipynb`: Gymnasium setup, environment exploration, SARSA theory, training with and without rendering, and testing
* `Q_Learning_Implementation.ipynb`: Q learning theory and how it differs from SARSA, training, testing, and the SARSA against Q learning report
* `README.md`: this file

## How to Run

```bash
pip install "gymnasium[toy-text]" numpy
jupyter notebook SARSA_Implementation.ipynb
```

The rendered cells open a game window through `pygame`.
