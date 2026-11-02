# Flappy Bird with a Deep Q Network (DQN)

A DQN written in PyTorch that learns to play Flappy Bird on its own. Nobody tells the bird how to fly. It plays thousands of games, collects rewards and penalties, and slowly learns whether flapping or doing nothing is worth more in each situation.

The project is part of my reinforcement learning work. Taxi gave me Q learning, Blackjack gave me exploration and averaging over many episodes, and Flappy Bird adds the problem that makes a Q table impossible: the state is a list of continuous numbers that never repeats exactly, so a neural network has to estimate the Q values instead.

## Files

| File | What it holds |
|---|---|
| `Flappy_Bird_DQN.ipynb` | the notebook where I build every piece step by step, with the theory next to the code |
| `agent.py` | the training and testing loop, and the `optimize` function |
| `dqn.py` | the `DQN` network class |
| `experience_replay.py` | the `ReplayMemory` class |
| `parameters.yaml` | all hyperparameters |
| `game_flappy_bird.py` | play the game yourself with the SPACE key |

## How it works

* **State.** 12 numbers: the positions of the three nearest pipes (horizontal position, where the gap starts and where it ends), plus the bird's height, vertical speed and rotation.
* **Actions.** `0` is do nothing, `1` is flap.
* **Rewards.** +0.1 for every frame alive, +1 for passing a pipe, -1 for dying, -0.5 for flying above the top of the screen.
* **Policy network.** `12 → 256 → 2` with a ReLU in the middle. It outputs one Q value per action.
* **Epsilon greedy with decay.** The bird starts fully random and shifts to using its network as epsilon falls, down to a floor of 0.05.
* **Experience replay.** Every step is stored as `(state, action, next_state, reward, terminated)` in a memory with a fixed size. Training uses random mini batches of 32, which breaks the similarity between consecutive frames.
* **Target network.** The training target is `reward + gamma * max Q(next_state)`, and it comes from a frozen copy of the policy network. The copy is refreshed every 1000 training steps, so the target does not move every time the network changes.
* **Learning on every step.** After each step of the game, once the memory holds more than a mini batch, the network trains on one random batch with an MSE loss and the Adam optimizer.
* **Saving the model.** The model is saved whenever the average reward of the last 50 episodes is a new best.

## Setup

```
conda activate pytorch
pip install flappy-bird-gymnasium gymnasium torch pyyaml matplotlib
```

## Run

Train from the terminal (no window, runs until `Ctrl + C`):

```
python agent.py flappybirdv0 --train
```

Test the best saved model (opens the game window, no random actions):

```
python agent.py flappybirdv0
```

The best model is saved to `runs/flappybirdv0.pt` and every new best is logged in `runs/flappybirdv0.log`. Use a new name in `parameters.yaml` if you want to train a second set of parameters without overwriting the first.

## Hyperparameters

| Parameter | Value | Meaning |
|---|---|---|
| `epsilon_init` | 1 | start fully random |
| `epsilon_min` | 0.05 | exploration never fully stops |
| `epsilon_decay` | 0.99 | epsilon is multiplied by this after every episode |
| `replay_memory_size` | 100,000 | how many experiences the memory keeps |
| `mini_batch_size` | 32 | experiences per training step |
| `network_sync_rate` | 1000 | training steps between target network refreshes |
| `alpha` | 0.001 | learning rate of Adam |
| `gamma` | 0.99 | discount factor |
| `reward_threshold` | 1000 | training stops when one episode reaches this reward |

## Results

In two training runs of about 15 minutes each on a CPU, the average training reward went from about -7 (a random bird) to about +3 over the first 1,000 episodes, about +9 to +10 after 3,000 to 4,000 episodes, and single games went above +50. Training rewards include the 5% random actions, so the greedy bird looks better when tested. Over 40 greedy test games, the saved models passed 18.1 pipes on average (median 11, best game 74) in one run and 6.5 pipes on average (median 5, best game 31) in the other.

Runs differ from each other, and the average can dip for a while before it recovers, so it is worth letting a run go for a few thousand episodes before judging it.

## What I tried along the way

| Setup | What I saw |
|---|---|
| lidar state (180 numbers), one update per episode, about 11,000 episodes | average reward about +0.8, no pipe passed |
| lidar state, one update per step, about 14,500 episodes | average reward about +2, no pipe passed |
| 12 numbers, one update per episode | learns, but slowly, with a training average of about 5 after the first few thousand episodes |
| 12 numbers, one update per step | training average of about 10 after 3,000 to 5,000 episodes |
| saving the model at the single best episode | unreliable, one saved bird passed about 1 pipe on average even though the training average was near 10 |
| saving the model at the best 50 episode average | much steadier |

What I learned from these runs:

* **The state matters.** The game can return 180 lidar distances instead of the 12 numbers (`use_lidar=True`). With lidar the bird never passed a pipe. Each step is also slow because 180 rays are cast every frame, and the 14,500 episode run took around 4 hours. Reading the game's source showed that lidar mode adds a -0.5 penalty on every frame where any ray is short, so flying through a pipe gap, which means being close to the pipe, is punished. The 12 numbers hand the pipe positions over directly.
* **Learning on every step helps.** My first version trained once per episode and synced the target network right after, so the target was always identical to the policy network. With the lidar state, training on every step did not change the outcome, because the state was the main problem. With the 12 numbers the difference was clear.
* **One lucky episode is a poor way to pick a model.** A single easy run of pipes can come from a weak network, so the model is now chosen by the average of the last 50 episodes.
