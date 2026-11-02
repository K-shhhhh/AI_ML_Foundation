# test_agent.py
# Checks every method of the Mario agent before the real training.
# Run from the project folder with:  python test_agent.py

from environment.mario_env import make_env
from agent.mario_agent import Mario


# The fully processed game, the same one train.py uses
env = make_env()

mario = Mario((4, 84, 84), 2)

# TEMPORARY, for this test only: a fast decay, so the exploration rate visibly falls.
# The real value comes from config/mario_parameters.yaml during training.
mario.exploration_rate_decay = 0.995


# Part 1: play, collect experience, and check act, cache, recall,
# td_estimate and td_target along the way

# Reset returns the first state and an info dictionary
current_state, info = env.reset()

max_steps = 1000
steps = 0
total_rewards = 0
episodes_finished = 0

while steps < max_steps:

    # The agent picks an action for the current state
    action = mario.act(current_state)

    next_state, reward, terminated, truncated, info = env.step(action)

    # Store this experience. It must happen before current_state is replaced.
    mario.cache(current_state, next_state, action, reward, terminated or truncated)

    # Add this step's reward first, so the death penalty is counted too
    total_rewards = total_rewards + reward
    steps = steps + 1

    if terminated or truncated:
        # The episode is over, report it and start a new one
        episodes_finished = episodes_finished + 1
        print("episode ended at step:", steps)
        print("total reward for this episode:", total_rewards)
        total_rewards = 0
        current_state, info = env.reset()
    else:
        # The episode continues, so the next state becomes the current state
        current_state = next_state

    # Print a short progress line every 200 steps
    if steps % 200 == 0:
        print("step:", steps)
        print("last action:", action)
        print("exploration rate:", mario.exploration_rate)
        print("agent step counter:", mario.curr_step)

    # Every 250 steps, test recall and the two learning numbers on a random batch
    if steps % 250 == 0:
        states, next_states, actions, rewards, dones = mario.recall()

        print("batch check at step:", steps)
        print("experiences in memory:", len(mario.memory))
        print("states shape:", states.shape, "dtype:", states.dtype)
        print("next_states shape:", next_states.shape, "dtype:", next_states.dtype)
        print("actions shape:", actions.shape, "dtype:", actions.dtype)
        print("rewards shape:", rewards.shape, "dtype:", rewards.dtype)
        print("dones shape:", dones.shape, "dtype:", dones.dtype)

        # What the online network currently predicts for the actions that were taken
        estimate = mario.td_estimate(states, actions)

        # What the Double DQN target says it should have predicted
        target = mario.td_target(rewards, next_states, dones)

        print("td_estimate shape:", estimate.shape)
        print("td_estimate requires_grad (should be True):", estimate.requires_grad)
        print("td_estimate first 3 values:", estimate[:3].tolist())

        print("td_target shape:", target.shape)
        print("td_target requires_grad (should be False):", target.requires_grad)
        print("td_target first 3 values:", target[:3].tolist())

        print("first 3 rewards in the batch:", rewards[:3].tolist())
        print("first 3 done flags in the batch:", dones[:3].tolist())

print("episodes finished:", episodes_finished)
print("experiences collected:", len(mario.memory))


# Part 2: overfit one batch to test update_q_online.
# The SAME 32 experiences are trained on again and again.
# If the update works, the loss must fall and the estimates must move
# toward their targets.

states, next_states, actions, rewards, dones = mario.recall()

# Look at the estimates and targets before any training
estimate_before = mario.td_estimate(states, actions)
target_before = mario.td_target(rewards, next_states, dones)
print("estimates before training (first 3):", estimate_before[:3].tolist())
print("targets before training (first 3):", target_before[:3].tolist())

training_rounds = 200

for round_number in range(1, training_rounds + 1):

    # What the online network predicts now (gradients on)
    estimate = mario.td_estimate(states, actions)

    # What the labels say (no gradients)
    target = mario.td_target(rewards, next_states, dones)

    # One learning update, returns the loss as a plain number
    loss_value = mario.update_q_online(estimate, target)

    # Print the loss now and then
    if round_number == 1 or round_number % 20 == 0:
        print("training round:", round_number, "loss:", loss_value)

# Look at the estimates and targets after training
estimate_after = mario.td_estimate(states, actions)
target_after = mario.td_target(rewards, next_states, dones)
print("estimates after training (first 3):", estimate_after[:3].tolist())
print("targets after training (first 3):", target_after[:3].tolist())

env.close()
