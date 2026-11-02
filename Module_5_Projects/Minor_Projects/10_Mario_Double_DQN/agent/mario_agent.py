import os
import torch
import random
import numpy as np
from collections import deque
from agent.network import MarioNet


class Mario():
    """
    The Mario agent.

    It chooses actions (act), remembers what happened (cache, recall),
    and learns from the memories with Double DQN (td_estimate, td_target,
    update_q_online). The learn method decides when to learn and when to
    copy the online network into the target network (sync_q_target).
    The save and load methods store and restore the training progress.
    """

    def __init__(self, state_dim, action_dim):

        # How much future rewards are worth compared to immediate ones
        self.gamma = 0.9

        # Memory of past experiences, the oldest are dropped when it is full
        self.memory = deque(maxlen=50000)

        # How many experiences are used in one learning step
        self.batch_size = 32

        # Loss function: gentle on big errors (like the -15 death penalty)
        self.loss_func = torch.nn.SmoothL1Loss()

        # Number of possible actions (2: right, right + jump)
        self.action_dim = action_dim

        # BURN IN: how many experiences the memory must hold BEFORE any learning starts.
        # At the very start the memory holds only a few experiences, all from the
        # first seconds of the level and all very similar. Learning from such a tiny,
        # repetitive memory would make the network memorise those few moments, so
        # the memory is filled up first. During burn in the agent only plays and
        # stores, so it runs fast.
        self.burnin = 10000

        # Learn only on every 3rd agent step.
        # One step here is one decision of the agent, which is 4 game frames.
        # A learning update (forward and backward pass on 32 states) costs far more
        # time than playing one step, so learning on every step would make training
        # about 3 times slower. Every experience still gets reused many times,
        # because each update samples 32 experiences from the whole memory.
        self.learn_every = 3

        # Copy the online network into the target network every 10000 agent steps
        self.sync_every = 10000

        # Pick the fastest hardware available
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("device set to mps")
        elif torch.cuda.is_available():
            self.device = "cuda"
            print("device set to cuda")
        else:
            self.device = "cpu"
            print("device set to cpu")

        # The online and target networks, moved to the chosen device
        self.net = MarioNet(state_dim, self.action_dim).to(self.device)

        # The optimizer must be created after the net, because it needs its parameters
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)

        # Epsilon greedy settings: start fully random, shrink after every action
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1

        # How many actions the agent has taken so far
        self.curr_step = 0

    def act(self, state):
        """Choose an action for one state, either random (explore) or best (exploit)."""

        if random.random() < self.exploration_rate:
            # Explore: pick a random action
            action = np.random.randint(self.action_dim)
        else:
            # Exploit: ask the online network, no gradients needed for acting
            with torch.no_grad():
                # LazyFrames to a normal array of shape (4, 84, 84)
                state_array = np.array(state)
                # To a tensor on the device, unsqueeze adds the batch dimension (1, 4, 84, 84)
                state_tensor = torch.tensor(state_array, dtype=torch.float32, device=self.device).unsqueeze(0)
                # One Q value per action, shape (1, 2)
                q_values = self.net(state_tensor, model="online")
                # The index of the largest Q value is the action to take
                action = torch.argmax(q_values, axis=1).item()

        # Shrink epsilon a little, but never below the minimum
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate * self.exploration_rate_decay)

        # Count this action
        self.curr_step = self.curr_step + 1

        return action

    def cache(self, state, next_state, action, reward, done):
        """Store one experience. States stay as LazyFrames to save memory."""

        self.memory.append((state, next_state, action, reward, done))

    def recall(self):
        """Take a random batch from memory and turn it into tensors."""

        # Pick batch_size random experiences, each is one 5 value tuple
        batch = random.sample(self.memory, self.batch_size)

        # Five empty lists, one for each kind of value
        states = []
        next_states = []
        actions = []
        rewards = []
        done_list = []

        # Split every experience into its five parts
        for experience in batch:
            state, next_state, action, reward, done = experience
            # np.array turns LazyFrames into a real array, only for these 32 states
            states.append(np.array(state))
            next_states.append(np.array(next_state))
            actions.append(action)
            rewards.append(reward)
            done_list.append(done)

        # np.stack glues 32 arrays into one (32, 4, 84, 84), then it becomes a tensor
        states_tensor = torch.tensor(np.stack(states, axis=0), dtype=torch.float32, device=self.device)
        next_states_tensor = torch.tensor(np.stack(next_states, axis=0), dtype=torch.float32, device=self.device)
        # Actions are used as positions later, so they must be integers (long)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        # Rewards and done flags are used in arithmetic, so they are floats
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        done_tensor = torch.tensor(done_list, dtype=torch.float32, device=self.device)

        return states_tensor, next_states_tensor, actions_tensor, rewards_tensor, done_tensor

    def td_estimate(self, state, action):
        """
        What the online network currently believes.
        Returns the Q value of the action that was really taken, shape (32,).
        Gradients flow here, because this is the side that gets trained.
        """

        # Q values for both actions of all 32 states, shape (32, 2)
        q_values = self.net(state, model="online")

        # Row i, column action[i]: keep only the Q value of the action taken
        return q_values[np.arange(0, self.batch_size), action]

    def td_target(self, reward, next_state, done):
        """
        What the online network should have predicted (the label), shape (32,).
        Double DQN: the online network picks the best next action,
        the target network scores it. No gradients flow here.
        """

        with torch.no_grad():

            # Online network picks: its Q values for the next states, shape (32, 2)
            next_q_online = self.net(next_state, model="online")
            # Index of the best next action in each row, shape (32,)
            best_action = torch.argmax(next_q_online, axis=1)

            # Target network scores: its Q values for the same next states, shape (32, 2)
            next_q_target = self.net(next_state, model="target")
            # Take the target's value at the action the online network chose
            next_q = next_q_target[np.arange(0, self.batch_size), best_action]

            # reward now + discounted future, and (1 - done) removes the future
            # when the episode ended
            return reward + (1 - done) * self.gamma * next_q

    def update_q_online(self, td_estimate, td_target):
        """
        The step where the network actually learns.
        Compares prediction with label and nudges the online weights.
        Returns the loss as a plain number for logging.
        """

        # Clear gradients left over from the previous update
        self.optimizer.zero_grad()

        # How wrong the predictions are, as one number
        loss = self.loss_func(td_estimate, td_target)

        # Work out how each weight contributed to the error
        loss.backward()

        # Move the weights a small step to reduce the error
        self.optimizer.step()

        return loss.item()

    def sync_q_target(self):
        """Copy the online network's weights into the target network."""

        # state_dict is the dictionary of a network's weights.
        # load_state_dict copies those weights into another network of the same shape.
        # After this the two networks are identical again.
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self):
        """
        Called once after every agent step.
        It decides if anything should happen on this step.
        Returns (mean Q estimate, loss) when it learned, otherwise (None, None).
        It always returns two values, so the caller can unpack it safely.
        """

        # Sync check. This is a plain "if", not an "elif", so the checks below
        # still run on a sync step and the method always returns two values.
        if self.curr_step % self.sync_every == 0:
            self.sync_q_target()

        # Burn in check: the memory is still small, so do not learn yet.
        # It uses the memory size, so it also works after resuming a saved run,
        # because the memory itself is not saved and starts empty again.
        if len(self.memory) < self.burnin:
            return None, None

        # Learn only when the step number divides by learn_every.
        # Example with learn_every = 3: step 10001 gives remainder 2 so skip,
        # step 10002 gives remainder 0 so learn, step 10003 skip, step 10005 learn.
        if self.curr_step % self.learn_every != 0:
            return None, None

        # From here on this step is a learning step.

        # Take a random batch of 32 experiences from memory as tensors
        state, next_state, action, reward, done = self.recall()

        # What the online network currently predicts for the actions taken
        td_est = self.td_estimate(state, action)

        # What the Double DQN label says it should have predicted
        td_tgt = self.td_target(reward, next_state, done)

        # One update of the online network, returns the loss as a plain number
        loss = self.update_q_online(td_est, td_tgt)

        # Return the average predicted Q value and the loss, both for logging
        return td_est.mean().item(), loss

    def save(self, path):
        """
        Save the training progress to a file.
        path is the file name chosen by the caller, for example "mario_latest.pt"
        or "mario_best.pt", so the same method can write different files.
        The memory is not saved because it is too big.
        """

        # One dictionary holds everything needed to restore the run later
        save_dict = {
            # Weights of both the online and the target network
            "model": self.net.state_dict(),
            # Adam's running averages, so a resumed run does not restart cold
            "optimizer": self.optimizer.state_dict(),
            # Where the exploration rate and step counter had reached
            "exploration_rate": self.exploration_rate,
            "curr_step": self.curr_step
        }

        # Write to a temporary file first, then rename it over the real file.
        # If the program is killed mid write, the old checkpoint stays intact.
        temp_path = path + ".tmp"

        # Save to the temporary file
        torch.save(save_dict, temp_path)

        # Rename the temporary file to the real name
        os.replace(temp_path, path)

    def load(self, path):
        """
        Restore the training progress from a file written by save.
        path is the file name, for example "mario_latest.pt".
        """

        # map_location puts the loaded tensors on the current device (mps, cuda or cpu)
        # path is whatever file path the caller passes in
        checkpoint = torch.load(path, map_location=self.device)

        # Restore the weights of both networks and the optimizer state
        self.net.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # Restore the exploration rate and the step counter, so the run continues
        # where it stopped. The keys must match exactly the ones used in save.
        self.exploration_rate = checkpoint["exploration_rate"]
        self.curr_step = checkpoint["curr_step"]
