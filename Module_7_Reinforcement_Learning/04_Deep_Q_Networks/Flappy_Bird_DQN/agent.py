"""
file for the agent
most of the code will be written here
steps we will follow are

    1. env. setup
    2. Define DQN architecture
    3. Create Experience Replay
    4. setup to train DQN on multiple episodes
    5. Epsilon Greedy (epsilon decay)
    6. Hyper parameter tuning
    7. Target Network
    8. Train and Test

we also have to select our device on which the game will run on, as
i am using a mac with m2 chip, it has a software called MPS for GPU

we also need to create a run() function which will help us do the
training as well as testing/visualizing etc

normally we do the training for a constant number of episodes but here we
will continue the training until our agent gets better and better and
reaches a ceiling.

We will manually stop the training using cmd + c or set a reward_threshold
limit to N and when that rewards limit is reached the training loop will end

For dynamicity we will put all the hyperparameters in a .yaml file and import from there

we will need to create epsilon greedy and epsilon decay policies

we will also need to convert the values into tensors to pass it to our DQN using pytorch
values to be converted into tensors are

    1. curr_state and next_state
    2. action
    3. reward

We will then define our target network in the training if condition only

we will also define our main optimize function where we will do the training
of our dqn and all the calculation of gradients, backpropagation etc. The steps are

    1. get experience from experience replay
    2. compute the target Q value (y_true or y_labels)
    3. compute the predictions using policy network (y_pred)
    4. compute loss between both
    5. backpropagate and optimize

in the end, we will write some conditions so our code is runnable in both training and testing mode from the terminal only

also, we will save all the runs and all in a separate directory


what the agent sees
    the game can return two kinds of state
    use_lidar = False gives 12 simple numbers (the positions of the next pipes, the height of the bird,
    its speed and its rotation), use_lidar = True gives 180 laser distances instead
    this project uses the 12 numbers, note 1 below explains why

when the network learns
    the network is trained on every single step of the game, inside the while loop, and the target network
    is refreshed after every network_sync_rate training steps, so it stays frozen in between (note 2 below)

notes from my experiments
    1. the lidar state (180 numbers). I trained with it first, once for 11,000 episodes and once for 14,500 episodes.
       The average reward stayed around +1 to +2 and the saved bird could not pass even one pipe.
       Reading the game's source showed one more thing. In lidar mode the game also gives -0.5 on every frame
       where any laser ray is short, which means the bird is close to some obstacle, so flying through a pipe gap
       is punished. On top of that every step is slow, because 180 rays are cast on every frame
       (around 4 hours for 14,500 episodes). The 12 number state avoids both problems.
    2. how often to learn. My first version called optimize only once per episode, below the while loop, and synced the
       target network right after it, so the target was always identical to the policy network. With the lidar state,
       moving optimize inside the loop did not fix anything, because the state was the real problem. With the 12 number
       state the difference was clear. After 3,000 to 5,000 episodes the average reward was about 10 when learning on every
       step and about 5 when learning once per episode, so that is the version in this file.
    3. which model to save. At first I saved the model whenever a single episode set a new best reward. That turned out to
       be unreliable, because one lucky game (for example an easy run of pipes) can come from a weak network, and in my test
       games that model often passed only one or two pipes. Now the model is saved when the average reward of the last
       50 episodes is a new best, which is a much steadier signal. The training rewards also include the few random actions
       that epsilon_min still allows, while testing is fully greedy.

how to run from the terminal
    python agent.py flappybirdv0 --train   (training mode)
    python agent.py flappybirdv0           (testing mode, loads the saved best model)
"""

# import libraries
import flappy_bird_gymnasium
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import itertools  # to make the training loop go indefinitely
from collections import deque  # to keep the rewards of the last few episodes
import yaml
import random
import os
import argparse  # to read the inputs typed in the terminal

# import DQN and experience replay classes
from dqn import DQN
from experience_replay import ReplayMemory

# check for mps, then cuda, otherwise use the cpu
if torch.backends.mps.is_available():
    device = "mps"
    print("device set to mps")
elif torch.cuda.is_available():
    device = "cuda"
    print("device set to cuda")
else:
    device = "cpu"
    print("device set to cpu")

# make directory for all the runs
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok = True)


class Agent:

    # define constructor to read all the params from the yaml file
    def __init__(self, param_set):

        # param_set is needed again below for the log and model file names
        self.param_set = param_set

        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        # store all the important values one by one
        self.env_id = params["env_id"]
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]
        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]
        self.network_sync_rate = params["network_sync_rate"]
        self.reward_threshold = params["reward_threshold"]

        # define loss function (MSE) and optimizer
        self.optimizer = None  # for now it is None, it becomes Adam inside run() when training
        self.loss_func = nn.MSELoss()

        # adding log and model files in the runs directory
        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.param_set}.pt")

    # defining the run() function
    # it will be used for both training and testing
    def run(self, is_training = True, render = False):

        # create env.
        # use_lidar = False gives the 12 simple numbers, use_lidar = True would give 180 laser distances (see note 1 at the top)
        # we dont want to render every training epoch so only when the render flag is true the render mode will turn to human
        env = gym.make(self.env_id, render_mode = "human" if render else None, use_lidar = False)

        # get the num of states and possible actions
        num_states = env.observation_space.shape[0]  # input dimensions (12 numbers)
        num_actions = env.action_space.n  # output dimensions
        hidden_dim = 256  # no. of neurons in the hidden layer

        # create the policy using the DQN class
        policy_dqn = DQN(num_states, num_actions, hidden_dim).to(device)

        # training only setup
        # all the things which are needed only for training
        if is_training:

            # Experience Replay
            memory = ReplayMemory(self.replay_memory_size)

            # initializing the epsilon value
            epsilon = self.epsilon_init

            # the target network
            # we will derive its parameters (weights and biases) from the policy dqn's
            # in short a copy of policy dqn
            target_dqn = DQN(num_states, num_actions, hidden_dim).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            # counting steps so that after every network_sync_rate steps we can copy policy_dqn into target_dqn
            steps = 0

            # initialising the optimizer from None to Adam
            self.optimizer = optim.Adam(policy_dqn.parameters(), lr = self.alpha)

            # tracking the best model
            # one lucky episode can be a very poor model, so the model is judged by the average reward
            # of the last 50 episodes and not by a single episode
            recent_rewards = deque([], maxlen = 50)
            best_average = float("-inf")

        # testing case, where we load the best saved model/policy
        # also putting the model in eval mode
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        # loop for many episodes
        for episode in itertools.count():

            # initialise these parameters
            terminated = False
            truncated = False
            episode_reward = 0

            # get the current state using env.reset
            # convert it into a tensor as well (float32 as the DQN layers use float32)
            curr_state, extra = env.reset()
            curr_state = torch.tensor(curr_state, dtype = torch.float32, device = device)

            # an episode ends when the game is over or when the reward threshold is reached
            while not (terminated or truncated) and episode_reward < self.reward_threshold:

                # using epsilon greedy only when in training
                if is_training and random.random() < epsilon:
                    # exploration, a random action converted to a tensor
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype = torch.int64, device = device)
                else:
                    # exploitation, the action with the highest Q value for the current state
                    # unsqueeze so the state gets the batch dimension the DQN expects and squeeze to remove it again
                    # as there is no learning here, we can use no_grad
                    # argmax already returns a tensor so no extra conversion is needed
                    with torch.no_grad():
                        action = policy_dqn(curr_state.unsqueeze(dim = 0)).squeeze().argmax()

                # Processing
                # .item() to get the plain value out of the action tensor
                next_state, reward, terminated, truncated, extra = env.step(action.item())

                # update rewards using the plain reward value, before it becomes a tensor
                episode_reward = episode_reward + reward

                # create tensors
                next_state = torch.tensor(next_state, dtype = torch.float32, device = device)
                reward = torch.tensor(reward, dtype = torch.float32, device = device)

                # push this step to the experience replay only while training
                # also add 1 to steps so that whenever it passes network_sync_rate we can update the target_dqn
                if is_training:
                    memory.push((curr_state, action, next_state, reward, terminated))
                    steps = steps + 1

                    # train on every single step, but only when we have
                    # more replays than the mini_batch_size fed into the neural network
                    if memory.buffer_size() > self.mini_batch_size:

                        # get sample
                        mini_batch = memory.sample(self.mini_batch_size)

                        # train and optimize it, most important function
                        # self. is used because optimize is a method of this Agent class
                        # it needs the Agent's own self.optimizer, self.loss_func and self.gamma
                        self.optimize(mini_batch, policy_dqn, target_dqn)

                        # sync the policy_dqn and target_dqn, steps now counts the training steps since the last sync
                        if steps > self.network_sync_rate:
                            target_dqn.load_state_dict(policy_dqn.state_dict())

                            # reset the steps again
                            steps = 0

                # update the state
                curr_state = next_state

            # printing total rewards for each episode (and epsilon when training)
            if is_training:
                print(f"Total rewards for episode {episode + 1}: {episode_reward}, epsilon used: {epsilon}")
            else:
                print(f"Total rewards for episode {episode + 1}: {episode_reward}")

            # everything below is for training only
            if is_training:

                # epsilon decay
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                # save the model and log it whenever the average of the last 50 episodes is a new best
                recent_rewards.append(episode_reward)
                average_reward = sum(recent_rewards) / len(recent_rewards)

                if len(recent_rewards) == recent_rewards.maxlen and average_reward > best_average:
                    best_average = average_reward
                    log_msg = f"best average reward of the last 50 episodes = {best_average} at episode : {episode + 1}"

                    # write in the log file
                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_msg + "\n")

                    # save model
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)

                # stop the training once the reward threshold is reached
                if episode_reward >= self.reward_threshold:
                    print(f"Reward threshold {self.reward_threshold} reached, training stopped")
                    break

        # only reached when the reward threshold stops the training
        # when we stop manually with cmd + c this line is not needed
        env.close()

    # optimize function, trains the policy dqn on one mini batch
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # mini_batch is a list of tuples like (state, action, next_state, reward, terminated)
        # the * unpacks the list into separate tuples and zip regroups them by position
        # so we get one tuple of all states, one of all actions and so on
        states, actions, next_states, rewards, terminated = zip(*mini_batch)

        # each state is a tensor of shape (180,)
        # stack joins them into one tensor of shape (batch_size, 180)
        # this lets the dqn process the whole batch in a single forward pass
        states = torch.stack(states)
        actions = torch.stack(actions)  # shape (batch_size,)
        next_states = torch.stack(next_states)
        rewards = torch.stack(rewards)  # shape (batch_size,)

        # terminated holds plain True/False values so we convert them to a float tensor of 1.0 and 0.0
        terminated = torch.tensor(terminated).float().to(device)

        # target Q values (y_label), no gradients needed as the target is only a label
        with torch.no_grad():
            # target_dqn(next_states) has shape (batch_size, num_actions) with the Q value of every action
            # max(dim = 1) takes the best action's Q value per row and returns (values, indices)
            # [0] keeps only the values, shape (batch_size,)
            # (1 - terminated) is 0 for finished games, which removes the future term
            # so the target becomes just the reward
            target_q = rewards + (1 - terminated) * self.gamma * target_dqn(next_states).max(dim = 1)[0]

        # predicted Q values (y_pred)
        # policy_dqn(states) has shape (batch_size, num_actions), the Q value of every action
        # we only want the Q value of the action that was actually taken in each experience
        # unsqueeze turns actions from (batch_size,) into (batch_size, 1) as gather needs the same number of dimensions
        # gather(dim = 1) picks from each row the column given by the index
        # squeeze turns the result from (batch_size, 1) back into (batch_size,)
        current_q = policy_dqn(states).gather(dim = 1, index = actions.unsqueeze(dim = 1)).squeeze()

        # calculate loss, MSE between predicted and target Q values (averaged over the batch)
        loss = self.loss_func(current_q, target_q)

        # reset the old gradients, backpropagate, then update the weights once for the whole batch
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# conditions for testing and training, so the code runs in both modes from the terminal
# this block only runs when the file is run directly
if __name__ == "__main__":

    # parse command line inputs
    parser = argparse.ArgumentParser(description = "Train or test model.")
    parser.add_argument("hyperparameters", help = "Name of the parameter set in parameters.yaml")
    parser.add_argument("--train", help = "Training mode", action = "store_true")
    args = parser.parse_args()

    # create the agent with the chosen parameter set
    dql = Agent(param_set = args.hyperparameters)

    # --train given: train without rendering (faster)
    if args.train:
        dql.run(is_training = True)
    # --train not given: test and render the game so we can watch it
    else:
        dql.run(is_training = False, render = True)