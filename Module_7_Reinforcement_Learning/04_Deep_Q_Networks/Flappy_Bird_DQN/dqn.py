import torch
import torch.nn as nn


class DQN(nn.Module):

    # state_dim is the input size, action_dim is the output size, hidden_dim is the neurons in the hidden layer
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(DQN, self).__init__()

        # one hidden layer with ReLU, the output layer gives one Q value per action
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    # pass the state through the network to get the Q values
    def forward(self, x):
        return self.model(x)
