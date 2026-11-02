import copy
import torch
import torch.nn as nn


class MarioNet(nn.Module):
    """
    The Q network for Mario.

    Input : a state of shape (4, 84, 84), meaning 4 stacked grey frames
    Output: one Q value per action, so shape (2,) for "right" and "right + jump"

    It holds two copies of the same CNN:
    online : the one that is trained and picks actions
    target : a frozen copy, refreshed from online every so often,
             used to build the learning targets
    """

    def __init__(self, input_dim, output_dim):
        # nn.Module must be set up before any layer is assigned to self
        super().__init__()

        # input_dim is the tuple (channels, height, width), for example (4, 84, 84)
        channels, height, width = input_dim

        # The layers below are built for 84 by 84 frames, so stop early if the size is wrong
        if height != 84:
            raise ValueError(f"Expected height 84, got: {height}")
        if width != 84:
            raise ValueError(f"Expected width 84, got: {width}")

        # Size after each conv layer is (input - kernel) / stride + 1
        # conv1: (84 - 8) / 4 + 1 = 20
        # conv2: (20 - 4) / 2 + 1 = 9
        # conv3: (9 - 3) / 1 + 1 = 7
        # After flatten there are 64 filters * 7 * 7 = 3136 numbers
        flattened_size = 64 * 7 * 7

        self.online = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4),
            nn.ReLU(), # output features = 32, output size = 20*20
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(), # output features = 64, output size = 9*9
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(), # output features = 64, output size = 7*7
            nn.Flatten(), # output size = 64*7*7 i.e output features * output size
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

        # The target starts as an exact copy of the online network
        self.target = copy.deepcopy(self.online)

        # Freeze the target so training never changes it directly.
        # It is only updated by copying the online weights into it.
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, input, model):
        # model decides which of the two networks reads the input
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
        else:
            raise ValueError(f"model must be 'online' or 'target', got: {model}")


if __name__ == "__main__":
    # Quick check with a random fake state, batch of 1
    net = MarioNet((4, 84, 84), 2)
    fake_state = torch.rand(1, 4, 84, 84)

    online_out = net(fake_state, model="online")
    target_out = net(fake_state, model="target")

    print("online output shape:", online_out.shape)
    print("target output shape:", target_out.shape)

    # Count how many target parameters can still be trained, should be 0
    trainable_target = 0
    for param in net.target.parameters():
        if param.requires_grad:
            trainable_target = trainable_target + 1
    print("trainable target parameters:", trainable_target)

    # Count how many online parameters can be trained, should be above 0
    trainable_online = 0
    for param in net.online.parameters():
        if param.requires_grad:
            trainable_online = trainable_online + 1
    print("trainable online parameters:", trainable_online)

    # The two networks start identical, so their outputs should match exactly
    print("outputs identical at the start:", torch.equal(online_out, target_out))
