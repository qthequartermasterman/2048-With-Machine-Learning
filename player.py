import torch
import torch.nn as nn


class Player(nn.Module):
    def __init__(self):
        super(Player, self).__init__()

        self.number_of_actions = 4
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        self.replay_memory_size = 10000
        self.minibatch_size = 1
        self.board_size = 4

        self.conv1 = nn.Conv2d(4, 16, 3)
        # self.fc1 = nn.Linear(self.board_size**2, 16)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, self.number_of_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        # out = torch.flatten(x)
        # out = self.relu(self.fc1(out))
        out = self.conv1(x)
        out = self.relu(out)
        out = torch.flatten(out)
        out = self.relu(self.fc2(out))
        out = nn.Sigmoid()(self.fc3(out))
        return out
