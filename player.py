import torch
import torch.nn as nn


class Player(nn.Module):
    def __init__(self):
        super(Player, self).__init__()
        self.fc1 = nn.Linear(4**2, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = torch.flatten(x)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = nn.Sigmoid()(self.fc3(out))
        return out
