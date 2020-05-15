import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channel):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, residual=None):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        post = x * y.expand_as(x)
        if residual is not None:
            return nn.Conv2d(post.shape[1] + residual.shape[1], x.shape[1], kernel_size=1)(
                torch.cat([post, residual], dim=1))
            # return (post+residual)/2  # add the skip connection from previous layer
        else:
            return post


class Player(nn.Module):
    def __init__(self):
        super(Player, self).__init__()

        self.number_of_actions = 4
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 200000000
        self.replay_memory_size = 10000
        self.minibatch_size = 16
        self.board_size = 4
        ''' #Jim v2
        self.conv1 = nn.Conv2d(4, 32, 3)
        # self.fc1 = nn.Linear(self.board_size**2, 16)
        self.conv2 = nn.Conv2d(32, 32, 2)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, self.number_of_actions)
        self.relu = nn.ReLU()
        '''

        self.conv1 = nn.Conv2d(4, 32, 2)
        self.se_block_1 = SEBlock(32)
        self.conv2 = nn.Conv2d(32, 64, 2)
        self.se_block_2 = SEBlock(64)
        self.conv3 = nn.Conv2d(64, 64, 2)
        self.se_block_3 = SEBlock(64)
        self.fc = nn.Linear(256, self.number_of_actions)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        '''Jim v2
        # out = torch.flatten(x)
        # out = self.relu(self.fc1(out))
        out = self.conv1(x)
        out = self.relu(out)
        out = self.relu(self.conv2(out))
        out = out.view(out.size()[0], -1)
        out = self.relu(self.fc2(out))
        out = nn.Sigmoid()(self.fc3(out))
        return out
        '''
        out = x
        out = self.relu(self.conv1(out))
        copy1 = out
        copy1 = self.se_block_1(copy1)
        out = out * copy1

        out = self.relu(self.conv2(out))
        copy2 = out
        copy2 = self.se_block_2(copy2)
        out = out * copy2

        copy3 = out
        out = self.relu(self.conv3(out))
        copy3 = self.se_block_3(copy3)
        # print(out.shape, copy.shape)
        out = out * copy3

        out = out.view(out.size()[0], -1)
        # print(out.shape)
        out = self.fc(out)
        out = self.softmax(out)
        return out
