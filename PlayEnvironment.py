import player
import gameboard
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn

gb = gameboard.Gameboard()
play = player.Player()

move_key = {0: 'left',
            1: 'right',
            2: 'up',
            3: 'down'}

prior_move = -1
prior_move_successful = True


class GameLoss(nn.Module):
    def __init__(self, previousScore=0):
        super(GameLoss).__init__()

    def forward(self, x):
        return


def choose_next_move(previous_move, previous_move_successful):
    # random_ratio represents how frequently, on average, the next move will be random
    board_tensor = gb.get_board_normalized()
    next_move_tensor = play(board_tensor)
    _, next_move = torch.max(next_move_tensor, -1)
    next_move = next_move.item()
    # print(board)
    print(next_move_tensor, next_move)
    return next_move if next_move != previous_move and not previous_move_successful else np.random.randint(0, 4)
    # Return the move chosen by the neural network if it isn't the previous move and the previous move wasn't successful
    # otherwise return a random move.


'''
while not gb.check_if_game_over():
    move = choose_next_move(prior_move, prior_move_successful)
    print('I am moving in direction {}'.format(move_key[move]))
    previous_move_result = gb.move(move_key[move], show_score=True)
    prior_move_successful = not not previous_move_result
    prior_move = move
    print('\n\n\n\n\n')
'''

net = play
optimizer = optim.SGD(net.parameters(), lr=.1, momentum=0.9, weight_decay=5e-4)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def try_move_and_get_loss():
    ''''''


def train(epoch_number):
    print('\nEpoch: %d   Learning rate: %f' % (epoch_number, optimizer.param_groups[0]['lr']))
    print("\nAllocated GPU memory:",
          torch.cuda.memory_allocated() if torch.cuda.is_available() else 'CUDA NOT AVAILABLE')
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    while True:  # TODO: find a better loop criteria
        inputs = gb.board_tensor  # Get inputs
        inputs = inputs.to(inputs)  # Send to the GPU, if available.

        optimizer.zero_grad()
        outputs = net(inputs)  # Get the results from the player network

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)  # Move the stuff to GPU after CPU work
        # print('input same:', inputs-inputs2)
        optimizer.zero_grad()
        outputs = net(inputs)  # Check if we should mask the images.
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    file_path = './records/cifar100/cifar_' + str(
        args.cifar) + '_' + args.netName + str(args.pre_process) + '_train.txt'
    record_str = str(epoch_number) + '\t' + "%.3f" % (train_loss / (batch_idx + 1)) + '\t' + "%.3f" % (
            100. * correct / total) + '\n'
    write_record(file_path, record_str)
