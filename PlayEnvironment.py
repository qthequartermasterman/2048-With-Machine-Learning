import player
import gameboard
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import random
import os
import time

move_key = {0: 'left',
            1: 'right',
            2: 'up',
            3: 'down'}

prior_move = -1
prior_move_successful = True


def train(model, start):
    gb = gameboard.Gameboard()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.MSELoss()

    replay_memory = []

    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    reward, terminal = gb.frame_step(action)  # image_data, reward, terminal = game_state.frame_step(action)
    board_data = gb.get_board_normalized().to(device)[None, ...]
    state = torch.cat((board_data, board_data, board_data, board_data)).unsqueeze(0)

    epsilon = model.initial_epsilon
    iteration = 0

    epsilon_decrements = np.linspace(model.initial_epsilon, model.final_epsilon, model.number_of_iterations)

    while iteration < model.number_of_iterations:
        # get output from the neural network
        print('ITERATION {} =================='.format(iteration + 1))

        output = model(state)
        print('Output: {}'.format(output))

        # initialize action
        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        action.to(device)

        # epsilon greedy exploration

        random_action = random.random() <= epsilon
        if random_action:
            print("Performed random action!")

        action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                        if random_action else torch.argmax(output)][0]
        print('First action index: {}'.format(action_index.item()))
        if not gb.simulate_move_successful(action_index.item()):
            # We played something that would not result in a move.
            print('Trying second option')
            output[torch.argmax(output)] = 0
            action_index = [torch.randint(model.number_of_actions, torch.Size([]), dtype=torch.int)
                            if random_action else torch.argmax(output)][0]
            print('Second action index: {}'.format(action_index.item()))

        action_index.to(device)

        action[action_index] = 1

        # get next state and reward
        reward, terminal = gb.frame_step(action)

        board_data_1 = gb.get_board_normalized()[None, ...]
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], board_data_1)).unsqueeze(0)

        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        # save transition to replay memory
        replay_memory.append((state, action, reward, state_1, terminal))

        # if replay memory is full, remove the oldest transition
        if len(replay_memory) > model.replay_memory_size:
            replay_memory.pop(0)

        # epsilon annealing
        epsilon = epsilon_decrements[iteration]

        # sample random minibatch
        minibatch = random.sample(replay_memory, min(len(replay_memory), model.minibatch_size))

        # unpack minibatch
        state_batch = torch.cat(tuple(d[0] for d in minibatch)).to(device)
        action_batch = torch.cat(tuple(d[1] for d in minibatch)).to(device)
        reward_batch = torch.cat(tuple(d[2] for d in minibatch)).to(device)
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch)).to(device)

        output_1_batch = model(state_1_batch)

        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + model.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch))))

        # extract Q-value
        output_state_batch = model(state_batch)
        blarg = output_state_batch * action_batch
        q_value = torch.sum(blarg, dim=1)

        # PyTorch accumulates gradients by default, so they need to be reset in each pass
        optimizer.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()

        # calculate loss
        loss = criterion(q_value, y_batch)
        print(loss)

        # do backward pass
        loss.backward()
        optimizer.step()

        # set state to be state_1
        state = state_1
        iteration += 1

        if iteration % 25 == 0:
            torch.save(model, "pretrained_model/current_model_" + str(iteration) + ".pth")

        print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
              action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
              np.max(output.cpu().detach().numpy()))


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


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)


def main(mode):
    cuda_is_available = torch.cuda.is_available()

    if mode == 'test':
        model = torch.load('pretrained_model/current_model_2000000.pth',
                           map_location='cpu' if not cuda_is_available else None).eval()

        if cuda_is_available:
            model = model.cuda()

        # test(model)

    elif mode == 'train':
        if not os.path.exists('pretrained_model/'):
            os.mkdir('pretrained_model/')

        model = player.Player()

        if cuda_is_available:  # put on GPU if CUDA is available
            model = model.cuda()

        model.apply(init_weights)
        start = time.time()

        train(model, start)


if __name__ == "__main__":
    main('train')
