import numpy as np
import math
import torch
import random
import copy


class Gameboard:
    def __init__(self):
        self.boardsize = 4  # The size of one side of the board.
        self.score = 0
        self.board = np.zeros((self.boardsize, self.boardsize), dtype=np.int32)
        self.board_tensor = torch.zeros((self.boardsize, self.boardsize), dtype=torch.int32)
        self.place_random()
        self.place_random()

    def copy(self):
        return copy.deepcopy(self)

    def print(self, show_score=False):
        print(self.board)
        if show_score:
            print('Score: {}'.format(self.score))
        print()

    def np_board(self):
        return self.board

    # Place a number in a random place on the board. If that random position is already filled, it chooses a different
    # position. The default number is 2.
    def place_random(self, number=2):
        (x_coordinate, y_coordinate) = np.random.randint(0, self.boardsize, 2)
        # print(x_coordinate, y_coordinate)
        if self.board[x_coordinate, y_coordinate] == 0:
            self.board[x_coordinate, y_coordinate] = number
        else:
            self.place_random(number)

    def collapse_right(self):
        # Loop over it three times so it moves blocks as far as possible
        for _ in range(0, 3):
            for i in range(0, self.boardsize):
                # print('Row {}:{}'.format(i, self.board[i]))
                for j in reversed(range(0, self.boardsize)):
                    # print('Checking if position {} is in range'.format(j+1))
                    if j+1 in range(0, self.boardsize):
                        # print('({},{}) has {}; ({},{}) has {}'.format(i, j, self.board[i, j],
                        # i, j + 1, self.board[i, j + 1]))
                        # print('Position {} is in range.'.format(j+1))
                        if self.board[i, j+1] == 0:
                            # print('Next position is 0')
                            self.board[i, j+1] = self.board[i, j]
                            self.board[i, j] = 0
                        elif self.board[i, j+1] == self.board[i, j]:
                            # print('Next position is identical')
                            self.score += self.board[i, j]
                            self.board[i, j+1] *= 2
                            self.board[i, j] = 0

    def collapse_left(self):
        # Loop over it three times so it moves blocks as far as possible
        for _ in range(0, 3):
            for i in range(0, self.boardsize):
                # print('Row {}:{}'.format(i, self.board[i]))
                for j in range(0, self.boardsize):
                    # print('Checking if position {} is in range'.format(j+1))
                    if j-1 in range(0, self.boardsize):
                        # print('({},{}) has {}; ({},{}) has {}'.format(i, j, self.board[i, j],
                        # i, j + 1, self.board[i, j + 1]))
                        # print('Position {} is in range.'.format(j+1))
                        if self.board[i, j-1] == 0:
                            # print('Next position is 0')
                            self.board[i, j-1] = self.board[i, j]
                            self.board[i, j] = 0
                        elif self.board[i, j-1] == self.board[i, j]:
                            # print('Next position is identical')
                            self.score += self.board[i, j]
                            self.board[i, j-1] *= 2
                            self.board[i, j] = 0

    def collapse_down(self):
        # Loop over it three times so it moves blocks as far as possible
        for _ in range(0, 3):
            for i in range(0, self.boardsize):
                # print('Column {}:{}'.format(i, self.board[:i]))
                for j in reversed(range(0, self.boardsize)):
                    # print('Checking if position {} is in range'.format(j+1))
                    if j+1 in range(0, self.boardsize):
                        # print('({},{}) has {}; ({},{}) has {}'.format(i, j, self.board[i, j],
                        # i, j + 1, self.board[i, j + 1]))
                        # print('Position {} is in range.'.format(j+1))
                        if self.board[j+1, i] == 0:
                            # print('Next position is 0')
                            self.board[j+1, i] = self.board[j, i]
                            self.board[j, i] = 0
                        elif self.board[j+1, i] == self.board[j, i]:
                            # print('Next position is identical')
                            self.score += self.board[j, i]
                            self.board[j+1, i] *= 2
                            self.board[j, i] = 0

    def collapse_up(self):
        # Loop over it three times so it moves blocks as far as possible
        for _ in range(0, 3):
            for i in range(0, self.boardsize):
                # print('Column {}:{}'.format(i, self.board[:i]))
                for j in range(0, self.boardsize):
                    # print('Checking if position {} is in range'.format(j+1))
                    if j-1 in range(0, self.boardsize):
                        # print('({},{}) has {}; ({},{}) has {}'.format(i, j, self.board[i, j],
                        # i, j - 1, self.board[i, j - 1]))
                        # print('Position {} is in range.'.format(j-1))
                        if self.board[j-1, i] == 0:
                            # print('Next position is 0')
                            self.board[j-1, i] = self.board[j, i]
                            self.board[j, i] = 0
                        elif self.board[j-1, i] == self.board[j, i]:
                            # print('Next position is identical')
                            self.score += self.board[j, i]
                            self.board[j-1, i] *= 2
                            self.board[j, i] = 0

    def simulate_collapse_right(self):
        # Loop over it three times so it moves blocks as far as possible
        simulated_board = self.board.copy()
        for _ in range(0, 3):
            for i in range(0, self.boardsize):
                # print('Row {}:{}'.format(i, self.board[i]))
                for j in reversed(range(0, self.boardsize)):
                    # print('Checking if position {} is in range'.format(j+1))
                    if j + 1 in range(0, self.boardsize):
                        # print('({},{}) has {}; ({},{}) has {}'.format(i, j, self.board[i, j],
                        # i, j + 1, self.board[i, j + 1]))
                        # print('Position {} is in range.'.format(j+1))
                        if simulated_board[i, j + 1] == 0:
                            # print('Next position is 0')
                            simulated_board[i, j + 1] = simulated_board[i, j]
                            simulated_board[i, j] = 0
                        elif simulated_board[i, j + 1] == simulated_board[i, j]:
                            # print('Next position is identical')
                            simulated_board[i, j + 1] *= 2
                            simulated_board[i, j] = 0

    def simulate_collapse_down(self):
        # Loop over it three times so it moves blocks as far as possible
        simulated_board = self.board.copy()
        for _ in range(0, 3):
            for i in range(0, self.boardsize):
                # print('Column {}:{}'.format(i, self.board[:i]))
                for j in reversed(range(0, self.boardsize)):
                    # print('Checking if position {} is in range'.format(j+1))
                    if j + 1 in range(0, self.boardsize):
                        # print('({},{}) has {}; ({},{}) has {}'.format(i, j, self.board[i, j],
                        # i, j + 1, self.board[i, j + 1]))
                        # print('Position {} is in range.'.format(j+1))
                        if simulated_board[j + 1, i] == 0:
                            # print('Next position is 0')
                            simulated_board[j + 1, i] = simulated_board[j, i]
                            simulated_board[j, i] = 0
                        elif simulated_board[j + 1, i] == simulated_board[j, i]:
                            # print('Next position is identical')
                            simulated_board[j + 1, i] *= 2
                            simulated_board[j, i] = 0

    def simulate_collapse_left(self):
        # Loop over it three times so it moves blocks as far as possible
        simulated_board = self.board.copy()
        for _ in range(0, 3):
            for i in range(0, self.boardsize):
                # print('Row {}:{}'.format(i, self.board[i]))
                for j in range(0, self.boardsize):
                    # print('Checking if position {} is in range'.format(j+1))
                    if j - 1 in range(0, self.boardsize):
                        # print('({},{}) has {}; ({},{}) has {}'.format(i, j, self.board[i, j],
                        # i, j + 1, self.board[i, j + 1]))
                        # print('Position {} is in range.'.format(j+1))
                        if simulated_board[i, j - 1] == 0:
                            # print('Next position is 0')
                            simulated_board[i, j - 1] = simulated_board[i, j]
                            simulated_board[i, j] = 0
                        elif simulated_board[i, j - 1] == simulated_board[i, j]:
                            # print('Next position is identical')
                            simulated_board[i, j - 1] *= 2
                            simulated_board[i, j] = 0

    def simulate_collapse_up(self):
        # Loop over it three times so it moves blocks as far as possible
        simulated_board = self.board.copy()
        for _ in range(0, 3):
            for i in range(0, self.boardsize):
                # print('Column {}:{}'.format(i, self.board[:i]))
                for j in range(0, self.boardsize):
                    # print('Checking if position {} is in range'.format(j+1))
                    if j - 1 in range(0, self.boardsize):
                        # print('({},{}) has {}; ({},{}) has {}'.format(i, j, self.board[i, j],
                        # i, j - 1, self.board[i, j - 1]))
                        # print('Position {} is in range.'.format(j-1))
                        if simulated_board[j - 1, i] == 0:
                            # print('Next position is 0')
                            simulated_board[j - 1, i] = simulated_board[j, i]
                            simulated_board[j, i] = 0
                        elif simulated_board[j - 1, i] == simulated_board[j, i]:
                            # print('Next position is identical')
                            simulated_board[j - 1, i] *= 2
                            simulated_board[j, i] = 0

    def rotate_board(self, board, number_of_rotations):
        # number of rotations is in quarter circles, with positive being counter-clockwise
        # returns a copy of the board, but rotated
        temporary_board = board
        return temporary_board.rot90(number_of_rotations)

    def simulate_move(self, direction):
        # Creates a copy of the board, rotates it so that the desired collapse directory is pointed down, collapses
        # down, then rotates it back to its proper orientation.
        # parameter direction is a string that says 'up', 'down', 'left', or 'right'
        move_dictionary = {'up': 2,
                           'down': 0,
                           'left': 1,
                           'right': -1}
        simulated_board = self.board_tensor  # make a copy
        simulated_board = self.rotate_board(simulated_board, move_dictionary[direction])  # rotate the copy
        # Do the collapse
        for _ in range(0, 3):
            for i in range(0, self.boardsize):
                # print('Column {}:{}'.format(i, self.board[:i]))
                for j in reversed(range(0, self.boardsize)):
                    # print('Checking if position {} is in range'.format(j+1))
                    if j + 1 in range(0, self.boardsize):
                        # print('({},{}) has {}; ({},{}) has {}'.format(i, j, self.board[i, j],
                        # i, j + 1, self.board[i, j + 1]))
                        # print('Position {} is in range.'.format(j+1))
                        if simulated_board[j + 1, i] == 0:
                            # print('Next position is 0')
                            simulated_board[j + 1, i] = simulated_board[j, i]
                            simulated_board[j, i] = 0
                        elif simulated_board[j + 1, i] == simulated_board[j, i]:
                            # print('Next position is identical')
                            simulated_board[j + 1, i] *= 2
                            simulated_board[j, i] = 0
        simulated_board = self.rotate_board(simulated_board, -1 * move_dictionary[direction])  # rotate back
        return simulated_board

    def simulate_move_successful(self, index):

        moves = {
            0: 'up',
            1: 'down',
            2: 'left',
            3: 'right',
            4: 'nothing'
        }
        temporary_board = np.copy(self.board)
        direction = moves[index]
        simulated_board = self.simulate_move(direction)
        if np.array_equal(simulated_board, temporary_board):
            # print('Simulated move not successful.')
            return False
        return True

    def collapse_nothing(self):
        return

    def is_board_full(self):
        board_full = not (self.boardsize ** 2 - np.count_nonzero(self.board))
        if board_full:
            for index in range(3):
                result = self.simulate_move_successful(index)
                if not result:
                    return True  # The move was not successful, so we return that the board is full
        return False

    def move(self, direction, show_score=False, print_board=True):
        # direction is a string representing the direction to collapse
        # show_score tells the print function to show the score or not after moving.

        # Exit codes:
        # 0: Move not successful (no change in board)
        # -1: board is full
        # 1: Move successful, tile added

        # parameter direction is a string that says 'up', 'down', 'left', or 'right'
        move_dictionary = {'up': self.collapse_up,
                           'down': self.collapse_down,
                           'left': self.collapse_left,
                           'right': self.collapse_right,
                           'nothing': self.collapse_nothing}

        # Make a temporary copy of the previous board
        temporary_board = np.copy(self.board)

        # Execute the proper collapse function for the given direction.
        if print_board:
            print('Moving in direction {}'.format(direction))
        move_dictionary[direction]()

        # print(self.board, temporary_board)

        # Add a random tile if a move was successful
        if self.is_board_full():
            if print_board:
                self.print(show_score)
                print('BOARD FULL')
            return -1
        if np.array_equal(self.board, temporary_board):
            if print_board:
                self.print()
                print('Move not successful. Score: {}'.format(show_score))
            return 0
        else:
            # print('Previous move successful')
            # print('Is board full: {}'.format(self.is_board_full()))
            if self.is_board_full():
                if print_board:
                    self.print(show_score)
                    print('BOARD FULL')
                return -1
            else:
                self.place_random(self.generate_random_tile())
                if print_board:
                    self.print(show_score)
                return 1

    def generate_random_tile(self):
        # Generate a 2, 90% of the time
        return 2 if np.random.random() < 0.9 else 4

    def get_highest_tile(self):
        return np.max(self.board)

    def get_board_total(self):
        return np.sum(self.board)

    def get_number_of_active_tiles(self):
        return np.count_nonzero(self.board)

    def calculate_score(self):
        return math.log(math.pow(self.get_highest_tile(), 2) / self.get_number_of_active_tiles(), 2)


    def check_if_game_over(self):
        if self.is_board_full():
            # print('Board is full.')
            horizontal_difference = np.diff(self.board)
            vertical_difference = np.diff(np.transpose(self.board))
            # print(np.count_nonzero(horizontal_difference))
            if np.count_nonzero(horizontal_difference) < self.boardsize**2-self.boardsize:
                # np.diff reduces the number of rows by one.
                return False
            elif np.count_nonzero(vertical_difference) < self.boardsize**2-self.boardsize:
                # np.diff reduces the number of rows by one.
                return False
            else:
                return True
        else:
            return False

    def get_board(self):
        board = self.board.astype(np.float32)
        board_tensor = torch.from_numpy(board)
        return board_tensor

    def get_board_log(self):
        board_tensor = self.get_board()
        board_tensor[board_tensor == 0] = 1
        board_tensor = torch.log2(board_tensor)
        return board_tensor

    def get_board_normalized(self):
        board = self.board.astype(np.float32)
        board_tensor = torch.from_numpy(board)
        board_tensor[board_tensor == 0] = 1
        board_tensor = torch.log2(board_tensor)
        mean = board_tensor.mean()
        std = board_tensor.std()
        return (board_tensor - mean) / std

    def frame_step(self, input_actions):
        reward = 0.1
        terminal = False

        moves = {
            0: 'up',
            1: 'down',
            2: 'left',
            3: 'right',
            4: 'nothing'
        }
        board_before_move = self.get_board_log().int().flatten()
        # tiles_before_move = self.get_number_of_active_tiles()
        tiles_before_move = torch.bincount(board_before_move, minlength=15)

        action_index = torch.argmax(input_actions).item()
        result = self.move(moves[action_index])
        # tiles_after_move=self.get_number_of_active_tiles()
        board_after_move = self.get_board_log().int().flatten()
        tiles_after_move = torch.bincount(board_after_move, minlength=15)

        difference = tiles_after_move - tiles_before_move
        positive_differences = torch.nn.ReLU()(difference).int()
        powers_of_two = 2 ** torch.linspace(0, 14, steps=15).int()
        powers_of_two[powers_of_two == 2] = 0
        powers_of_two[powers_of_two == 1] = 0
        # print(positive_differences)

        if result == -1:
            terminal = True
            reward = -30
            self.__init__()
        elif result == 0:
            terminal = False
            reward = -10
        elif result == 1:
            terminal = False
            reward = torch.matmul(positive_differences, powers_of_two).item()

        return reward, terminal

    def get_score(self):
        return self.score()

    def play_random_game(self, ai=None, length=None):
        move_dictionary = {0: 'up',
                           1: 'down',
                           2: 'left',
                           3: 'right'}
        if ai is None:
            lost = False
            clone = self.copy()
            iterations = 0
            while (not lost) and (length is None or iterations < length):
                random_move = move_dictionary[random.randint(0, 3)]
                clone.move(random_move, print_board=False)
                lost = clone.is_board_full()
                iterations += 1

            return clone.score
        else:
            ''''''

    def play_multiple_random_games(self, number=100, ai=None, length=None):
        agregate_score = 0
        for _ in range(0, number):
            agregate_score += self.play_random_game(ai=ai, length=length)
        return agregate_score / number  # return the average score of all the games


'''
gb = Gameboard()
gb.print()

for i in range(0, 1000):
    print('Step number {}'.format(i))
    if gb.check_if_game_over():
        break
    if not (gb.move('right', show_score=True) or gb.move('down', show_score=True)):
        gb.move('left', show_score=True)
'''
