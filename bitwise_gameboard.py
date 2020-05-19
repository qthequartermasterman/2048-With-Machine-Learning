'''This gameboard should function identically to the outside world, but uses bitwise operations instead of tensor
operations, meaning that it should be /much/ faster. Inspired by @nneonneo's 2048 cpp structure at
https://github.com/nneonneo/2048-ai, used and adapted under the MIT license'''

import numpy as np
import torch
import copy
import math
import random
from bitwisetables import *
from bitwise_helper_functions import *
import time


class BitwiseGameboard:
    def __init__(self, starting_array=None, board_size=4):
        self.board_size = board_size  # The size of one side of the board.
        self.score = 0
        self.scorepenalty = 0  # Our score tables assume we always get random 2, so this will help us weed out random 4

        '''Initialize the values on the board'''
        if starting_array is None:
            # 64 bit int, where every 4 bits represents a cell, and every 16 bits represents a row
            self.board = 0
            self.place_random(number=2)
            self.place_random(number=2)
        else:
            self.board = self.np_array_to_uint64(starting_array)

    def copy(self):
        return copy.copy(self)

    def np_array_to_uint64(self, starting_array):
        """starting_array must be an np_array"""
        print('Converting np array to unint64')

        board_tensor = starting_array
        board_tensor[board_tensor == 0] = 1
        board_tensor = np.log2(board_tensor)

        board = 0
        i = 0
        for row in board_tensor:
            for c in row:
                board |= int(c) << (4 * i)
                i += 1
        return board

    def np_board(self, board=None):
        int_board = board if board is not None else self.board
        new_board = np.array([[0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0],
                              [0, 0, 0, 0]])
        for i in range(4):
            for j in range(4):
                power_value = int_board & 0xf
                # Print the power_value, unless it's zero. In which case print 1. Also makes sure to not go to a newline
                value = 0 if power_value == 0 else 1 << power_value  # If the power is 0, we want to print 0, not 1. If the power isn't 0, then we want 2**power_value
                new_board[i][j] = value
                int_board >>= 4
        return new_board

    def print(self, show_score=False):
        board = self.board  # We copy it, since we'll be bit-shifting it like crazy.
        print('64 bit representation: {}'.format(str(hex(board))))
        for i in range(4):
            for j in range(4):
                power_value = board & 0xf
                # Print the power_value, unless it's zero. In which case print 1. Also makes sure to not go to a newline
                print('{:6d}'.format(0 if power_value == 0 else 1 << power_value), end='')
                board >>= 4
            print('', end='\n')  # Make a new line
        print('', end='\n')  # Make a new line

        if show_score:
            print('Score: {}'.format(self.score))
        print()

    def count_empty(self):
        x = self.board
        # The rest of this function is directly from @nneonneo's, but adapted to python
        x |= (x >> 2) & 0x3333333333333333
        x |= (x >> 1)
        x = ~x & 0x1111111111111111
        # At this point each nibble is:
        #  0 if the original nibble was non-zero
        #  1 if the original nibble was zero
        # Next sum them all
        x += x >> 32
        x += x >> 16
        x += x >> 8
        x += x >> 4  # this can overflow to the next nibble if there were 16 empty positions
        return x & 0xf

    # Place a number in a random place on the board. If that random position is already filled, it chooses a different
    # position.
    def place_random(self, number=None):
        tile = number if number is not None else self.generate_random_tile()
        index = random.randint(0,
                               self.count_empty())  # We need to choose a random empty tile, so we will use the index-th empty tile
        tmp = self.board
        while True:
            # since tile is formatted as an empty board except for the random tile in the top left, we're going to shift
            # it around until it's in the right spot. Then we bitwise_or it with the board to place it.
            while (tmp & 0xf) != 0:
                tmp >>= 4
                tile <<= 4
            if index == 0:
                break
            index -= 1
            tmp >>= 4
            tile <<= 4
        self.board = self.board | tile

    def transpose(self, x):
        """x is some board as an int64"""
        a1 = x & 0xF0F00F0FF0F00F0F
        a2 = x & 0x0000F0F00000F0F0
        a3 = x & 0x0F0F00000F0F0000
        a = a1 | (a2 << 12) | (a3 >> 12)
        b1 = a & 0xFF00FF0000FF00FF
        b2 = a & 0x00FF00FF00000000
        b3 = a & 0x00000000FF00FF00
        return b1 | (b2 >> 24) | (b3 << 24)

    def collapse_up(self):
        ret = self.board
        t = self.transpose(self.board)
        ret ^= col_up_table[(t >> 0) & ROW_MASK] << 0
        ret ^= col_up_table[(t >> 16) & ROW_MASK] << 4
        ret ^= col_up_table[(t >> 32) & ROW_MASK] << 8
        ret ^= col_up_table[(t >> 48) & ROW_MASK] << 12
        return ret

    def collapse_left(self):
        board = self.board
        ret = self.board
        ret ^= row_left_table[(board >> 0) & ROW_MASK] << 0
        ret ^= row_left_table[(board >> 16) & ROW_MASK] << 16
        ret ^= row_left_table[(board >> 32) & ROW_MASK] << 32
        ret ^= row_left_table[(board >> 48) & ROW_MASK] << 48
        return ret

    def collapse_down(self):
        ret = self.board
        t = self.transpose(self.board)
        ret ^= col_down_table[(t >> 0) & ROW_MASK] << 0
        ret ^= col_down_table[(t >> 16) & ROW_MASK] << 4
        ret ^= col_down_table[(t >> 32) & ROW_MASK] << 8
        ret ^= col_down_table[(t >> 48) & ROW_MASK] << 12
        return ret

    def collapse_right(self):
        board = self.board
        ret = self.board
        ret ^= row_right_table[(board >> 0) & ROW_MASK] << 0
        ret ^= row_right_table[(board >> 16) & ROW_MASK] << 16
        ret ^= row_right_table[(board >> 32) & ROW_MASK] << 32
        ret ^= row_right_table[(board >> 48) & ROW_MASK] << 48
        return ret

    def rotate_board(self, board, number_of_rotations):
        # number of rotations is in quarter circles, with positive being counter-clockwise
        # returns a copy of the board, but rotated
        temporary_board = board
        return temporary_board.rot90(number_of_rotations)

    def collapse_nothing(self):
        return

    def is_board_full(self):
        number_empty = self.count_empty()
        board_full = number_empty == 0  # If there are empty tiles, the board can't be full
        return board_full

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
        temporary_board = self.board

        # Execute the proper collapse function for the given direction.
        self.board = move_dictionary[direction]() % int(np.uint64(
            -1) + 1)  # Because python precision is weird, sometimes, the board will be too large. So we can shrink it to keep this faster

        # Update score
        self.score = score_board(self.board) - self.scorepenalty

        if print_board:
            print('Moving in direction {}'.format(direction))
            # self.print(show_score=show_score)

        # Move unsuccessful
        if temporary_board == self.board:
            if print_board:
                self.print()
                print('Move not successful. Score: {}'.format(self.score))
            return 0
        else:
            # If move was successful, but now it's game over
            if self.check_if_game_over():
                if print_board:
                    self.print(show_score)
                    print('Game Over')
                return -1
            else:
                # Add a random tile if a move was successful
                tile = self.generate_random_tile()
                if tile == 2:
                    self.scorepenalty += 4  # If tile==2, then the board gets a new 2^2=4 tile, which means our score_table over counts
                self.place_random(number=tile)
                if print_board:
                    self.print(show_score)
                return 1

    def simulate_move_successful(self, move):
        move_dictionary = {'up': self.collapse_up,
                           'down': self.collapse_down,
                           'left': self.collapse_left,
                           'right': self.collapse_right,
                           'nothing': self.collapse_nothing}
        result = move_dictionary[move]()
        if result == self.board:
            return False
        else:
            return True

    def generate_random_tile(self):
        # Generate a 2, 90% of the time
        # a 2 on the user end is actually a 1 for the board (since 2=2^1)
        # Similarly, 4 =2^2 , so we return a 2
        return 1 if np.random.random() < 0.9 else 2

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
            list_of_moves = ['up', 'down', 'left', 'right']
            for move in list_of_moves:
                if self.simulate_move_successful(move):  # If any move is successful, game is not over
                    return False
            return True  # All moves were unsuccessful
        else:
            return False

    def get_board(self):
        board = self.np_board().astype(np.float32)
        board_tensor = torch.from_numpy(board)
        return board_tensor

    def get_board_log(self):
        board_tensor = self.get_board()
        board_tensor[board_tensor == 0] = 1
        board_tensor = torch.log2(board_tensor)
        return board_tensor

    def get_board_normalized(self):
        board_tensor = self.get_board_log()
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
        }
        # Calculate the desired move, first by choosing the action index (the highest move score)
        action_index = torch.argmax(input_actions).item()
        move = moves[action_index]

        # Save a temporary copy of the board, so we can calculate the reward
        board_before_move = self.get_board_log().int().flatten()
        tiles_before_move = torch.bincount(board_before_move, minlength=15)  # Gives us the number of each type of tile

        # do the move, then save a copy of the board
        result = self.move(move)
        board_after_move = self.get_board_log().int().flatten()
        tiles_after_move = torch.bincount(board_after_move, minlength=15)

        if result == -1:  # Move caused game to end
            terminal = True
            reward = -0.5
            self.__init__()
        elif result == 0:  # Move was invalid, i.e. no change
            terminal = False
            reward = 0
        elif result == 1:
            # If the move was successful, we can calculate the reward
            difference = tiles_after_move - tiles_before_move
            positive_differences = torch.nn.ReLU()(
                difference).int()  # We only care about how many /new/ tiles were made, not how many were destroyed
            powers_of_two = 2 ** torch.linspace(0, 14, steps=15).int()
            powers_of_two[powers_of_two == 2] = 0  # Ignore all the new 2 tiles (they were randomly generated)
            powers_of_two[powers_of_two == 1] = 0  # Ignore all the new 1 tiles (they are actually 0 tiles)
            sum_of_new_tiles = torch.matmul(positive_differences,
                                            powers_of_two).item()  # Take the dot product of the differences and the powers of two

            reward = sum_of_new_tiles
            terminal = False

        return reward, terminal

    def get_score(self):
        return self.score

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
