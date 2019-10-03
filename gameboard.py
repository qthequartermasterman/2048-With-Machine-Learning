import numpy as np


class Gameboard:
    def __init__(self):
        self.boardsize = 4  # The size of one side of the board.
        self.board = np.zeros((self.boardsize, self.boardsize), dtype=np.int32)
        self.place_random()
        self.place_random()

    def print(self):
        print(self.board)
        print()

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
                            self.board[j-1, i] *= 2
                            self.board[j, i] = 0

    def collapse_nothing(self):
        return


    def is_board_full(self):
        return not (self.boardsize**2 - np.count_nonzero(self.board))

    def move(self, direction):
        # parameter direction is a string that says 'up', 'down', 'left', or 'right'
        move_dictionary = {'up': self.collapse_up,
                           'down': self.collapse_down,
                           'left': self.collapse_left,
                           'right': self.collapse_right,
                           'nothing': self.collapse_nothing}

        # Make a temporary copy of the previous board
        temporary_board = np.copy(self.board)

        # Execute the proper collapse function for the given direction.
        print('Moving in direction {}'.format(direction))
        move_dictionary[direction]()

        # print(self.board, temporary_board)

        # Add a random tile if a move was successful
        if np.array_equal(self.board, temporary_board):
            self.print()
            return 0
        else:
            print('Previous move successful')
        if self.is_board_full():
            self.print()
            print('BOARD FULL')
            return -1
        else:
            self.place_random(self.generate_random_tile())
            self.print()
            return 1

    def generate_random_tile(self):
        # Generate a 2, 90% of the time
        return 2 if np.random.random() < 0.9 else 4




gb = Gameboard()
gb.print()
for _ in range(0, 1000):
    if not (gb.move('right') or gb.move('down')):
        gb.move('left')


