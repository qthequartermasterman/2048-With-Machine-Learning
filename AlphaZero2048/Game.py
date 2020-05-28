import sys

sys.path.append('..')  # BitwiseGameboard is in a file in the parent directory

from bitwise_gameboard import BitwiseGameboard
import numpy as np


class Game():
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """

    def __init__(self):
        pass

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        pass

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        pass

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        pass

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        pass

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        pass

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        pass

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        pass

    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        pass

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        pass


class Game2048(Game):

    def __init__(self):
        self.gameboard = BitwiseGameboard()
        self.moves = {
            0: 'up',
            1: 'left',
            2: 'down',
            3: 'right',
        }

        self.moves_lookup = {direction: number for (number, direction) in self.moves.items()}

    def getInitBoard(self):
        self.gameboard = BitwiseGameboard()
        # self.gameboard.set_board(0x081834239)
        return self.gameboard.board

    def getBoardSize(self):
        return self.gameboard.board_size

    def getActionSize(self):
        return self.gameboard.number_of_actions

    def getNextState(self, board, player, action):
        if board is not None:
            self.gameboard.set_board(board)
        self.gameboard.move(self.moves[action], print_board=False)
        return self.gameboard.get_board_log_divided_by_16(), player

    def getValidMoves(self, board, player):
        if board is not None:
            self.gameboard.set_board(board)
        list_of_valid_moves = self.gameboard.get_valid_moves()

        valid_moves_one_hot_encoding = [0] * self.getActionSize()

        for move in list_of_valid_moves:
            valid_moves_one_hot_encoding[self.moves_lookup[move]] = 1

        return np.array(valid_moves_one_hot_encoding)

    def getGameEnded(self, board, player):
        if board is not None:
            self.gameboard.set_board(board)
        game_over = self.gameboard.check_game_status()
        return game_over

    def getCanonicalForm(self, board, player):
        # return self.gameboard.get_board_log_divided_by_16()
        if board is not None:
            self.gameboard.set_board(board)
        return self.gameboard.board

    def getSymmetries(self, board, pi):
        list_of_symmetries = [(board, pi)]
        for i in range(1, 4):
            temp_board = self.gameboard.rotate_board(board, i)
            temp_pi = np.roll(pi, i)
            list_of_symmetries.append((temp_board, temp_pi))  # Append the rotated board and actions to the list of sym
        return list_of_symmetries

    def stringRepresentation(self, board):
        if board is not None:
            self.gameboard.set_board(board)
        return str(hex(self.gameboard.board))

    def string_representation_readable(self, board):
        if board is not None:
            self.gameboard.set_board(board)
        return self.gameboard.string_readable()

    def display(self, board, show_score=False):
        if board is not None:
            self.gameboard.set_board(board)
        self.gameboard.print(show_score=show_score)

    def estimateScore(self, board):
        return self.gameboard.estimate_score(board)
