import numpy as np


class Player():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        pass


class RandomPlayer(Player):
    def __init__(self, game):
        super(RandomPlayer, self).__init__(game)

    def play(self, board):
        action = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[action] != 1:
            action = np.random.randint(self.game.getActionSize())
        return action
