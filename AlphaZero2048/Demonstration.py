import Game
import Players
import MCTS
import Arena
import numpy as np
import NNet
from utils import *

game = Game.Game2048()
player = Players.RandomPlayer(game)
nnet_player = NNet.NNetWrapper(game)
nnet_player.load_checkpoint('./temp', 'best.pth.tar')
args = dotdict({
    'numIters': 1000,
    'numEps': 100,  # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,  #
    'updateThreshold': 0.6,
    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,  # Number of game examples to train the neural networks.
    'numMCTSSims': 25,  # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,  # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

mcts = MCTS.MCTS(game, nnet_player, args)
nnet_player_function = lambda x: np.argmax(mcts.getActionProb(x))
random_arena = Arena.Arena(player.play, None, game, game.display)
nnet_arena = Arena.Arena(nnet_player_function, None, game, game.display)
nnet_arena.playGame(True)
