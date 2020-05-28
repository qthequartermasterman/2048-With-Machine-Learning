import logging

from tqdm import tqdm

from multiprocessing import Pool

log = logging.getLogger(__name__)


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display
        self.pool = Pool(processes=8)

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player2, None, self.player1]

        def run_a_game(curPlayer):
            print('Starting game')
            # curPlayer = 1
            board = self.game.getInitBoard()
            it = 0
            while self.game.getGameEnded(board, curPlayer) == 0:
                it += 1
                if verbose:
                    assert self.display
                    print("Turn ", str(it), "Player ", str(curPlayer))
                    self.game.display(board)
                action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))

                valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

                if valids[action] == 0:
                    log.error(f'Action {action} is not valid!')
                    log.debug(f'valids = {valids}')
                    assert valids[action] > 0
                board, curPlayer = self.game.getNextState(board, curPlayer, action)
            if verbose:
                assert self.display
                print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
                self.display(board)
            # return curPlayer * self.game.getGameEnded(board, curPlayer)
            # player_1_score = self.game.estimateScore(board)
            return self.game.estimateScore(board)

        # curPlayer *= -1
        '''
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            if verbose:
                assert self.display
                print("Turn ", str(it), "Player ", str(curPlayer))
                self.game.display(board)
            action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))

            valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
            # board, curPlayer = self.game.getNextState(None, curPlayer, action)  # The game object already has the new board saved. It's faster to just let it do its thing.
        if verbose:
            assert self.display
            print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
            self.display(board)
        # return curPlayer * self.game.getGameEnded(board, curPlayer)
        player_2_score = self.game.estimateScore(board)
        '''
        scores = self.pool.map(run_a_game, [-1, 1])
        player_2_score = scores[0]
        player_1_score = scores[1]

        print("Player 1 score: {}, Player 2 score: {}".format(player_1_score, player_2_score))
        return player_1_score, player_2_score
        '''
        if player_1_score == player_2_score:
            return 0  # It's a draw, since both players had the same result
        elif player_1_score > player_2_score:
            return 1  # Player 1 won this round
        elif player_2_score < player_2_score:
            return -1 # Player 2 won this round
        '''

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """
        '''
        num = int(num/2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == -1:
                oneWon += 1
            elif gameResult == 1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws
        '''
        player1_total_score = 0
        player2_total_score = 0

        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            player1_score, player2_score = self.playGame(verbose=verbose)
            player1_total_score += player1_score
            player2_total_score += player2_score
        player1_total_score = player1_total_score / num  # Take averages
        player2_total_score = player2_total_score / num  # Take averages

        return player1_total_score, player2_total_score
