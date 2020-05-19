from unittest import TestCase
from gameboard import Gameboard
from bitwise_gameboard import BitwiseGameboard
import numpy as np


class TestGameboard(TestCase):
    def setUp(self):
        self.game_board = BitwiseGameboard(np.array([[0, 0, 0, 2],
                                                     [2, 0, 2, 0],
                                                     [4, 0, 2, 0],
                                                     [2, 2, 0, 4]]))

    def test_print(self):
        self.game_board.print()
        self.game_board.print(True)
        pass

    def test_place_random(self):
        pass

    def test_collapse_right(self):
        board = self.game_board.np_board(board=self.game_board.collapse_right())
        np.testing.assert_array_equal(board,
                                      np.array([[0, 0, 0, 2],
                                                [0, 0, 0, 4],
                                                [0, 0, 4, 2],
                                                [0, 0, 4, 4]]))

    def test_collapse_left(self):
        board = self.game_board.np_board(board=self.game_board.collapse_left())
        np.testing.assert_array_equal(board,
                                      np.array([[2, 0, 0, 0],
                                                [4, 0, 0, 0],
                                                [4, 2, 0, 0],
                                                [4, 4, 0, 0]]))

    def test_collapse_down(self):
        board = self.game_board.np_board(board=self.game_board.collapse_down())
        np.testing.assert_array_equal(board,
                                      np.array([[0, 0, 0, 0],
                                                [2, 0, 0, 0],
                                                [4, 0, 0, 2],
                                                [2, 2, 4, 4]]))

    def test_collapse_up(self):
        print(self.game_board.np_board())
        board_int = self.game_board.collapse_up()
        board = self.game_board.np_board(board=board_int)
        print('After move')
        print(board)
        np.testing.assert_array_equal(board,
                                      np.array([[2, 2, 4, 2],
                                                [4, 0, 0, 4],
                                                [2, 0, 0, 0],
                                                [0, 0, 0, 0]]))

    def test_is_move_unsuccessful(self):
        # give it a board that it cannot collapse_right on. If it triggers as successful, fail.
        self.game_board = BitwiseGameboard(np.array([[0, 0, 0, 2],
                                                     [0, 0, 0, 4],
                                                     [0, 0, 4, 2],
                                                     [0, 0, 0, 8]]))
        self.assertEqual(self.game_board.move('right'), 0)

    def test_is_board_full(self):
        # Initialize board
        self.assertFalse(self.game_board.is_board_full())
        # Reinitialize board
        self.game_board = BitwiseGameboard(np.array([[2, 4, 16, 2],
                                                     [4, 2, 64, 4],
                                                     [2, 8, 16, 32],
                                                     [8, 16, 64, 128]]))
        self.assertTrue(self.game_board.is_board_full())

    def test_check_if_game_over(self):
        # With initialized board
        self.assertFalse(self.game_board.check_if_game_over())

        # With custom boards
        # Full board with no collapses
        self.game_board = BitwiseGameboard(np.array([[2, 4, 16, 2],
                                                     [4, 2, 64, 4],
                                                     [2, 8, 16, 32],
                                                     [8, 16, 64, 128]]))
        self.assertTrue(self.game_board.check_if_game_over())

        # Full board with collapses
        self.game_board = BitwiseGameboard(np.array([[2, 4, 16, 2],
                                                     [2, 2, 64, 4],
                                                     [2, 8, 16, 32],
                                                     [8, 16, 64, 128]]))
        self.assertFalse(self.game_board.check_if_game_over())
