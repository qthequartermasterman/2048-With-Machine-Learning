from unittest import TestCase
from gameboard import Gameboard
import numpy as np


class TestGameboard(TestCase):
    def setUp(self):
        self.gameboard = Gameboard()
        self.gameboard.board = np.array([[0, 0, 0, 2],
                                         [2, 0, 2, 0],
                                         [4, 0, 2, 0],
                                         [2, 2, 0, 4]])

    def test_print(self):
        pass

    def test_place_random(self):
        pass

    def test_collapse_right(self):
        self.gameboard.collapse_right()
        np.testing.assert_array_equal(self.gameboard.board,
                                      np.array([[0, 0, 0, 2],
                                                [0, 0, 0, 4],
                                                [0, 0, 4, 2],
                                                [0, 0, 0, 8]]))

    def test_collapse_left(self):
        self.gameboard.collapse_left()
        np.testing.assert_array_equal(self.gameboard.board,
                                      np.array([[2, 0, 0, 0],
                                                [4, 0, 0, 0],
                                                [4, 2, 0, 0],
                                                [8, 0, 0, 0]]))

    def test_collapse_down(self):
        self.gameboard.collapse_down()
        np.testing.assert_array_equal(self.gameboard.board,
                                      np.array([[0, 0, 0, 0],
                                                [2, 0, 0, 0],
                                                [4, 0, 0, 2],
                                                [2, 2, 4, 4]]))

    def test_collapse_up(self):
        self.gameboard.collapse_up()
        np.testing.assert_array_equal(self.gameboard.board,
                                      np.array([[2, 2, 4, 2],
                                                [4, 0, 0, 4],
                                                [2, 0, 0, 0],
                                                [0, 0, 0, 0]]))

    def test_is_move_unsuccessful(self):
        # give it a board that it cannot collapse_right on. If it triggers as successful, fail.
        self.gameboard.board = np.array([[0, 0, 0, 2],
                                         [0, 0, 0, 4],
                                         [0, 0, 4, 2],
                                         [0, 0, 0, 8]])
        self.assertEqual(self.gameboard.move('right'), 0)

    def test_is_board_full(self):
        # Initialize board
        self.assertFalse(self.gameboard.is_board_full())
        # Reinitialize board
        self.gameboard.board = np.array([[2,   4,  16,   2],
                                         [4,   2,  64,   4],
                                         [2,   8,  16,  32],
                                         [8,  16,  64, 128]])
        self.assertTrue(self.gameboard.is_board_full())

    def test_check_if_game_over(self):
        # With initialized board
        self.assertFalse(self.gameboard.check_if_game_over())

        # With custom boards
        # Full board with no collapses
        self.gameboard.board = np.array([[2,   4,  16,   2],
                                         [4,   2,  64,   4],
                                         [2,   8,  16,  32],
                                         [8,  16,  64, 128]])
        self.assertTrue(self.gameboard.check_if_game_over())

        # Full board with collapses
        self.gameboard.board = np.array([[2,   4,  16,   2],
                                         [2,   2,  64,   4],
                                         [2,   8,  16,  32],
                                         [8,  16,  64, 128]])
        self.assertFalse(self.gameboard.check_if_game_over())


