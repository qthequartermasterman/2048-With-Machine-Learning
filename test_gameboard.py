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
