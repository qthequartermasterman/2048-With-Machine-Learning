from bitwisetables import *

ROW_MASK = 0xFFFF
COL_MASK = 0x000F000F000F000


def reverse_row(row):
    maximum_length_of_row = 0x10000  # Because of the unlimited bits in python, we can just mod out 0xFFFF so we only get the last 16 bits.

    one = (row >> 12) % maximum_length_of_row
    two = ((row >> 4) & 0x00F0) % maximum_length_of_row
    three = ((row << 4) & 0x0F00) % maximum_length_of_row
    four = (row << 12) % maximum_length_of_row
    # print('{:16b}\n{:16b}\n{:16b}\n{:16b}'.format(one, two, three, four))
    reverse = one | two | three | four
    print('Reversing: ', '{:4x}'.format(row), '{:4x}'.format(reverse))
    return reverse

    # return (row >> 12) | ((row >> 4) & 0x00F0) | ((row << 4) & 0x0F00) | (row << 12)


def unpack_col(row):
    tmp = row
    return (tmp | (tmp << 12) | (tmp << 24) | (tmp << 36)) & COL_MASK


def score_helper(board, table):
    return table[(board >> 0) & ROW_MASK] + table[(board >> 16) & ROW_MASK] + table[(board >> 32) & ROW_MASK] + table[(board >> 48) & ROW_MASK]


def score_board(board):
    return score_helper(board, score_table)
