from Config import BOARD_SIZE
import numpy as np
import torch


def preprocess(game):
    board = game.board
    player = game.current_player
    mat = np.zeros((3, BOARD_SIZE, BOARD_SIZE))
    for m, p in board.items():
        row, col = divmod(m, BOARD_SIZE)
        if p == player:
            mat[0, row, col] = 1
        else:
            mat[1, row, col] = 1
    if board:
        mat[2, row, col] = 1

    mat = torch.tensor(mat, dtype=torch.float).unsqueeze(0)
    return mat

