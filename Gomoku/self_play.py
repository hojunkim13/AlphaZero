from network import ResNet
from mcts import MCTS
from game import Gomoku
import os
import torch
from utils import preprocess
from Config import *


def play(mcts):
    data = []
    game = Gomoku()
    while True:
        done, winner = game.is_done()
        if done:
            break
        move, prob = mcts.get_move(game)
        state = preprocess(game)
        data.append([state, prob, None])
        game.play(move)
    if winner == -1:
        value = 0
    else:
        value = 1 if winner == 0 else -1
    for i in range(len(data)):
        data[i][2] = value
        value *= -1
    return data


def self_play(net, n_play):
    try:
        net.load_state_dict(torch.load(os.path.join(weight_path, "best.pt")))
    except:
        print("Can't find best weight.")
        os.makedirs(weight_path, exist_ok=True)
        torch.save(net.state_dict(), os.path.join(weight_path, "best.pt"))
    mcts = MCTS(net, self_play=True)
    datum = []
    for e in range(n_play):
        data = play(mcts)
        datum.extend(data)
    return datum


if __name__ == "__main__":
    self_play(1)
