from game import Gomoku
from network import ResNet
from mcts import MCTS
import torch
from shutil import copy
from Config import *


def compare(policies):
    total_point = 0
    for e in range(EVAL_N_PLAY):
        if e % 2 == 0:
            total_point += play(policies)
        else:
            total_point += 1 - play(policies[::-1])
        print(f"\rEvaluate {e+1}/{EVAL_N_PLAY}", end="")
    return total_point / EVAL_N_PLAY


def play(policies):
    game = Gomoku()
    while True:
        done, winner = game.is_done()
        if done:
            break
        move = policies[game.current_player](game)
        game.play(move)
    if winner == -1:
        return 0.5
    else:
        return 1 if winner == 0 else 0


def update_weight():
    copy(weight_path + "lastest.pt", weight_path + "best.pt")
    print("best weight is updated")


def evaluate():
    net_lastest = ResNet()
    net_lastest.load_state_dict(torch.load(weight_path + "lastest.pt"))
    mcts_lastest = MCTS(net_lastest, log=True)

    net_best = ResNet()
    net_best.load_state_dict(torch.load(weight_path + "best.pt"))
    mcts_best = MCTS(net_best, log=True)

    policy_lastest = lambda game: mcts_lastest.get_move(game)[0]
    policy_best = lambda game: mcts_best.get_move(game)[0]
    policies = [policy_lastest, policy_best]
    win_rate = compare(policies)
    print(f" with Best player, Win rate : {win_rate}")

    if win_rate > 0.5:
        update_weight()


if __name__ == "__main__":
    evaluate()
