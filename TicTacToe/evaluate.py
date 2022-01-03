from game import TicTacToe, alpha_beta_player
from network import CNN
from mcts import MCTS
import torch
from shutil import copy

N_PLAY = 10

lastest_weight = "./TicTacToe/weight/lastest.pt"
best_weight = "./TicTacToe/weight/best.pt"


def compare(policies):
    total_point = 0
    for e in range(N_PLAY):
        if e % 2 == 0:
            total_point += play(policies)
        else:
            total_point += 1 - play(policies[::-1])
        print(f"\rEvaluate {e+1}/{N_PLAY}", end="")
    return total_point / N_PLAY


def play(policies):
    game = TicTacToe()
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
    copy(lastest_weight, best_weight)
    print("best weight is updated")


def evaluate():
    net_lastest = CNN()
    net_lastest.load_state_dict(torch.load(lastest_weight))
    mcts_lastest = MCTS(net_lastest)

    net_best = CNN()
    net_best.load_state_dict(torch.load(best_weight))
    mcts_best = MCTS(net_best)

    policy_lastest = lambda game: mcts_lastest.get_move(game)[0]
    policy_best = lambda game: mcts_best.get_move(game)[0]
    policies = [policy_lastest, policy_best]
    win_rate = compare(policies)
    print(f" with Best player, Win rate : {win_rate}")

    if win_rate > 0.5:
        update_weight()
        win_rate_ = compare([policy_lastest, alpha_beta_player])
        print(f" with Strong player, Win rate : {win_rate_}")


if __name__ == "__main__":
    net_lastest = CNN()
    net_lastest.load_state_dict(torch.load(lastest_weight))
    mcts_lastest = MCTS(net_lastest)

    net_best = CNN()
    net_best.load_state_dict(torch.load(best_weight))
    mcts_best = MCTS(net_best)

    policy_lastest = lambda game: mcts_lastest.get_move(game)[0]
    policy_best = lambda game: mcts_best.get_move(game)[0]

    win_rate = compare([policy_best, alpha_beta_player])
    print(f" with Strong player, Win rate : {win_rate}")
