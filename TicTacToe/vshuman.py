import torch
from game import *
from network import CNN
from mcts import MCTS
import random


def play(policies):
    game = TicTacToe()
    while True:
        done, winner = game.is_done()
        if done:
            break
        print(game)
        action = policies[game.current_player](game)
        game.play(action)
    print(game)
    if winner == -1:
        print(f"Tie!")
    elif policies[winner].__name__ == "<lambda>":
        print(f"Winner: Bot # {'Black' if winner == 0 else 'White'}")
    else:
        print(f"Winner: Player # {'Black' if winner == 0 else 'White'}")


def main():
    net = CNN()
    net.load_state_dict(torch.load("./TicTacToe/weight/best.pt"))
    mcts = MCTS(net)
    bot_policy = lambda x: mcts.get_move(x)[0]
    policies = [alpha_beta_player, bot_policy]
    policies = policies if random.random() > 0.5 else policies[::-1]
    play(policies)


main()
