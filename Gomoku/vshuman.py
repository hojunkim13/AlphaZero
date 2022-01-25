import torch
from game import *
from network import ResNet
from mcts import MCTS
import random
from Config import *


def play(policies):
    game = Gomoku()
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
    net = ResNet()
    net.load_state_dict(torch.load(weight_path + "best.pt"))
    mcts = MCTS(net, log=True)
    bot_policy = lambda x: mcts.get_move(x, temp=0)[0]
    policies = [human_action, bot_policy]
    policies = policies if random.random() > 0.7 else policies[::-1]
    play(policies)


main()
