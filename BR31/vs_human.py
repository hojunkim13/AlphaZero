import torch
from game import BR31, strong_player
from network import DNN
from mcts import MCTS
import random


def human_action(x):
    while True:
        action = input("Please input number [1 ~ 3] :")
        try:
            action = int(action)
            break
        except:
            pass
    return action - 1


def play(policies):
    game = BR31()
    while not game.is_done():
        print(game)
        action = policies[game.turn](game)
        game.play(action)
    if policies[game.turn].__name__ == "<lambda>":
        print(f"Winner: Bot # {'Black' if game.turn == 0 else 'White'}")
    else:
        print(f"Winner: Player # {'Black' if game.turn == 0 else 'White'}")


def main():
    net = DNN(31, 3)
    net.load_state_dict(torch.load("./BR31/weight/best.pt"))
    mcts = MCTS(net)
    bot_policy = lambda x: mcts.get_action(x)[0]
    policies = [bot_policy, strong_player]
    policies = policies if random.random() > 0 else policies[::-1]
    play(policies)


main()

