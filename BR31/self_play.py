from network import DNN
from mcts import MCTS
from game import BR31
import os
import pickle
import time
import torch


def play(mcts):
    data = []
    game = BR31()
    while not game.is_done():
        act, prob = mcts.get_action(game)
        data.append([game.n, prob, None])
        game.play(act)
    value = 1 if not game.turn else -1
    for i in range(len(data)):
        data[i][2] = value
        value *= -1
    return data


def save_data(data):
    os.makedirs("./BR31/data", exist_ok=True)
    date = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime(time.time()))
    with open(f"./BR31/data/{date}.data", mode="wb") as f:
        pickle.dump(data, f)
    n_files = len(os.listdir("./BR31/data/"))
    if n_files > 5:
        for _ in range(n_files - 5):
            os.remove("./BR31/data/" + os.listdir("./BR31/data/")[0])
    print("Data were generated.")


def self_play(n_play):
    net = DNN(31, 3)  # Don't use gpu when self-play. it will make bottle-neck effect
    net.load_state_dict(torch.load("./BR31/weight/best.pt"))
    mcts = MCTS(net)
    datum = []
    for e in range(n_play):
        print(f"\rSelf Play {e+1}/{n_play}", end="")
        data = play(mcts)
        datum.extend(data)
    print("")
    save_data(datum)


if __name__ == "__main__":
    self_play(1)
