from network import ResNet
from mcts import MCTS
from game import Gomoku
import os
import pickle
import time
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


def save_data(data):
    os.makedirs(data_path, exist_ok=True)
    date = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime(time.time()))
    with open(os.path.join(data_path, f"{date}.data"), mode="wb") as f:
        pickle.dump(data, f)
    n_files = len(os.listdir(data_path))
    if n_files > 5:
        for _ in range(n_files - 5):
            os.remove(os.path.join(data_path, os.listdir(data_path)[0]))
    print("Data were generated.")


def self_play(n_play):
    net = ResNet()
    try:
        net.load_state_dict(torch.load(os.path.join(weight_path, "best.pt")))
    except:
        os.makedirs(weight_path, exist_ok=True)
        torch.save(net.state_dict(), os.path.join(weight_path, "best.pt"))
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
