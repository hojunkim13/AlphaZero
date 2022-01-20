from self_play import self_play
from evaluate import evaluate
from train import train
from Config import *


if __name__ == "__main__":
    for i in range(ITERATION):
        print("Train", i + 1, "===================================")
        self_play(N_SELF_PLAY)
        train(TRAIN_EPOCH)
        evaluate()
