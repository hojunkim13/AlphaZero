from self_play import self_play
from evaluate import evaluate
from train import train

ITERATION = 10
N_SELF_PLAY = 500
BATCHSIZE = 512
TRAIN_EPOCH = 100


if __name__ == "__main__":
    for i in range(ITERATION):
        print("Train", i + 1, "===================================")
        self_play(N_SELF_PLAY)
        train(BATCHSIZE, TRAIN_EPOCH)
        evaluate()
