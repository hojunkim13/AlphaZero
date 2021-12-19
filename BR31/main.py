from self_play import self_play

from evaluate import evaluate
from train import train

BATCHSIZE = 512
N_PLAY = 500
EPOCH = 100

from network import DNN
import torch

net = DNN(31, 3)
torch.save(net.state_dict(), "./BR31/weight/best.pt")

if __name__ == "__main__":
    for i in range(10):
        print("Train", i + 1, "==================================")
        self_play(N_PLAY)
        train(BATCHSIZE, EPOCH)
        evaluate()
