from self_play import self_play
from evaluate import evaluate
from train import train
from Config import *
from collections import deque
from network import ResNet


def main():
    i = 0
    data = deque(maxlen=3000)
    net = ResNet()
    while True:
        history = self_play(net, 1)
        data.extend(history)
        if len(data) * 10 >= BATCHSIZE:
            loss = train(net, data, epoch=20)
        else:
            loss = 0
        print(f"Iter {i+1}, Loss: {loss:.2E}")
        if (i + 1) % 50 == 0:
            evaluate()
        i += 1


if __name__ == "__main__":
    main()
