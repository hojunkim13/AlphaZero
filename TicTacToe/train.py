from network import CNN
import os
import numpy as np
import torch
import pickle
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(batch_size=512, epoch=100):
    net = CNN().to(device)
    net.load_state_dict(torch.load("./TicTacToe/weight/best.pt"))
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    data_loader = get_data_loader(batch_size)
    for e in range(epoch):
        losses = []
        for state, prob, value in data_loader:
            optimizer.zero_grad()
            pred_prob, pred_value = net(state)
            value_loss = torch.mean(torch.square(pred_value.squeeze() - value))
            policy_loss = -torch.mean(prob * torch.log(pred_prob + 1e-8))
            loss = value_loss + policy_loss

            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        loss_ = np.mean(losses)
        print(f"\rTrain {e+1}/{epoch}, Loss : {loss_:.3e}", end="")
    print("")
    torch.save(net.state_dict(), "./TicTacToe/weight/lastest.pt")


def load_data():
    file = sorted(os.listdir("./TicTacToe/data/"))[-1]
    with open("./TicTacToe/data/" + file, mode="rb") as f:
        data = pickle.load(f)
    return data


def get_data_loader(batch_size):
    data = load_data()
    state, prob, value = zip(*data)

    prob = np.array(prob)
    value = np.array(value)

    state = torch.vstack(state).to(device)
    prob = torch.tensor(prob, dtype=torch.float).to(device)
    value = torch.tensor(value, dtype=torch.float).to(device)

    dataset = TensorDataset(state, prob, value)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    return loader


if __name__ == "__main__":
    train()
