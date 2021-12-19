from network import DNN
import os
import numpy as np
import torch
import pickle
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(batch_size=512, epoch=100):
    net = DNN(31, 3).to(device)
    net.load_state_dict(torch.load("./BR31/weight/best.pt"))
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    data_loader = get_data_loader(batch_size)
    for e in range(epoch):
        losses = []
        for state, prob, value in data_loader:
            optimizer.zero_grad()
            pred_prob, pred_value = net(state)
            value_loss = torch.mean(torch.square(pred_value.squeeze() - value))
            policy_loss = -torch.mean(prob * torch.log(pred_prob))
            loss = value_loss + policy_loss

            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        loss_ = np.mean(losses)
        print(f"\rTrain {e+1}/{epoch}, Loss : {loss_:.3e}", end="")
    print("")
    torch.save(net.state_dict(), "./BR31/weight/lastest.pt")


def load_data():
    file = sorted(os.listdir("./BR31/data/"))[-1]
    with open("./BR31/data/" + file, mode="rb") as f:
        data = pickle.load(f)
    return data


def get_data_loader(batch_size):
    data = load_data()
    state, prob, value = zip(*data)
    state = np.eye(31)[np.array(state)]
    prob = np.array(prob)
    value = np.array(value)

    state = torch.tensor(state, dtype=torch.float).to(device)
    prob = torch.tensor(prob, dtype=torch.float).to(device)
    value = torch.tensor(value, dtype=torch.float).to(device)

    dataset = TensorDataset(state, prob, value)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    return loader


if __name__ == "__main__":
    train()
