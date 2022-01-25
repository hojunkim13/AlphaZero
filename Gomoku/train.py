import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from Config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(net, data, epoch):
    net = net.to(device).train()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=L2_const)
    data_loader = get_data_loader(data, BATCHSIZE)
    mean_loss = 0
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
        mean_loss += np.mean(losses)
    torch.save(net.state_dict(), weight_path + "lastest.pt")
    return mean_loss / epoch


def get_data_loader(data, batch_size):
    state, prob, value = zip(*data)

    prob = np.array(prob)
    value = np.array(value)

    state = torch.vstack(state).to(device)
    prob = torch.tensor(prob, dtype=torch.float).to(device)
    value = torch.tensor(value, dtype=torch.float).to(device)

    dataset = HistoryDataset((state, prob, value), transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    return loader


class HistoryDataset(Dataset):
    """TensorDataset with support of transforms.
    """

    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        state = self.tensors[0][index]
        prob = self.tensors[1][index]

        if self.transform:
            state, prob = self.transform(state, prob)

        value = self.tensors[2][index]

        return state, prob, value

    def __len__(self):
        return self.tensors[0].size(0)


def transform(state, prob):
    k = np.random.randint(4)
    prob = prob.view(BOARD_SIZE, BOARD_SIZE)
    state = torch.rot90(state, k=k, dims=[1, 2])
    prob = torch.rot90(prob, k=k)

    if np.random.rand() > 0.5:
        state[0] = state[0].fliplr()
        state[1] = state[1].fliplr()
        prob = prob.fliplr()

    if np.random.rand() > 0.5:
        state[0] = state[0].flipud()
        state[1] = state[1].flipud()
        prob = prob.flipud()
    prob = prob.flatten()
    return state, prob


if __name__ == "__main__":
    train()
