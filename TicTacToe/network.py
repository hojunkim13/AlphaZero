import torch
import torch.nn as nn
from utils import preprocess
from game import BOARD_SIZE


class CNN(nn.Module):
    def __init__(self, hidden_dim=64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, hidden_dim, 3, 1, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )

        self.policy_net = nn.Sequential(
            nn.Conv2d(hidden_dim, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE),
            nn.Softmax(-1),
        )
        self.value_net = nn.Sequential(
            nn.Conv2d(hidden_dim, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(BOARD_SIZE * BOARD_SIZE, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.net(x)
        prob = self.policy_net(x)
        value = self.value_net(x)
        return prob, value

    def predict(self, x):
        x = preprocess(x)
        with torch.no_grad():
            prob, value = self.forward(x)
        prob = prob.squeeze().cpu().numpy()
        value = value.squeeze().cpu().item()
        return prob, value
