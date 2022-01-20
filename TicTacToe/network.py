import torch
import torch.nn as nn
from utils import preprocess
from game import BOARD_SIZE


class conv_layer(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                dim_in,
                dim_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self, hidden_dim=64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            conv_layer(2, hidden_dim, 3, 1, 1),
            *[conv_layer(hidden_dim, hidden_dim, 3, 1, 1) for _ in range(3)],
            nn.Flatten(),
        )

        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE),
            nn.Softmax(-1),
        )
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim * BOARD_SIZE * BOARD_SIZE, 1), nn.Tanh()
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
