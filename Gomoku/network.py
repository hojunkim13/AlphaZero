import torch
import torch.nn as nn
from utils import preprocess
from Config import BOARD_SIZE


class residual_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernerl_size, strides, padding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernerl_size,
                stride=strides,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size=kernerl_size,
                stride=strides,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        out = self.net(x)
        out += x
        out = torch.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, hidden_dim=128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            *[residual_block(hidden_dim, hidden_dim, 3, 1, 1) for _ in range(9)]
        )

        self.policy_net = nn.Sequential(
            nn.Conv2d(hidden_dim, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE),
            nn.LogSoftmax(-1),
        )
        self.value_net = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(BOARD_SIZE * BOARD_SIZE, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
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
            log_prob, value = self.forward(x)
        prob = torch.exp(log_prob).squeeze().cpu().numpy()
        value = value.squeeze().cpu().item()
        return prob, value
