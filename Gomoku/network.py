import torch
import torch.nn as nn
from utils import preprocess
from game import BOARD_SIZE


class residual_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernerl_size, strides, padding) -> None:
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
    def __init__(self, hidden_dim=64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            residual_block(hidden_dim, hidden_dim, 3, 1, 1),
            residual_block(hidden_dim, hidden_dim, 3, 1, 1),
            residual_block(hidden_dim, hidden_dim, 3, 1, 1),
        )

        self.policy_net = nn.Sequential(
            nn.Conv2d(hidden_dim, 4, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE),
            nn.Softmax(-1),
        )
        self.value_net = nn.Sequential(
            nn.Conv2d(hidden_dim, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
        self.cuda()

    def forward(self, x):
        x = self.net(x)
        prob = self.policy_net(x)
        value = self.value_net(x)
        return prob, value

    def predict(self, x):
        x = preprocess(x)
        with torch.no_grad():
            prob, value = self.forward(x.cuda())
        prob = prob.squeeze().cpu().numpy()
        value = value.squeeze().cpu().item()
        return prob, value
