import torch
import torch.nn as nn
from utils import preprocess
from game import BOARD_SIZE


class ResNet(nn.Module):
    def __init__(self, hidden_dim=32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 3, hidden_dim * 4, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.policy_net = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, 4, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * BOARD_SIZE * BOARD_SIZE, BOARD_SIZE * BOARD_SIZE),
            nn.LogSoftmax(-1),
        )
        self.value_net = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, 2, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * BOARD_SIZE * BOARD_SIZE, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 1),
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
            log_prob, value = self.forward(x.cuda())
        prob = torch.exp(log_prob).squeeze().cpu().numpy()
        value = value.squeeze().cpu().item()
        return prob, value
