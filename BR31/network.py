import torch
import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
        )
        self.policy_net = nn.Sequential(nn.Linear(128, output_dim), nn.Softmax(-1))
        self.value_net = nn.Sequential(nn.Linear(128, 1), nn.Tanh())

    def forward(self, x):
        x = self.net(x)
        prob = self.policy_net(x)
        value = self.value_net(x)
        return prob, value

    def predict(self, x):
        with torch.no_grad():
            prob, value = self.forward(x)
        prob = prob.squeeze().cpu().numpy()
        value = value.squeeze().cpu().item()
        return prob, value
