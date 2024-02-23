import torch
import torch.nn as nn


class NuraleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cl1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.cl2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(2, 2)

        self.flatten = nn.Flatten()
        self.ln1 = nn.Linear(64*32, 1000)  # 32 + 32 = 64
        self.ln2 = nn.Linear(1000, 500)
        self.ln3 = nn.Linear(500, 10)

        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.cl1(x))
        x = self.maxpool(x)
        x = self.activation(self.cl2(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.activation(self.ln1(x))
        x = self.activation(self.ln2(x))
        x = self.ln3(x)

        return x
