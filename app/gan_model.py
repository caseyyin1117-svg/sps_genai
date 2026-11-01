import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(z_dim, 7 * 7 * 128),
            nn.ReLU(True),
            nn.Unflatten(1, (128, 7, 7)),

            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                64, 1, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                1, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                64, 128, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)