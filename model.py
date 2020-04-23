import torch.nn as nn
import torch.nn.functional as F

class Autoencoder50(nn.Module):
    def __init__(self):
        super(Autoencoder50, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),  # b, 16, 32, 32
            nn.ReLU(True),
            nn.Conv2d(16, 8, 4, stride=2, padding=1),  # b, 8, 16, 16
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=1, padding=1),  # b, 16, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=1, padding=1),  # b, 8, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 4, stride=2, padding=1),  # b, 3, 32, 32
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Autoencoder25(nn.Module):
    def __init__(self):
        super(Autoencoder25, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),  # b, 16, 32, 32
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 16, 16
            nn.Conv2d(16, 8, 4, stride=2, padding=1),  # b, 8, 8, 8
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=1, padding=1),  # b, 16, 8, 8
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),  # b, 8, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 4, stride=2, padding=1),  # b, 3, 32, 32
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x