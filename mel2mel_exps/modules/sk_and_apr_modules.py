import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, n_channles, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(n_channles)
        self.net = nn.Sequential(nn.Conv1d(n_channles, n_channles, 3, padding=1),
                                 nn.LeakyReLU(0.1),
                                 nn.Conv1d(n_channles, n_channles, 1),
                                 nn.Dropout(dropout))

    def forward(self, x):
        # x - b, c, l
        xt = self.net(x)
        return self.layer_norm((xt + x).transpose(1, 2)).transpose(1, 2)


class SpeakerClassification(nn.Module):
    def __init__(self, n_mels, n_channels, n_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(n_mels, n_channels, 3, padding=1),
                                 nn.LeakyReLU(0.1),
                                 ResBlock(n_channels),
                                 nn.Conv1d(n_channels, 2 * n_channels, 3, 2, 1),
                                 ResBlock(2 * n_channels),
                                 nn.Conv1d(2 * n_channels, 4 * n_channels, 3, 2, 1),
                                 ResBlock(4 * n_channels),
                                 nn.Conv1d(4 * n_channels, 4 * n_channels, 3, 2, 1))

        self.proj = nn.Linear(4 * n_channels, n_classes)

    def forward(self, mel):
        f_map = self.net(mel)  # b, 4 * n_channels, T // 8
        proj = self.proj(f_map.transpose(1, 2))  # b, T // 8, n_classes
        return torch.mean(proj, dim=1)

