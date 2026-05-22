import torch
import torch.nn as nn


class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_channels, branch_channels, kernels=(7, 15, 31)):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_channels, branch_channels, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(branch_channels),
                    nn.GELU(),
                    nn.Conv1d(branch_channels, branch_channels, kernel_size=k, padding=k // 2),
                    nn.BatchNorm1d(branch_channels),
                    nn.GELU(),
                )
                for k in kernels
            ]
        )
        fused_channels = branch_channels * len(kernels)
        self.fusion = nn.Sequential(
            nn.Conv1d(fused_channels, fused_channels, kernel_size=1),
            nn.BatchNorm1d(fused_channels),
            nn.GELU(),
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.fusion(x)
        return self.pool(x)


class PlantTimeDomainEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.block1 = MultiScaleConvBlock(16, 16)
        self.block2 = MultiScaleConvBlock(48, 32)
        self.sequence_projection = nn.Sequential(
            nn.Conv1d(96, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        self.bi_gru = nn.GRU(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.sequence_projection(x)
        x = x.permute(0, 2, 1)
        _, hidden = self.bi_gru(x)
        return torch.cat((hidden[-2], hidden[-1]), dim=1)
