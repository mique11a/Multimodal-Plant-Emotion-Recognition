import torch
import torch.nn as nn
import torch.nn.functional as F

from PlantBiphasicActivation import PlantBiphasicActivation
from PlantTimeDomainEncoder import PlantTimeDomainEncoder


class ImpedanceEncoder(nn.Module):
    def __init__(self, dropout=0.15):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 64),
            nn.GELU(),
        )

    def forward(self, x):
        return self.network(x)


class GatedFusionLayer(nn.Module):
    def __init__(self, dropout=0.15):
        super().__init__()
        self.gate_generator = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 128),
        )
        self.impedance_projection = nn.Sequential(nn.Linear(64, 128))
        self.fusion_projection = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 128),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, voltage_features, impedance_features):
        gate = torch.sigmoid(self.gate_generator(impedance_features))
        impedance_projection = self.impedance_projection(impedance_features)
        fused = voltage_features * gate + impedance_projection
        fused = self.fusion_projection(fused)
        return self.dropout(F.gelu(fused))


class PlantFusionBackbone(nn.Module):
    def __init__(self, num_classes=3, voltage_channels=1, dropout=0.15):
        super().__init__()
        self.time_encoder = PlantTimeDomainEncoder(in_channels=voltage_channels, hidden_dim=64)
        self.imp_branch = ImpedanceEncoder(dropout=dropout)
        self.fusion = GatedFusionLayer(dropout=dropout)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, volt, imp):
        voltage_features = self.time_encoder(volt)
        impedance_features = self.imp_branch(imp)
        fused_features = self.fusion(voltage_features, impedance_features)
        logits = self.classifier(fused_features)
        return logits, fused_features


class PlantFusionNet(nn.Module):
    def __init__(self, num_classes=3, voltage_channels=1, impedance_dim=1, dropout=0.15):
        super().__init__()
        self.model_config = {
            "num_classes": num_classes,
            "voltage_channels": voltage_channels,
            "impedance_dim": impedance_dim,
            "dropout": dropout,
        }
        self.backbone = PlantFusionBackbone(
            num_classes=num_classes,
            voltage_channels=voltage_channels,
            dropout=dropout,
        )
        self.output_layer = PlantBiphasicActivation(in_features=128, num_classes=num_classes)

    def forward(self, volt, imp):
        logits, fused_features = self.backbone(volt, imp)
        probs, fast, slow, alpha = self.output_layer(logits, fused_features)
        return probs, fast, slow, alpha
