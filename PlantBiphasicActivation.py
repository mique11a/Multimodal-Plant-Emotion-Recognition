import torch
import torch.nn as nn
import torch.nn.functional as F


class PlantBiphasicActivation(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.k_fast_raw = nn.Parameter(torch.zeros(num_classes))
        self.theta_fast = nn.Parameter(torch.zeros(num_classes))
        self.theta_slow = nn.Parameter(torch.zeros(num_classes))
        self.alpha_predictor = nn.Linear(in_features, num_classes)

    def forward(self, logits, fused_features):
        k_fast = F.softplus(self.k_fast_raw) + 1.0
        fast = F.softplus(k_fast * (logits - self.theta_fast))
        slow = F.softplus(logits - self.theta_slow)
        alpha = torch.sigmoid(self.alpha_predictor(fused_features))
        phi = alpha * fast + (1.0 - alpha) * slow
        probs = phi / (torch.sum(phi, dim=1, keepdim=True) + 1e-8)
        return probs, fast, slow, alpha
