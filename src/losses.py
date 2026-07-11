from __future__ import annotations

import torch
import torch.nn as nn

from .utils import total_variation


class UncertaintyGuidedLoss(nn.Module):
    def __init__(self, lambda_recon=0.1, lambda_tv=0.005):
        super().__init__()
        self.lambda_recon = lambda_recon
        self.lambda_tv = lambda_tv

    def forward(self, pred, target, log_var):
        error = torch.abs(pred - target).mean(dim=1, keepdim=True)

        # uncertainty loss
        loss_unc = (torch.exp(-log_var) * error + log_var).mean()

        # small stabilizer 
        loss_recon = error.mean()

        # smooth map
        loss_tv = total_variation(log_var)

        loss = loss_unc + self.lambda_recon * loss_recon + self.lambda_tv * loss_tv

        return loss, {
            "loss_total": loss.detach(),
            "loss_unc": loss_unc.detach(),
            "loss_recon": loss_recon.detach(),
            "logvar_mean": log_var.mean().detach(),
        }
