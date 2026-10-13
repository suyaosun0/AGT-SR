"""Eqs. (15)-(21), per-utterance confidence and unnormalized sequence NLLs."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from model import padding_mask


class AGTSRLoss(nn.Module):
    def __init__(self, lambda_cao=0.3, gamma=1.0, epsilon=1e-8):
        super().__init__()
        self.lambda_cao, self.gamma, self.epsilon = lambda_cao, gamma, epsilon
        self.ctc = nn.CTCLoss(blank=0, reduction="none", zero_infinity=False)

    def forward(self, output, batch):
        logits = output["decoder_logits"]
        targets = batch["decoder_target"]
        # Paper Eq. (20) SUMS tokens. Average sequence sums across the batch only.
        ce = F.cross_entropy(logits.transpose(1, 2), targets, ignore_index=0, reduction="none").sum(1)
        lengths = output["encoder_lengths"]
        layer_ids = sorted(output["ctc_log_probs"])
        nlls, confidences = [], []
        for layer in layer_ids:
            lp = output["ctc_log_probs"][layer].float()  # [B,T,V]
            nll = self.ctc(lp.transpose(0, 1), batch["ctc_targets"], lengths,
                           batch["ctc_lengths"])
            if not torch.isfinite(nll).all():
                raise ValueError("Non-finite CTC: check target lengths/repeats vs acoustic grid; no silent zeroing")
            nlls.append(nll)
            with torch.no_grad():
                probabilities = lp.exp()
                entropy = -(probabilities * (probabilities + self.epsilon).log()).sum(-1)
                valid = ~padding_mask(lengths, lp.size(1))
                c = 1 - (entropy * valid).sum(1) / lengths / math.log(lp.size(-1))
                confidences.append(c.clamp(0, 1))
        nlls, confidences = torch.stack(nlls), torch.stack(confidences)  # [layers,B]
        weights = (confidences + self.epsilon).pow(self.gamma)
        alpha = (weights / weights.sum(0, keepdim=True)).detach()
        cao = (alpha * nlls).sum(0)
        total = (ce + self.lambda_cao * cao).mean()
        # The final-layer CTC loss is ALREADY included; never add it separately.
        stats = {"total": total.detach(), "ce": ce.mean().detach(), "cao": cao.mean().detach()}
        for i, layer in enumerate(layer_ids):
            stats[f"ctc_{layer}"] = nlls[i].mean().detach()
            stats[f"confidence_{layer}"] = confidences[i].mean()
            stats[f"alpha_{layer}"] = alpha[i].mean()
        return total, stats
