"""AGT-SR, revised manuscript v2: Eqs. (2)-(14), Table 1.

Inputs are waveforms and EXTERNALLY estimated 80-D FMLLR features.
No log-Mel fallback, learned fusion weights, or extra final CTC head.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    checkpoint: str = "facebook/wav2vec2-base"
    fmllr_dim: int = 80
    pathway_dim: int = 256
    d_model: int = 512
    encoder_layers: int = 12
    decoder_layers: int = 6
    heads: int = 8
    ff_dim: int = 2048
    dropout: float = 0.1
    ctc_layers: tuple = (4, 8, 10, 12)
    eta: float = 0.1
    local_kernel: int = 3
    # The following choices are NOT specified by the manuscript.
    activation: str = "relu"
    norm_first: bool = False
    shared_grid: str = "wav2vec"  # alternative: fmllr
    interpolation_align_corners: bool = False
    max_positions: int = 20000


def padding_mask(lengths, width):
    return torch.arange(width, device=lengths.device)[None] >= lengths[:, None]


class SinusoidalPE(nn.Module):
    """Standard sinusoidal PE; its exact type is unspecified in the paper."""
    def __init__(self, dim, max_positions):
        super().__init__()
        position = torch.arange(max_positions).float()[:, None]
        scale = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000) / dim))
        pe = torch.zeros(max_positions, dim)
        pe[:, 0::2] = torch.sin(position * scale)
        pe[:, 1::2] = torch.cos(position * scale)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        if x.size(1) > self.pe.size(0):
            raise ValueError("Sequence exceeds configured max_positions")
        return x + self.pe[:x.size(1)].to(x.dtype)


class DualPathFrontend(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers import Wav2Vec2Model
        self.cfg = cfg
        self.ssl = Wav2Vec2Model.from_pretrained(cfg.checkpoint)
        if self.ssl.config.hidden_size != 768 or self.ssl.config.num_hidden_layers != 12:
            raise ValueError("Paper requires wav2vec2 Base (768 dimensions, 12 layers)")
        # Disable checkpoint's built-in SpecAugment: augmentation is explicit in data.py.
        self.ssl.config.apply_spec_augment = False
        self.f_proj = nn.Linear(cfg.fmllr_dim, cfg.pathway_dim)
        self.z_proj = nn.Linear(768, cfg.pathway_dim)
        self.fusion = nn.Linear(2 * cfg.pathway_dim, cfg.d_model)
        self.stage = 1
        self.set_stage(1)

    def set_stage(self, epoch):
        """Epochs 1-10: freeze ALL SSL. Epochs 11-40: fine-tune Transformer only."""
        self.stage = 1 if epoch <= 10 else 2
        for p in self.ssl.parameters():
            p.requires_grad_(False)
        if self.stage == 2:
            for p in self.ssl.encoder.parameters():
                p.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # Keep all frozen SSL components deterministic. Stage 2 encoder uses training mode.
        self.ssl.eval()
        if mode and self.stage == 2:
            self.ssl.encoder.train()
        return self

    def forward(self, waveform, wave_lengths, fmllr, fmllr_lengths):
        # Per-utterance SSL extraction avoids group-norm contamination by batch padding.
        # This is slower than padded extraction, but does not change feature values with batch shape.
        zs = []
        for i, n in enumerate(wave_lengths.tolist()):
            context = torch.no_grad() if self.stage == 1 else torch.enable_grad()
            # Preserve outer no_grad() during validation/inference.
            if not torch.is_grad_enabled():
                context = torch.no_grad()
            with context:
                hidden = self.ssl(waveform[i:i+1, :n], output_hidden_states=True).hidden_states
                # hidden[0] is the input to the first Transformer layer.
                zs.append((0.2 * hidden[6] + 0.3 * hidden[9] + 0.5 * hidden[12])[0])
        fused, lengths = [], []
        for i, z in enumerate(zs):
            f = fmllr[i, :int(fmllr_lengths[i])]
            t = z.size(0) if self.cfg.shared_grid == "wav2vec" else f.size(0)
            if self.cfg.shared_grid not in ("wav2vec", "fmllr"):
                raise ValueError("Unknown shared temporal grid")
            def resize(v):
                return F.interpolate(v.T[None], size=t, mode="linear",
                                     align_corners=self.cfg.interpolation_align_corners)[0].T
            # Eqs. (3)-(5): independent 256-D projections, concatenate, LINEAR fusion.
            fused.append(self.fusion(torch.cat((self.f_proj(resize(f)), self.z_proj(resize(z))), -1)))
            lengths.append(t)
        return nn.utils.rnn.pad_sequence(fused, batch_first=True), wave_lengths.new_tensor(lengths)


class APGQI(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.eta = cfg.eta
        self.local = nn.Conv1d(d, d, cfg.local_kernel, padding=cfg.local_kernel // 2)
        self.prior = nn.Linear(2 * d, d)
        self.init_projection = nn.Linear(d, d)

    @staticmethod
    def retrieval_indices(memory_lengths, estimated_lengths, steps):
        """Eq. (9), one-based floor/clip converted to zero-based tensor indices."""
        u = torch.arange(1, steps + 1, device=memory_lengths.device, dtype=torch.float64)[None]
        t = memory_lengths[:, None]
        length = estimated_lengths[:, None]
        tau = torch.floor((u - 0.5) * t / length.clamp_min(1) + 0.5).long()
        tau = tau.clamp_min(1).minimum(t)
        tau = torch.where((length == 0) & (u == 1), torch.ones_like(tau), tau)
        return tau - 1

    def forward(self, token_representation, memory, memory_lengths, estimated_lengths):
        mask = padding_mask(memory_lengths, memory.size(1))
        valid_memory = memory.masked_fill(mask[..., None], 0)
        global_prior = valid_memory.sum(1) / memory_lengths[:, None]
        local = self.local(valid_memory.transpose(1, 2)).transpose(1, 2)
        indices = self.retrieval_indices(memory_lengths, estimated_lengths, token_representation.size(1))
        local_prior = local.gather(1, indices[..., None].expand(-1, -1, memory.size(2)))
        g = global_prior[:, None].expand_as(local_prior)
        p = torch.tanh(self.prior(torch.cat((g, local_prior), -1)))
        # Eqs. (10)-(13), acoustic injection occurs only before decoder layer 1.
        return self.init_projection(token_representation + self.eta * p)


class AGTSR(nn.Module):
    def __init__(self, cfg, token_count):
        """Lexical IDs: 1..token_count. CTC blank/padding=0.

        Decoder uses the same lexical IDs, plus BOS=token_count+1, EOS=token_count+2.
        CTC has NO BOS/EOS classes; decoder has no blank/BOS output classes.
        """
        super().__init__()
        self.cfg = cfg
        self.token_count = token_count
        self.bos_id, self.eos_id = token_count + 1, token_count + 2
        self.frontend = DualPathFrontend(cfg)
        self.pe = SinusoidalPE(cfg.d_model, cfg.max_positions)
        self.input_dropout = nn.Dropout(cfg.dropout)
        self.encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(cfg.d_model, cfg.heads, cfg.ff_dim, cfg.dropout,
                                       activation=cfg.activation, batch_first=True,
                                       norm_first=cfg.norm_first)
            for _ in range(cfg.encoder_layers)
        ])
        if cfg.encoder_layers not in cfg.ctc_layers:
            raise ValueError("Final encoder layer must belong to CAO supervision set")
        if any(l < 1 or l > cfg.encoder_layers for l in cfg.ctc_layers):
            raise ValueError("Invalid CTC layer index")
        self.ctc_heads = nn.ModuleDict({str(l): nn.Linear(cfg.d_model, token_count + 1)
                                       for l in cfg.ctc_layers})
        self.embedding = nn.Embedding(token_count + 3, cfg.d_model, padding_idx=0)
        self.apgqi = APGQI(cfg)
        layer = nn.TransformerDecoderLayer(cfg.d_model, cfg.heads, cfg.ff_dim, cfg.dropout,
                                           activation=cfg.activation, batch_first=True,
                                           norm_first=cfg.norm_first)
        self.decoder = nn.TransformerDecoder(layer, cfg.decoder_layers)
        self.output = nn.Linear(cfg.d_model, token_count + 3)

    def encode(self, waveform, wave_lengths, fmllr, fmllr_lengths, final_only=False):
        h, lengths = self.frontend(waveform, wave_lengths, fmllr, fmllr_lengths)
        mask = padding_mask(lengths, h.size(1))
        h = self.input_dropout(self.pe(h))
        heads = {}
        for layer_index, layer in enumerate(self.encoder, 1):
            h = layer(h, src_key_padding_mask=mask)
            # Prevent padded queries from affecting local Conv1D near utterance boundary.
            h = h.masked_fill(mask[..., None], 0)
            if str(layer_index) in self.ctc_heads and (not final_only or layer_index == self.cfg.encoder_layers):
                heads[layer_index] = self.ctc_heads[str(layer_index)](h).float().log_softmax(-1)
        return h, lengths, heads

    def decode(self, tokens, memory, memory_lengths, estimated_lengths):
        token_repr = self.pe(self.embedding(tokens))  # Eq. (11), no extra embedding scale
        query = self.input_dropout(self.apgqi(token_repr, memory, memory_lengths, estimated_lengths))
        causal = torch.ones(tokens.size(1), tokens.size(1), device=tokens.device, dtype=torch.bool).triu(1)
        hidden = self.decoder(query, memory, tgt_mask=causal,
                              tgt_key_padding_mask=tokens.eq(0),
                              memory_key_padding_mask=padding_mask(memory_lengths, memory.size(1)))
        logits = self.output(hidden)
        invalid = torch.zeros(logits.size(-1), device=logits.device, dtype=torch.bool)
        invalid[0] = invalid[self.bos_id] = True
        return logits.masked_fill(invalid, float("-inf"))

    def forward(self, batch):
        h, lengths, heads = self.encode(batch["waveform"], batch["wave_lengths"],
                                        batch["fmllr"], batch["fmllr_lengths"])
        # Training U includes EOS in decoder targets; CTC excludes EOS (documented choice).
        logits = self.decode(batch["decoder_input"], h, lengths, batch["decoder_lengths"])
        return {"decoder_logits": logits, "ctc_log_probs": heads, "encoder_lengths": lengths}

    @staticmethod
    def ctc_output_lengths(log_probs, lengths):
        """Merge repeats BEFORE removing blanks, Sec. 3.3. Estimate once per utterance."""
        estimates = []
        for row, n in zip(log_probs.argmax(-1), lengths.tolist()):
            collapsed = torch.unique_consecutive(row[:n])
            estimates.append(int(collapsed.ne(0).sum()))
        return lengths.new_tensor(estimates)
