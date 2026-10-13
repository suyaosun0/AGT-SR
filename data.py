"""Manifest IO, external FMLLR transforms, MUSAN and SpecAugment.

FMLLR statistics/transform estimation is deliberately outside neural training.
The manuscript does not specify augmentation policy or fbank extraction options;
their explicit choices live in config.json and README.md.
"""
import json
from pathlib import Path
import random

import numpy as np
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def read_audio(path, sample_rate=16000):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != sample_rate or audio.shape[1] != 1:
        raise ValueError(f"Expected mono {sample_rate} Hz audio: {path}, got {sr} Hz/{audio.shape[1]} channels")
    if len(audio) < 400 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid or too short waveform: {path}")
    return torch.from_numpy(audio[:, 0].copy())


def apply_fmllr(features, transform):
    """Eq. (1). transform .npz must contain A[80,80] and b[80]."""
    with np.load(transform, allow_pickle=False) as values:
        a = torch.tensor(values["A"], dtype=features.dtype)
        b = torch.tensor(values["b"], dtype=features.dtype)
    if a.shape != (80, 80) or b.shape != (80,):
        raise ValueError("FMLLR A/b must have shapes (80,80)/(80,)")
    return features @ a.T + b


def fbank(waveform, options):
    from torchaudio.compliance.kaldi import fbank as kaldi_fbank
    return kaldi_fbank(waveform[None] * 32768.0, num_mel_bins=80,
                      sample_frequency=16000, frame_length=options["frame_length_ms"],
                      frame_shift=options["frame_shift_ms"], dither=0.0,
                      use_energy=False, snip_edges=True)


def add_noise(waveform, noise, snr):
    if noise.numel() < waveform.numel():
        noise = noise.repeat((waveform.numel() + noise.numel() - 1) // noise.numel())
    start = random.randint(0, noise.numel() - waveform.numel())
    noise = noise[start:start + waveform.numel()]
    speech_power, noise_power = waveform.square().mean(), noise.square().mean()
    if noise_power <= 0:
        raise ValueError("MUSAN file is silent")
    mixed = waveform + noise * torch.sqrt(speech_power / noise_power / (10 ** (snr / 10)))
    # Joint rescaling avoids clipping while preserving SNR; unspecified implementation choice.
    return mixed / mixed.abs().max().clamp_min(1.0)


def specaugment(features, policy):
    features = features.clone()
    for axis, count, width in [(0, policy["time_masks"], policy["time_width"]),
                               (1, policy["frequency_masks"], policy["frequency_width"])]:
        for _ in range(count):
            w = random.randint(0, min(width, features.size(axis)))
            start = random.randint(0, features.size(axis) - w)
            if axis == 0:
                features[start:start+w] = 0
            else:
                features[:, start:start+w] = 0
    return features


class SpeechDataset(Dataset):
    def __init__(self, manifest, vocabulary, config, training=False):
        self.path = Path(manifest).resolve()
        self.rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.rows:
            raise ValueError(f"Empty manifest: {self.path}")
        self.vocab, self.config, self.training = vocabulary, config, training
        self.noises = []
        if training and config["augmentation"]["musan"]["enabled"]:
            noise_root = Path(config["augmentation"]["musan"]["root"])
            if not noise_root.is_dir():
                raise FileNotFoundError(f"MUSAN directory does not exist: {noise_root}")
            self.noises = sorted(noise_root.rglob("*.wav"))
            if not self.noises:
                raise ValueError("MUSAN directory contains no WAV files")
            if any("transform" not in row for row in self.rows):
                raise ValueError("Online MUSAN needs external speaker A/b transforms for ALL train utterances")

    def resolve(self, name):
        path = Path(name)
        return path if path.is_absolute() else self.path.parent / path

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        waveform = read_audio(self.resolve(row["audio"]))
        musan = self.config["augmentation"]["musan"]
        noisy = self.training and bool(self.noises) and random.random() < musan["probability"]
        if noisy:
            noise = read_audio(random.choice(self.noises))
            waveform = add_noise(waveform, noise, random.uniform(*musan["snr_db"]))
        if "transform" in row:
            # Same acoustic observation feeds BOTH branches after waveform augmentation.
            acoustic = apply_fmllr(fbank(waveform, self.config["fbank"]), self.resolve(row["transform"]))
        elif "fmllr" in row and not noisy:
            acoustic = torch.from_numpy(np.load(self.resolve(row["fmllr"]), allow_pickle=False).astype("float32"))
        else:
            raise ValueError("Provide external FMLLR features or GMM-HMM-estimated A/b; no log-Mel fallback")
        if acoustic.ndim != 2 or acoustic.size(1) != 80 or acoustic.size(0) == 0:
            raise ValueError("FMLLR features must be nonempty [T,80]")
        if not torch.isfinite(acoustic).all():
            raise ValueError("FMLLR contains NaN/Inf")
        if self.training and self.config["augmentation"]["specaugment"]["enabled"]:
            acoustic = specaugment(acoustic, self.config["augmentation"]["specaugment"])
        # wav2vec2 normalization per utterance; do not normalize FMLLR implicitly.
        waveform = (waveform - waveform.mean()) / torch.sqrt(waveform.var(unbiased=False) + 1e-7)
        ids = row.get("token_ids")
        if ids is None:
            tokens = row.get("tokens")
            if tokens is None:
                raise ValueError("Manifest requires token_ids or explicit tokens; tokenizer is unspecified by paper")
            ids = [self.vocab["token_to_id"][token] for token in tokens]
        if not ids or any(not isinstance(i, int) or i < 1 or i > len(self.vocab["token_to_id"]) for i in ids):
            raise ValueError("Targets must contain nonempty lexical IDs in [1,N], no blank/BOS/EOS")
        return {"waveform": waveform, "fmllr": acoustic, "tokens": torch.tensor(ids, dtype=torch.long)}


class Collator:
    def __init__(self, token_count):
        self.token_count = token_count

    def __call__(self, items):
        return _collate(items, self.token_count)


def _collate(items, n):
    wave = [x["waveform"] for x in items]
    features = [x["fmllr"] for x in items]
    targets = [x["tokens"] for x in items]
    return {
        "waveform": pad_sequence(wave, batch_first=True),
        "wave_lengths": torch.tensor([len(x) for x in wave]),
        "fmllr": pad_sequence(features, batch_first=True),
        "fmllr_lengths": torch.tensor([len(x) for x in features]),
        "ctc_targets": torch.cat(targets),
        "ctc_lengths": torch.tensor([len(x) for x in targets]),
        "decoder_input": pad_sequence([torch.cat((torch.tensor([n+1]), x)) for x in targets], batch_first=True),
        "decoder_target": pad_sequence([torch.cat((x, torch.tensor([n+2]))) for x in targets], batch_first=True),
        "decoder_lengths": torch.tensor([len(x)+1 for x in targets]),
    }
