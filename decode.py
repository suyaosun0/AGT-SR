"""Reference joint CTC/attention beam search, Eq. (22), no external LM.

All lexical extensions are scored (no attention-only candidate preselection).
CTC prefixes use the probability of collapsed paths STARTING with that prefix;
completed hypotheses use exact CTC sequence probability. This reference version
recomputes DP and full decoder prefixes, favoring readability over speed.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from data import read_audio
from model import AGTSR, ModelConfig


def ctc_score(log_probs, labels, prefix=False):
    """Log-domain CTC DP. log_probs: NumPy [T,V], lexical labels only."""
    if not labels:
        return 0.0 if prefix else float(log_probs[:, 0].sum())
    extended = [0]
    for label in labels:
        extended += [label, 0]
    if prefix:
        extended = extended[:-1]  # last label is the absorbing first-hit boundary
    previous = np.full(len(extended), -np.inf)
    previous[0] = 0.0
    hit_mass = -np.inf
    for frame in log_probs:
        current = np.full_like(previous, -np.inf)
        for s, label in enumerate(extended):
            incoming = previous[s]
            if s >= 1:
                incoming = np.logaddexp(incoming, previous[s-1])
            if s >= 2 and label != 0 and label != extended[s-2]:
                incoming = np.logaddexp(incoming, previous[s-2])
            value = incoming + frame[label]
            if prefix and s == len(extended) - 1:
                # previous[last] is always -inf: count FIRST completion once.
                # Any remaining acoustic tail is unrestricted and marginalizes to 1.
                hit_mass = np.logaddexp(hit_mass, value)
            else:
                current[s] = value
        previous = current
    return float(hit_mass if prefix else np.logaddexp(previous[-1], previous[-2]))


@torch.no_grad()
def beam_search(model, waveform, wave_lengths, fmllr, fmllr_lengths,
                beam_size=10, beta=0.3, max_tokens=500):
    if waveform.size(0) != 1:
        raise ValueError("Reference search expects one utterance")
    if beam_size < 1 or not 0 <= beta <= 1:
        raise ValueError("Invalid beam_size or beta")
    model.eval()
    memory, lengths, heads = model.encode(waveform, wave_lengths, fmllr, fmllr_lengths, final_only=True)
    final = heads[model.cfg.encoder_layers]
    estimated = model.ctc_output_lengths(final, lengths)  # computed ONCE, never from hypothesis length
    lp = final[0, :int(lengths[0])].cpu().double().numpy()
    cache = {}
    def combine(ctc, attention):
        # Avoid 0 * -inf when callers explicitly disable one scoring branch.
        if beta == 0:
            return attention
        if beta == 1:
            return ctc
        return beta * ctc + (1-beta) * attention
    def acoustic_score(labels, prefix):
        key = (tuple(labels), prefix)
        if key not in cache:
            cache[key] = ctc_score(lp, labels, prefix)
        return cache[key]
    # tuple: lexical tokens, attention log score, joint score, ended
    beams = [([], 0.0, 0.0, False)]
    for _ in range(max_tokens + 1):
        candidates = []
        for labels, attention, joint, ended in beams:
            if ended:
                candidates.append((labels, attention, joint, True))
                continue
            tokens = torch.tensor([[model.bos_id] + labels], device=memory.device)
            next_lp = model.decode(tokens, memory, lengths, estimated)[0, -1].float().log_softmax(-1).cpu()
            eos_attention = attention + float(next_lp[model.eos_id])
            candidates.append((labels, eos_attention,
                               combine(acoustic_score(labels, False), eos_attention), True))
            if len(labels) >= max_tokens:
                continue
            for token in range(1, model.token_count + 1):
                extension = labels + [token]
                score_att = attention + float(next_lp[token])
                score = combine(acoustic_score(extension, True), score_att)
                if math.isfinite(score):
                    candidates.append((extension, score_att, score, False))
        beams = sorted(candidates, key=lambda h: h[2], reverse=True)[:beam_size]
        if beams and all(h[3] for h in beams):
            break
    complete = [h for h in beams if h[3] and math.isfinite(h[2])]
    if not complete:
        raise RuntimeError("No finite completed hypothesis; increase search limit/check model")
    best = max(complete, key=lambda h: h[2])
    return {"token_ids": best[0], "joint_score": best[2], "ctc_length_estimate": int(estimated[0])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--fmllr", required=True, help="External 80-D FMLLR .npy for this SAME waveform")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    vocab = checkpoint["vocabulary"]["token_to_id"]
    model = AGTSR(ModelConfig(**checkpoint["model_config"]), len(vocab)).to(args.device)
    model.load_state_dict(checkpoint["model"])
    waveform = read_audio(args.audio)
    waveform = (waveform - waveform.mean()) / torch.sqrt(waveform.var(unbiased=False) + 1e-7)
    features = torch.tensor(np.load(args.fmllr, allow_pickle=False), dtype=torch.float32)
    if features.ndim != 2 or features.size(1) != 80 or not torch.isfinite(features).all():
        raise ValueError("Expected finite FMLLR [T,80]")
    options = checkpoint["config"]["decoding"]
    result = beam_search(model, waveform[None].to(args.device),
                         torch.tensor([len(waveform)], device=args.device),
                         features[None].to(args.device), torch.tensor([len(features)], device=args.device),
                         options["beam_size"], options["ctc_weight"], options["max_tokens"])
    inverse = {i: token for token, i in vocab.items()}
    result["tokens"] = [inverse[i] for i in result["token_ids"]]
    # Detokenization depends on user's tokenizer. Do not guess it from language.
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
