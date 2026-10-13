"""40-epoch, two-stage AGT-SR training; supports torchrun DDP. Does not auto-launch."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from data import SpeechDataset, Collator
from losses import AGTSRLoss
from model import AGTSR, ModelConfig


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def move(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def validate(model, criterion, loader, device, world_size):
    # Bare module allows unequal validation shard sizes without DDP collective deadlocks.
    model.eval()
    sums = torch.zeros(4, dtype=torch.float64, device=device)
    for raw in loader:
        batch = move(raw, device)
        _, stats = criterion(model(batch), batch)
        b = batch["waveform"].size(0)
        sums[:3] += torch.stack([stats[k] for k in ("total", "ce", "cao")]).double() * b
        sums[3] += b
    if world_size > 1:
        dist.all_reduce(sums)
    if sums[3] == 0:
        raise ValueError("Empty validation dataset")
    return dict(zip(("total", "ce", "cao"), (sums[:3] / sums[3]).tolist()))


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def append_log(path, record):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--resume")
    parser.add_argument("--allow-other-world-size", action="store_true",
                        help="Explicitly allow hardware different from paper; keep global batch 32")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    # Resolve all configured paths relative to config, independent of current working directory.
    def resolve(name):
        path = Path(name)
        return str(path if path.is_absolute() else config_path.parent / path)
    for key in ("train_manifest", "valid_manifest", "vocabulary"):
        cfg["data"][key] = resolve(cfg["data"][key])
    cfg["augmentation"]["musan"]["root"] = resolve(cfg["augmentation"]["musan"]["root"])
    cfg["training"]["output_dir"] = resolve(cfg["training"]["output_dir"])
    training = cfg["training"]
    if training["epochs"] != 40:
        raise ValueError("Manuscript schedule requires 40 epochs (10 frozen + 30 fine-tuned)")
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world != training["expected_world_size"] and not args.allow_other_world_size:
        raise ValueError("Paper uses four GPUs; use torchrun --nproc_per_node=4 or explicitly --allow-other-world-size")
    if world > 1:
        dist.init_process_group("nccl" if os.name != "nt" and torch.cuda.is_available() else "gloo")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    base_seed = training["seed"]
    random.seed(base_seed + rank)
    np.random.seed(base_seed + rank)
    torch.manual_seed(base_seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(base_seed + rank)
    batch_size = training["global_batch_size"]
    if batch_size % world:
        raise ValueError("Global batch size must be divisible by world size")
    per_rank = batch_size // world
    vocab = json.loads(Path(cfg["data"]["vocabulary"]).read_text(encoding="utf-8"))
    token_count = len(vocab["token_to_id"])
    if sorted(vocab["token_to_id"].values()) != list(range(1, token_count + 1)):
        raise ValueError("Vocabulary IDs must be consecutive 1..N")
    train_data = SpeechDataset(cfg["data"]["train_manifest"], vocab, cfg, training=True)
    valid_data = SpeechDataset(cfg["data"]["valid_manifest"], vocab, cfg, training=False)
    sampler = DistributedSampler(train_data, num_replicas=world, rank=rank,
                                 shuffle=True, seed=base_seed, drop_last=True)
    loader_options = dict(batch_size=per_rank, num_workers=training["num_workers"],
                          pin_memory=device.type == "cuda", collate_fn=Collator(token_count),
                          worker_init_fn=seed_worker)
    train_loader = DataLoader(train_data, sampler=sampler, drop_last=True, **loader_options)
    valid_loader = DataLoader(Subset(valid_data, list(range(rank, len(valid_data), world))),
                              shuffle=False, **loader_options)
    if len(train_loader) == 0:
        raise ValueError("Training data insufficient for one global batch")
    model_cfg = ModelConfig(**cfg["model"])
    model = AGTSR(model_cfg, token_count).to(device)
    criterion = AGTSRLoss(training["lambda_cao"], training["gamma"], training["epsilon"])
    # Include frozen parameters now, so stage-2 parameters already belong to optimizer.
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                 betas=tuple(training["betas"]), eps=training["adam_epsilon"],
                                 weight_decay=training["weight_decay"])
    warmup = training["warmup_steps"]
    total_steps = len(train_loader) * training["epochs"]
    def lr_multiplier(step):
        s = step + 1
        if s <= warmup:
            return s / warmup
        if training["after_warmup"] == "constant":
            return 1.0
        if training["after_warmup"] == "inverse_sqrt":
            return (warmup / s) ** 0.5
        raise ValueError("Unsupported after_warmup schedule")
    if training["after_warmup"] not in ("constant", "inverse_sqrt"):
        raise ValueError("Unsupported after_warmup schedule")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    start_epoch, global_step, best = 1, 0, float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["config"] != cfg or checkpoint["vocabulary"] != vocab:
            raise ValueError("Resume requires identical configuration and vocabulary")
        if checkpoint["world_size"] != world:
            raise ValueError("Resume requires same world size to restore per-rank RNG")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch, global_step, best = checkpoint["epoch"] + 1, checkpoint["step"], checkpoint["best"]
        restore_rng(checkpoint["rng_by_rank"][rank])
    output = Path(training["output_dir"])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "resolved_config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"world={world}, per_rank_batch={per_rank}, global_batch={batch_size}, total_steps={total_steps}")
        if total_steps < warmup:
            print("Note: selected dataset is too small to finish the paper's 20,000-step warm-up in 40 epochs")
    active_stage, wrapped = None, None
    for epoch in range(start_epoch, training["epochs"] + 1):
        stage = 1 if epoch <= 10 else 2
        if stage != active_stage:
            # DDP captures trainable parameters at construction. Rebuild when unfreezing.
            if world > 1:
                dist.barrier()
            del wrapped
            model.frontend.set_stage(epoch)
            wrapped = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None,
                          find_unused_parameters=True) if world > 1 else model
            active_stage = stage
        sampler.set_epoch(epoch)
        wrapped.train()
        for raw in train_loader:
            batch = move(raw, device)
            optimizer.zero_grad(set_to_none=True)
            loss, stats = criterion(wrapped(batch), batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training objective")
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            if global_step % training["log_every"] == 0:
                values = torch.stack(list(stats.values()))
                if world > 1:
                    dist.all_reduce(values)
                    values /= world
                if rank == 0:
                    record = {"epoch": epoch, "step": global_step, "stage": stage,
                              "next_lr": scheduler.get_last_lr()[0],
                              **dict(zip(stats, values.cpu().tolist()))}
                    append_log(output / "train.jsonl", record)
                    print(json.dumps(record))
        metrics = validate(model, criterion, valid_loader, device, world)
        improved = metrics["total"] < best
        best = min(best, metrics["total"])
        states = [None] * world
        if world > 1:
            dist.all_gather_object(states, rng_state())
        else:
            states[0] = rng_state()
        if rank == 0:
            append_log(output / "valid.jsonl", {"epoch": epoch, "step": global_step, **metrics})
            checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(), "epoch": epoch, "step": global_step,
                          "best": best, "config": cfg, "model_config": asdict(model_cfg),
                          "vocabulary": vocab, "world_size": world, "rng_by_rank": states}
            torch.save(checkpoint, output / "last.pt")
            if improved:
                torch.save(checkpoint, output / "best.pt")
            print(f"epoch={epoch}, stage={stage}, validation={metrics}")
        if world > 1:
            dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
