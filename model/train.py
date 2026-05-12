"""
Apollo-Helios training loop.

Near-direct port of Apollo/model/train.py with two additions:
  1. --preset flag for ablation presets (baseline, rope, rmsnorm, qknorm, swiglu, modern)
  2. --optimizer {adamw, muon} for the Muon ablation

Throttle fix: clamps sleep duration to be non-negative to avoid race condition
between the while-loop time check and the sleep call when target time has
already passed.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import build_model, ABLATION_PRESETS  # noqa: E402
from muon import Muon, partition_params           # noqa: E402


# ---------- helpers ----------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_lr(it, *, base_lr, min_lr, warmup, decay_to):
    if it < warmup:
        return base_lr * (it + 1) / warmup
    if it >= decay_to:
        return min_lr
    decay_ratio = (it - warmup) / (decay_to - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


def get_batch(split, train_data, val_data, device, batch_size, block_size):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, device, batch_size, block_size, eval_iters):
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, train_data, val_data, device, batch_size, block_size)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def fmt_secs(s):
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s/60:.1f}m"
    return f"{s/3600:.2f}h"


def safe_sleep_until(end_at, chunk=0.5):
    """Sleep in chunks until end_at, never with a negative duration."""
    while True:
        remaining = end_at - time.time()
        if remaining <= 0:
            return
        time.sleep(min(chunk, remaining))


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', required=True, choices=list(ABLATION_PRESETS.keys()))
    ap.add_argument('--optimizer', default='adamw', choices=['adamw', 'muon'])
    ap.add_argument('--data-dir', default='data')
    ap.add_argument('--out-dir', default='out')
    ap.add_argument('--vocab-size', type=int, default=32000)
    ap.add_argument('--block-size', type=int, default=256)
    ap.add_argument('--batch-size', type=int, default=24)
    ap.add_argument('--iters', type=int, default=8000)
    ap.add_argument('--hours', type=float, default=8.0)
    ap.add_argument('--lr', type=float, default=2.5e-4)
    ap.add_argument('--muon-lr', type=float, default=0.02)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--eval-interval', type=int, default=250)
    ap.add_argument('--eval-iters', type=int, default=25)
    ap.add_argument('--log-interval', type=int, default=25)
    ap.add_argument('--save-interval', type=int, default=1000)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--weight-decay', type=float, default=0.1)
    ap.add_argument('--grad-clip', type=float, default=1.0)
    args = ap.parse_args()

    device = get_device()

    print(f"== Apollo-Helios train ==")
    print(f"  device: {device}")
    print(f"  preset: {args.preset} | optimizer: {args.optimizer}")
    print(f"  iters: {args.iters} | batch: {args.batch_size} | block: {args.block_size}")
    print(f"  lr: {args.lr:.2e} | dropout: {args.dropout}")
    if args.hours is not None:
        print(f"  target hours: {args.hours:.2f} (throttled)")

    # Load data
    data_dir = Path(args.data_dir)
    train_path = data_dir / 'train.bin'
    val_path = data_dir / 'val.bin'
    if not train_path.exists() or not val_path.exists():
        sys.exit(f"missing {train_path} or {val_path}; run prepare.py first")
    train_data = np.fromfile(str(train_path), dtype=np.uint32)
    val_data = np.fromfile(str(val_path), dtype=np.uint32)
    print(f"  vocab_size: {args.vocab_size}")
    print(f"  train tokens: {len(train_data):,}, val tokens: {len(val_data):,}")

    # Build model
    model = build_model(args.preset, vocab_size=args.vocab_size,
                        block_size=args.block_size, dropout=args.dropout)
    model = model.to(device)
    body = model.num_params(exclude_embeddings=True)
    total = model.num_params(exclude_embeddings=False)
    print(f"  params: {total/1e6:.2f}M total | {body/1e6:.2f}M body")

    # Optimizer setup
    if args.optimizer == 'adamw':
        decay, no_decay = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if p.dim() < 2:
                no_decay.append(p)
            else:
                decay.append(p)
        optimizers = [
            torch.optim.AdamW(
                [
                    {"params": decay, "weight_decay": args.weight_decay},
                    {"params": no_decay, "weight_decay": 0.0},
                ],
                lr=args.lr,
                betas=(0.9, 0.95),
            )
        ]
    else:
        muon_params, adamw_params = partition_params(model)
        mc = sum(p.numel() for p in muon_params)
        ac = sum(p.numel() for p in adamw_params)
        print(f"  muon params: {mc/1e6:.2f}M | adamw params: {ac/1e6:.2f}M")
        optimizers = [
            Muon(muon_params, lr=args.muon_lr, momentum=0.95,
                 weight_decay=args.weight_decay),
            torch.optim.AdamW(adamw_params, lr=args.lr,
                              weight_decay=0.0, betas=(0.9, 0.95)),
        ]

    # Output dir
    run_name = f"{args.preset}_{args.optimizer}"
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'log.jsonl'
    cfg_path = out_dir / 'config.json'
    ckpt_path = out_dir / 'best.pt'
    cfg_path.write_text(json.dumps({
        'preset': args.preset,
        'preset_flags': ABLATION_PRESETS[args.preset],
        'optimizer': args.optimizer,
        'hparams': vars(args),
        'param_count_body': body,
        'param_count_total': total,
    }, indent=2))
    print(f"  out: {ckpt_path}")

    warmup = max(100, args.iters // 25)
    decay_to = args.iters
    min_lr = args.lr / 10.0

    target_seconds = args.hours * 3600 if args.hours is not None else None
    start_time = time.time()
    best_val = float('inf')
    evals_since_improvement = 0
    stopped_early = False

    log_f = open(log_path, 'w')

    for it in range(args.iters + 1):
        for opt in optimizers:
            for pg in opt.param_groups:
                base = args.muon_lr if isinstance(opt, Muon) else args.lr
                opt_min = base / 10.0
                pg['lr'] = get_lr(it, base_lr=base, min_lr=opt_min,
                                  warmup=warmup, decay_to=decay_to)

        if it % args.eval_interval == 0 or it == args.iters:
            losses = estimate_loss(model, train_data, val_data, device,
                                   args.batch_size, args.block_size, args.eval_iters)
            elapsed = time.time() - start_time
            improved = losses['val'] < best_val
            marker = ' *' if improved else ''
            cur_lr = optimizers[0].param_groups[0]['lr']
            print(
                f"iter {it:5d} | lr {cur_lr:.2e} | "
                f"train {losses['train']:.4f} | val {losses['val']:.4f} | "
                f"elapsed {fmt_secs(elapsed)}{marker}"
            )
            log_f.write(json.dumps({
                'iter': it,
                'lr': cur_lr,
                'train_loss': losses['train'],
                'val_loss': losses['val'],
                'elapsed_h': elapsed / 3600,
            }) + '\n')
            log_f.flush()

            if improved:
                best_val = losses['val']
                evals_since_improvement = 0
                torch.save({
                    'model': model.state_dict(),
                    'config': model.cfg.__dict__,
                    'iter': it,
                    'val_loss': best_val,
                }, ckpt_path)
            else:
                evals_since_improvement += 1
                if args.patience > 0 and evals_since_improvement >= args.patience:
                    print(
                        f"\nearly stop: val loss has not improved for "
                        f"{args.patience} consecutive evals (best val={best_val:.4f}). "
                        f"halting at iter {it}."
                    )
                    stopped_early = True
                    break

        if it == args.iters:
            break

        X, Y = get_batch("train", train_data, val_data, device,
                         args.batch_size, args.block_size)
        _, loss = model(X, Y)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        for opt in optimizers:
            opt.step()

        if it % args.log_interval == 0 and it > 0:
            cur_lr = optimizers[0].param_groups[0]['lr']
            print(f"  step {it:5d} | loss {loss.item():.4f} | lr {cur_lr:.2e}")

        # Throttle: sleep until target wall time for this iter, with safe sleep
        if target_seconds is not None and args.iters > 0:
            elapsed = time.time() - start_time
            target_at_this_iter = (it + 1) / args.iters * target_seconds
            slack = target_at_this_iter - elapsed
            if slack > 0:
                end_at = time.time() + slack
                safe_sleep_until(end_at)

    log_f.close()
    total_elapsed = time.time() - start_time
    if stopped_early:
        print(f"\nstopped early in {fmt_secs(total_elapsed)} | best val: {best_val:.4f}")
    else:
        print(f"\nDONE: best val {best_val:.4f}    ckpt: {ckpt_path}")


if __name__ == "__main__":
    main()
