"""
Apollo-Helios ablation runner.

Runs the full 7-experiment matrix in sequence:

    A0  baseline       (Apollo v2 reproduction: old arch, AdamW)
    A1  rope           (+ RoPE only)
    A2  rmsnorm        (+ RMSNorm only)
    A3  qknorm         (+ QK-norm only)
    A4  swiglu         (+ SwiGLU only)
    A5  modern         (all four modern arch components, AdamW)
    A6  modern_muon    (all four + Muon)

Use --resume to skip runs that already have a best.pt checkpoint.
Each run is its own python subprocess (no state bleed, clean stop/resume).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


EXPERIMENTS = [
    ('A0_baseline',     'baseline', 'adamw'),
    ('A1_rope',         'rope',     'adamw'),
    ('A2_rmsnorm',      'rmsnorm',  'adamw'),
    ('A3_qknorm',       'qknorm',   'adamw'),
    ('A4_swiglu',       'swiglu',   'adamw'),
    ('A5_modern',       'modern',   'adamw'),
    ('A6_modern_muon',  'modern',   'muon'),
]


def already_done(out_dir: Path, preset: str, optimizer: str) -> bool:
    return (out_dir / f"{preset}_{optimizer}" / 'best.pt').exists()


def run_one(run_name, preset, optimizer, hours, out_dir, extra_args):
    print(f"\n{'='*60}\n  RUN: {run_name}  (preset={preset}, opt={optimizer})\n{'='*60}")
    cmd = [
        sys.executable, 'model/train.py',
        '--preset', preset,
        '--optimizer', optimizer,
        '--hours', str(hours),
        '--out-dir', str(out_dir),
    ] + extra_args
    print(' '.join(cmd))
    start = time.time()
    result = subprocess.run(cmd)
    elapsed = (time.time() - start) / 3600
    if result.returncode != 0:
        print(f"  FAILED with code {result.returncode} after {elapsed:.2f}h")
        return False
    print(f"  done in {elapsed:.2f}h")
    return True


def collect_results(out_dir: Path):
    rows = []
    for run_name, preset, optimizer in EXPERIMENTS:
        run_dir = out_dir / f"{preset}_{optimizer}"
        log_path = run_dir / 'log.jsonl'
        cfg_path = run_dir / 'config.json'
        if not log_path.exists() or not cfg_path.exists():
            rows.append({'run': run_name, 'status': 'missing'})
            continue
        cfg = json.loads(cfg_path.read_text())
        best_val = float('inf')
        last_step = 0
        last_elapsed = 0
        with log_path.open() as f:
            for line in f:
                e = json.loads(line)
                # Handle both 'iter' (Apollo-style) and 'step' (old-style) keys
                step_val = e.get('iter', e.get('step', 0))
                val_loss = e.get('val_loss', float('inf'))
                if val_loss < best_val:
                    best_val = val_loss
                last_step = step_val
                last_elapsed = e.get('elapsed_h', 0)
        rows.append({
            'run': run_name,
            'preset': preset,
            'optimizer': optimizer,
            'body_M': round(cfg.get('param_count_body', 0) / 1e6, 2),
            'best_val': round(best_val, 4),
            'last_step': last_step,
            'wall_h': round(last_elapsed, 2),
            'status': 'ok',
        })
    return rows


def write_results_md(rows, out_path):
    lines = [
        '# Apollo-Helios ablation results',
        '',
        '| Run | Preset | Optimizer | Body params | Best val | Steps | Wall h |',
        '|-----|--------|-----------|-------------|----------|-------|--------|',
    ]
    for r in rows:
        if r.get('status') != 'ok':
            lines.append(f"| {r['run']} | - | - | - | - | - | (missing) |")
            continue
        lines.append(
            f"| {r['run']} | {r['preset']} | {r['optimizer']} | "
            f"{r['body_M']}M | {r['best_val']} | {r['last_step']} | {r['wall_h']} |"
        )
    out_path.write_text('\n'.join(lines) + '\n')
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hours', type=float, default=8.0)
    ap.add_argument('--out-dir', default='out')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--only', nargs='+', default=None)
    ap.add_argument('--collect-only', action='store_true')
    args, extra = ap.parse_known_args()
    out_dir = Path(args.out_dir)

    if not args.collect_only:
        for run_name, preset, optimizer in EXPERIMENTS:
            if args.only and run_name not in args.only:
                continue
            if args.resume and already_done(out_dir, preset, optimizer):
                print(f"SKIP {run_name} (already done)")
                continue
            ok = run_one(run_name, preset, optimizer, args.hours, out_dir, extra)
            if not ok:
                print(f"\nstopping after failed run {run_name}")
                break

    rows = collect_results(out_dir)
    write_results_md(rows, out_dir / 'results.md')
    print('\nResults:')
    for r in rows:
        if r.get('status') == 'ok':
            print(f"  {r['run']:20s} val={r['best_val']:.4f}  "
                  f"steps={r['last_step']}  wall={r['wall_h']}h")
        else:
            print(f"  {r['run']:20s} (missing)")


if __name__ == '__main__':
    main()
