# Apollo-Modern

Sibling project to Apollo. Same architecture body size (8L/8H/512, ~25M params), same training corpus (reuses Apollo's `data/`), same wall-clock budget (8h M4 throttled).

The point: cleanly attribute the contribution of each modern transformer component (RoPE, RMSNorm, QK-norm, SwiGLU, Muon) at the 25M-param scale, in single-variable ablations.

This is something only a constrained-compute project can do well. Big labs publish these components together and you can't tell which one did the work.

## Why this fork instead of editing Apollo

Apollo's thesis is "architecture held constant, attribute changes to corpus and tokenizer." Apollo-Modern's thesis is "modern architecture held constant, isolate the contribution of each component." Both are valid. Forking keeps each project's narrative clean.

## What's in here

```
Apollo-Modern/
├── README.md
├── ABLATIONS.md               experimental protocol + results
├── ablation_runner.py         runs the 7-experiment matrix
└── model/
    ├── model.py               transformer with config flags per ablation
    ├── muon.py                Muon optimizer
    └── train.py               training loop with mixed Muon/AdamW
```

What's reused from Apollo (symlink or copy):
- `data/train.bin`, `data/val.bin`, `data/tokenizer.model`, `data/meta.json`
- `prepare.py` (only if regenerating data)
- `sample.py` (works with Apollo-Modern checkpoints if you point it at the new model file)

## Quick start

```bash
# From ~/Desktop/projectvik/aivik/Apollo-Modern/

# Reuse Apollo's prepared data (or run ../Apollo/model/prepare.py if missing)
mkdir -p data
ln -s ../Apollo/data/train.bin data/train.bin
ln -s ../Apollo/data/val.bin data/val.bin
ln -s ../Apollo/data/tokenizer.model data/tokenizer.model
ln -s ../Apollo/data/meta.json data/meta.json

# Activate Apollo's venv (Python 3.14, all deps already there)
source ../Apollo/.venv/bin/activate

# Sanity check param counts
python model/model.py
# should print body params for each preset, all near 25.3M

# Single ablation (smoke test, 1 hour)
python model/train.py --preset modern --optimizer muon --hours 1 --iters 1000 --eval-interval 100

# Full 7-run matrix, 8h each, ~7 overnights
python ablation_runner.py --hours 8 --resume

# Collect results without re-running
python ablation_runner.py --collect-only
```

## Experiment matrix

| Run | Preset | Optimizer | Notes |
|-----|--------|-----------|-------|
| A0  | baseline | AdamW | Apollo v2 reproduction; should land near val 3.95 |
| A1  | rope     | AdamW | + RoPE only |
| A2  | rmsnorm  | AdamW | + RMSNorm only |
| A3  | qknorm   | AdamW | + QK-norm only |
| A4  | swiglu   | AdamW | + SwiGLU only |
| A5  | modern   | AdamW | All four arch components |
| A6  | modern   | Muon  | All four + Muon |

Single-variable ablations (A1-A4) tell you the marginal contribution of each component. A5 shows whether they compose. A6 isolates the optimizer's contribution given the modern stack.

Total wall: 56h split across overnights. Resumable.

## Output structure

```
out/
├── baseline_adamw/
│   ├── best.pt
│   ├── log.jsonl       per-eval val loss
│   └── config.json     full run config
├── rope_adamw/
│   └── ...
└── results.md          summary table after collect-only or all runs done
```

## What Phase 2 looks like (after these runs)

Phase 2 is the register-token mechanistic probe. It needs a trained checkpoint, so it can only start after Phase 1 finishes (or at minimum after A6 finishes). Probes will go in a separate `probes/` directory and won't require new training runs — they hook into the trained model and analyze attention/MLP behavior conditional on the leading register tag.

The Phase 2 questions:
1. At which layer does the register signal first measurably affect predictions?
2. Are there register-specific attention heads (heads whose pattern changes most when the register tag changes)?
3. Are there register-specific MLP neurons?
4. Can we causally intervene on the register tag mid-sequence and switch generation register?

These probes are data-cheap and compute-cheap (forward passes only). Will be written once we have at least the modern_muon checkpoint.
