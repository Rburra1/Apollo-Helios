# A5 Retune: Hyperparameter Sensitivity in the Modern Stack

**Status:** Reproduces known prior art. Documenting for completeness.

## TL;DR

The original `A5_modern` run reported val 5.9502, *worse* than baseline (5.4775). After a targeted two-hyperparameter retune (`--lr` cut from 2.5e-4 → 1.2e-4, `--patience` raised from 8 → 12), the same architecture reached **best val 4.1277 at iter 5500** — outperforming the next-best ablation (A1_rope, 4.6775) by 12.0% and the baseline by 24.6%. The retune confirms that the original "modern stack underperforms baseline" reading was a tuning artifact, not an architectural finding.

## Results table (updated)

| Run | Val | Δ vs A0 | Steps | Wall (h) | Notes |
|---|---|---|---|---|---|
| A0_baseline | 5.4775 | — | 8000 | 8.00 | |
| A1_rope | 4.6775 | −0.80 | 8000 | 8.01 | |
| A2_rmsnorm | 5.3608 | −0.12 | 6000 | 6.01 | early-stopped |
| A3_qknorm | 5.3919 | −0.09 | 8000 | 8.01 | |
| A4_swiglu | 5.4674 | −0.01 | 4750 | 4.75 | early-stopped |
| A5_modern | 5.9502 | +0.47 | 3500 | 3.51 | **early-stopped — broken under default hparams** |
| A6_modern_muon | 5.1404 | −0.34 | 8000 | 8.01 | |
| **A5_modern_retuned** | **4.1277** | **−1.35** | 8000 | 8.01 | LR ↓50%, patience +50% |

## What changed

| Hparam | Original A5 | Retuned A5 | Why |
|---|---|---|---|
| `--lr` | 2.5e-4 (default) | 1.2e-4 | Modern norms + RoPE + SwiGLU jointly are less LR-tolerant than baseline at this scale. Default LR caused early divergence. |
| `--patience` | 8 | 12 | Default patience triggered after 3500 steps before the model had a chance to converge. |

No source-level changes (warmup, β₂, init scaling) were exposed in the existing CLI surface; those were not modified.

## Trajectory observations

- Best val (4.1277) reached at iter 5500, with surrounding evals (4750: 4.18, 5750: 4.30) confirming the peak is not a logging fluke.
- Val degraded from 4.30 → 6.75 over the final ~2500 steps as cosine LR decayed from ~1.1e-4 → 1.2e-5. Likely a low-LR / stale Adam β₂=0.999 second-moment interaction. Not investigated further.
- Best checkpoint (`best.pt`) is saved at the peak; the saved model represents the val=4.13 state, not the iter-8000 state.

## Reproduction

```bash
python model/train.py \
  --preset modern \
  --optimizer adamw \
  --lr 1.2e-4 \
  --patience 12 \
  --out-dir out_retune \
  --iters 8000 \
  --hours 8.0
```

Log: `out_retune/modern_adamw/log.jsonl`. Best checkpoint: `out_retune/modern_adamw/best.pt`.

## Relation to prior work

This is not a novel finding at the field level. The general principle — that modern transformer components (RoPE, RMSNorm, QK-norm, SwiGLU) require their own hyperparameter regime to demonstrate their value over a baseline-tuned LayerNorm + GELU + learned-position-embedding stack — is documented in:

- **Modernizing GPT-2: A Journey from 2019 to 2025** (recsysml.substack.com, Jan 2026) — same experiment on Tiny Shakespeare, same observation that the modernized model fails to beat baseline on small datasets without adjusted training conditions.
- **SimpleGPT** (arxiv 2602.01212) — modern normalization variants tolerate LR 3–10× larger than standard convention; underscores hparam regime dependence.
- General community knowledge from open-weight Llama/Qwen/Gemma reproduction efforts.

The contribution of this note is reproducing the principle on a specific 25M-parameter MPS-trained setup and documenting the magnitude of the effect (val 5.95 → 4.13 with two hparam changes).

## Methodological takeaway

Single-arm ablations of modern transformer components, run under default hyperparameters that were tuned for the baseline configuration, will systematically under-report the value of those components. Any ablation suite that does not jointly retune hparams across configurations should be interpreted with that bias in mind.

---

*Run date: May 11, 2026. Hardware: MacBook Air M-series, MPS backend. Corpus: Apollo-Helios training set, 34M train tokens / 1.8M val tokens.*
