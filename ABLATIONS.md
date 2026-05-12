# Apollo-Modern: Ablation Protocol

## Research question

At the 25M-param decoder scale, in single-variable ablations on a fixed corpus and tokenizer, what is the contribution of each modern transformer component to validation loss and to the named OOD failure modes inherited from Apollo v2?

## Why this matters

The 2026 small-LM consensus stack (RoPE + RMSNorm + QK-norm + SwiGLU + Muon) is published as a bundle in nearly every modern paper. SmolLM2, IMU-1, OLMo 3, Gemma 3 270M, Liquid LFM2 all ship the bundle. None of them isolate component contributions cleanly because they're at 270M+ params and can't afford to.

A 25M-param model trained on a laptop in 8 hours can. That's the unique thing this project contributes.

## Held constant across all runs

- **Body params:** ~25.3M (8 layers, 8 heads, 512 embed). SwiGLU hidden is set to 1344 to match GELU's 4x hidden=2048 in total parameter count within ~2%.
- **Block size:** 256
- **Vocab:** 32000 (Apollo v2 SentencePiece BPE)
- **Corpus:** Apollo v2's prepared `train.bin` + `val.bin`. Token mix 34/35/31 lit/wiki/code.
- **Iterations:** 8000 max, with patience-8 early stopping
- **Batch size:** 24
- **Wall clock:** 8h M4-MPS throttled
- **Seed:** 1337
- **LR schedule:** linear warmup over 320 steps → cosine decay
- **Weight decay:** 0.1 on 2D matrix params, 0 elsewhere
- **Gradient clip:** 1.0

## Variables

| Component | Baseline | Modern |
|-----------|----------|--------|
| Position encoding | Learned absolute | RoPE |
| Norm | LayerNorm | RMSNorm |
| Attention stability | none | QK-norm (RMSNorm on Q, K) |
| FFN activation | GELU, 4x hidden | SwiGLU, 8/3 hidden |
| Optimizer | AdamW | Muon (matrices) + AdamW (embeds/norms/biases) |

## Run matrix

| Run | RoPE | RMSNorm | QK-norm | SwiGLU | Muon | Hypothesis |
|-----|------|---------|---------|--------|------|------------|
| A0 baseline | — | — | — | — | — | Reproduces Apollo v2 val ≈ 3.95 |
| A1 rope | ✓ | — | — | — | — | Δval ≈ -0.03 to -0.08. RoPE shines at longer context; minor gain at 256. |
| A2 rmsnorm | — | ✓ | — | — | — | Δval ≈ 0 ± 0.02. Pure speedup, equivalent quality. |
| A3 qknorm | — | — | ✓ | — | — | Δval ≈ 0 ± 0.02 at 8 layers. Insurance, not a contributor at this depth. |
| A4 swiglu | — | — | — | ✓ | — | Δval ≈ -0.05 to -0.15. Largest single arch contributor. |
| A5 modern | ✓ | ✓ | ✓ | ✓ | — | Δval ≈ -0.10 to -0.25. Components compose roughly additively. |
| A6 modern_muon | ✓ | ✓ | ✓ | ✓ | ✓ | Δval ≈ -0.20 to -0.50. Muon is ~2× compute-efficient → faster convergence at fixed wall. |

These hypotheses are pre-registered. Update with actual numbers after each run completes; do not edit the hypothesis column.

## Failure-mode evaluation

In addition to val loss, every checkpoint runs the same 6-prompt stress test as Apollo (3 in-distribution, 3 OOD per register). Pre-registered binary outcomes:

| Failure mode | A0 | A1 | A2 | A3 | A4 | A5 | A6 |
|--------------|----|----|----|----|----|----|----|
| Illustration loop on lit OOD | YES | ? | ? | ? | ? | ? | ? |
| Fake-French on wiki OOD | YES | ? | ? | ? | ? | ? | ? |
| Test-fixture reversion on code OOD | YES (v1) / NO (v2) | ? | ? | ? | ? | ? | ? |

For Apollo-Modern using Apollo v2's corpus, A0 should reproduce v2's pattern: NO test-fixture reversion (v2 corpus already excludes tests), but the literature and wiki failures may or may not appear depending on whether they were corpus-driven or arch-driven. **This itself is a finding** — if A0 reproduces wiki/lit failures, they are corpus-rooted. If A0 doesn't reproduce them but the modern stack still has issues, the failures may be arch-rooted.

## What "single-variable" means here

A1-A4 each toggle ONE component vs the baseline (A0), with everything else held at A0 settings. This is single-variable in the strict statistical sense.

A5 toggles all four arch components together. This isn't single-variable but it IS necessary to test compositionality — do the components add up linearly, or do they interact non-trivially? If A5 - A0 ≈ (A1-A0) + (A2-A0) + (A3-A0) + (A4-A0), they're additive. If A5 is much better, there's positive interaction. If much worse, there's negative interaction.

A6 toggles only the optimizer vs A5. Single-variable comparison, A6 - A5 = isolated Muon contribution.

## Limitations stated upfront

1. **Single-seed.** All runs use seed=1337. Δval less than ~0.05 is below noise floor and shouldn't be over-interpreted. Future work: 3-seed runs of the most interesting comparisons.
2. **Single corpus.** Results may not transfer to other corpora. Apollo v2's lit/wiki/code mix is specific.
3. **Single scale.** 25M params. Component contributions can scale-shift; some matter more at 1B+.
4. **Wall-clock comparison, not iteration comparison.** Muon vs AdamW is compared at fixed 8h wall, not fixed iter count. This is the practically relevant comparison but conflates "Muon converges faster per iter" with "Muon is faster per iter."
5. **Param-count drift.** SwiGLU vs GELU param counts differ by ~2% under our hidden-dim choice. Documented above.

## Results table

(Filled in by `ablation_runner.py --collect-only` after runs complete.)

| Run | Best val | Steps | Wall h | Δ vs A0 |
|-----|----------|-------|--------|---------|
| A0 baseline | _pending_ | _pending_ | _pending_ | 0 |
| A1 rope | _pending_ | _pending_ | _pending_ | _pending_ |
| A2 rmsnorm | _pending_ | _pending_ | _pending_ | _pending_ |
| A3 qknorm | _pending_ | _pending_ | _pending_ | _pending_ |
| A4 swiglu | _pending_ | _pending_ | _pending_ | _pending_ |
| A5 modern | _pending_ | _pending_ | _pending_ | _pending_ |
| A6 modern_muon | _pending_ | _pending_ | _pending_ | _pending_ |

## After Phase 1

Phase 2 = register-token mechanistic probe. Uses the A6 (or best-performing) checkpoint. No additional training runs.
