# Apollo-Helios ablation results

| Run | Preset | Optimizer | Body params | Best val | Steps | Wall h |
|-----|--------|-----------|-------------|----------|-------|--------|
| A0_baseline | baseline | adamw | 25.18M | 5.4775 | 8000 | 8.0 |
| A1_rope | rope | adamw | 25.18M | 4.6775 | 8000 | 8.01 |
| A2_rmsnorm | rmsnorm | adamw | 25.17M | 5.3608 | 6000 | 6.01 |
| A3_qknorm | qknorm | adamw | 25.18M | 5.3919 | 8000 | 8.01 |
| A4_swiglu | swiglu | adamw | 24.92M | 5.4674 | 4750 | 4.75 |
| A5_modern | modern | adamw | 24.91M | 5.9502 | 3500 | 3.51 |
| A6_modern_muon | modern | muon | 24.91M | 5.1404 | 8000 | 8.01 |
