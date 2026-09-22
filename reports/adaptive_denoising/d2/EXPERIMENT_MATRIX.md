# Seed-42 experiment matrix

| Arm | Objective | First execution |
|---|---|---|
| D20 | reconstruction only | CPU smoke, then seed-42 pilot |
| D21 | D20 + vessel ROI | only if gate diagnostics request it |
| D22 | D21 + vessel boundary | seed-42 pilot |
| D23 | D22 + CNR | only if gate diagnostics request it |
| D24 | D23 + frozen segmentation teacher | seed-42 pilot |
| D25 | D24 + identity/residual/leakage | CPU smoke, CUDA overfit, seed-42 pilot |

All arms use the same evidence-bound D1-best initialization, train/validation
pool, seed, sampling and augmentation plan, optimizer budget, P0 threshold 0.5,
and sealed test policy. D24/D25 require a separately identified legal teacher.

