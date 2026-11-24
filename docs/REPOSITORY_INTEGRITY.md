# Repository integrity review — 24 November 2026

Reviewed GitHub `origin/main` at `13d1aa0` against Abel's original baseline commit `e7d2e3a` and Maheen's latest ablation commit `f665c56`.

## Findings

- Every tracked path from both reference commits is still present. No deleted tracked files were found in the fetched history.
- Abel's frozen inputs, preparation notebook, baseline guide and complete baseline results are unchanged from `e7d2e3a`. Baseline executable code is also unchanged; an AI-assistance comment was removed and has now been restored.
- Maheen's analysis code and saved results matched `f665c56` before this repair. Only her default input/output paths and the matching documentation were updated.
- Pad's cleaning work, Hann's feature preparation, Louis's neural scripts, Abel's report and the separate threshold experiment remain present.
- The README contained a leftover merge separator and lacked a guide to Maheen's contribution. These integration issues do not demonstrate that Maheen deleted anyone's work.

## Repairs

- Replaced the malformed README introduction and documented all team contributions and run order.
- Made the default ablation command read `data/frozen/` and write `results/ablation/`.
- Added `docs/ABLATION.md` with inputs, outputs, scale conventions and reproducibility commands.
- Corrected MLP/CNN cleaned-CSV paths without changing their models or evaluation procedures.

## Scope limits

This review covers files in the fetched Git history and current checkout, not uncommitted work on other people's machines. Fitted baseline `.joblib` files are intentionally ignored by Git and can be regenerated. The external `flood10k/` image dataset is not included; `CNN 2D.py` depends on it. TensorFlow is unavailable in the verification environment, so neural training has not been verified.
