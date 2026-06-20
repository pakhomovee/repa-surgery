# results/analysis

Folder-driven analysis of KID sweeps produced by `training/evaluate.py`.

```bash
python results/analysis/analyze_kid.py <results-folder>
# e.g.
python results/analysis/analyze_kid.py results/celeba
```

It reads every `kid_*.csv` in `<results-folder>` (one run each, named
`kid_<mode>_<N>.csv`, e.g. `kid_repa_25k.csv`), then:

- prints a **plateau table** (KID×10³, mean of the last `--plateau` checkpoints,
  with standard error `std/√subsets`),
- prints **pairwise gaps** vs `baseline` and `repa` with significance flags
  (`*` = |Δ| > 2× combined SE),
- writes three figures into the folder (or `--output`):
  `kid_curve_full.png` (log-y, full run), `kid_plateau.png` (zoomed, SE bars),
  `kid_plateau_bar.png` (final-KID bars).

Known modes (`baseline`, `repa`, `haste`, `sigma`/`repa-sigma`) get fixed
colors/labels/order; anything else is picked up automatically. New-format CSVs
carry `kid_se`; older ones derive SE from `kid_std` via `--subsets` (default 100).

Options: `--plateau N`, `--zoom-from STEP`, `--subsets N`, `--output DIR`,
`--title STR`.
