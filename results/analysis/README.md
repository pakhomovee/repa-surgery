# results/analysis

Folder-driven analysis of KID sweeps produced by `training/evaluate.py`.

```bash
python results/analysis/analyze_kid.py <results-folder>
# e.g.
python results/analysis/analyze_kid.py results/celeba
```

It reads every `kid_*.csv` in `<results-folder>` (one run each, named
`kid_<mode>_<N>.csv`, e.g. `kid_repa_25k.csv`), then for **KID** (and **FID**, if
the CSV has a `fid` column — newer `evaluate.py` writes both):

- prints a **plateau table** (mean of the last `--plateau` checkpoints; KID×10³
  with SE `std/√subsets`, FID with plateau-spread SE),
- prints **pairwise gaps** vs `baseline` and `repa` with significance flags
  (`*` = |Δ| > 2× combined SE),
- writes figures into the folder (or `--output`): `kid_curve_full.png`,
  `kid_plateau.png`, `kid_plateau_bar.png`, and the `fid_*.png` equivalents.

Known modes (`baseline`, `repa`, `haste`, `pcgrad`/`repa-PCGrad`) get fixed
colors/labels/order; anything else is picked up automatically. New-format CSVs
carry `kid_se`; older ones derive SE from `kid_std` via `--subsets` (default 100).

Options: `--plateau N`, `--zoom-from STEP`, `--subsets N`, `--output DIR`,
`--title STR`.
