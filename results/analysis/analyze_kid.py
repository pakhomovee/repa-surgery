#!/usr/bin/env python3
"""Analyze a folder of KID sweeps and plot.

Reads every kid_*.csv in the given folder (one run each, ideally named
kid_<mode>_<N>.csv, e.g. kid_repa_25k.csv), reports KID x10^3 with standard
errors (SE = std/sqrt(subsets)), tests plateau differences vs baseline / REPA,
and writes figures.

Usage:
    python results/analysis/analyze_kid.py results/celeba
    python results/analysis/analyze_kid.py results/imagenet100 --plateau 5 --output /tmp
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCALE = 1e3  # report KID x10^3

# Known modes: prettified label + color + sort order. Unknown modes fall back to
# the raw token and the default color cycle, sorted after the known ones.
PRETTY = {"baseline": "baseline", "repa": "REPA", "haste": "HASTE",
          "sigma": "REPA-σ", "repa-sigma": "REPA-σ"}
COLOR = {"baseline": "#888888", "repa": "#1f77b4", "haste": "#ff7f0e",
         "sigma": "#2ca02c", "repa-sigma": "#2ca02c"}
ORDER = ["baseline", "repa", "haste", "sigma", "repa-sigma"]
_CYCLE = ["#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22"]


def parse_name(path: Path):
    """kid_<mode>_<N>.csv -> (mode_key, label). N (e.g. '25k') is optional."""
    stem = path.stem
    stem = stem[4:] if stem.startswith("kid_") else stem
    parts = stem.split("_")
    count = None
    if len(parts) > 1 and re.fullmatch(r"\d+k?", parts[-1]):
        count = parts.pop()
    mode = "_".join(parts)
    key = mode if mode in PRETTY else mode.replace("_", "-")
    label = PRETTY.get(key, mode)
    if count:
        label = f"{label} ({count})"
    return key, label


def load(path: Path, num_subsets: int):
    steps, mean, se = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            mean.append(float(row["kid_mean"]) * SCALE)
            if row.get("kid_se"):                 # new evaluate.py CSV
                se.append(float(row["kid_se"]) * SCALE)
            else:                                  # old CSV: derive from std
                se.append(float(row["kid_std"]) / (num_subsets ** 0.5) * SCALE)
    order = sorted(range(len(steps)), key=lambda i: steps[i])
    return [steps[i] for i in order], [mean[i] for i in order], [se[i] for i in order]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="Folder containing kid_*.csv files.")
    ap.add_argument("--plateau", type=int, default=5, help="Avg last N checkpoints.")
    ap.add_argument("--subsets", type=int, default=100,
                    help="KID subsets used (for SE from old CSVs without kid_se).")
    ap.add_argument("--zoom-from", type=int, default=100000,
                    help="Plateau plot starts at this step.")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output dir for PNGs (default: the input folder).")
    ap.add_argument("--title", default=None, help="Title prefix (default: folder name).")
    args = ap.parse_args()

    files = sorted(args.folder.glob("kid_*.csv"))
    if not files:
        raise SystemExit(f"No kid_*.csv found in {args.folder}")
    out = args.output or args.folder
    out.mkdir(parents=True, exist_ok=True)
    title = args.title or args.folder.name

    runs = []  # (key, label, color, (steps, mean, se))
    cyc = iter(_CYCLE)
    for fp in files:
        key, label = parse_name(fp)
        color = COLOR.get(key) or next(cyc, "#333333")
        runs.append((key, label, color, load(fp, args.subsets)))
    runs.sort(key=lambda r: (ORDER.index(r[0]) if r[0] in ORDER else len(ORDER), r[1]))

    # ---- plateau stats ---------------------------------------------------- #
    plateau = {}
    print(f"=== {title}: plateau KID x10^3 (mean of last {args.plateau} ckpts) ===")
    for key, label, _, (_, m, _) in runs:
        tail = m[-args.plateau:]
        mu = statistics.mean(tail)
        se = statistics.pstdev(tail) / (len(tail) ** 0.5) if len(tail) > 1 else 0.0
        plateau[key] = (label, mu, se)
        print(f"  {label:16s}: {mu:6.3f} ± {se:.3f}")

    refs = [k for k in ("baseline", "repa") if k in plateau]
    if refs:
        print("\n=== gaps (lower = better;  *significant = |Δ| > 2·combined SE) ===")
        for key, label, _, _ in runs:
            _, mu, se = plateau[key]
            cells = []
            for rk in refs:
                rlabel, rmu, rse = plateau[rk]
                d = rmu - mu
                c = (se ** 2 + rse ** 2) ** 0.5
                sig = "*" if c and abs(d) > 2 * c else " "
                cells.append(f"vs {PRETTY.get(rk, rk):8s} {d:+.3f}{sig}({d/c:+.1f}σ)"
                             if c else f"vs {rk} n/a")
            print(f"  {label:16s} " + "   ".join(cells))

    # ---- Fig 1: full curve, log-y ----------------------------------------- #
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for _, label, color, (s, m, se) in runs:
        ax.plot(s, m, marker="o", ms=4, color=color, label=label)
        ax.fill_between(s, [a - b for a, b in zip(m, se)],
                        [a + b for a, b in zip(m, se)], color=color, alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("training step"); ax.set_ylabel(r"KID $\times 10^3$  (log)")
    ax.set_title(f"{title} — KID vs training step")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out / "kid_curve_full.png", dpi=140); plt.close(fig)

    # ---- Fig 2: plateau zoom, linear -------------------------------------- #
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for _, label, color, (s, m, se) in runs:
        idx = [i for i, st in enumerate(s) if st >= args.zoom_from]
        if not idx:
            continue
        ax.errorbar([s[i] for i in idx], [m[i] for i in idx],
                    yerr=[se[i] for i in idx], marker="o", ms=4, capsize=2,
                    color=color, label=label)
    ax.set_xlabel("training step"); ax.set_ylabel(r"KID $\times 10^3$")
    ax.set_title(f"{title} — KID plateau (step ≥ {args.zoom_from:,})")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out / "kid_plateau.png", dpi=140); plt.close(fig)

    # ---- Fig 3: plateau bar chart ----------------------------------------- #
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    labels = [plateau[k][0] for k, _, _, _ in runs]
    mus = [plateau[k][1] for k, _, _, _ in runs]
    ses = [plateau[k][2] for k, _, _, _ in runs]
    cols = [c for _, _, c, _ in runs]
    ax.bar(labels, mus, yerr=ses, capsize=4, color=cols)
    for i, (mu, se) in enumerate(zip(mus, ses)):
        ax.text(i, mu + se, f"{mu:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(r"plateau KID $\times 10^3$  (lower = better)")
    if mus:
        ax.set_ylim(min(mus) - 0.25, max(mus) + 0.3)
    ax.set_title(f"{title} — final KID (mean of last {args.plateau} ckpts)")
    ax.grid(True, axis="y", alpha=0.3); plt.xticks(rotation=15, ha="right")
    fig.tight_layout(); fig.savefig(out / "kid_plateau_bar.png", dpi=140); plt.close(fig)

    print(f"\nWrote kid_curve_full.png, kid_plateau.png, kid_plateau_bar.png -> {out}")


if __name__ == "__main__":
    main()
