#!/usr/bin/env python3
"""Overlay KID-vs-step curves from multiple runs to compare training modes.

Reads each run's eval/kid.csv (written by evaluate.py), plots them together with
standard-error bands, and prints each run's plateau KID (mean over the last
--plateau checkpoints) plus pairwise gaps, so modes can be compared directly.

The error bar is the standard error (std/sqrt(subsets)), NOT the raw subset std.

Example:
    python training/compare_kid.py \
        ../runs/imagenet100_sit-b_2_baseline \
        ../runs/imagenet100_sit-b_2_repa \
        ../runs/imagenet100_sit-b_2_haste \
        ../runs/imagenet100_sit-b_2_repa-PCGrad
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def resolve(p: str):
    path = Path(p)
    if path.is_dir():
        return path / "eval" / "kid.csv", path.name
    return path, path.resolve().parent.parent.name  # .../<run>/eval/kid.csv


def load_csv(path: Path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            mean = float(row["kid_mean"])
            if row.get("kid_se"):
                se = float(row["kid_se"])
            else:  # old CSV without SE: derive assuming 100 subsets
                se = float(row["kid_std"]) / 10.0
            rows.append((int(row["step"]), mean, se))
    rows.sort()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="Run dirs (or kid.csv paths) to compare.")
    ap.add_argument("--plateau", type=int, default=5,
                    help="Average the last N checkpoints for the plateau number.")
    ap.add_argument("--output", type=Path, default=Path("kid_compare.png"))
    args = ap.parse_args()

    series = []
    plateaus = []
    print(f"=== plateau KID x10^3 (mean of last {args.plateau} checkpoints) ===")
    for r in args.runs:
        csvp, label = resolve(r)
        if not csvp.exists():
            print(f"  {label:40s}: MISSING {csvp}")
            continue
        rows = load_csv(csvp)
        series.append((label, rows))
        tail = rows[-args.plateau:]
        vals = [x[1] * 1e3 for x in tail]
        m = statistics.mean(vals)
        # SE of the plateau average from the spread of those checkpoints.
        se = (statistics.pstdev(vals) / len(vals) ** 0.5) if len(vals) > 1 else tail[0][2] * 1e3
        plateaus.append((label, m, se))
        print(f"  {label:40s}: {m:7.3f} ± {se:.3f}")

    if len(plateaus) > 1:
        print("\n=== pairwise gaps (row - col), x10^3; |gap| >> combined SE => real ===")
        labels = [p[0] for p in plateaus]
        w = max(len(l) for l in labels)
        print(" " * (w + 2) + "  ".join(f"{l[:10]:>10s}" for l in labels))
        for li, mi, si in plateaus:
            cells = []
            for _lj, mj, sj in plateaus:
                gap = mi - mj
                comb = (si ** 2 + sj ** 2) ** 0.5
                sig = "" if abs(gap) <= 2 * comb or comb == 0 else "*"
                cells.append(f"{gap:+9.3f}{sig}")
            print(f"{li:>{w}}  " + "  ".join(cells))
        print("  (* = |gap| > 2x combined SE)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        for label, rows in series:
            steps = [x[0] for x in rows]
            means = [x[1] * 1e3 for x in rows]
            ses = [x[2] * 1e3 for x in rows]
            plt.errorbar(steps, means, yerr=ses, marker="o", capsize=3, label=label)
        plt.xlabel("training step")
        plt.ylabel("KID x10^3")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.output, dpi=120)
        print(f"\nWrote {args.output}")
    except Exception as e:
        print(f"(skipped plot: {e})")


if __name__ == "__main__":
    main()
