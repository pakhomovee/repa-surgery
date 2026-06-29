#!/usr/bin/env python3
"""Render the gradient-conflict diagnostics from a run's rho.csv.

rho.csv (written by train.py --log-grad-conflict) has one row per (step, noise
bin): cos(g_diff, g_repa) plus gradient norms. This produces:
  - rho_heatmap.png : cos over (noise level t, training step) -- red = cooperate,
                      blue = conflict. The central mechanism figure.
  - rho_slices.png  : cos vs t at the first / middle / last logged step.
  - prints t*(step): the noise level where cos crosses zero (conflict boundary).

Usage:
    python results/analysis/analyze_rho.py runs/celeba_sit-b_2_repa
    python results/analysis/analyze_rho.py path/to/rho.csv --output /tmp
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(csvp: Path):
    rows = []
    with open(csvp) as f:
        for r in csv.DictReader(f):
            rows.append((int(r["step"]), float(r["t_center"]), float(r["rho"])))
    steps = sorted({r[0] for r in rows})
    centers = sorted({round(r[1], 6) for r in rows})
    M = np.full((len(centers), len(steps)), np.nan)
    si = {s: j for j, s in enumerate(steps)}
    ci = {c: i for i, c in enumerate(centers)}
    for s, c, rho in rows:
        M[ci[round(c, 6)], si[s]] = rho
    return steps, centers, M


def zero_crossing(centers, col):
    """Lowest t where cos crosses from - to + (the conflict boundary t*)."""
    for i in range(len(centers) - 1):
        a, b = col[i], col[i + 1]
        if np.isfinite(a) and np.isfinite(b) and a < 0 <= b:
            return centers[i] + (centers[i + 1] - centers[i]) * (-a) / (b - a + 1e-12)
    return np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="Run dir (uses <run>/rho.csv) or a rho.csv path.")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    csvp = args.run / "rho.csv" if args.run.is_dir() else args.run
    if not csvp.exists():
        raise SystemExit(f"No rho.csv at {csvp}")
    out = args.output or csvp.parent
    out.mkdir(parents=True, exist_ok=True)
    name = csvp.parent.name

    steps, centers, M = load(csvp)

    # ---- heatmap ---------------------------------------------------------- #
    vmax = float(np.nanmax(np.abs(M))) or 1.0
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=[steps[0], steps[-1], centers[0], centers[-1]])
    # zero-crossing curve t*(step)
    tstar = [zero_crossing(centers, M[:, j]) for j in range(len(steps))]
    ax.plot(steps, tstar, "k--", lw=1.4, label="t* (cos=0)")
    ax.set_xlabel("training step"); ax.set_ylabel("noise level  t")
    ax.set_title(f"gradient conflict  cos(g_diff, g_repa) — {name}")
    cb = fig.colorbar(im, ax=ax); cb.set_label("cosine  (red>0 cooperate · blue<0 conflict)")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(out / "rho_heatmap.png", dpi=140); plt.close(fig)

    # ---- slices ----------------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j, lab in [(0, "first"), (len(steps) // 2, "mid"), (len(steps) - 1, "last")]:
        ax.plot(centers, M[:, j], marker="o", ms=3, label=f"{lab} (step {steps[j]:,})")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("noise level  t"); ax.set_ylabel("cos(g_diff, g_repa)")
    ax.set_title(f"conflict vs noise level — {name}")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out / "rho_slices.png", dpi=140); plt.close(fig)

    finite = [t for t in tstar if np.isfinite(t)]
    if finite:
        print(f"t* (conflict boundary): first={finite[0]:.3f}  last={finite[-1]:.3f}  "
              f"(cos<0 below t*, cos>0 above)")
    print(f"Wrote {out}/rho_heatmap.png and rho_slices.png")


if __name__ == "__main__":
    main()
