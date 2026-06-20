#!/usr/bin/env python3
"""Analyze a folder of KID/FID sweeps and plot.

Reads every kid_*.csv in the given folder (one run each, ideally named
kid_<mode>_<N>.csv, e.g. kid_repa_25k.csv), reports KID x10^3 (± SE) and, if the
CSV has a `fid` column, FID too; tests plateau differences vs baseline / REPA and
writes figures for each metric. Also runs the REPA-style "a fixed quality level is
reached faster" analysis: iterations to hit target KID/FID levels and the speedup
vs the reference run (baseline, or REPA when baseline lacks the metric).

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

KID_SCALE = 1e3  # report KID x10^3

PRETTY = {"baseline": "baseline", "repa": "REPA", "haste": "HASTE",
          "pcgrad": "REPA-PCGrad", "repa-PCGrad": "REPA-PCGrad"}
COLOR = {"baseline": "#888888", "repa": "#1f77b4", "haste": "#ff7f0e",
         "pcgrad": "#2ca02c", "repa-PCGrad": "#2ca02c"}
ORDER = ["baseline", "repa", "haste", "pcgrad", "repa-PCGrad"]
_CYCLE = ["#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22"]


def parse_name(path: Path):
    """kid_<mode>_<N>.csv -> (mode_key, label). N (e.g. '25k') is optional."""
    stem = path.stem
    stem = stem[4:] if stem.startswith("kid_") else stem
    parts = stem.split("_")
    count = parts.pop() if len(parts) > 1 and re.fullmatch(r"\d+k?", parts[-1]) else None
    mode = "_".join(parts)
    key = mode if mode in PRETTY else mode.replace("_", "-")
    label = PRETTY.get(key, mode)
    return key, (f"{label} ({count})" if count else label)


def load(path: Path, num_subsets: int):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            kse = (float(row["kid_se"]) if row.get("kid_se")
                   else float(row["kid_std"]) / (num_subsets ** 0.5))
            fid = float(row["fid"]) if row.get("fid") not in (None, "") else None
            rows.append((int(row["step"]), float(row["kid_mean"]) * KID_SCALE,
                         kse * KID_SCALE, fid))
    rows.sort()
    steps = [r[0] for r in rows]
    kid = [r[1] for r in rows]
    kse = [r[2] for r in rows]
    fid = [r[3] for r in rows] if all(r[3] is not None for r in rows) else None
    return {"steps": steps, "kid": kid, "kse": kse, "fid": fid}


def plateau(values, n):
    tail = values[-n:]
    mu = statistics.mean(tail)
    se = statistics.pstdev(tail) / (len(tail) ** 0.5) if len(tail) > 1 else 0.0
    return mu, se


def crossing_step(steps, values, thresh):
    """First step at which `values` first reaches <= `thresh` (linear interp in step).

    Returns None if the run never gets that low. This is the REPA "iterations to
    reach a fixed quality" quantity; the first crossing is used (later dips back
    above the threshold are ignored, matching how speedup is reported in REPA)."""
    if values[0] <= thresh:
        return float(steps[0])
    for i in range(1, len(values)):
        if values[i] <= thresh:
            v0, v1, s0, s1 = values[i - 1], values[i], steps[i - 1], steps[i]
            if v0 == v1:
                return float(s1)
            frac = (v0 - thresh) / (v0 - v1)
            return s0 + frac * (s1 - s0)
    return None


def report_and_plot(runs, vkey, sekey, name, ylabel, title, out, args):
    """One metric: print plateau table + gaps, write curve/plateau/bar figures."""
    out.mkdir(parents=True, exist_ok=True)
    plats = {}
    print(f"\n=== {title}: plateau {name} (mean of last {args.plateau} ckpts) ===")
    for r in runs:
        mu, se = plateau(r[vkey], args.plateau)
        plats[r["key"]] = (r["label"], mu, se)
        print(f"  {r['label']:16s}: {mu:7.3f} ± {se:.3f}")

    refs = [k for k in ("baseline", "repa") if k in plats]
    if refs:
        print(f"  -- gaps (lower = better; * = |Δ| > 2·SE) --")
        for r in runs:
            _, mu, se = plats[r["key"]]
            cells = []
            for rk in refs:
                _, rmu, rse = plats[rk]
                c = (se ** 2 + rse ** 2) ** 0.5
                d = rmu - mu
                cells.append(f"vs {PRETTY.get(rk, rk):8s} {d:+7.3f}{'*' if c and abs(d) > 2*c else ' '}"
                             + (f"({d/c:+.1f}σ)" if c else ""))
            print(f"    {r['label']:16s} " + "  ".join(cells))

    # curve (log-y), plateau zoom (linear), bar
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for r in runs:
        ax.plot(r["steps"], r[vkey], marker="o", ms=4, color=r["color"], label=r["label"])
        if sekey:
            ax.fill_between(r["steps"], [a - b for a, b in zip(r[vkey], r[sekey])],
                            [a + b for a, b in zip(r[vkey], r[sekey])], color=r["color"], alpha=0.2)
    ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_ylabel(f"{ylabel} (log)")
    ax.set_title(f"{title} — {name} vs step"); ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out / "curve_full.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for r in runs:
        idx = [i for i, st in enumerate(r["steps"]) if st >= args.zoom_from]
        if not idx:
            continue
        ax.errorbar([r["steps"][i] for i in idx], [r[vkey][i] for i in idx],
                    yerr=([r[sekey][i] for i in idx] if sekey else None),
                    marker="o", ms=4, capsize=2, color=r["color"], label=r["label"])
    ax.set_xlabel("training step"); ax.set_ylabel(ylabel)
    ax.set_title(f"{title} — {name} plateau (step ≥ {args.zoom_from:,})")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out / "plateau.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    labels = [plats[r["key"]][0] for r in runs]
    mus = [plats[r["key"]][1] for r in runs]
    ses = [plats[r["key"]][2] for r in runs]
    ax.bar(labels, mus, yerr=ses, capsize=4, color=[r["color"] for r in runs])
    for i, (mu, se) in enumerate(zip(mus, ses)):
        ax.text(i, mu + se, f"{mu:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(f"plateau {ylabel}  (lower = better)")
    if mus:
        pad = (max(mus) - min(mus)) * 0.5 + max(ses) + 1e-6
        ax.set_ylim(min(mus) - pad, max(mus) + pad)
    ax.set_title(f"{title} — final {name} (last {args.plateau} ckpts)")
    ax.grid(True, axis="y", alpha=0.3); plt.xticks(rotation=15, ha="right")
    fig.tight_layout(); fig.savefig(out / "plateau_bar.png", dpi=140); plt.close(fig)


def speedup(runs, vkey, name, ylabel, title, out, args):
    """REPA-style "fixed quality reached faster": iterations to hit a target metric.

    Picks a reference run (baseline if present, else REPA), turns its plateau into
    a ladder of targets (factor x plateau), and reports the first step each run
    reaches each target plus the speedup = ref_steps / run_steps vs the reference.
    Writes a steps-to-target curve and a headline speedup bar."""
    out.mkdir(parents=True, exist_ok=True)
    keys = [r["key"] for r in runs]
    ref_key = next((k for k in ("baseline", "repa") if k in keys), runs[0]["key"])
    ref = next(r for r in runs if r["key"] == ref_key)
    p_ref, _ = plateau(ref[vkey], args.plateau)
    # Targets the reference actually reaches (so its speedup is the 1.0 anchor).
    levels = sorted({round(p_ref * f, 4) for f in args.speedup_factors}, reverse=True)
    ref_steps = {lv: crossing_step(ref["steps"], ref[vkey], lv) for lv in levels}
    levels = [lv for lv in levels if ref_steps[lv] is not None]
    if not levels:
        print(f"\n=== {title}: {name} speedup — reference {ref['label']} reaches no target, skipped ===")
        return

    print(f"\n=== {title}: steps to reach a fixed {name} "
          f"(targets = factor x {ref['label']} plateau {p_ref:.3f}) ===")
    header = "  " + " " * 16 + "".join(f"{('<=' + format(lv, '.2f')):>13s}" for lv in levels)
    print(header)
    steps_by_run = {}  # key -> {level: step}
    for r in runs:
        cells, sr = [], {}
        for lv in levels:
            s = crossing_step(r["steps"], r[vkey], lv)
            sr[lv] = s
            if s is None:
                cells.append(f"{'—':>13s}")
            else:
                rs = ref_steps[lv]
                spd = f"({rs / s:.2f}x)" if (rs and s) else ""
                cells.append(f"{s / 1000:6.0f}k{spd:>6s}")
        steps_by_run[r["key"]] = sr
        print(f"  {r['label']:16s}" + "".join(cells))
    print(f"  (speedup vs {ref['label']} in parentheses; >1 = reaches that {name} in fewer steps)")

    # Steps-to-target curve: harder targets to the right (x inverted).
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for r in runs:
        xs = [lv for lv in levels if steps_by_run[r["key"]][lv] is not None]
        ys = [steps_by_run[r["key"]][lv] for lv in xs]
        if xs:
            ax.plot(xs, ys, marker="o", ms=5, color=r["color"], label=r["label"])
    ax.invert_xaxis()
    ax.set_xlabel(f"target {ylabel}  (harder ->)"); ax.set_ylabel("training steps to reach")
    ax.set_title(f"{title} — steps to reach a fixed {name}")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out / "speedup_curve.png", dpi=140); plt.close(fig)

    # Headline speedup bar at the hardest target the reference reaches.
    lv = levels[-1]
    rs = ref_steps[lv]
    labels, spds, colors = [], [], []
    for r in runs:
        s = steps_by_run[r["key"]][lv]
        labels.append(r["label"]); colors.append(r["color"])
        spds.append(rs / s if (s and rs) else 0.0)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(labels, spds, color=colors)
    ax.axhline(1.0, color="#888888", ls="--", lw=1)
    for i, s in enumerate(spds):
        ax.text(i, s, f"{s:.2f}x" if s else "—", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(f"speedup vs {ref['label']}  (higher = faster)")
    ax.set_title(f"{title} — steps to reach {name} <= {lv:.2f}")
    ax.grid(True, axis="y", alpha=0.3); plt.xticks(rotation=15, ha="right")
    fig.tight_layout(); fig.savefig(out / "speedup_bar.png", dpi=140); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="Folder containing kid_*.csv files.")
    ap.add_argument("--plateau", type=int, default=5, help="Avg last N checkpoints.")
    ap.add_argument("--subsets", type=int, default=100,
                    help="KID subsets (for SE from old CSVs without kid_se).")
    ap.add_argument("--zoom-from", type=int, default=100000, help="Plateau plot start step.")
    ap.add_argument("--speedup-factors", type=float, nargs="+", default=[2.0, 1.5, 1.25, 1.1],
                    help="Targets for the steps-to-fixed-quality analysis, as multiples "
                         "of the reference run's plateau (REPA-style speedup).")
    ap.add_argument("--output", type=Path, default=None, help="PNG output dir (default: folder).")
    ap.add_argument("--title", default=None, help="Title prefix (default: folder name).")
    args = ap.parse_args()

    files = sorted(args.folder.glob("kid_*.csv"))
    if not files:
        raise SystemExit(f"No kid_*.csv found in {args.folder}")
    out = args.output or args.folder
    out.mkdir(parents=True, exist_ok=True)
    title = args.title or args.folder.name

    runs, cyc = [], iter(_CYCLE)
    for fp in files:
        key, label = parse_name(fp)
        d = load(fp, args.subsets)
        d.update(key=key, label=label, color=COLOR.get(key) or next(cyc, "#333333"))
        runs.append(d)
    runs.sort(key=lambda r: (ORDER.index(r["key"]) if r["key"] in ORDER else len(ORDER), r["label"]))

    # KID uses every run; FID uses only the runs whose CSV has a `fid` column.
    report_and_plot(runs, "kid", "kse", "KID x10^3", r"KID $\times 10^3$", title, out / "kid", args)
    speedup(runs, "kid", "KID x10^3", r"KID $\times 10^3$", title, out / "kid", args)

    fid_runs = [r for r in runs if r["fid"] is not None]
    if fid_runs:
        report_and_plot(fid_runs, "fid", None, "FID", "FID", title, out / "fid", args)
        speedup(fid_runs, "fid", "FID", "FID", title, out / "fid", args)
        skipped = [r["label"] for r in runs if r["fid"] is None]
        note = f"  (FID skipped for: {', '.join(skipped)})" if skipped else ""
        print(f"\nWrote {out}/kid/ and {out}/fid/ plots{note}")
    else:
        print(f"\nWrote {out}/kid/ plots  (no `fid` column in any CSV)")


if __name__ == "__main__":
    main()
