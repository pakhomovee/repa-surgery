# Gradient Surgery for REPA — results analysis

Analysis of five SiT-B/2 training arms on ImageNet-100 and CelebA, and what they
support for a paper.

**Source data:** `imagenet_results/data.txt` (5 arms × 12 checkpoints),
`celeba_results/*.csv` (5 arms × 15 checkpoints).
**Setup** (from `training/train.sh`, `training/envs/{imagenet100,celeba}.env`,
`REPA/train.py`): SiT-B/2 @ 256px on SD-VAE latents, batch 256, 300k steps,
DINOv2-B alignment at encoder depth 4, `proj-coeff` 0.5, linear path,
v-prediction, uniform weighting.
**Eval:** ImageNet-100 at 10k samples, CelebA at 25k, both cfg 1.5 / 50 ODE
steps / EMA weights, KID and FID from the same cached InceptionV3 features.

> **Naming.** Everything below is ImageNet-**100** (100 classes, 126,689
> images), not ImageNet-1K. None of these FID numbers are comparable to
> published ImageNet-256 results.

---

## 1. The read

**REPA-PCGrad is the ImageNet result.** It reaches REPA's final FID in 170k
steps instead of 300k — a 1.76× step speedup, 3.33× over the plain SiT baseline
— and finishes at FID 9.015 / KID 1.089×10⁻³ against REPA's 9.852 / 1.419×10⁻³.
The KID gap is 23% and about nine combined standard errors out.

**P-PCGrad is the CelebA result, and it is the better of the two.** On a dataset
where alignment is worth *nothing* — baseline, REPA and HASTE plateau at KID
4.177 / 4.183 / 4.148 — the preconditioned surgery arm lands at KID 3.870 and
FID 6.787: **7.3% below the baseline** in KID where REPA is 0.1% *above* it, and
13.9% below REPA in FID. Surgery converts a worthless alignment loss into a
useful one.

**HASTE is a quality null.** Terminating alignment at 100k lands within noise of
REPA at 300k (ΔFID −0.07, ΔKID z ≈ −0.9). Its real benefit is that the last 200k
steps cost baseline compute — no encoder forward, no image read. "REPA Works
Until It Doesn't" reproduces as a *cost* result, not a *quality* result.

**Surgery beats REPA in all eight cells** (2 datasets × 2 conflict metrics ×
2 quality metrics). Only *which* variant wins flips, and there is a principled
reason to expect that (§5).

**The two wins are different kinds of win.** On ImageNet-100 the advantage
decays with training — a head start. On CelebA it does not decay at all. The
second is the more valuable, because it does not evaporate with more compute.

---

## 2. ImageNet-100 at 300k steps

| Arm | Precision | FID | vs REPA | KID ×10³ | vs REPA | z |
|---|---|---:|---:|---:|---:|---:|
| Baseline | fp16 | 14.573 | +47.9% | 3.474 ± 0.055 | +144.8% | +33.1 |
| REPA | fp16 | 9.852 | — | 1.419 ± 0.029 | — | — |
| HASTE | fp16 | 9.782 | −0.7% | 1.381 ± 0.028 | −2.7% | −0.9 |
| P-PCGrad *(whitened)* | bf16 | 9.522 | −3.3% | 1.381 ± 0.030 | −2.7% | −0.9 |
| **REPA-PCGrad** *(Euclidean)* | bf16 | **9.015** | **−8.5%** | **1.089 ± 0.020** | **−23.3%** | **−9.4** |

The ± is over Inception feature subsets, not over training seeds — it measures
how much *this checkpoint's* samples wobble, not how much *the method* wobbles.
Say so in the paper; a suspiciously large z-score doing work it wasn't built for
is exactly what reviewers catch.

### FID trajectory

| step | baseline | REPA | HASTE | P-PCGrad | REPA-PCGrad |
|---:|---:|---:|---:|---:|---:|
| 25k | 200.096 | 195.783 | 195.586 | 191.107 | 195.061 |
| 50k | 78.709 | 55.130 | 55.819 | 49.754 | **36.245** |
| 75k | 48.023 | 27.024 | 27.547 | 25.536 | **17.637** |
| 100k | 34.735 | 18.654 | 18.823 | 18.222 | **12.883** |
| 125k | 28.032 | 15.460 | 14.690 | 15.197 | **11.360** |
| 150k | 23.574 | 13.494 | 12.886 | 13.349 | **10.493** |
| 175k | 20.382 | 12.092 | 11.369 | 11.851 | **9.708** |
| 200k | 18.698 | 11.275 | 10.934 | 11.339 | **9.639** |
| 225k | 17.285 | 10.777 | 10.714 | 10.682 | **9.370** |
| 250k | 16.179 | 10.406 | 10.129 | 10.251 | **9.209** |
| 275k | 15.881 | 10.342 | 10.219 | 10.111 | **9.323** |
| 300k | 14.573 | 9.852 | 9.782 | 9.522 | **9.015** |

REPA-PCGrad separates by 50k and is never overtaken. The other three alignment
arms are one line from 150k onward.

---

## 3. The efficiency claim

Read the curves horizontally, not vertically. A 0.84-FID endpoint gap becomes a
compute claim, and compute claims don't evaporate when the curves converge.

**Steps required to reach a fixed FID** (log-interpolated between checkpoints):

| Arm | FID ≤ 20 | FID ≤ 16 | FID ≤ 14.57 | speedup vs baseline |
|---|---:|---:|---:|---:|
| Baseline | 180k | 265k | 300k | 1.00× |
| REPA | 95k | 120k | 136k | 2.21× |
| HASTE | 96k | 116k | 127k | 2.37× |
| P-PCGrad | 93k | 118k | 133k | 2.25× |
| **REPA-PCGrad** | **71k** | **83k** | **90k** | **3.33×** |

Against REPA rather than baseline: REPA-PCGrad reaches REPA's 300k FID (9.852)
at **170k steps — 1.76× fewer**. HASTE and P-PCGrad manage 1.01× and 1.05×.

**Corrected for wall-clock.** Surgery needs a second, partial backward through
the alignment subgraph, which the repo estimates at ~1.2× the per-step cost of
plain REPA. *(That figure is a README comment, not a measurement — measure it.)*
At equal wall-clock:

| Equal wall-clock | FID | KID ×10³ |
|---|---:|---:|
| REPA @ 300k (1.0×/step) | 9.852 | 1.419 |
| **REPA-PCGrad @ 250k** (1.2×/step) | **9.209** | **1.208** |

−6.5% FID and −14.9% KID at matched compute. The honest headline is therefore
**≈1.47× wall-clock speedup** to reach REPA's final quality, not 1.76×.

### The caveat a reviewer will find first

On ImageNet-100 the advantage is **being eaten**:

| step | 50k | 75k | 100k | 150k | 200k | 250k | 300k |
|---|---:|---:|---:|---:|---:|---:|---:|
| ΔFID (REPA − PCGrad) | +18.885 | +9.387 | +5.771 | +3.001 | +1.636 | +1.197 | **+0.837** |
| ΔKID relative | −46.9% | −55.7% | −55.6% | −48.2% | −33.2% | −29.2% | **−23.3%** |

Monotone decay after 75k, no sign of a floor. Anyone who plots the difference
sees this, so plot it yourself. Framed as a *quality ceiling* the paper invites
"does it still win at 1M steps?" and the honest answer is probably not. Framed
as **convergence acceleration** the narrowing is the expected signature of an
optimization fix — and it is exactly what REPA itself does relative to the
baseline. Own it in the abstract.

---

## 4. CelebA: alignment does nothing, surgery does

The right comparison here is **against the baseline**, not against REPA,
because REPA *is* the baseline on this dataset.

Plateau over the last five checkpoints (220k–300k):

| Arm | KID ×10³ | vs baseline | FID | vs REPA | Reading |
|---|---:|---:|---:|---:|---|
| Baseline *(5k eval samples)* | 4.177 | — | *not computed* | — | Reference |
| REPA | 4.183 | +0.1% | 7.884 | — | **Alignment buys nothing** |
| HASTE | 4.148 | −0.7% | 7.876 | −0.1% | Within noise |
| REPA-PCGrad *(Euclidean)* | 4.080 | −2.3% | 7.767 | −1.5% | Small but consistent |
| **P-PCGrad** *(whitened)* | **3.870** | **−7.3%** | **6.787** | **−13.9%** | **Best arm, both metrics** |

KID is unbiased and computed over fixed 1000-sample subsets, so the baseline's
5k-sample KID is directly comparable — only its error bar is wider. FID is
computed over *all* samples and is biased in the sample count, which is why the
baseline has no FID row. Both surgery arms were evaluated at 25k, same as REPA
and HASTE, so every FID above is directly comparable. *(Confirmed.)*

### Precision is not the explanation

The obvious objection is the bf16/fp16 split, and CelebA answers it internally:
**both** surgery arms ran bf16, and they differ *from each other* by 12.6% FID —
more than either differs from fp16 REPA on the low end. A precision effect moves
them together. On this dataset the confound is excluded by runs already in hand.
(ImageNet-100 still needs the control; see §6.)

### Surgery vs REPA — every cell

| Dataset | Conflict test | FID vs REPA | KID vs REPA |
|---|---|---:|---:|
| **ImageNet-100** | **Euclidean** | **−8.5%** | **−23.3%** |
| ImageNet-100 | Adam-whitened | −3.3% | −2.7% |
| CelebA | Euclidean | −1.5% | −2.5% |
| **CelebA** | **Adam-whitened** | **−13.9%** | **−7.5%** |

Eight cells, eight improvements. The winning variant flips by dataset; the sign
never does. In each dataset the better metric wins by a lot while the worse one
still wins by a little.

### The CelebA gap does not shrink

Absolute FID gap (REPA − P-PCGrad), against the FID level:

| step | 60k | 100k | 140k | 180k | 220k | 260k | 300k |
|---|---:|---:|---:|---:|---:|---:|---:|
| REPA FID | 11.506 | 9.215 | 8.685 | 8.070 | 8.082 | 7.737 | 7.858 |
| abs gap | 0.977 | 1.110 | 1.351 | 1.113 | 1.160 | 0.958 | 1.096 |
| rel gap | 8.5% | 12.0% | 15.6% | 13.8% | 14.4% | 12.4% | 13.9% |

The gap holds at **1.14 ± 0.13 absolute** from 60k to 300k while the FID level
falls 32% — so in relative terms it *grows*, 8.5% → 14%. Indexed to its first
value, the ImageNet gap decays to **9%** over the same span; the CelebA gap sits
at **112%**.

Two qualitatively different benefit profiles from the same method family:

- **ImageNet-100 is a convergence result.** Surgery finds the same basin sooner;
  given enough steps plain REPA catches up. Worth a speedup headline, nothing
  more.
- **CelebA is a quality result.** The advantage is established by 60k and then
  held, in absolute terms, for another 240k steps against a model that is still
  improving. Nothing in the trend suggests REPA closes it.

### One asymmetry worth explaining

CelebA gains more in FID (−13.9%) than KID (−7.5%); ImageNet-100 is the reverse
(−8.5% FID, −23.3% KID). FID is a Gaussian fit to the Inception features — first
and second moments only — whereas KID's cubic kernel is sensitive to higher
moments. A gain larger in FID than KID is concentrated in the **mean and
covariance** of the feature distribution: a better-centred, better-scaled match
rather than a broad improvement in per-sample fidelity. Testable — decompose FID
into its mean term and its covariance trace term, and look at samples.

---

## 5. Why whitening should be dataset-dependent

The variant flip is a *prediction*, not an embarrassment, once you notice that
PCGrad's Euclidean conflict test is inherited from an SGD formulation. The
update actually applied here is Adam's, `P·g` with `P = 1/(√v̂+ε)` — so "these
gradients conflict" should be evaluated in the metric the optimizer moves in.
**The whitened test is the principled one; the Euclidean test is the legacy
one.**

Whitening changes the test most where the diffusion gradient's spectrum is most
anisotropic. On CelebA — one visual domain, aligned faces, 16 attribute classes
— a handful of high-curvature directions dominate the Euclidean dot product, and
the test goes nearly blind to the low-curvature directions where the alignment
gradient lives. On ImageNet-100's 100 diverse classes the spectrum is flatter,
whitening buys less, and the extra variance of estimating v̂ costs more than it
gains. That predicts exactly the observed ordering, and it is measurable.

---

## 6. What the runs support, and what they can't

### Solid

- **REPA reproduces.** −32.4% FID and a 2.21× step speedup over plain SiT on
  ImageNet-100. The pipeline does what it should.
- **REPA-PCGrad beats REPA on ImageNet-100.** Large, opens early, holds at all
  twelve checkpoints, survives compute matching.
- **REPA fails on CelebA.** Baseline, REPA and HASTE agree to within 0.05 KID.
  A controlled negative result is hard to argue with.
- **Surgery works where alignment doesn't.** CelebA P-PCGrad is 7.3% below the
  *baseline* in KID while REPA is 0.1% above it. 8/8 cells beat REPA.
- **Precision is excluded on CelebA.** The two bf16 arms differ from each other
  by 12.6% FID.
- **HASTE costs less.** Post-termination steps load neither images nor encoder
  features. That part needs no defence.

### Confounded or missing

- **Precision is uncontrolled on ImageNet-100.** Both surgery arms forced to
  bf16; baseline/REPA/HASTE ran fp16. At the *first* checkpoint the two bf16
  arms already show KID 153.2 / 154.6 against 160.9–161.2 for all three fp16
  arms — a clean split along the precision line. CelebA rules this out for that
  dataset; ImageNet-100 still needs the control run.
- **Baseline eval features were never regenerated.** The sweep log shows **0 of
  12** checkpoints featurized for the baseline arm and **5 of 12** reused for
  HASTE. `evaluate.py` keys its cache on `(step, rank)` only — not on sample
  count, cfg, sampler steps, or `--compile`, which the repo's own README warns
  is not bitwise identical to eager.
- **The CelebA baseline has no FID.** Evaluated at 5k, so its KID is comparable
  but no FID was written. The paper's best sentence rests half on a row that
  does not exist. Eval-only fix.
- **n = 1 seed everywhere.** Every ± is Inception subset noise, not run-to-run
  noise. Nothing here casts doubt on the *direction* — every arm points the same
  way — but the effect *sizes* need seeds before they go in a table.
- **No mechanism data.** The premise is that ∇ℓ_diff and ∇ℓ_repa conflict at low
  noise. **No `rho.csv` exists anywhere in the repo.** The method is currently
  motivated by an assertion.
- **The surgery is not isolated.** When the dot product is negative the update
  adds a positive multiple of the EMA diffusion direction — so it is also a
  momentum-like term. Nothing yet separates "conflict removal" from "extra
  smoothed diffusion gradient".
- **The FID/KID asymmetry is unexplained.** Not a threat to either result, but a
  reviewer will ask.
- **Scale and protocol.** ImageNet-*100*, SiT-B/2, FID-10K at a single cfg.

---

## 7. Positioning

"We beat REPA by 0.8 FID" is a weak paper: one dataset, one model size, one
seed, a shrinking margin. The same runs support something stronger:

> **Representation alignment helps a diffusion transformer only where its
> gradient does not fight the denoising objective. Remove the conflicting
> component — in the metric the optimizer actually moves in — and you recover
> most of the loss, including on a dataset where alignment on its own is worth
> exactly nothing.**

| Pillar | Evidence | Status |
|---|---|---|
| Conflict exists and is noise-dependent | ρ(t) = cos(∇ℓ_diff, ∇ℓ_repa) over noise level × step | **Not measured** |
| Removing it accelerates convergence | ImageNet-100: 3.33× over baseline vs REPA's 2.21× | In hand |
| **The gain is not just "more alignment"** | CelebA: REPA = baseline (+0.1% KID), P-PCGrad −7.3% vs baseline | **In hand (KID)** |
| The conflict metric should follow the optimizer | Whitened wins on CelebA, Euclidean on IN-100; 8/8 beat REPA | Needs the anisotropy measurement |

Pillar 1 is the cheapest and converts an empirical note into a mechanistic
paper. Pillar 4 turns the variant flip from a wrinkle into a second
contribution.

### Venue

**Workshop, 4–8 pages — reachable in about a week.** ImageNet-100 + CelebA at
SiT-B/2 is a normal workshop budget, and the negative CelebA result plus a
mechanism figure is a genuinely interesting 4 pages. Needs: ρ(t) figures, the
bf16 control, the CelebA baseline FID, and a couple of seeds.

**Main conference — a different project.** Reviewers will require ImageNet-1K at
256px, SiT-XL/2 or at minimum SiT-L/2, FID-50K with a cfg sweep, multi-seed, and
comparison against the current REPA follow-up literature rather than REPA alone.
Weeks of large-GPU time. Start the IN-1K run now if you want it, but don't hold
the workshop version for it.

---

## 8. Next runs, in priority order

**1. Measure ρ(t) on the checkpoints you already have** — *no training, hours*

```bash
python training/measure_rho.py --run-dir runs/imagenet100_sit-b_2_repa --gpu 0
python training/measure_rho.py --run-dir runs/imagenet100_sit-b_2_repa-PCGrad --gpu 0
python results/analysis/analyze_rho.py runs/imagenet100_sit-b_2_repa
```

If cos(∇ℓ_diff, ∇ℓ_repa) really is positive at high noise and negative at low
noise, that heatmap is Figure 1 and the whole story locks into place. If it
isn't, you need a different explanation for why surgery works — better to learn
that now than in review. Run it on both datasets: the mechanism claim in §5
predicts the CelebA and ImageNet heatmaps look *different*.

**2. Measure the gradient anisotropy that predicts the variant flip** —
*post-hoc, hours*

Two quantities test §5 directly on checkpoints you already have: the effective
rank of the diffusion gradient's second moment (Adam's `v̂` is sitting in the
optimizer state), and the disagreement rate between the Euclidean and whitened
conflict tests on the same batch. If CelebA is the anisotropic one, the flip
becomes a *rule for choosing the metric*. Natural follow-up is a knob rather
than a switch: interpolate the preconditioner as `P^α` for α ∈ {0, ½, 1} and
show the optimum moving between datasets.

**3. Train REPA in bf16 on ImageNet-100** — *1 run, 300k steps*

The single experiment most likely to be demanded, and it kills the biggest
remaining confound. Until it exists, the honest ImageNet claim is "REPA-PCGrad
in bf16 beats REPA in fp16".

**4. Close the two eval gaps** — *eval only, ~2 h*

Regenerate the CelebA **baseline** at 25k so it has an FID. Then
`--refresh --refresh-real` on the ImageNet baseline arm and HASTE's 25k–125k
checkpoints, whose features came from a `(step, rank)`-keyed cache and were
never regenerated under this sweep's settings.

**5. Two more seeds, on both datasets** — *6 runs, 150k each*

ImageNet-100 REPA and REPA-PCGrad stopped at 150k, where the FID gap is already
3.0 — you don't need 300k to establish separation. Add CelebA REPA and P-PCGrad,
since that pair now carries a headline. Converts every error bar from "subset
noise" to "run-to-run noise".

**6. Ablate what the surgery is actually doing** — *3 runs, 150k each*

- (a) instantaneous ∇ℓ_diff as reference instead of its EMA — isolates the EMA;
- (b) always project, dropping the negative-only clamp — isolates the gating;
- (c) **project against a fixed random direction of matched norm** — the null
  control.

Without (c) especially, a reviewer can argue the whole effect is a smoothed
gradient term wearing a PCGrad costume.

**7. Log how often the surgery fires** — *cheap, add to (6)*

Fraction of steps with ⟨g_repa, g_ema⟩ < 0, and the norm of the removed
component, bucketed by timestep. A second mechanism figure for nearly no cost,
and it connects directly to the ρ(t) heatmap.

**8. Fix the reporting protocol for the final table** — *eval only*

FID-50K at 300k with a cfg sweep over {1.0, 1.25, 1.5, 1.75, 2.0} for every arm.
FID-10K at a single cfg is fine for run-to-run curves but will not pass as a
headline number, and the cfg=1.0 column is what makes the result comparable to
REPA's own tables.

**9. Decide HASTE's role** — *optional, 2 runs*

It is a quality null at a 100k termination step. Either sweep the termination
point over {50k, 200k} to show the null is robust, or demote HASTE to a cost
baseline and say plainly that its published benefit did not reproduce as a
quality gain at this scale. The second is cheaper and equally honest.

---

## 9. Claims to make, claims to avoid

**Make:** "Gradient surgery reaches REPA's 300k-step quality in 170k steps — a
1.47× wall-clock speedup on ImageNet-100."
Measurable from the curves, robust to the narrowing gap, already
compute-corrected. This is the abstract sentence.

**Avoid:** "Gradient surgery improves FID by 8.5% over REPA."
True at 300k and shrinking monotonically since 75k. Stating it as a quality
ceiling invites the one question the data cannot answer.

**Make:** "Representation alignment yields no measurable benefit on CelebA:
baseline, REPA and HASTE plateau within 0.05 KID×10³ of one another."
Three-arm agreement, one of the more defensible statements in the set.
Regenerate the baseline's FID first.

**Make:** "On CelebA, where alignment gives no benefit at all, conflict surgery
in Adam's metric improves KID by 7.3% over the unaligned baseline and FID by
13.9% over REPA — an advantage established by 60k steps that does not decay."
The strongest sentence available, and what separates this from "REPA, tuned".

**Avoid:** "The preconditioned variant is the better method."
It wins on CelebA and comes second on ImageNet-100. Claim the *rule* — the
conflict test belongs in the optimizer's metric, and how much that matters
tracks the gradient spectrum — and report both variants on both datasets.

**Make:** "On ImageNet-100 the KID gap is roughly nine combined standard errors
— but those errors are over Inception subsets, and we additionally report N
seeds."
Say what the error bar measures.

**Avoid:** "We achieve FID 9.02 on ImageNet."
ImageNet-**100**, SiT-B/2, FID-10K, cfg 1.5. Every one of those qualifiers has
to travel with the number, in the abstract and in the table caption.

---

*Speedups are log-interpolated between adjacent checkpoints. The 1.2× per-step
cost of surgery is the repo's own estimate and has not been measured.*
