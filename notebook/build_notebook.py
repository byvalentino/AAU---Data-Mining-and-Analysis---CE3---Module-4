#!/usr/bin/env python3
"""Build Module 4's demonstration notebook, and execute it.

    python "Module 4/notebook/build_notebook.py"
    python "Module 4/notebook/build_notebook.py" --no-run

Reads `exercises/data/bus_slice.csv.gz` — the committed extract of the vehicle
telemetry, which is the whole population this module's grain uses: one shuttle,
both days, 48,290 readings. It used to read `data/bus.csv`, the instructor's copy
of the archive, which is deliberately not in git, so nobody but the instructor
could run this notebook. The two give identical numbers on this grain, and the
extract is the one that ships.

The phone traces are not opened, so this notebook needs no aggregate-only
discipline and can print whatever is useful.

The grain is fixed and printed: one shuttle, five-minute windows of at least 300
readings, the first day as reference.

Executed from `Module 4/exercises`, so its relative paths resolve exactly as the
labs' do.
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent
EXERCISES = HERE.parent / "exercises"
OUTPUT = HERE / "Module4_demonstration.ipynb"
MARKDOWN, CODE = "markdown", "code"

CELLS = [
(MARKDOWN, """# Module 4 — Statistics for detecting change

**Data Mining and Analysis (course code CE3) · Aalborg University, Copenhagen**

One inequality and four small mathematical objects — an interval, a divergence,
a symmetrised index and a distance — built from their definitions and run on the
real archive.

> **The grain, stated once.** Shuttle VJRD1A10224000055 only, the one that ran on
> both days; five-minute tumbling windows on `utc_time` holding at least 300
> readings; 22 January as the reference against 23 January; the target is mean
> payload per window.
>
> Why one vehicle: the other shuttle ran on the first day and not the second.
> Pool both on day one against one on day two and part of what you call drift is
> a vehicle going to the depot. An earlier version of this course's plan did
> exactly that and got the *sign* of the target's movement wrong.
>
> Why at least 300 readings: a window holding a handful of readings is the edge
> of the day rather than a window. That choice is worth 0.21 of a standard
> deviation on this module's headline number, which is why it is printed here
> rather than left in the code."""),

(MARKDOWN, """## Hook

The second day looks different from the first. Prove it — and then work out
whether it matters, which is a different question with, here, a different
answer.

Start somewhere smaller, because the shape of the answer is already visible in a
journey."""),

(CODE, '''# A shuttle covers 30 km at 20 km/h, then the same 30 km at 60 km/h.
# What was its average speed?
LEG_KM, FIRST_KMH, SECOND_KMH = 30.0, 20.0, 60.0

mean_of_speeds = (FIRST_KMH + SECOND_KMH) / 2
hours = LEG_KM / FIRST_KMH + LEG_KM / SECOND_KMH
journey_speed = 2 * LEG_KM / hours

print(f"average of the two speeds : {mean_of_speeds:.1f} km/h")
print(f"total distance over total time: {journey_speed:.1f} km/h  ({2 * LEG_KM:.0f} km in {hours:.1f} h)")
print("\\nNobody made an arithmetic error. Speed and journey time are joined by a")
print("curve, and averaging before the curve is not averaging after it. That gap")
print("is Jensen's inequality (Jensen, 1906), and everything below rests on it.")'''),

(CODE, '''import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.stats import entropy as scipy_entropy, wasserstein_distance, ttest_ind

warnings.filterwarnings("ignore", category=FutureWarning)

# The course palette: reference blue, current orange, neutral grey, and red only
# for what fails.
BLUE, ORANGE, GREY, RED = "#2A78D6", "#E07B39", "#52514E", "#C0392B"

# The figures load plotly from a content delivery network rather than embedding a
# copy of it in this file: the embedded copy is four and a half megabytes, once,
# and a notebook that large is a notebook nobody opens on a train. The portable
# network graphics beside it are the offline copy.
pio.renderers.default = "notebook_connected"

FIGURES = Path("../notebook/figures")   # the working directory is Module 4/exercises
FIGURES.mkdir(parents=True, exist_ok=True)

def show(fig, name, width=900, height=500):
    """Render inline and keep a portable network graphic beside the notebook."""
    fig.update_layout(template="plotly_white", width=width, height=height)
    fig.write_image(str(FIGURES / f"{name}.png"), scale=2)
    fig.show()

VEHICLE = "VJRD1A10224000055"
WINDOW = "5min"
MINIMUM_READINGS = 300
REFERENCE_DAY, CURRENT_DAY = "2020-01-22", "2020-01-23"
DEFAULT_BINS = 5
PSI_EPSILON = 1e-6          # the floor under an empty bin's share; see below
MATERIAL_SHIFT_SD = 2.0     # a stated choice of this course's
# The index half of the rule is NOT a constant. It is derived below, per feature,
# from the null the index produces when the reference is compared against a
# resample of itself. BORROWED_INDEX is credit scoring's 0.25, kept only so that
# borrowing it can be priced.
BORROWED_INDEX = 0.25
NULL_RESAMPLES, NULL_QUANTILE = 1000, 0.99
SEED = 20200122

# The committed extract: one vehicle, both days, every reading this grain uses.
bus = pd.read_csv(Path("data/bus_slice.csv.gz"), low_memory=False)
one = bus[bus["vehicle_id"] == VEHICLE].copy()
one["_t"] = pd.to_datetime(one["utc_time"], utc=True)
one["window"] = one["_t"].dt.floor(WINDOW)
one["day"] = one["_t"].dt.date.astype(str)

table = one.groupby("window").agg(
    mean_speed=("speed", "mean"), sd_speed=("speed", "std"),
    sd_payload=("payload", "std"), human_driven=("mode", lambda v: float((v == "manual").mean())),
    mean_payload=("payload", "mean"), readings=("speed", "size"),
    manual_readings=("mode", lambda v: int((v == "manual").sum())),
).reset_index()
table["day"] = table["window"].dt.date.astype(str)
table = table[table["readings"] >= MINIMUM_READINGS]

reference = table[table["day"] == REFERENCE_DAY]
current = table[table["day"] == CURRENT_DAY]
print(f"{len(one):,} readings -> {len(reference)} windows on {REFERENCE_DAY}, "
      f"{len(current)} on {CURRENT_DAY}")'''),

(MARKDOWN, """## Core Concept

### An interval, and what it costs

From tomorrow there are no labels. Accuracy cannot be computed, only bought —
somebody checks a sample by hand. So the first question is arithmetic: how
many?"""),

(MARKDOWN, """> **Definition — the Wald interval.** The observed share of successes, plus and
> minus the standard normal quantile times the standard error read at that same
> observed share.
>
> `p̂ ± z·√( p̂(1−p̂)/n ), with p̂ = k/n successes in n trials`
>
> Choices: the ninety-five per cent level, so z = 1.96, and the normal
> approximation to the binomial law (Brown, Cai & DasGupta, 2001; Agresti &
> Coull, 1998).

> **Definition — the Wilson score interval.** The true rates that would have
> produced an observation at least as extreme as the one seen, at that level —
> the test inverted rather than an error bar hung on the estimate.
>
> `( p̂ + z²/2n ± z·√( p̂(1−p̂)/n + z²/4n² ) ) / ( 1 + z²/n )`
>
> Same inputs, same level, one more line of code (Wilson, 1927)."""),

(CODE, '''import math
Z = 1.959963984540054

def naive_interval(k, n):
    p = k / n
    half = Z * math.sqrt(p * (1 - p) / n)
    return (p - half, p + half)

def wilson_interval(k, n):
    centre = (k + Z**2 / 2) / (n + Z**2)
    half = (Z / (n + Z**2)) * math.sqrt(k * (n - k) / n + Z**2 / 4)
    return (centre - half, centre + half)

for k, n in ((34, 40), (40, 40)):
    lo_n, hi_n = naive_interval(k, n)
    lo_w, hi_w = wilson_interval(k, n)
    print(f"{k} of {n}:  naive [{lo_n:.3f}, {hi_n:.3f}]   Wilson [{lo_w:.3f}, {hi_w:.3f}]")
print("\\nForty out of forty: the naive interval claims certainty from forty")
print("observations. Only one of these two declines to say so.")'''),

(MARKDOWN, """> **Definition — coverage, and what a half-width costs.** Coverage is the promise
> measured: the long-run share of intervals that contain the true rate, over many
> samples drawn at that rate. The label count is the same arithmetic read
> backwards, at the worst case.
>
> `coverage(p, n) = (1/R)·Σ_{r=1}^{R} 1{ low_r ≤ p ≤ high_r }` and
> `n = ⌈ 0.25·(z/h)² ⌉ at the worst case p = ½`
>
> Choices: R = 4000 samples of n = 40 at each rate, seed 20200122; the worst case
> p = ½, where p(1−p) is largest; rounding up, because labels come whole (Brown,
> Cai & DasGupta, 2001)."""),

(CODE, '''# A 95% interval makes a testable claim. Test it.
def coverage(interval, true_rate, trials=40, repeats=4000, seed=SEED):
    rng = np.random.default_rng(seed)
    successes = rng.binomial(trials, true_rate, repeats)
    contained = [interval(int(k), trials)[0] <= true_rate <= interval(int(k), trials)[1]
                 for k in successes]
    return float(np.mean(contained)), float(np.mean(successes == 0))

print(f"{'true rate':>10} {'naive':>8} {'Wilson':>8} {'samples with no successes':>28}")
for rate in (0.01, 0.02, 0.05, 0.10, 0.30, 0.50):
    naive_cover, empty = coverage(naive_interval, rate)
    wilson_cover, _ = coverage(wilson_interval, rate)
    print(f"{rate:10.2f} {naive_cover:8.3f} {wilson_cover:8.3f} {empty:28.3f}")
print("\\nBoth promise 0.95. One of them delivers it. The worst point is the lowest")
print("rate: two thirds of those samples hold no successes at all, the naive")
print("interval is then [0, 0], and an interval of zero width contains nothing.")'''),

(CODE, '''# What precision costs, in hand-checks.
def labels_needed(half_width):
    return int(math.ceil(0.25 * (Z / half_width) ** 2))

print(f"{'half-width':>12} {'labels':>8}")
for half in (0.10, 0.05, 0.025, 0.01):
    print(f"{half:12.3f} {labels_needed(half):8,}")
print("\\nHalving the width costs four times the labels. The square root never stops.")'''),

(CODE, '''# Every result forty hand-checks could produce, both ways.
counts = np.arange(0, 41)
wald = np.array([naive_interval(int(k), 40) for k in counts])
wilson = np.array([wilson_interval(int(k), 40) for k in counts])

figure = go.Figure()
figure.add_scatter(x=counts, y=wilson[:, 1], mode="lines", showlegend=False,
                   line=dict(color=BLUE, width=2))
figure.add_scatter(x=counts, y=wilson[:, 0], mode="lines", name="Wilson interval",
                   line=dict(color=BLUE, width=2), fill="tonexty",
                   fillcolor="rgba(42, 120, 214, 0.22)")
figure.add_scatter(x=counts, y=wald[:, 1], mode="lines", showlegend=False,
                   line=dict(color=GREY, width=2, dash="dot"))
figure.add_scatter(x=counts, y=wald[:, 0], mode="lines", name="Wald interval",
                   line=dict(color=GREY, width=2, dash="dot"), fill="tonexty",
                   fillcolor="rgba(82, 81, 78, 0.18)")
figure.add_scatter(x=[0, 40], y=[0.0, 1.0], mode="markers", name="Wald: no width at all",
                   marker=dict(color=RED, size=12, symbol="x"))
figure.update_layout(title="Forty hand-checks: what each method reports",
                     xaxis_title="hand-checked predictions found correct, out of n = 40",
                     yaxis_title="interval reported for the true rate",
                     legend=dict(x=0.03, y=0.97))
show(figure, "wilson_against_wald")'''),

(CODE, '''# And the same choice on the archive's own windows, where the rate being
# estimated is the share of readings a person drove. At both edges -- no manual
# reading at all, or nothing but manual readings -- the Wald interval is a point.
share = (table["manual_readings"] / table["readings"]).to_numpy()
pairs = list(zip(table["manual_readings"], table["readings"]))
wald = np.array([naive_interval(int(k), int(n)) for k, n in pairs])
wilson = np.array([wilson_interval(int(k), int(n)) for k, n in pairs])
collapsed = (wald[:, 1] - wald[:, 0]) < 1e-12
position = np.arange(1, len(table) + 1)

print(f"{collapsed.sum()} of {len(table)} windows carry a Wald interval of no width")
print(f"Wilson on a window with no manual reading and 600 readings: "
      f"[{wilson_interval(0, 600)[0]:.4f}, {wilson_interval(0, 600)[1]:.4f}]")

figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                       subplot_titles=("every window, both days",
                                       "the same windows, magnified onto the floor"))
for row in (1, 2):
    figure.add_scatter(x=position, y=share, mode="markers",
                       marker=dict(color=np.where(collapsed, RED, GREY), size=6),
                       error_y=dict(type="data", symmetric=False,
                                    array=wald[:, 1] - share, arrayminus=share - wald[:, 0],
                                    color=GREY, thickness=1.4, width=0),
                       name="Wald (red where it has no width)", showlegend=row == 1,
                       row=row, col=1)
    figure.add_scatter(x=position + 0.3, y=share, mode="markers",
                       marker=dict(color=BLUE, size=5),
                       error_y=dict(type="data", symmetric=False,
                                    array=wilson[:, 1] - share,
                                    arrayminus=share - wilson[:, 0],
                                    color=BLUE, thickness=1.4, width=0),
                       name="Wilson", showlegend=row == 1, row=row, col=1)
figure.update_yaxes(title_text="share of readings in manual mode", range=[-0.03, 1.06],
                    row=1, col=1)
figure.update_yaxes(title_text="share of readings", range=[-0.002, 0.02], row=2, col=1)
figure.update_xaxes(title_text="five-minute window, in order of time", row=2, col=1)
figure.update_layout(legend=dict(orientation="h", x=0.25, y=1.14))
show(figure, "wilson_on_the_archive", height=640)'''),

(MARKDOWN, """### Surprise, and the two ways to average it

Give an outcome probability p and observing it carries −log(p) of surprise.
Average that surprise under **today's own** distribution and you have the
entropy: how varied today was. Average it under **yesterday's** and you have the
cross-entropy: what today cost you, given what you believed.

The gap between them is the Kullback–Leibler divergence (Kullback & Leibler,
1951) — the *excess* surprise, not the surprise. Calling the cross-entropy a
divergence is the commonest error in this material, and it was in this deck's
own first draft."""),

(MARKDOWN, """> **Definition — entropy and cross-entropy.** Entropy is the average surprise of
> a distribution under itself; cross-entropy is the average surprise of the same
> data under a different distribution.
>
> `H(P) = −Σ_i P(i)·log P(i)` and `H(P,Q) = −Σ_i P(i)·log Q(i)`, natural
> logarithm, in nats (Shannon, 1948; Murphy, 2022, §6.1.2).

> **Definition — the Kullback–Leibler divergence.** The excess average surprise
> of holding the wrong distribution.
>
> `D(P ‖ Q) = Σ_i P(i)·log( P(i) / Q(i) ) = H(P,Q) − H(P), in nats`
>
> Never negative and nought exactly when the two match — Jensen's inequality
> applied to −log, which is Gibbs' inequality by MacKay's name for it (Kullback &
> Leibler, 1951; MacKay, 2003, §2.6).

> **Definition — Jensen's inequality.** `f( E[X] ) ≤ E[ f(X) ] for convex f, with
> equality only where f is straight or X never varies` (Jensen, 1906; Wasserman,
> 2004, Theorem 4.9). It is the hook at the top of this notebook, and the reason
> the divergence has a floor at all."""),

(CODE, '''def entropy(p):
    p = np.asarray(p, dtype=float); live = p > 0
    return float(-np.sum(p[live] * np.log(p[live])))

def cross_entropy(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float); live = p > 0
    if np.any(q[live] == 0): return float("inf")
    return float(-np.sum(p[live] * np.log(q[live])))

def kl_divergence(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float); live = p > 0
    if np.any(q[live] == 0): return float("inf")
    return float(np.sum(p[live] * np.log(p[live] / q[live])))

TODAY, YESTERDAY = [0.9, 0.1], [0.5, 0.5]
print(f"entropy of today        H(P)    = {entropy(TODAY):.3f} nats")
print(f"cross-entropy           H(P, Q) = {cross_entropy(TODAY, YESTERDAY):.3f} nats")
print(f"divergence  H(P,Q) - H(P)       = {cross_entropy(TODAY, YESTERDAY) - entropy(TODAY):.3f} nats")
print(f"divergence, summed directly     = {kl_divergence(TODAY, YESTERDAY):.3f} nats")
print(f"\\nthe other way round D(Q||P)     = {kl_divergence(YESTERDAY, TODAY):.3f} nats  <- not the same number")
print(f"and against scipy               = {float(scipy_entropy(TODAY, YESTERDAY)):.3f} nats")
print(f"\\nin bits, divide by log(2) = {math.log(2):.4f}: {kl_divergence(TODAY, YESTERDAY) / math.log(2):.3f} bits")'''),

(CODE, '''# The three numbers side by side: the gap between the two averages of surprise
# is the divergence, and it is the only one of the three with a fixed floor.
figure = go.Figure(go.Bar(
    x=["entropy H(P)", "cross-entropy H(P,Q)", "divergence D(P||Q)"],
    y=[entropy(TODAY), cross_entropy(TODAY, YESTERDAY), kl_divergence(TODAY, YESTERDAY)],
    marker_color=[BLUE, ORANGE, GREY],
    text=[f"{value:.3f}" for value in (entropy(TODAY), cross_entropy(TODAY, YESTERDAY),
                                       kl_divergence(TODAY, YESTERDAY))],
    textposition="outside"))
figure.update_layout(title="Today [0.9, 0.1] under yesterday [0.5, 0.5]",
                     yaxis_title="nats", showlegend=False)
show(figure, "surprise_decomposition", height=440)'''),

(MARKDOWN, """## Worked Example

### Three cases where the divergence and the distance disagree

The divergence measures how wrong your beliefs were. The distance measures how
far the world went. Those are different questions, and here are three pairs of
samples that make the difference impossible to miss.

> **Definition — the Wasserstein-1 distance.** The area between two cumulative
> distribution functions, which in one dimension is also the cheapest cost of
> moving one pile of mass into the shape of the other — in the variable's own
> units.
>
> `W₁(P,Q) = ∫ |F_P(x) − F_Q(x)| dx = ∫₀¹ |F_P⁻¹(u) − F_Q⁻¹(u)| du`
>
> For two sorted samples of equal size it is the mean of |a_(i) − b_(i)|
> (Vallender, 1974; Peyré & Cuturi, 2019, Remarks 2.30 and 2.28).

> **Definition — the two binnings.** All four measures on one pair of samples
> bin twice, on purpose:
>
> `H and D: equal-width edges over both samples · J: the reference's quantile
> edges, opened at both ends`
>
> A pair in front of you can be binned over both; a monitor watching one
> reference for months needs a yardstick cut once (Siddiqi, 2006; Yurdakul &
> Naranjo, 2020)."""),

(CODE, '''def shares_over_both(a, b, bins=DEFAULT_BINS):
    """Equal-width bins spanning both samples, so a value the reference never
    held gets a bin of its own instead of hiding inside an existing one."""
    edges = np.histogram_bin_edges(np.concatenate([a, b]), bins=bins)
    return (np.histogram(a, bins=edges)[0] / len(a),
            np.histogram(b, bins=edges)[0] / len(b))

# 1. No overlap: yesterday between 0 and 2 m/s, today between 5 and 7.
rng = np.random.default_rng(SEED)
yesterday = rng.uniform(0.0, 2.0, 400)
today = yesterday + 5.0
share_yesterday, share_today = shares_over_both(yesterday, today, bins=10)

print("no overlap:")
print(f"  yesterday {yesterday.min():.2f}-{yesterday.max():.2f} m/s, "
      f"today {today.min():.2f}-{today.max():.2f} m/s")
print(f"  divergence  {kl_divergence(share_today, share_yesterday)}  "
      "<- the same answer for a gap of five and a gap of five hundred")
print(f"  distance    {wasserstein_distance(yesterday, today):.3f} m/s  <- the gap itself")'''),

(CODE, '''# 2. Shuffled bins: one pair of distributions, and the bins relabelled so that
#    every share keeps its partner and only the value each bin stands for changes.
centres = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
reference_shares = np.array([0.40, 0.30, 0.20, 0.07, 0.03])
current_shares = np.array([0.03, 0.07, 0.20, 0.30, 0.40])
order = np.argsort(np.random.default_rng(SEED).permutation(len(centres)))

before_divergence = kl_divergence(current_shares, reference_shares)
after_divergence = kl_divergence(current_shares[order], reference_shares[order])
before_distance = wasserstein_distance(centres, centres, reference_shares, current_shares)
after_distance = wasserstein_distance(centres, centres,
                                      reference_shares[order], current_shares[order])

print("shuffled bins:")
print(f"  divergence  {before_divergence:.3f} -> {after_divergence:.3f}   "
      "<- unchanged; it never knew the values were ordered")
print(f"  distance    {before_distance:.3f} -> {after_distance:.3f}   "
      "<- moved; it is built on the geometry, and the geometry is what changed")'''),

(CODE, '''# 3. Moved, or squeezed. Both cases above end in an infinity or in a relabelling,
#    and both are easy to dismiss as pathologies. This one is neither: the same
#    law moved sideways, against the same law squeezed about its own centre,
#    chosen so that the distance cannot tell them apart. The mean and the spread
#    are the archive's own; the two changes are constructed from them.
mean = float(reference["mean_speed"].mean())
spread = float(reference["mean_speed"].std(ddof=1))
factor = 0.25                       # a quarter of the spread, a stated choice
shift = (1 - factor) * spread * np.sqrt(2 / np.pi)   # E|Z| = sqrt(2/pi) for a normal

law = np.random.default_rng(SEED).normal(mean, spread, 20000)
moved, squeezed = law + shift, mean + factor * (law - mean)

share_moved = shares_over_both(law, moved)
share_squeezed = shares_over_both(law, squeezed)
print("moved, or squeezed:")
print(f"  distance    {wasserstein_distance(law, moved):.3f} m/s moved, "
      f"{wasserstein_distance(law, squeezed):.3f} m/s squeezed  <- the same, by construction")
print(f"  divergence  {kl_divergence(share_moved[1], share_moved[0]):.3f} nats moved, "
      f"{kl_divergence(share_squeezed[1], share_squeezed[0]):.3f} nats squeezed  <- not the same")
print("\\nIn closed form on the normal law, where nothing is binned, the two divergences")
print("are 0.179 and 0.918 nats: binning pulls both down, and that is what the binning")
print("costs in exchange for staying finite.")'''),

(CODE, '''# The three cases in one picture: distributions above, cumulative curves below.
figure = make_subplots(rows=2, cols=3, vertical_spacing=0.14, horizontal_spacing=0.08,
                       subplot_titles=("no overlap", "bins relabelled", "moved, or squeezed",
                                       "cumulative curves", "the same, relabelled",
                                       "cumulative curves"))

def cumulative(sample, grid):
    return np.searchsorted(np.sort(sample), grid, side="right") / len(sample)

grid = np.linspace(-0.2, 7.2, 400)
edges = np.linspace(0.0, 7.0, 36)
mids = (edges[:-1] + edges[1:]) / 2
for sample, colour, name in ((yesterday, BLUE, "reference"), (today, ORANGE, "current")):
    figure.add_bar(x=mids, y=np.histogram(sample, bins=edges)[0] / len(sample),
                   marker_color=colour, name=name, row=1, col=1)
    figure.add_scatter(x=grid, y=cumulative(sample, grid), mode="lines",
                       line=dict(color=colour, width=2), showlegend=False, row=2, col=1)

for row, (a, b) in enumerate(((reference_shares, current_shares),
                              (reference_shares[order], current_shares[order])), start=1):
    figure.add_bar(x=centres, y=a, marker_color=BLUE, width=0.38, offset=-0.4,
                   showlegend=False, row=row, col=2)
    figure.add_bar(x=centres, y=b, marker_color=ORANGE, width=0.38, offset=0.02,
                   showlegend=False, row=row, col=2)

fine = np.linspace(mean - 4 * spread, mean + 4 * spread, 400)
for sample, colour, dash in ((law, BLUE, "solid"), (moved, ORANGE, "solid"),
                             (squeezed, GREY, "dash")):
    counts, bin_edges = np.histogram(sample, bins=60, range=(fine[0], fine[-1]))
    figure.add_scatter(x=(bin_edges[:-1] + bin_edges[1:]) / 2, y=counts / len(sample),
                       mode="lines", line=dict(color=colour, width=2, dash=dash),
                       showlegend=False, row=1, col=3)
    figure.add_scatter(x=fine, y=cumulative(sample, fine), mode="lines",
                       line=dict(color=colour, width=2, dash=dash), showlegend=False,
                       row=2, col=3)

figure.update_xaxes(title_text="metres per second", row=2, col=1)
figure.update_xaxes(title_text="bin centre, metres per second", row=2, col=2)
figure.update_xaxes(title_text="window mean speed, metres per second", row=2, col=3)
figure.update_yaxes(title_text="share", row=1, col=1)
figure.update_yaxes(title_text="cumulative share", row=2, col=1)
figure.update_layout(barmode="overlay", legend=dict(orientation="h", x=0.3, y=1.12))
show(figure, "divergence_against_distance", width=1100, height=620)'''),

(MARKDOWN, """### The verdict — the same table that reaches the slide

Three measures per feature and one rule: material when the shift reaches 2.0
reference standard deviations **or** the index reaches **the threshold derived
from that feature's own null**. The shift bound is a stated choice of this
course's. The index bound is a measurement, and there is deliberately no 0.25
anywhere in the rule.

> **Definition — the symmetrised index.** `J(P,Q) = Σ_i ( P(i) − Q(i) )·log(
> P(i) / Q(i) ) = D(P‖Q) + D(Q‖P)`, over binned shares, with edges from the
> reference's own quantiles and a floor of 1e-06 under every share (Jeffreys,
> 1946). Under no change (1/n + 1/m)⁻¹·J is approximately chi-squared with one
> fewer degree of freedom than bins, so any threshold depends on the two sample
> sizes and the bin count (Yurdakul & Naranjo, 2020).

> **Definition — the standardised shift, and the rule it is read against.**
> `Δ = ( mean_current − mean_reference ) / s_reference, ddof = 1` (Glass, 1976),
> and `material when |Δ| ≥ 2.0 or J ≥ threshold(B, q) derived from this feature's
> own null` (Yurdakul & Naranjo, 2020). The reference period's own spread, not a
> pooled one; either instrument may fire.

> **Definition — the materiality threshold, derived from the floor.**
> `threshold(B, q) = Quantile_q { J( reference, resample of the reference ) at B
> bins }, over the same R resamples` (Yurdakul & Naranjo, 2020). The floor is the
> median of that same null; the threshold is a point in its tail, and `q` is a
> false-alarm rate said out loud rather than a convention nobody derived."""),

(CODE, '''def population_stability_index(reference_sample, current_sample, bins=DEFAULT_BINS):
    """Symmetric, binned on the REFERENCE's quantiles -- never on today's -- and
    floored at PSI_EPSILON so one empty bin does not make the whole index infinite.

    Returns (index, measured). A reference whose quantile edges collapse to fewer
    than three distinct values has one bin, both shares are 1.0, and the index is
    exactly nought whatever the column did. That is not "no change", it is "no
    measurement", so it comes back flagged rather than as a reassuring zero.
    """
    edges = np.unique(np.quantile(reference_sample, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0, False
    edges[0], edges[-1] = -np.inf, np.inf
    share_a = np.clip(np.histogram(reference_sample, bins=edges)[0] / len(reference_sample),
                      PSI_EPSILON, None)
    share_b = np.clip(np.histogram(current_sample, bins=edges)[0] / len(current_sample),
                      PSI_EPSILON, None)
    return float(np.sum((share_b - share_a) * np.log(share_b / share_a))), True

FEATURES = ["mean_speed", "sd_speed", "sd_payload", "human_driven", "mean_payload"]
TARGET = "mean_payload"

def index_threshold(reference_sample, current_sample, bins=DEFAULT_BINS,
                    resamples=NULL_RESAMPLES, quantile=NULL_QUANTILE, seed=SEED):
    """The floor the index reads under no change, and the threshold derived from it.

    Comparing the reference against a resample of ITSELF is two samples from one
    world, so the distribution that comes back is what "nothing happened" looks
    like at this bin count and these sample sizes. Its median is the floor; the
    stated quantile of its upper tail is the threshold, and that quantile is a
    false-alarm rate somebody chose out loud.
    """
    stream = np.random.default_rng(seed)
    null = np.array([population_stability_index(
        reference_sample, stream.choice(reference_sample, size=len(current_sample),
                                        replace=True), bins)[0]
        for _ in range(resamples)])
    return {"noise_floor": float(np.median(null)),
            "threshold": float(np.quantile(null, quantile)),
            "share_above_borrowed": float(np.mean(null >= BORROWED_INDEX)),
            "bins": bins, "resamples": resamples, "quantile": quantile, "seed": seed}

def verdict(reference_frame, current_frame, features=FEATURES, thresholds=None):
    thresholds = dict(thresholds or {})
    rows = {}
    for feature in features:
        before = reference_frame[feature].dropna().to_numpy()
        after = current_frame[feature].dropna().to_numpy()
        shift = (after.mean() - before.mean()) / before.std(ddof=1)
        index, measured = population_stability_index(before, after)
        # The null is built out of the reference alone, so a caller sweeping the
        # current day can derive it once and hand it back. A column that cannot
        # be binned has no null and therefore no threshold either.
        derived = thresholds.get(feature) if measured else None
        if measured and derived is None:
            derived = index_threshold(before, after)
        rows[feature] = {
            "shift": shift, "index": index, "index_measured": measured,
            "noise_floor": derived["noise_floor"] if measured else None,
            "threshold": derived["threshold"] if measured else None,
            "wasserstein": wasserstein_distance(before, after),
            "material": bool(abs(shift) >= MATERIAL_SHIFT_SD
                             or (measured and index >= derived["threshold"])),
        }
    return rows

def print_verdict(rows):
    print(f"{'feature':14}{'shift SD':>10}{'index':>13}{'floor':>13}"
          f"{'threshold':>13}{'material':>10}")
    for feature, row in rows.items():
        index = f"{row['index']:.3f}" if row["index_measured"] else "unmeasured"
        floor = f"{row['noise_floor']:.3f}" if row["index_measured"] else "unmeasured"
        threshold = f"{row['threshold']:.3f}" if row["index_measured"] else "unmeasured"
        marker = "  <- target" if feature == TARGET else ""
        print(f"{feature:14}{row['shift']:+10.2f}{index:>13}{floor:>13}"
              f"{threshold:>13}{str(row['material']):>10}{marker}")

rows = verdict(reference, current)
print_verdict(rows)
material = [f for f, r in rows.items() if r["material"]]
borrowed = [f for f, r in rows.items()
            if abs(r["shift"]) >= MATERIAL_SHIFT_SD
            or (r["index_measured"] and r["index"] >= BORROWED_INDEX)]
largest = max((f for f, r in rows.items() if r["index_measured"]),
              key=lambda f: rows[f]["index"])
print(f"\\n{len(material)} of {len(FEATURES)} material: {', '.join(material)}")
print(f"with banking's {BORROWED_INDEX} instead it would be {len(borrowed)}: "
      f"{', '.join(borrowed)}")
print(f"the extra alarm is {', '.join(sorted(set(borrowed) - set(material)))}, whose "
      f"index {rows['sd_payload']['index']:.3f} sits just under its own threshold "
      f"of {rows['sd_payload']['threshold']:.3f}")
print(f"largest index of all: {largest}, not mean_speed. Two measures, two orderings.")
print(f"\\nand the target: index {rows[TARGET]['index']:.3f} against its own measured "
      f"noise floor of {rows[TARGET]['noise_floor']:.3f} -- BELOW the floor, so the "
      f"instrument cannot tell it apart from a day on which nothing happened.")'''),

(MARKDOWN, """One input moved a great deal. The target moved three hundredths of a standard
deviation.

A monitor watching inputs would have fired. A monitor watching the target would
not. **Both would have been right** — and the operator's question is not "did
anything change?" but "must we do anything?"

Note the row the index could not measure at all. `human_driven` is nought in
most reference windows, so its quantile edges collapse to one bin and the index
reads exactly nought — for the column that explains the whole event.

### The cause is in a column nobody was watching"""),

(CODE, '''window_share = {day: round(float(group["human_driven"].mean()) * 100, 1)
                for day, group in table.groupby("day")}
reading_share = {day: round(float((group["mode"] == "manual").mean()) * 100, 2)
                 for day, group in one.groupby("day")}
print("manual mode, mean of the per-window shares:", window_share)
print("manual mode, share of readings            :", reading_share)

degenerate = reference["human_driven"].dropna().to_numpy()
edges = np.unique(np.quantile(degenerate, np.linspace(0, 1, DEFAULT_BINS + 1)))
print(f"\\nreference windows holding nought: {(degenerate == 0).sum()} of {len(degenerate)}")
print(f"distinct quantile edges surviving: {len(edges)} -> {len(edges) - 1} bin")
print("\\nThe speed distribution moved because a person was driving four times as")
print("often. That is the `mode` column, nothing was monitoring it, and the index")
print("cannot see it.")'''),

(MARKDOWN, """> **Definition — the positive control.** `verdict( reference, current +
> k·s_reference ) must return material, with k stated beside the result`. Without
> one, "nothing changed" is a claim about an instrument nobody has tested
> (Saltelli et al., 2019).

> **Definition — the detection limit.** `detection limit = min{ k in the swept
> sizes : verdict( reference, current + j·s_reference ) is material for every
> swept j ≥ k }` (Currie, 1968). A control at one size says the detector detects
> that size. The sweep says what it is blind to, and that belongs in the
> report."""),

(MARKDOWN, """### The positive control

"The target did not move" is an absence claim. An absence claim from an
instrument nobody has tested is an opinion, so inject a shift of a known size
and re-run the **unchanged** verdict."""),

(CODE, '''INJECTED_SHIFT_SD = 1.5
DETECTION_SIZES = [round(0.05 * step, 2) for step in range(0, 31)]

payload = reference[TARGET].dropna().to_numpy()
spread = payload.std(ddof=1)
# Derived once: the null resamples the reference against itself, so nothing the
# sweep does to the current day can change it. Holding it fixed is what "the
# unchanged verdict" means.
target_threshold = {TARGET: index_threshold(payload, current[TARGET].dropna().to_numpy())}

def inject(size):
    moved = current.copy()
    moved[TARGET] = moved[TARGET] + size * spread
    return verdict(reference, moved, [TARGET], target_threshold)[TARGET]

control = inject(INJECTED_SHIFT_SD)
print(f"injected {INJECTED_SHIFT_SD} reference standard deviations into the target")
print(f"  shift     {control['shift']:+.2f} SD")
print(f"  index     {control['index']:.3f}")
print(f"  threshold {control['threshold']:.3f}")
print(f"  material  {control['material']}")

swept = [inject(size) for size in DETECTION_SIZES]
fired = [row["material"] for row in swept]
first = next(size for size, hit in zip(DETECTION_SIZES, fired) if hit)
limit = next(size for position, size in enumerate(DETECTION_SIZES) if all(fired[position:]))
print(f"\\nsweep {DETECTION_SIZES[0]} to {DETECTION_SIZES[-1]} SD in steps of 0.05:")
print(f"  first firing at        {first} SD -- and it falls back at the next size")
print(f"  material from          {limit} SD upwards, and at every larger size")
print(f"  detection limit        {limit} reference SD = {limit * spread:.1f} kilograms")
print("\\nThe shift rule does not fire at 1.5, so what fired is the index. The same")
print("code, on the same grain, detects movement when there is some -- and now we")
print("can also say what it would have missed: anything under "
      f"{limit * spread:.1f} kilograms of mean payload per five-minute window.")

curve = go.Figure()
curve.add_scatter(x=DETECTION_SIZES, y=[row["index"] for row in swept],
                  mode="lines+markers", line=dict(color=BLUE, width=2.5),
                  name="index of the injected target")
curve.add_hline(y=control["threshold"], line=dict(color=ORANGE, dash="dash"),
                annotation_text=f"derived threshold {control['threshold']:.3f}")
curve.add_hline(y=rows[TARGET]["noise_floor"], line=dict(color=GREY, dash="dot"),
                annotation_text=f"measured floor {rows[TARGET]['noise_floor']:.3f}")
curve.add_vline(x=limit, line=dict(color=RED, width=2),
                annotation_text=f"detection limit {limit} SD")
curve.update_layout(title="How small a shift in the target this instrument can still see",
                    showlegend=False, yaxis_type="log")
curve.update_xaxes(title_text="shift injected into the target, in reference standard deviations")
curve.update_yaxes(title_text="symmetrised index (nats, logarithmic)")
show(curve, "detection_limit", width=1000, height=560)'''),

(MARKDOWN, """### The threshold you borrowed does not fit your sample

Banking reads an index above 0.25 as a material shift. Those thresholds come
from scorecard populations of many thousands. Here there are about forty windows
a day — so measure what the index reads when **nothing** has changed, and take
the threshold out of that instead of out of a handbook."""),

(CODE, '''# Compare the reference against a RESAMPLE OF ITSELF -- no change at all.
# The median of what comes back is the floor; its 0.99 quantile is the threshold;
# and the share of it above 0.25 is the false-alarm rate borrowing that number buys.
today = current[TARGET].dropna().to_numpy()
observed = {bins: population_stability_index(payload, today, bins)[0]
            for bins in (3, 5, 10, 20)}
print(f"{'bins':>5}{'floor':>9}{'threshold':>11}{'passes 0.25':>13}{'fires on nothing':>18}")
for bins in (3, 5, 10, 20):
    derived = index_threshold(payload, today, bins)
    fires = observed[bins] >= derived["threshold"]
    print(f"{bins:5}{derived['noise_floor']:9.3f}{derived['threshold']:11.3f}"
          f"{derived['share_above_borrowed']:13.3f}{str(fires):>18}")
print(f"\\nAt five bins, one comparison in ten in which NOTHING changed already passes")
print("0.25. At ten bins it is closer to three in five, and the last column is the")
print("measurement that settles the bin count: at ten bins the untouched target")
print(f"reads {observed[10]:.3f} against a threshold of "
      f"{index_threshold(payload, today, 10)['threshold']:.3f}, so the verdict calls")
print(f"it material -- a false alarm on a column that moved {rows[TARGET]['shift']:+.2f} SD.")
print(f"At five bins it reads {observed[5]:.3f} against {rows[TARGET]['threshold']:.3f} and")
print("does not. Find your own floor before trusting anybody's number.")'''),

(MARKDOWN, """### Significance is not size

> **Definition — Welch's t-test, and the bootstrap to the reading grain.**
> `t = ( m₁ − m₂ ) / √( s₁²/n₁ + s₂²/n₂ )` (Welch, 1947), with no common variance
> assumed; and the same difference resampled with replacement to n readings, seed
> 20200122 (Efron, 1979) — only the sample size changes.
>
> **Definition — Cohen's d.** `d = ( m₁ − m₂ ) / s_pooled, s_pooled = √(
> ((n₁−1)s₁² + (n₂−1)s₂²) / (n₁+n₂−2) )` (Cohen, 1988). The pooling is weighted
> by degrees of freedom; the unweighted root mean square is a different number
> whenever the samples differ in size, and here they do."""),

(CODE, '''before = reference["mean_speed"].dropna().to_numpy()
after = current["mean_speed"].dropna().to_numpy()

windows = ttest_ind(before, after, equal_var=False)
readings = ttest_ind(one.loc[one["day"] == REFERENCE_DAY, "speed"],
                     one.loc[one["day"] == CURRENT_DAY, "speed"], equal_var=False)
# Cohen's pooling, weighted by degrees of freedom -- 45 windows against 35, so
# the unweighted root mean square of the two variances is a different number.
pooled = np.sqrt(((len(before) - 1) * before.var(ddof=1)
                  + (len(after) - 1) * after.var(ddof=1))
                 / (len(before) + len(after) - 2))
unweighted = np.sqrt((before.var(ddof=1) + after.var(ddof=1)) / 2)

# A p-value is never exactly nought. The reading-grain one underflows the
# smallest double there is, so what gets printed is a bound: "p = 0" claims
# something no test can support.
reported = (f"{readings.pvalue:.3g}" if readings.pvalue > 0
            else f"less than 1e-300 (it underflowed to exactly {readings.pvalue})")
print(f"at {len(before) + len(after):>6,} windows : p = {windows.pvalue:.3g}")
print(f"at {len(one):>6,} readings: p = {reported}")
print(f"\\nCohen's d, pooled by degrees of freedom : {(after.mean() - before.mean()) / pooled:.3f}")
print(f"the unweighted pooling would give        : {(after.mean() - before.mean()) / unweighted:.3f}")

# And the reading grain is not worth 48,290 independent observations anyway:
# consecutive readings half a second apart correlate at 0.997 (measured in
# Module 1), which leaves n(1-rho)/(1+rho).
rho = 0.997
print(f"effective sample size at rho = {rho}: {len(one) * (1 - rho) / (1 + rho):.0f}")
print("\\nThe difference did not change. Only how much of it we looked at -- and the")
print("larger sample was mostly the same shuttle, half a second later.")'''),

(MARKDOWN, """### The test the required reading argues for

> **Definition — the classifier two-sample test.** `the two samples differ when
> the interval around a held-out classifier's accuracy lies above chance, chance
> = 1/2 on balanced classes` (Rabanser, Günnemann & Lipton, 2019). Balanced to
> the smaller day so chance is one half; half of each trains and half is held
> out; features standardised by the training reference alone; Wilson's interval
> at 95 per cent.

Rabanser, Günnemann and Lipton put the marginal tests this module builds against
a domain classifier and found the classifier hard to beat. So run it here, on
the same pair of days, and read the interval rather than the accuracy."""),

(CODE, '''def classifier_two_sample_test(reference_frame, current_frame, features=FEATURES,
                              seed=SEED):
    """A domain classifier as a two-sample test, with block one's interval on it.

    Nearest class centroid on standardised features: a linear rule in six lines,
    which is enough to make the reading's point. Standardising with the training
    reference alone rather than with both days is not fussiness -- standardising
    with both would let the held-out rows inform the scaling, and the test would
    then be partly about itself.
    """
    stream = np.random.default_rng(seed)
    before = reference_frame[list(features)].dropna().to_numpy(dtype=float)
    after = current_frame[list(features)].dropna().to_numpy(dtype=float)

    size = min(len(before), len(after))
    before = before[stream.permutation(len(before))[:size]]
    after = after[stream.permutation(len(after))[:size]]

    train = size // 2
    centre = before[:train].mean(axis=0)
    scale = np.where(before[:train].std(axis=0, ddof=1) > 0,
                     before[:train].std(axis=0, ddof=1), 1.0)
    reference_centroid = ((before[:train] - centre) / scale).mean(axis=0)
    current_centroid = ((after[:train] - centre) / scale).mean(axis=0)

    def says_current(sample):
        standardised = (sample - centre) / scale
        return (((standardised - current_centroid) ** 2).sum(axis=1)
                < ((standardised - reference_centroid) ** 2).sum(axis=1))

    correct = int((~says_current(before[train:])).sum() + says_current(after[train:]).sum())
    held_out = len(before[train:]) + len(after[train:])
    low, high = wilson_interval(correct, held_out)
    return {"correct": correct, "held_out": held_out, "accuracy": correct / held_out,
            "interval": (low, high), "chance": 0.5, "detected": bool(low > 0.5)}

joint = classifier_two_sample_test(reference, current)
alone = classifier_two_sample_test(reference, current, [TARGET])
for name, outcome in (("all five features", joint), ("the target alone", alone)):
    print(f"{name:20} {outcome['correct']:3} of {outcome['held_out']} held out, "
          f"accuracy {outcome['accuracy']:.3f}, interval "
          f"[{outcome['interval'][0]:.3f}, {outcome['interval'][1]:.3f}], "
          f"detected {outcome['detected']}")
print("\\nIt detects on all five -- by three thousandths at the lower bound, with an")
print("interval nearly three tenths wide. On the target alone it does not, which is")
print("the index's answer reached by an entirely different route.")
print("\\nThe reading's result stands; its conditions do not hold here. They test on")
print("thousands of samples. We have 36 held-out windows, and the honest comparison")
print("is that at this sample size the classifier is not better -- it is untestable,")
print("and the four measures also say WHICH column moved.")'''),

(MARKDOWN, """### What the law asks of a monitor

Regulation (EU) 2024/1689, the European Union Artificial Intelligence Act,
Article 15 — accuracy, robustness and cybersecurity.

- **Article 15(3):** the levels of accuracy and the relevant accuracy metrics of
  a high-risk system *shall be declared in the instructions for use*. Declared,
  to whoever operates the system — not measured internally and filed.
- **Article 15(4):** the system shall be as resilient as possible regarding
  errors, faults or inconsistencies that may occur within it or in the
  environment in which it operates. A drifting input distribution is that
  environment moving.

Two sentences worth saying plainly. **The positive control is how you evidence
that a monitor works** — a monitor nobody has shown can fire is not evidence of
anything. And **a declared metric without a measured floor is not a
declaration**: "the index is below 0.25" says nothing until somebody says what
the index reads when nothing has changed, and how small a change would still
have been missed.

The Annex III high-risk obligations fall due on **2 December 2027** after the
Digital Omnibus deferral. That is a deadline for evidence, and the evidence is
the kind of measurement in the cells above."""),

(MARKDOWN, """## Practice

1. **Does the verdict survive a different grain?** Recompute the table at one
   minute and at fifteen minutes. Does the target ever become material? What does
   that tell you about quoting a shift without its grain?
2. **Where is the noise floor for the Wasserstein distance?** Resample the
   reference against itself and find the distribution of distances. Is the
   target's 20.9 kilograms inside it?
3. **How small a shift would the control still catch?** Repeat the positive
   control at 1.0, 0.5 and 0.25 standard deviations. At what size does the
   detector stop firing, and what does that say about what your null result
   actually established?

Answers in the Appendix."""),

(CODE, '''# Your workings here.
'''),

(MARKDOWN, """## Appendix

### Answers"""),

(CODE, '''# 1. The verdict is stable across grains -- but the numbers are not, which is
#    exactly why the grain is printed beside them.
for window in ("1min", "5min", "15min"):
    grouped = one.assign(w=one["_t"].dt.floor(window)).groupby("w").agg(
        mean_payload=("payload", "mean"), readings=("speed", "size")).reset_index()
    grouped["day"] = grouped["w"].dt.date.astype(str)
    # The same rule as the main table: at least half a full window of readings,
    # at two readings a second. Using a different floor here would make the
    # 5-minute row disagree with the table above for no reason.
    floor_reads = {"1min": 60, "5min": 300, "15min": 900}[window]
    grouped = grouped[grouped["readings"] >= floor_reads]
    a = grouped.loc[grouped["day"] == REFERENCE_DAY, "mean_payload"]
    b = grouped.loc[grouped["day"] == CURRENT_DAY, "mean_payload"]
    print(f"{window:>6}: {len(a):3} vs {len(b):3} windows, target shift "
          f"{(b.mean() - a.mean()) / a.std(ddof=1):+.2f} SD")

# 2. The target's distance sits inside the null distribution -- it is noise.
rng = np.random.default_rng(SEED)
distances = [wasserstein_distance(payload, rng.choice(payload, size=35, replace=True))
             for _ in range(500)]
observed = wasserstein_distance(payload, current[TARGET].dropna().to_numpy())
print(f"\\nnull distances: median {np.median(distances):.1f}, "
      f"95th percentile {np.quantile(distances, 0.95):.1f} kg")
print(f"observed target distance: {observed:.1f} kg -> "
      f"{'inside the noise' if observed < np.quantile(distances, 0.95) else 'outside'}")

# 3. How small a shift the control still catches -- and the sweep above already
#    answered it: the limit is the smallest size from which the answer STAYS
#    material, not the first size at which it fires.
print()
for size in (1.5, 1.0, 0.5, 0.25):
    row = inject(size)
    print(f"injected {size:>4} SD -> index {row['index']:7.3f}   material {row['material']}")
print(f"\\nand the sustained limit, off the sweep: {limit} reference SD "
      f"= {limit * spread:.1f} kilograms per window")'''),

(MARKDOWN, """## References

- Jensen, J. L. W. V. (1906). *Sur les fonctions convexes et les inégalités entre les valeurs moyennes.* Acta Mathematica 30, 175–193. https://doi.org/10.1007/BF02418571
- Wilson, E. B. (1927). *Probable inference, the law of succession, and statistical inference.* Journal of the American Statistical Association 22(158), 209–212. https://doi.org/10.1080/01621459.1927.10502953
- Kantorovich, L. V. (1942). *On the translocation of masses.* Doklady Akademii Nauk SSSR 37(7–8), 227–229; English reprint, Management Science 5(1), 1958, 1–4. https://doi.org/10.1287/mnsc.5.1.1
- Bayley, G. V. & Hammersley, J. M. (1946). *The "effective" number of independent observations in an autocorrelated time series.* Supplement to the Journal of the Royal Statistical Society 8(2), 184–197. https://doi.org/10.2307/2983560
- Jeffreys, H. (1946). *An invariant form for the prior probability in estimation problems.* Proceedings of the Royal Society A 186, 453–461. https://doi.org/10.1098/rspa.1946.0056
- Welch, B. L. (1947). *The generalization of "Student's" problem when several different population variances are involved.* Biometrika 34(1/2), 28–35. https://doi.org/10.1093/biomet/34.1-2.28
- Shannon, C. E. (1948). *A Mathematical Theory of Communication.* Bell System Technical Journal 27(3), 379–423. https://doi.org/10.1002/j.1538-7305.1948.tb01338.x
- Kullback, S. & Leibler, R. A. (1951). *On Information and Sufficiency.* Annals of Mathematical Statistics 22(1), 79–86. https://doi.org/10.1214/aoms/1177729694
- Page, E. S. (1954). *Continuous Inspection Schemes.* Biometrika 41(1/2), 100–115. https://doi.org/10.1093/biomet/41.1-2.100
- Vallender, S. S. (1974). *Calculation of the Wasserstein distance between probability distributions on the line.* Theory of Probability and Its Applications 18(4), 784–786. https://doi.org/10.1137/1118101
- Glass, G. V. (1976). *Primary, secondary, and meta-analysis of research.* Educational Researcher 5(10), 3–8. https://doi.org/10.3102/0013189X005010003
- Efron, B. (1979). *Bootstrap methods: another look at the jackknife.* Annals of Statistics 7(1), 1–26. https://doi.org/10.1214/aos/1176344552
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed. Lawrence Erlbaum.
- Currie, L. A. (1968). *Limits for qualitative detection and quantitative determination.* Analytical Chemistry 40(3), 586–593 — the detection limit, and the discipline of never reporting "not detected" without it. https://doi.org/10.1021/ac60259a007
- Lewis, E. M. (1994). *An Introduction to Credit Scoring.* Athena Press — the origin of the 0.1 and 0.25 rule of thumb.
- Benjamini, Y. & Hochberg, Y. (1995). *Controlling the False Discovery Rate.* Journal of the Royal Statistical Society B 57(1), 289–300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
- Agresti, A. & Coull, B. A. (1998). *Approximate is better than "exact" for interval estimation of binomial proportions.* The American Statistician 52(2), 119–126. https://doi.org/10.1080/00031305.1998.10480550
- Brown, L. D., Cai, T. T. & DasGupta, A. (2001). *Interval Estimation for a Binomial Proportion.* Statistical Science 16(2), 101–133. https://doi.org/10.1214/ss/1009213286
- MacKay, D. J. C. (2003). *Information Theory, Inference, and Learning Algorithms*, §2.6. Cambridge University Press. https://www.inference.org.uk/itprnn/book.pdf
- Wasserman, L. (2004). *All of Statistics.* Springer — ch. 5 for the central limit theorem, Theorem 4.9 for Jensen's inequality.
- Cover, T. M. & Thomas, J. A. (2006). *Elements of Information Theory*, 2nd ed., ch. 2. Wiley — entropy and the divergence; Theorem 2.6.3 is the information inequality, and it defines neither the cross-entropy nor "Gibbs' inequality" by name. https://doi.org/10.1002/047174882X
- Siddiqi, N. (2006). *Credit Risk Scorecards.* Wiley; and (2017) *Intelligent Credit Scoring*, 2nd ed. https://doi.org/10.1002/9781119282396
- Wasserstein, R. & Lazar, N. (2016). *The ASA Statement on p-Values.* The American Statistician 70(2), 129–133. https://doi.org/10.1080/00031305.2016.1154108
- Ramdas, A., García Trillos, N. & Cuturi, M. (2017). *On Wasserstein Two-Sample Testing and Related Families of Nonparametric Tests.* Entropy 19(2), 47 — the two-sample test, not the closed form. https://doi.org/10.3390/e19020047
- Peyré, G. & Cuturi, M. (2019). *Computational Optimal Transport.* Foundations and Trends in Machine Learning 11(5–6), 355–607 — Remark 2.30 and Remark 2.28. https://doi.org/10.1561/2200000073
- Rabanser, S., Günnemann, S. & Lipton, Z. (2019). *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift.* NeurIPS 32. https://arxiv.org/abs/1810.11953
- Saltelli, A. et al. (2019). *Why so many published sensitivity analyses are false.* Environmental Modelling and Software 114, 29–39. https://doi.org/10.1016/j.envsoft.2019.01.012
- Truong, C., Oudre, L. & Vayatis, N. (2020). *Selective review of offline change point detection methods.* Signal Processing 167, 107299. https://doi.org/10.1016/j.sigpro.2019.107299
- Yurdakul, B. & Naranjo, J. (2020). *Statistical properties of the population stability index.* Journal of Risk Model Validation 14(4), 89–100. https://doi.org/10.21314/JRMV.2020.227
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*, §6.1.2. MIT Press. https://probml.github.io/pml-book/book1.html
- Regulation (EU) 2024/1689 of the European Parliament and of the Council laying down harmonised rules on artificial intelligence (Artificial Intelligence Act), Article 15. https://eur-lex.europa.eu/eli/reg/2024/1689/oj

*All output above is Author's own, computed from
`Module 4/exercises/data/bus_slice.csv.gz` — the committed extract of
`data/bus.csv`, vehicle VJRD1A10224000055 on 22–23 January 2020, vehicle data
only and no personal data — by this notebook, on the grain printed at the top.
The figures are plotly and are also saved under `notebook/figures/`. The lag-1
autocorrelation of 0.997 is read from Module 1 rather than re-measured, and the
two closed-form divergences quoted in the third contrast are computed in
`Module 4/slides/make_figs.py`.*"""),
]


def main(*arguments):
    notebook = new_notebook(cells=[
        new_markdown_cell(text) if kind == MARKDOWN else new_code_cell(text)
        for kind, text in CELLS])
    notebook.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}})

    if "--no-run" not in arguments:
        from nbclient import NotebookClient
        # Executed from exercises/, so `data/bus_slice.csv.gz` resolves exactly as
        # it does for the labs sitting next to it.
        NotebookClient(notebook, timeout=1800,
                       resources={"metadata": {"path": str(EXERCISES)}}).execute()

    OUTPUT.write_text(nbformat.writes(notebook))
    executed = sum(1 for cell in notebook.cells if cell.get("outputs"))
    print(f"wrote {OUTPUT.name} — {len(CELLS)} cells, {executed} with output")


if __name__ == "__main__":
    main(*sys.argv[1:])
