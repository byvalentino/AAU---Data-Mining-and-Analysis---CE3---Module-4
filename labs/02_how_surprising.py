"""Lab 2 — Surprise, and the gap between two averages of it.

Why this lab exists: "the data has changed" is a claim, and a claim needs a
number with a fixed floor, so that "nothing happened" is sayable. You build the
four small objects that make it sayable, and you find out by measurement that the
threshold everybody quotes for the fourth is a statement about somebody else's
sample size rather than about your data.
Where it sits: Block two — "Surprise, and the two ways to average it", and the
definition slides "Definition — entropy and cross-entropy", "Definition — the
Kullback–Leibler divergence", "Definition — the symmetrised index" and
"Definition — the index's noise floor".
What the check grades: your cross-entropy minus your entropy must equal your
divergence to ten decimal places on five pairs; all three against scipy; the
divergence infinite where the reference gave no mass; the index built on the
reference's own quantile edges, nought against itself, above 0.25 on mean speed,
and refusing the degenerate column; and a default bin count low enough that a
no-change comparison reads as no change.
Needs: numpy, and the constants in lab_support.

Twenty-five minutes.

"The data has changed" is a claim, and a claim needs a number.

Start with surprise. If an outcome has probability p, observing it carries
−log(p) of surprise. A certainty surprises you not at all; a one-in-a-thousand
event surprises you a great deal. The logarithm is there so that surprises add:
two independent events surprise you by the sum of their surprises. Everything
here uses the natural logarithm, so the unit is the **nat**; divide by log(2)
for bits.

Now average that surprise, and notice that there are two ways to do it.

    entropy         H(P)    = − Σ P(i) · log( P(i) )

        Today's surprise averaged under **today's own** distribution: how
        varied today was, and nothing else.

    cross-entropy   H(P, Q) = − Σ P(i) · log( Q(i) )

        Today's surprise averaged under **yesterday's** distribution: what
        today cost you, given what you believed.

Subtract them and you have the third object, the one that monitors change
(Kullback & Leibler, 1951):

    divergence      D(P ‖ Q) = Σ P(i) · log( P(i) / Q(i) ) = H(P, Q) − H(P)

That identity is not decoration. It is the first test the check runs, because
writing the cross-entropy where the divergence belongs is the commonest error in
this material -- this course's own first draft made it -- and the two differ by
an entropy term that moves.

Three properties you must feel rather than memorise:

  never negative   The best you can do is nought, when the distributions match.
                   Believing the wrong distribution can never cost you less than
                   believing the right one.

  asymmetric       D(P‖Q) is not D(Q‖P). It matters which one you call the
                   reference, and monitoring systems that forget this compare
                   today against yesterday one week and yesterday against today
                   the next, and wonder why the number jumped.

  infinite         If today contains a value the reference never contained,
                   Q(i) is nought, the ratio is unbounded, and so is the answer.
                   Not a bug -- it is the correct reply to "how surprising is
                   something you said was impossible?" But it is useless on a
                   dashboard, so people bin the data, and the binning quietly
                   hides the very thing the measure was trying to tell them.

The fourth object takes the divergence both ways and adds them (Jeffreys, 1946).
Credit scoring calls it the **Population Stability Index**:

    index = Σ ( P(i) − Q(i) ) · log( P(i) / Q(i) )

which is D(P‖Q) + D(Q‖P), and therefore symmetric. Its conventional thresholds —
below 0.1 no material change, 0.1 to 0.25 worth investigating, above 0.25 a
material shift — are convention, not derivation, and they were established on
populations of many thousands. You have about forty windows a day.

What you write: entropy(p), cross_entropy(p, q), kl_divergence(p, q) and
population_stability_index(reference, current, bins).
"""
from __future__ import annotations

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from lab_support import (BORROWED_INDEX, DegenerateReference,   # noqa: E402
                         MINIMUM_EDGES, NotSolved, NULL_QUANTILE, NULL_RESAMPLES,
                         PSI_EPSILON, SEED)

LAB = 2
# Five, and the choice is printed here because the number depends on it. On the
# forty-odd windows this module works with, ten bins put the index of the reference
# against a resample of *itself* -- no change whatsoever -- above banking's 0.25
# threshold, so every feature reads as material and the measure says nothing at all.
# Block 2 measures that floor rather than asserting it.
DEFAULT_BINS = 5


def entropy(p) -> float:
    """H(P) in nats, over one probability vector.

    Terms where P(i) is nought contribute nothing: as P(i) approaches nought so
    does P(i)·log(P(i)), so dropping them is the limit rather than a convenience.

    Definition graded by the check:
        H(P) = −Σ_i P(i)·log P(i)
        (Shannon, 1948; Murphy, 2022, §6.1.2). Choices: the natural logarithm, so
        the unit is the nat, and the limit convention for P(i) = 0. Slide:
        "Definition — entropy and cross-entropy".
    Needs: numpy
    """
    # TODO: -sum(p * log(p)), skipping the zeros.
    raise NotSolved("entropy(p) still raises instead of returning a number")


def cross_entropy(p, q) -> float:
    """H(P, Q) in nats: today's surprise, averaged under yesterday's beliefs.

    The same sum as the entropy with one substitution -- the logarithm reads Q
    where the entropy reads P. That single substitution is the whole difference
    between "how varied was today" and "what did today cost me".

    Definition graded by the check:
        H(P,Q) = −Σ_i P(i)·log Q(i)
        (Murphy, 2022, §6.1.2; Shannon, 1948). Choices: the natural logarithm and
        the same limit convention; a zero in Q where P has mass makes it
        infinite. Slide: "Definition — entropy and cross-entropy".
    Needs: numpy
    """
    # TODO: -sum(p * log(q)), skipping the zeros in p.
    raise NotSolved("cross_entropy(p, q) still raises instead of returning a number")


def kl_divergence(p, q) -> float:
    """D(P || Q) in nats, over two probability vectors of the same length.

    Use the natural logarithm, so the check can compare you against scipy's
    entropy(). Terms where P(i) is nought contribute nothing; a zero in Q where P
    has mass makes the answer infinite, and Python can say that.

    Whatever you write here, cross_entropy(p, q) minus entropy(p) must equal it.
    The check requires that to ten decimal places on five pairs.

    Definition graded by the check:
        D(P ‖ Q) = Σ_i P(i)·log( P(i) / Q(i) ) = H(P,Q) − H(P), in nats
        (Kullback & Leibler, 1951; MacKay, 2003, §2.6). Choices: the natural
        logarithm; nought where the two match; infinite where Q gives no mass to
        something P does. Slide: "Definition — the Kullback–Leibler divergence".

    It is never negative, and that is Jensen's inequality applied to −log:
        f( E[X] ) ≤ E[ f(X) ] for convex f, with equality only where f is straight or X never varies
        (Jensen, 1906; Wasserman, 2004, Theorem 4.9). The check requires
        non-negativity on every pair, both ways round. Slide: "Definition —
        Jensen's inequality".
    Needs: numpy
    """
    # TODO: sum p * log(p / q), skipping the zeros in p.
    raise NotSolved("kl_divergence(p, q) still raises instead of returning a number")


def population_stability_index(reference, current, bins: int = DEFAULT_BINS) -> float:
    """The symmetric index, over two samples of raw values.

    Three decisions, and every one of them decides the answer:

    1. **The edges come from the reference**, not from today -- quantiles of the
       reference are the usual choice, so each reference bin holds about the same
       share. Take today's quantiles instead and the yardstick moves every time
       you measure, so a stable world produces a wandering index. Open the outer
       two edges to -inf and +inf, so that today cannot fall off the end of
       yesterday's range.

    2. **A floor under empty bins**, PSI_EPSILON, or one empty bin makes the
       whole index infinite. The floor buys a usable number and costs you the
       signal that a new value appeared, so watch for new values separately.

    3. **A refusal.** Count the edges that survive `numpy.unique`. If fewer than
       MINIMUM_EDGES remain there are not two bins to compare, so raise
       DegenerateReference. Returning 0.0 would say "no change" where the truth
       is "no measurement", and those two are opposites.

    Definition graded by the check:
        J(P,Q) = Σ_i ( P(i) − Q(i) )·log( P(i) / Q(i) ) = D(P‖Q) + D(Q‖P)
        (Jeffreys, 1946; Yurdakul & Naranjo, 2020). Choices: bin edges from the
        reference's own quantiles, opened at both ends; a floor of PSI_EPSILON
        under every share; a refusal when fewer than MINIMUM_EDGES survive.
        Slide: "Definition — the symmetrised index".

    What this reads when nothing has changed at all is the next function's
    business, and the threshold this module judges by is derived from it rather
    than taken from a handbook.
    Needs: numpy
    """
    # TODO: refuse a degenerate reference, then bin on reference quantiles and
    # sum (p - q) * log(p / q).
    raise NotSolved("population_stability_index(reference, current, bins) still raises "
                    "instead of returning a number")


def index_threshold(reference, current, bins: int = DEFAULT_BINS,
                    resamples: int = NULL_RESAMPLES,
                    quantile: float = NULL_QUANTILE, seed: int = SEED) -> dict:
    """The floor the index reads when nothing changed, and the threshold derived from it.

    This is the function that stops this module borrowing a number. Credit
    scoring calls anything above 0.25 a material shift. That figure was settled
    on populations of many thousands; you have about forty windows a day, and on
    forty windows the index reads something well above nought even when the two
    samples come from the identical world.

    So measure that. Compare the reference against a resample of **itself** --
    no change whatsoever -- `resamples` times, and you have the whole
    distribution of the index under "nothing happened". Two numbers come out of
    it:

        noise_floor   its median. Below this the index is measuring your sample
                      size, not the world.
        threshold     its `quantile` point. Above this the index reaches only
                      once in every 1/(1 - quantile) comparisons of a world that
                      did not change, which is a stated false-alarm rate rather
                      than a convention.

    Print all four choices beside whatever you report -- the bin count, the
    number of resamples, the quantile and the seed -- because every one of them
    moves the answer. That is standing rule 2, and it is the whole difference
    between a threshold and a habit.

    Return a dictionary holding at minimum "noise_floor", "threshold", "bins",
    "resamples", "quantile" and "seed".

    Definition graded by the check:
        floor(B) = median over R resamples of J( reference, resample of the reference ) at B bins
        (Yurdakul & Naranjo, 2020). Choices: resamples drawn with replacement
        from the reference, each the size of the current sample, so the null
        carries the sample sizes the comparison will really have; the median
        rather than the mean, because the null is skewed; the windows are
        resampled as if they were exchangeable, which is optimistic if they
        are autocorrelated at the window grain the way the underlying readings
        are at the reading grain -- printed here because it is unmeasured, not
        because it is believed to be zero. Slide: "Definition — the index's
        noise floor".

    And the threshold the verdict is read against, derived from that same null
    rather than borrowed from anybody:
        threshold(B, q) = Quantile_q { J( reference, resample of the reference ) at B bins }, over the same R resamples
        (Yurdakul & Naranjo, 2020). Choices: q = NULL_QUANTILE, R = NULL_RESAMPLES,
        the seed in the signature, and the bin count you are comparing at — a
        threshold derived at one bin count says nothing at another. Slide:
        "Definition — the materiality threshold, derived from the floor".
    Needs: numpy, and the constants in lab_support
    """
    # TODO: build the null distribution by resampling the reference against
    # itself, then read the median and the quantile off it.
    raise NotSolved("index_threshold(reference, current, ...) still raises instead of "
                    "returning the measured floor and the threshold derived from it")


if __name__ == "__main__":
    from lab_support import reference_and_current

    print("identical distributions ->", kl_divergence([0.5, 0.5], [0.5, 0.5]))
    print("asymmetry:", kl_divergence([0.9, 0.1], [0.5, 0.5]),
          "against", kl_divergence([0.5, 0.5], [0.9, 0.1]))
    print("today [0.9, 0.1] believing yesterday [0.5, 0.5]:")
    print("  entropy      ", entropy([0.9, 0.1]))
    print("  cross-entropy", cross_entropy([0.9, 0.1], [0.5, 0.5]))
    print("  divergence   ", kl_divergence([0.9, 0.1], [0.5, 0.5]))
    print("unseen value ->", kl_divergence([0.5, 0.5], [1.0, 0.0]))

    reference, current = reference_and_current()
    for feature in ("mean_speed", "mean_payload", "human_driven"):
        try:
            index = population_stability_index(reference[feature].dropna(),
                                               current[feature].dropna())
            print(f"{feature:14} population stability index {index:.3f}")
        except DegenerateReference as refused:
            print(f"{feature:14} unmeasured — {refused}")
