"""Lab 1 — How sure are you?

Why this lab exists: from tomorrow nobody labels anything, so accuracy can no
longer be computed — it can only be bought, by having somebody check a sample by
hand, and the first question you will be asked is how many. You prove here that
the interval everybody writes fails exactly where a monitor lives, at the edges,
and that the interval that holds up there costs one more line of code.
Where it sits: Block one — "The interval everybody writes, and where it fails",
and the definition slides "Definition — the Wald interval", "Definition — the
Wilson score interval" and "Definition — coverage, and what a half-width costs".
What the check grades: your Wilson interval against an independent implementation
of the published formula on five cases; your Wald interval against its own
formula, including the zero-width collapse on forty out of forty; a coverage
experiment that puts the Wald interval far below its promise near a true rate of
0.02 and Wilson near 0.95; and the label count at three half-widths, one of which
this file never quotes.
Needs: math, numpy.

Twenty-five minutes.

From tomorrow there are no labels. The only way to know whether the service is
still right is to buy the truth: somebody checks a sample by hand. So the
question becomes arithmetic — how many do they have to check?

You hand-check forty predictions and thirty-four are right. Is the true accuracy
85 per cent? No. It is somewhere, and the interval says where.

The obvious interval is the one everybody writes:

    p ± 1.96 · sqrt( p(1-p) / n )

It is wrong in a way that matters. Near nought or one it is too narrow, and at
the edges it collapses entirely: check forty and get forty right, and it reports
an interval of zero width around 1.0 — perfect certainty from forty
observations. You will measure how often it actually contains the truth, and the
answer near the edge is nothing like the 95 per cent it promises.

The Wilson interval fixes it by asking a different question: not "what is the
error around my estimate?" but "which true rates would plausibly have produced
what I saw?" (Wilson, 1927).

    centre = (k + z²/2) / (n + z²)
    half   = z / (n + z²) · sqrt( k(n-k)/n + z²/4 )

    interval = centre ± half

which is the definition slide's single expression, multiplied out.

What you write: wilson_interval(successes, trials, z) and naive_interval(...),
then coverage(...) which measures how often each contains a known truth.

And answer one thing in labels_needed(): to halve the width of an interval, how
many more labels do you need? The square root in the formula is the whole
answer, and it is why bought truth gets expensive fast.
"""
from __future__ import annotations

import math
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from lab_support import NotSolved  # noqa: E402

LAB = 1
Z_95 = 1.959963984540054


def naive_interval(successes: int, trials: int, z: float = Z_95):
    """The interval everybody writes. Return (low, high).

    Definition graded by the check:
        p̂ ± z·√( p̂(1−p̂)/n ), with p̂ = k/n successes in n trials
        (Brown, Cai & DasGupta, 2001; Agresti & Coull, 1998). Choices: the
        ninety-five per cent level, so z = 1.96, and the normal approximation to
        the binomial law. Slide: "Definition — the Wald interval".
    Needs: math
    """
    # TODO: the observed share, plus and minus z times its standard error.
    raise NotSolved("naive_interval(successes, trials, z) still raises instead of "
                    "returning (low, high)")


def wilson_interval(successes: int, trials: int, z: float = Z_95):
    """The interval that holds up near the edges. Return (low, high).

    Definition graded by the check:
        ( p̂ + z²/2n ± z·√( p̂(1−p̂)/n + z²/4n² ) ) / ( 1 + z²/n )
        (Wilson, 1927; Brown, Cai & DasGupta, 2001). Choices: the same level and
        the same approximation; what changes is the question — which true rates
        could have produced what was seen. Slide: "Definition — the Wilson score
        interval".
    Needs: math
    """
    # TODO: the centre and half-width in the module docstring, which are that
    # single expression multiplied out.
    raise NotSolved("wilson_interval(successes, trials, z) still raises instead of "
                    "returning (low, high)")


def coverage(interval, true_rate: float, trials: int, repeats: int = 4000,
             seed: int = 20200122) -> float:
    """How often `interval` contains `true_rate`, over `repeats` samples.

    Draw `repeats` samples of `trials` observations at `true_rate`, build the
    interval from each, and return the share that contained the truth. A 95 per
    cent interval should return about 0.95.

    Definition graded by the check:
        coverage(p, n) = (1/R)·Σ_{r=1}^{R} 1{ low_r ≤ p ≤ high_r }
        (Brown, Cai & DasGupta, 2001; Agresti & Coull, 1998). Choices: R = 4000
        samples of n = 40 at each true rate, and numpy's default_rng(20200122),
        so the answer is the same every time. Slide: "Definition — coverage, and
        what a half-width costs".
    Needs: numpy, rng.binomial
    """
    # TODO: simulate, build the interval from each sample, count.
    raise NotSolved("coverage(interval, true_rate, trials) still raises instead of "
                    "returning a share")


def labels_needed(half_width: float) -> int:
    """How many labels for a 95 per cent interval of this half-width?

    Compute it for 0.05 and for 0.025 and look at the ratio. That ratio is why
    buying truth gets expensive faster than anybody expects.

    Definition graded by the check:
        n = ⌈ 0.25·(z/h)² ⌉ at the worst case p = ½
        (Brown, Cai & DasGupta, 2001). Choices: the worst case p = ½, where
        p(1−p) is largest; the normal approximation, so the count is itself an
        approximation; and rounding up, because labels come whole. Slide:
        "Definition — coverage, and what a half-width costs".
    Needs: math
    """
    # TODO: rearrange half = z·√(0.25/n) for n, and round up.
    raise NotSolved("labels_needed(half_width) still raises instead of returning a count")


if __name__ == "__main__":
    print("34 of 40 right")
    print("  naive :", naive_interval(34, 40))
    print("  Wilson:", wilson_interval(34, 40))
    print("\n40 of 40 right")
    print("  naive :", naive_interval(40, 40))
    print("  Wilson:", wilson_interval(40, 40))
    print("\ncoverage at a true rate of 0.02, 40 trials")
    print("  naive :", coverage(naive_interval, 0.02, 40))
    print("  Wilson:", coverage(wilson_interval, 0.02, 40))
    print("\nlabels for a half-width of 0.05:", labels_needed(0.05))
