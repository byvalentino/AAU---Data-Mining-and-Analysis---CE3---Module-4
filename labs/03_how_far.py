"""Lab 3 — How far, in the variable's own units.

Why this lab exists: a divergence of 0.03 cannot be acted on, because it is
measured in nothing; an operations manager needs "the speed distribution moved by
half a metre per second". You write the one measure that answers in the
variable's own units, and you run all four on the same pair so that you can say
which question each one answers.
Where it sits: Block three — "A divergence of 0.03 is unactionable", and the
definition slides "Definition — the Wasserstein-1 distance" and "Definition — the
two binnings inside compare_four".
What the check grades: your distance against scipy on four cases including
samples of different lengths, exactly five when a sample is shifted by five,
symmetry and nought against itself; and compare_four returning all four measures,
its index equal to Lab 2's on the same pair, its divergence infinite on two
samples that do not overlap, and its distance moving when the bins are relabelled
while the divergence does not.
Needs: numpy, and lab_support.load_lab to reuse Lab 2.

Twenty-five minutes.

A divergence of 0.03 is unactionable. Report it to an operations manager and the
honest reply is "is that a lot?" -- and you cannot answer, because it is measured
in nothing.

The **Wasserstein distance** answers in the variable's own units. Its picture is
the reason it is worth knowing: think of each distribution as a pile of earth,
and ask how much work it takes to move one pile into the shape of the other,
where work is mass times the distance it travels. The answer comes out in metres
per second, or kilograms, or whatever the variable was measured in.

For one dimension it has a closed form that is easier than the picture suggests:

    W(a, b) = the area between the two cumulative distribution functions

and with two samples of the same size, sorted, it is simply

    mean( |a_sorted − b_sorted| )

which is three lines of code and no optimisation at all, despite the
optimal-transport pedigree (Ramdas et al., 2017).

Then the comparison, which is the examinable part. Four measures, one pair of
samples, and each answers a different question:

    cross-entropy   what today cost you, given what you believed
    divergence      how much of that cost was avoidable -- the excess
    index           the same, symmetrised, so no reference order is needed
    distance        how far the world moved, in units somebody can act on

The first three need the data binned; the last does not, and that is why it
alone can see that the values are ordered.

What you write: wasserstein(a, b), and compare_four(reference, current) which
runs all four on the same pair and returns them together.
"""
from __future__ import annotations

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from lab_support import NotSolved, load_lab  # noqa: E402

LAB = 3
# The same five Lab 2 settles on, and for the same reason: on this many windows,
# ten bins put a no-change comparison above the threshold Lab 4 judges against.
DEFAULT_BINS = 5


def wasserstein(a, b) -> float:
    """The distance between two samples, in the units they were measured in.

    Samples may be different lengths. The general one-dimensional form is the
    area between the two cumulative distribution functions; a clean way to get
    it is to take every value either sample holds as a breakpoint, read both
    cumulative shares between consecutive breakpoints, and add up the gaps times
    the widths. Evaluating both quantile functions on a fine common grid of
    probabilities and averaging the absolute difference gets you the same number.

    One property is the whole point, and the check tests it: add five to every
    value in one sample and the distance is five.

    Definition graded by the check:
        W₁(P,Q) = ∫ |F_P(x) − F_Q(x)| dx = ∫₀¹ |F_P⁻¹(u) − F_Q⁻¹(u)| du
        (Vallender, 1974; Peyré & Cuturi, 2019, Remark 2.30). Choices: no binning
        at all, so the answer is in the variable's own units; for two sorted
        samples of equal size it is the mean of |a_(i) − b_(i)| (Remark 2.28).
        Slide: "Definition — the Wasserstein-1 distance".
    Needs: numpy
    """
    # TODO: the area between the two cumulative distribution functions.
    raise NotSolved("wasserstein(a, b) still raises instead of returning a distance")


def compare_four(reference, current, bins: int = DEFAULT_BINS) -> dict:
    """All four measures on the same pair of samples.

    Returns:
        {"cross_entropy": float,                # in nats
         "kl_divergence": float,                # in nats, and sometimes infinite
         "population_stability_index": float,   # in nats, symmetric
         "wasserstein": float}                  # in the variable's own units

    Import the first three from Lab 2 rather than writing them twice:

        from lab_support import load_lab
        lab2 = load_lab(2)

    Two different binnings, and each has a reason you should be able to give:

        the index         edges from the **reference's** quantiles, opened at
                          both ends. A monitor's yardstick is cut from the
                          reference and must not move, which is Lab 2's rule and
                          the one that reaches the verdict in Lab 4.

        the cross-entropy equal-width edges spanning **both** samples, from
        and divergence    numpy.histogram_bin_edges on the two concatenated.
                          Here you are comparing two samples in front of you
                          rather than watching one against a fixed past, so the
                          bins cover everything either sample holds. A value the
                          reference never held then falls in a bin whose
                          reference share is nought, and the divergence answers
                          "infinite" instead of hiding it inside an existing bin.

    The second binning is why this function can tell you something the index
    cannot: the index floors empty bins and stays finite, by design.

    Definition graded by the check:
        H and D: equal-width edges over both samples · J: the reference's quantile edges, opened at both ends
        (Siddiqi, 2006; Yurdakul & Naranjo, 2020). Choices: numpy's
        histogram_bin_edges over the two samples concatenated for the entropy and
        the divergence, and Lab 2's own index — reused rather than rewritten —
        for the third. Slide: "Definition — the two binnings inside compare_four".
    Needs: numpy, lab_support.load_lab
    """
    # TODO: four measures, one dictionary.
    raise NotSolved("compare_four(reference, current) still raises instead of "
                    "returning four measures")


if __name__ == "__main__":
    from lab_support import reference_and_current

    reference, current = reference_and_current()
    for feature in ("mean_speed", "mean_payload"):
        measures = compare_four(reference[feature].dropna(), current[feature].dropna())
        print(f"{feature:14}", {k: round(v, 3) for k, v in measures.items()})

    print("\nsame sample, shifted by five:")
    base = reference["mean_payload"].dropna().to_numpy()
    print("  distance:", round(wasserstein(base, base + 5.0), 6))
