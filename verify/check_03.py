#!/usr/bin/env python3
"""Check 3 — the distance, against scipy, and the two cases that separate the four."""
import math
import re
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _harness import run, close, not_ready                           # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
try:
    import numpy as np                                               # noqa: E402
    from scipy.stats import wasserstein_distance                     # noqa: E402
    from lab_support import load_lab, reference_and_current          # noqa: E402
except ImportError as unready:                                       # noqa: E402
    not_ready(unready)

MEASURES = ("cross_entropy", "kl_divergence", "population_stability_index",
            "wasserstein")


def sample_at(centres, shares, size=2000):
    """A sample whose share in each bin is the one asked for."""
    return np.repeat(centres, np.round(np.asarray(shares) * size).astype(int))


def refuse_the_library(written):
    """A lab that imports the library it is graded against is graded against itself.

    scipy.stats.wasserstein_distance is exactly what this check compares you
    with. Calling it here would make the comparison a tautology, and the point of
    the lab is that this measure is one sort and one loop rather than a magic
    import.
    """
    source = pathlib.Path(written.__file__).read_text()
    assert not re.search(r"^\s*(?:from|import)\s+scipy\b", source, re.M), (
        "labs/03_how_far.py imports scipy, which holds the very function this check "
        "compares your distance against. Write the area between the two cumulative "
        "distribution functions yourself -- it is three lines -- and keep scipy for "
        "the checking.")


def body(lab):
    refuse_the_library(lab)

    rng = np.random.default_rng(7)

    # Against the library, on samples of equal and unequal length.
    cases = [
        (rng.normal(0, 1, 500), rng.normal(0, 1, 500)),
        (rng.normal(0, 1, 500), rng.normal(2, 1, 500)),
        (rng.normal(0, 1, 400), rng.normal(0.5, 2, 900)),
        (rng.exponential(1, 600), rng.exponential(3, 600)),
    ]
    for index, (a, b) in enumerate(cases, start=1):
        close(lab.wasserstein(a, b), float(wasserstein_distance(a, b)), 5e-3,
              f"wasserstein on case {index}, against scipy "
              f"({len(a)} against {len(b)} observations)")

    close(lab.wasserstein([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 0.0, 1e-9,
          "a sample against itself is zero distance")

    # It must be in the variable's own units: shift everything by a constant and
    # the distance is that constant.
    base = rng.normal(10, 2, 800)
    close(lab.wasserstein(base, base + 5.0), 5.0, 1e-2,
          "shifting a sample by 5 must give a distance of 5 — the whole point is that "
          "the answer is in the variable's own units")

    assert lab.wasserstein(base, base + 5.0) >= 0, "a distance is never negative"
    close(lab.wasserstein(base, base + 5.0), lab.wasserstein(base + 5.0, base), 1e-6,
          "the distance is symmetric")

    # All four together, on the real archive.
    reference, current = reference_and_current()
    speed_before = reference["mean_speed"].dropna()
    speed_after = current["mean_speed"].dropna()
    speed = lab.compare_four(speed_before, speed_after)
    for key in MEASURES:
        assert key in speed, (
            f"compare_four() returned no '{key}'. All four measures on the same "
            f"pair: {', '.join(MEASURES)}")

    close(speed["wasserstein"], float(wasserstein_distance(speed_before, speed_after)),
          5e-3, "compare_four → wasserstein on mean speed")
    close(speed["population_stability_index"],
          load_lab(2).population_stability_index(speed_before, speed_after), 1e-9,
          "compare_four → population_stability_index on mean speed, against Lab 2's "
          "own function — reuse it rather than writing the index a second time")
    assert speed["cross_entropy"] >= speed["kl_divergence"], (
        f"your cross-entropy is {speed['cross_entropy']:.3f} and your divergence is "
        f"{speed['kl_divergence']:.3f}. The first is the second plus today's own "
        "entropy, which is never negative, so it cannot be the smaller of the two.")

    # No overlap. Yesterday between 0 and 2 metres per second, today between 5
    # and 7: the divergence is infinite whether the gap is five or five hundred,
    # and the distance is the gap.
    yesterday = np.random.default_rng(20200122).uniform(0.0, 2.0, 400)
    today = yesterday + 5.0
    disjoint = lab.compare_four(yesterday, today)
    assert math.isinf(disjoint["kl_divergence"]), (
        f"on two samples with no overlap your divergence came out "
        f"{disjoint['kl_divergence']:.3f}. Today's values all sit where the reference "
        "had no mass at all, so the ratio is unbounded and the honest answer is "
        "infinity. If yours is finite, your bins are wide enough to hold both "
        "samples in one — which is precisely the binning that hides a new value.")
    close(disjoint["wasserstein"], 5.0, 1e-9, (
        "on those same two samples the distance must be the gap, 5 metres per "
        "second. That is the difference between the two measures: the divergence "
        "cannot tell a gap of five from a gap of five hundred, and this can"))

    # Shuffled bins. One pair of distributions, and the bins relabelled so that
    # every share keeps its partner and only the value each bin stands for
    # changes. The divergence sums over pairs and cannot notice; the distance is
    # built on the values and notices completely.
    centres = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    reference_shares = np.array([0.40, 0.30, 0.20, 0.07, 0.03])
    current_shares = np.array([0.03, 0.07, 0.20, 0.30, 0.40])
    order = np.argsort(np.random.default_rng(20200122).permutation(len(centres)))

    before = lab.compare_four(sample_at(centres, reference_shares),
                              sample_at(centres, current_shares))
    after = lab.compare_four(sample_at(centres, reference_shares[order]),
                             sample_at(centres, current_shares[order]))

    close(after["kl_divergence"], before["kl_divergence"], 1e-9, (
        "relabelling which bin is which left every share with its partner, so the "
        "divergence must not move at all — it never knew the values were ordered"))
    moved = abs(after["wasserstein"] - before["wasserstein"])
    assert moved > 0.2 * before["wasserstein"], (
        f"the same relabelling moved your distance from {before['wasserstein']:.3f} "
        f"to {after['wasserstein']:.3f}, a change of {moved:.3f}. It should move a "
        "great deal: the mass now sits at different values, and the distance is "
        "built on the geometry the divergence cannot see.")


run(3, "03_how_far", "wasserstein", body,
    requires=[(2, lambda lab: lab.kl_divergence([0.5, 0.5], [0.5, 0.5]))])
