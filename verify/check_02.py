#!/usr/bin/env python3
"""Check 2 — the four objects, the identity that binds them, and the refusal."""
import math
import re
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _harness import run, close, not_ready                           # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
try:
    import numpy as np                                               # noqa: E402
    from scipy.stats import entropy as scipy_entropy                 # noqa: E402
    from lab_support import (BORROWED_INDEX, NotSolved,              # noqa: E402
                             NULL_QUANTILE, NULL_RESAMPLES, PSI_EPSILON,
                             reference_and_current)                  # noqa: E402
except ImportError as unready:                                       # noqa: E402
    not_ready(unready)

# Pairs where the reference gives every outcome some mass, so the decomposition
# is finite and can be required to close exactly. The fourth has a zero in P,
# which tests the limit convention; a zero in Q is tested separately, because
# there the right answer is an infinity rather than a number.
PAIRS = [
    ([0.5, 0.5], [0.5, 0.5]),
    ([0.9, 0.1], [0.5, 0.5]),
    ([0.5, 0.5], [0.9, 0.1]),
    ([0.2, 0.3, 0.5], [0.1, 0.6, 0.3]),
    ([0.0, 1.0], [0.4, 0.6]),
]


def refuse_the_library(written):
    """A lab that imports the library it is graded against is graded against itself.

    scipy is the independent second opinion this check compares you with, and the
    reason the labs implement rather than import is on the slide: a measure you
    cannot write is one you cannot debug at three in the morning.
    """
    source = pathlib.Path(written.__file__).read_text()
    assert not re.search(r"^\s*(?:from|import)\s+scipy\b", source, re.M), (
        "labs/02_how_surprising.py imports scipy, which is what this check compares "
        "your answers against. Comparing scipy with scipy proves nothing about your "
        "code. Write the four sums from their definitions -- each is one line -- and "
        "keep scipy for the checking.")


def body(lab):
    refuse_the_library(lab)

    # The identity first, because it is the one that catches the error this
    # material is famous for: cross-entropy written where the divergence belongs.
    # The two differ by today's entropy, which is small on tidy examples and
    # never zero on real ones, so nothing else here would notice.
    for p, q in PAIRS:
        gap = lab.cross_entropy(p, q) - lab.entropy(p)
        close(gap, lab.kl_divergence(p, q), 1e-10, (
            f"on P={p} against Q={q}, your cross_entropy minus your entropy is "
            f"{gap:.10f} and your kl_divergence is {lab.kl_divergence(p, q):.10f}. "
            "Those must be the same number to ten decimal places — the divergence "
            "IS the gap between the two averages of surprise. One of the three is "
            "not what you think it is, and the usual culprit is a cross-entropy "
            "written where the divergence belongs"))

    # Each of the three against the library, so agreement is not agreement with
    # your own arithmetic.
    for p, q in PAIRS:
        close(lab.kl_divergence(p, q), float(scipy_entropy(p, q)), 1e-9,
              f"kl_divergence({p}, {q}) against scipy.stats.entropy")
        close(lab.entropy(p), float(scipy_entropy(p)), 1e-9,
              f"entropy({p}) against scipy.stats.entropy")

    close(lab.kl_divergence([0.5, 0.5], [0.5, 0.5]), 0.0, 1e-12,
          "identical distributions must give exactly zero")
    close(lab.entropy([1.0, 0.0]), 0.0, 1e-12,
          "a certainty carries no surprise, so its entropy is exactly zero")
    close(lab.entropy([0.5, 0.5]), math.log(2), 1e-12,
          "two equally likely outcomes carry log(2) nats — one bit")

    forward = lab.kl_divergence([0.9, 0.1], [0.5, 0.5])
    backward = lab.kl_divergence([0.5, 0.5], [0.9, 0.1])
    assert abs(forward - backward) > 1e-6, (
        f"D(P||Q) = {forward:.6f} and D(Q||P) = {backward:.6f} came out equal. The "
        "divergence is asymmetric — a symmetric answer is a different object, and "
        "the symmetric one you want has its own name and its own function below.")
    assert forward >= 0 and backward >= 0, "a divergence is never negative"

    # The cross-entropy's floor is today's entropy, and it moves. That is the
    # whole reason the divergence is the thing on the monitor.
    assert lab.cross_entropy([0.9, 0.1], [0.9, 0.1]) > 0.3, (
        "the cross-entropy of a distribution against itself came out near zero. "
        "It bottoms out at that distribution's own entropy, not at nought — which "
        "is exactly why it cannot tell you that nothing changed.")

    # The infinity, and that it is deliberate.
    assert lab.kl_divergence([0.5, 0.5], [1.0, 0.0]) == float("inf"), (
        "a value the reference gave probability zero must give an infinite "
        "divergence — that is the correct answer to 'how surprising is something "
        "you said was impossible?'")

    # The index: symmetric, zero on itself, and it moves the right way.
    reference, current = reference_and_current()
    speed_before = reference["mean_speed"].dropna().to_numpy()
    speed_after = current["mean_speed"].dropna().to_numpy()

    self_index = lab.population_stability_index(speed_before, speed_before)
    close(self_index, 0.0, 1e-6, "the index of a sample against itself must be zero")

    forward = lab.population_stability_index(speed_before, speed_after)
    backward = lab.population_stability_index(speed_after, speed_before)
    assert forward >= 0, f"the index must not be negative; got {forward}"
    assert forward > 0.25, (
        f"mean speed between the two days gave an index of {forward:.3f}. This is the "
        "one feature that genuinely moved — expected well above the 0.25 threshold.")

    payload_index = lab.population_stability_index(
        reference["mean_payload"].dropna().to_numpy(),
        current["mean_payload"].dropna().to_numpy())
    assert payload_index < forward, (
        f"the target moved less than mean speed in every other measure, but your index "
        f"gives {payload_index:.3f} for the target against {forward:.3f} for speed")

    # Edges must come from the reference, not from the current sample. The check
    # computes both variants itself on a deliberately awkward pair -- a skewed
    # reference against a flat current -- where the two give clearly different
    # answers, and insists the student's matches the reference-edged one.
    def index_with_edges_from(source, reference_sample, current_sample, bins=5):
        edges = np.unique(np.quantile(source, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        share_a = np.clip(np.histogram(reference_sample, bins=edges)[0]
                          / len(reference_sample), PSI_EPSILON, None)
        share_b = np.clip(np.histogram(current_sample, bins=edges)[0]
                          / len(current_sample), PSI_EPSILON, None)
        return float(np.sum((share_b - share_a) * np.log(share_b / share_a)))

    rng = np.random.default_rng(11)
    skewed = rng.exponential(1.0, 4000)
    flat = rng.uniform(0.0, 6.0, 4000)

    from_reference = index_with_edges_from(skewed, skewed, flat)
    from_current = index_with_edges_from(flat, skewed, flat)
    assert abs(from_reference - from_current) > 0.2, (
        "the check's own fixture failed to separate the two — tell the instructor")

    student = lab.population_stability_index(skewed, flat, 5)
    assert abs(student - from_reference) < abs(student - from_current), (
        f"your index gave {student:.3f}. Edges taken from the reference give "
        f"{from_reference:.3f}; edges taken from the current sample give "
        f"{from_current:.3f}, and yours is closer to the second. Bin on the "
        "reference's quantiles — take today's and the yardstick moves every time you "
        "measure, so a stable world produces a wandering index.")

    # The noise floor, measured on the student's own implementation. With about
    # forty windows and ten bins, comparing the reference against a resample of
    # ITSELF -- no change at all -- already exceeds banking's 0.25 threshold. The
    # thresholds assume a large sample.
    payload = reference["mean_payload"].dropna().to_numpy()
    floor_rng = np.random.default_rng(0)
    null_ten = np.median([lab.population_stability_index(
        payload, floor_rng.choice(payload, size=35, replace=True), 10) for _ in range(120)])
    assert null_ten > 0.15, (
        f"resampling the reference against itself at ten bins gave a median index of "
        f"{null_ten:.3f}. On forty observations it should be substantial — that noise "
        "floor is why this module bins in five and why a borrowed threshold has to be "
        "checked against the sample size it is used on.")

    # ------------------------------------------------------------------
    # The threshold, derived rather than borrowed.
    # ------------------------------------------------------------------
    # This is the graded part of the module's own headline lesson. A student who
    # writes 0.25 here has read the slide that says not to.
    today = current["mean_payload"].dropna().to_numpy()
    derived = lab.index_threshold(payload, today)
    for key in ("noise_floor", "threshold", "bins", "resamples", "quantile", "seed"):
        assert key in derived, (
            f"index_threshold() returned no '{key}'. Every choice that moved the "
            "number is reported beside it, or the number cannot be defended: the bin "
            "count, the number of resamples, the quantile and the seed.")

    assert derived["threshold"] > derived["noise_floor"] > 0, (
        f"you report a floor of {derived['noise_floor']!r} and a threshold of "
        f"{derived['threshold']!r}. The floor is the middle of the null and the "
        "threshold is a point in its upper tail, so the threshold is strictly the "
        "larger, and both are above nought on any sample this small.")

    # The check builds the same null its own way, from its own stream, so that a
    # threshold is graded as a measurement rather than as one particular draw.
    def own_null(sample, size, bins, seed):
        stream = np.random.default_rng(seed)
        return np.array([lab.population_stability_index(
            sample, stream.choice(sample, size=size, replace=True), bins)
            for _ in range(NULL_RESAMPLES)])

    reference_null = own_null(payload, len(today), derived["bins"], 4321)
    own_floor = float(np.median(reference_null))
    own_threshold = float(np.quantile(reference_null, NULL_QUANTILE))
    assert abs(derived["noise_floor"] - own_floor) <= 0.4 * own_floor, (
        f"your noise floor is {derived['noise_floor']:.4f}; resampling the reference "
        f"against itself here gives a median of {own_floor:.4f}. The floor is the "
        "median of the index over resamples of the reference at the size of the "
        "current sample — not a constant, and not the mean of a skewed null.")
    assert abs(derived["threshold"] - own_threshold) <= 0.3 * own_threshold, (
        f"your threshold is {derived['threshold']:.4f}; the {NULL_QUANTILE} quantile of "
        f"the same null measured here is {own_threshold:.4f}. Note what this rejects: "
        f"the {BORROWED_INDEX} of credit scoring, which is what the deck spends six "
        "bullets on. The threshold has to come out of the null you measured, at the "
        "bin count and the sample sizes you are actually comparing at.")

    # A derivation answers to its inputs; a constant does not.
    assert lab.index_threshold(payload, today)["threshold"] == derived["threshold"], (
        "two identical calls to index_threshold() gave different thresholds. Seed the "
        "resampling with the seed in the signature — a threshold that moves between "
        "runs cannot be written into a report.")
    coarse = lab.index_threshold(payload, today, bins=3)["threshold"]
    fine = lab.index_threshold(payload, today, bins=10)["threshold"]
    assert coarse < derived["threshold"] < fine, (
        f"your thresholds at three, {derived['bins']} and ten bins are {coarse:.3f}, "
        f"{derived['threshold']:.3f} and {fine:.3f}. More bins on the same forty-odd "
        "windows means emptier bins and a noisier index, so the threshold has to rise "
        "with the bin count. One number that serves every bin count is the borrowed "
        "kind of number.")
    elsewhere = lab.index_threshold(
        reference["sd_speed"].dropna().to_numpy(),
        current["sd_speed"].dropna().to_numpy())["threshold"]
    assert abs(elsewhere - derived["threshold"]) > 0.02, (
        f"the threshold you derive on sd_speed ({elsewhere:.4f}) is the one you derive "
        f"on the target ({derived['threshold']:.4f}). Two columns with different shapes "
        "and different numbers of ties do not have the same null, and a threshold that "
        "does not notice the difference is a constant wearing a function's name.")

    # And the bin count itself, bound to a measurement rather than to a habit:
    # at the default the student carries into Labs 3 and 4, the detector must not
    # fire on a target that did not move. At ten bins on this archive it does.
    index_today = lab.population_stability_index(payload, today)
    assert index_today < derived["threshold"], (
        f"at your default of {getattr(lab, 'DEFAULT_BINS', 'unknown')} bins the "
        f"target's index today is {index_today:.3f} and the threshold derived from its "
        f"own null is {derived['threshold']:.3f}, so your verdict fires on a column "
        "that moved by three hundredths of a standard deviation. That is a false alarm "
        "on the null, and at ten bins on this archive it is what happens. The "
        "arithmetic is not the problem — the bin count is.")

    # The refusal. human_driven is nought in 39 of the 45 reference windows, so
    # its quantile edges collapse to one bin and every correct implementation of
    # the arithmetic returns exactly 0.0 -- for the column that explains the whole
    # event. Nought means "did not move"; the truth here is "cannot be measured",
    # and the two must not be reported with the same number.
    degenerate = reference["human_driven"].dropna().to_numpy()
    moved = current["human_driven"].dropna().to_numpy()
    refused = False
    try:
        outcome = lab.population_stability_index(degenerate, moved)
    except NotSolved:
        raise
    except Exception:
        refused = True
        outcome = None
    assert refused or outcome is None or (
        isinstance(outcome, float) and math.isnan(outcome)), (
        f"on the human_driven column your index returned {outcome!r}. Its reference "
        "day is nought in 39 of 45 windows, so the quantile edges collapse to a "
        "single bin, both shares are 1.0, and the arithmetic gives exactly 0.0 for a "
        "column that moved by 1.24 reference standard deviations and caused "
        "everything else in this archive. Count the surviving edges and refuse: "
        "raise DegenerateReference, or return None. An index of nought there says "
        "'no change' where the truth is 'no measurement'.")


run(2, "02_how_surprising", "kl_divergence", body)
