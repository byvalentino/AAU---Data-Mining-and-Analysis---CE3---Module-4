"""Lab 2, solved — with the reasoning, not only the code.

Run it: `python3 solutions/lab_02.py` from the exercises directory. It narrates
the worked decomposition and writes the two pictures of block two under out/.
"""
from __future__ import annotations

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from lab_support import (BORROWED_INDEX, DegenerateReference,   # noqa: E402
                         MINIMUM_EDGES, NULL_QUANTILE, NULL_RESAMPLES,
                         PSI_EPSILON, SEED)

LAB = 2
DEFAULT_BINS = 5   # not ten -- see the note below

# Why five bins and not the customary ten. There are about forty windows a day
# here. Ten bins leaves three or four observations in each, several of them
# empty, and the index then measures the binning rather than the world:
# comparing the reference against a RESAMPLE OF ITSELF -- no change whatsoever --
# gives a median index of 0.280 at ten bins against 0.105 at five.
#
# But the criterion that settles it is not that comparison. It is this: at ten
# bins the threshold derived from that same null calls the UNTOUCHED target
# material -- a false alarm on a column that moved three hundredths of a standard
# deviation. At five bins it does not. check_02 grades exactly that, and nothing
# in this module compares a bin count against a threshold out of a handbook.
#
# The lesson is not "use five". It is that the conventional thresholds assume a
# large sample, and on a small one you must find your own noise floor by
# simulation before you trust any threshold at all.


def entropy(p) -> float:
    """How varied today was, and nothing else.

    The zeros contribute nothing because the limit says so: as P(i) goes to
    nought, so does P(i)·log(P(i)). Dropping those terms is the limit rather
    than a convenience, and it is the same convention the divergence uses.

    This is also the term that makes the cross-entropy a poor monitor. It moves
    with today's data, so the cross-entropy's floor moves with it, and a measure
    whose floor moves cannot tell you that nothing happened.

    Definition graded by the check:
        H(P) = −Σ_i P(i)·log P(i)
        (Shannon, 1948; Murphy, 2022, §6.1.2). Choices: the natural logarithm, so
        the unit is the nat, and the limit convention for P(i) = 0. Slide:
        "Definition — entropy and cross-entropy".
    Needs: numpy.asarray, numpy.log, numpy.sum
    """
    p = np.asarray(p, dtype=float)
    live = p > 0
    return float(-np.sum(p[live] * np.log(p[live])))


def cross_entropy(p, q) -> float:
    """What today cost you, given what you believed yesterday.

    One substitution away from the entropy: the logarithm reads Q where the
    entropy reads P. That is also why it is the loss you have already trained a
    classifier with -- minimising it minimises the divergence, because the
    entropy term does not depend on the model.

    Definition graded by the check:
        H(P,Q) = −Σ_i P(i)·log Q(i)
        (Murphy, 2022, §6.1.2; Shannon, 1948). Choices: the natural logarithm and
        the same limit convention; a zero in Q where P has mass makes it
        infinite. Slide: "Definition — entropy and cross-entropy".
    Needs: numpy.asarray, numpy.log, numpy.sum
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    live = p > 0
    if np.any(q[live] == 0):
        return float("inf")
    return float(-np.sum(p[live] * np.log(q[live])))


def kl_divergence(p, q) -> float:
    """The excess surprise: what the wrong belief cost over the right one.

    Written directly as the sum, but it is exactly cross_entropy(p, q) minus
    entropy(p), and the check requires those two routes to agree to ten decimal
    places. That test exists because writing the cross-entropy where the
    divergence belongs is the commonest error in this material -- it was in this
    course's own first draft of these slides -- and little else catches it: both
    are non-negative, both rise when the world moves, and they differ only by an
    entropy term that is small on tidy examples.

    The zeros in P contribute nothing, as in the entropy. The zeros in Q are a
    different matter entirely: they make the answer infinite, which is the
    correct reply to "how surprising is something you called impossible?"

    Definition graded by the check:
        D(P ‖ Q) = Σ_i P(i)·log( P(i) / Q(i) ) = H(P,Q) − H(P), in nats
        (Kullback & Leibler, 1951; MacKay, 2003, §2.6). Choices: the natural
        logarithm; nought where the two match; infinite where Q gives no mass to
        something P does. Slide: "Definition — the Kullback–Leibler divergence".

    It is never negative, and that is Jensen's inequality applied to −log:
        f( E[X] ) ≤ E[ f(X) ] for convex f, with equality only where f is straight or X never varies
        (Jensen, 1906; Wasserman, 2004, Theorem 4.9). Slide: "Definition —
        Jensen's inequality".
    Needs: numpy.asarray, numpy.log, numpy.sum
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    live = p > 0
    if np.any(q[live] == 0):
        return float("inf")
    return float(np.sum(p[live] * np.log(p[live] / q[live])))


def population_stability_index(reference, current, bins: int = DEFAULT_BINS) -> float:
    """The symmetric index, on bins taken from the reference.

    Three decisions here, and each is the kind that quietly decides the answer:

    1. **The edges come from the reference, not from today.** Take today's
       quantiles and the yardstick moves every time you measure, so a stable
       world produces a wandering index. The reference is fixed for the same
       reason Module 5 fixes its baseline: a moving baseline drifts with the
       problem.

    2. **A small floor on empty bins.** Without it, one empty bin makes the whole
       index infinite -- mathematically right, operationally useless, and the
       reason binned monitors quietly hide the appearance of a genuinely new
       value. The floor buys a usable number and costs you that signal, so watch
       for new values separately rather than trusting the index to shout. It is
       not cosmetic either: at twenty bins on this sample size, 0.867 of what the
       index reports comes from bins the floor invented.

    3. **A refusal when there is nothing to measure.** On this archive the
       human_driven column is nought in 39 of the 45 reference windows, so all
       six requested quantiles collapse to two distinct edges, one bin holds
       everything, both shares are 1.0, and the index is exactly nought -- for a
       column that moved by 1.24 reference standard deviations and explains the
       whole event. Nought there does not mean "no change", it means "no
       measurement", so this raises instead. A measure that cannot fail visibly
       will fail invisibly.

    Definition graded by the check:
        J(P,Q) = Σ_i ( P(i) − Q(i) )·log( P(i) / Q(i) ) = D(P‖Q) + D(Q‖P)
        (Jeffreys, 1946; Yurdakul & Naranjo, 2020). Choices: bin edges from the
        reference's own quantiles, opened at both ends; a floor of PSI_EPSILON
        under every share; a refusal when fewer than MINIMUM_EDGES survive.
        Slide: "Definition — the symmetrised index".

    What this reads when nothing changed at all is index_threshold's business,
    below, and the threshold Lab 4 judges by is derived from it.
    Needs: numpy.quantile, numpy.unique, numpy.histogram, numpy.clip, numpy.log
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < MINIMUM_EDGES:
        raise DegenerateReference(
            f"{len(edges)} distinct quantile edges survive on this reference, which "
            f"is {len(edges) - 1} bin: there is nothing to compare and no index to "
            "report. Watch this column another way -- the share itself moved.")
    edges[0], edges[-1] = -np.inf, np.inf

    reference_share = np.histogram(reference, bins=edges)[0] / len(reference)
    current_share = np.histogram(current, bins=edges)[0] / len(current)

    reference_share = np.clip(reference_share, PSI_EPSILON, None)
    current_share = np.clip(current_share, PSI_EPSILON, None)

    return float(np.sum((current_share - reference_share)
                        * np.log(current_share / reference_share)))


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
    reference = np.asarray(reference, dtype=float)
    size = len(np.asarray(current, dtype=float))

    # Resampling the reference against itself is the null: two samples from one
    # world. Whatever the index reads here is what "nothing happened" looks like
    # at this bin count and these two sample sizes, and nothing below it is
    # signal. A degenerate reference raises out of this loop rather than
    # returning a floor, which is right -- a column with no measurement has no
    # threshold either.
    rng = np.random.default_rng(seed)
    null = np.array([population_stability_index(
        reference, rng.choice(reference, size=size, replace=True), bins)
        for _ in range(resamples)])

    return {
        # The median rather than the mean: the null has a long right tail, and a
        # mean pulled by it would call the middle of the distribution "typical".
        "noise_floor": float(np.median(null)),
        "threshold": float(np.quantile(null, quantile)),
        "bins": int(bins),
        "resamples": int(resamples),
        "quantile": float(quantile),
        "seed": int(seed),
        "sample_size": int(size),
    }


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    import plotly.graph_objects as go                                # noqa: E402
    from plotly.subplots import make_subplots                        # noqa: E402
    import pandas as pd                                              # noqa: E402
    from _narrate import narrator, show_table, save_figure           # noqa: E402
    from lab_support import reference_and_current                    # noqa: E402

    BLUE, ORANGE, GREY, RED = "#2A78D6", "#E07B39", "#52514E", "#C0392B"

    say = narrator(LAB)
    say.info("Lab 2 — two averages of surprise, the gap between them, and the "
             "floor under the gap")

    # 1. The worked pair, in nats, with the identity closed out loud.
    today, yesterday = [0.9, 0.1], [0.5, 0.5]
    entropy_today = entropy(today)
    cross = cross_entropy(today, yesterday)
    divergence = kl_divergence(today, yesterday)
    say.info("generated, the worked pair: today %s against yesterday %s, natural "
             "logarithms", today, yesterday)
    say.info("entropy of today %.3f nats — how varied today was, and nothing else",
             entropy_today)
    say.info("cross-entropy %.3f nats — what today cost under yesterday's beliefs", cross)
    say.info("divergence %.3f nats — the excess, and %.3f - %.3f = %.3f closes the "
             "identity to %.1e", divergence, cross, entropy_today, cross - entropy_today,
             abs((cross - entropy_today) - divergence))
    say.info("the other way round: D(yesterday || today) = %.3f nats, which is a "
             "different number — the reference is a decision",
             kl_divergence(yesterday, today))
    say.info("a value the reference called impossible: D = %s",
             kl_divergence([0.5, 0.5], [1.0, 0.0]))

    bars = go.Figure()
    bars.add_bar(x=["entropy H(P)", "cross-entropy H(P,Q)", "divergence D(P||Q)"],
                 y=[entropy_today, cross, divergence],
                 marker_color=[BLUE, ORANGE, GREY],
                 text=[f"{value:.3f}" for value in (entropy_today, cross, divergence)],
                 textposition="outside")
    bars.update_layout(
        title="Today [0.9, 0.1] under yesterday [0.5, 0.5]: the gap is the divergence",
        yaxis_title="nats", xaxis_title="", showlegend=False)
    save_figure(bars, "surprise_decomposition", LAB, logger=say)

    # 2. The archive, where the index has to decide something.
    reference, current = reference_and_current()
    say.info("archive slice, %d reference windows against %d current windows",
             len(reference), len(current))
    rows = []
    for feature in ("mean_speed", "sd_speed", "mean_payload", "human_driven"):
        before = reference[feature].dropna().to_numpy()
        after = current[feature].dropna().to_numpy()
        try:
            index = population_stability_index(before, after)
            measured = True
        except DegenerateReference as refused:
            index, measured = None, False
            say.info("%s: the index refuses — %s", feature, refused)
        rows.append({"feature": feature, "index": index, "measured": measured})
    show_table(pd.DataFrame(rows), "the symmetrised index at %d bins, edges from the "
                                   "reference" % DEFAULT_BINS, logger=say)

    # 3. The noise floor, and the threshold derived from it. Both come out of the
    #    same null: the reference against a resample of itself, which is two
    #    samples from one world.
    payload = reference["mean_payload"].dropna().to_numpy()
    today = current["mean_payload"].dropna().to_numpy()
    bins_tried = [3, 5, 8, 10, 15, 20]
    derived = [index_threshold(payload, today, bins=bins) for bins in bins_tried]
    floor = [item["noise_floor"] for item in derived]
    threshold = [item["threshold"] for item in derived]
    for bins, item in zip(bins_tried, derived):
        say.info("no change at all, %2d bins: floor (median) %.3f, threshold (the %.2f "
                 "quantile of the same null) %.3f, over %d resamples of %d, seed %d",
                 bins, item["noise_floor"], item["quantile"], item["threshold"],
                 item["resamples"], item["sample_size"], item["seed"])

    here = bins_tried.index(DEFAULT_BINS)
    say.info("the target's own index today is %.3f, which is below its own noise floor "
             "of %.3f at %d bins — the instrument cannot tell it apart from nothing "
             "happening", population_stability_index(payload, today), floor[here],
             DEFAULT_BINS)
    # The same null, the same stream, one question of it: how often does a world
    # in which nothing changed pass the threshold credit scoring hands out?
    borrowed_rng = np.random.default_rng(SEED)
    borrowed_rate = float(np.mean(
        [population_stability_index(
            payload, borrowed_rng.choice(payload, size=len(today), replace=True),
            DEFAULT_BINS) >= BORROWED_INDEX for _ in range(NULL_RESAMPLES)]))
    say.info("and the threshold this module judges by is %.3f, derived from that null, "
             "against the %.2f credit scoring hands out — which %.1f per cent of "
             "no-change comparisons pass, so borrowing it buys about one false alarm "
             "in ten, per feature, per run", threshold[here], BORROWED_INDEX,
             100 * borrowed_rate)

    curve = go.Figure()
    curve.add_scatter(x=bins_tried, y=floor, mode="lines+markers",
                      line=dict(color=BLUE, width=2.5), name="measured floor (median)")
    curve.add_scatter(x=bins_tried, y=threshold, mode="lines+markers",
                      line=dict(color=ORANGE, width=2.5),
                      name=f"derived threshold ({NULL_QUANTILE} quantile)")
    curve.add_scatter(x=bins_tried, y=[BORROWED_INDEX] * len(bins_tried), mode="lines",
                      line=dict(color=RED, width=2, dash="dash"),
                      name=f"the borrowed {BORROWED_INDEX} threshold")
    curve.update_layout(
        title="What the index reads when nothing has changed at all, and the "
              "threshold that comes out of it",
        xaxis_title=f"number of bins (this module uses {DEFAULT_BINS})",
        yaxis_title=f"index over {NULL_RESAMPLES} resamples (nats)")
    save_figure(curve, "noise_floor_by_bins", LAB, logger=say)

    say.info("what the check grades: cross-entropy minus entropy equals the divergence "
             "to ten decimal places on five pairs, all three against scipy, the index "
             "built on reference quantiles, the refusal on human_driven, a threshold "
             "derived from your own null rather than borrowed, and a bin count at which "
             "the detector does not fire on a target that did not move")
