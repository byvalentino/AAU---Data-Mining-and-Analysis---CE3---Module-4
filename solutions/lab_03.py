"""Lab 3, solved — with the reasoning, not only the code.

Run it: `python3 solutions/lab_03.py` from the exercises directory. It narrates
all four measures on the archive and draws block three's three-panel contrast
under out/.
"""
from __future__ import annotations

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from lab_support import load_lab  # noqa: E402

LAB = 3
DEFAULT_BINS = 5


def wasserstein(a, b) -> float:
    """The area between two cumulative distribution functions.

    In one dimension the earth-moving picture collapses into something easy.
    Take every value either sample holds as a breakpoint, read both cumulative
    shares on each strip between consecutive breakpoints, and add the gaps times
    the widths. That is the same integral as the picture, taken along the value
    axis, and it handles samples of different lengths without any special
    pleading.

    Why bother, when a divergence already gives you a number: because this one
    comes out in the variable's own units. "The speed distribution moved by 0.46
    metres per second" is a sentence an operations manager can act on. "The
    divergence is 0.031" is a sentence that ends the conversation.

    Definition graded by the check:
        W₁(P,Q) = ∫ |F_P(x) − F_Q(x)| dx = ∫₀¹ |F_P⁻¹(u) − F_Q⁻¹(u)| du
        (Vallender, 1974; Peyré & Cuturi, 2019, Remark 2.30). Choices: no binning
        at all, so the answer is in the variable's own units; for two sorted
        samples of equal size it is the mean of |a_(i) − b_(i)| (Remark 2.28).
        Slide: "Definition — the Wasserstein-1 distance".
    Needs: numpy.sort, numpy.concatenate, numpy.diff, numpy.searchsorted, numpy.sum
    """
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if len(a) == 0 or len(b) == 0:
        return float("nan")

    breakpoints = np.sort(np.concatenate([a, b]))
    widths = np.diff(breakpoints)
    cumulative_a = np.searchsorted(a, breakpoints[:-1], side="right") / len(a)
    cumulative_b = np.searchsorted(b, breakpoints[:-1], side="right") / len(b)
    return float(np.sum(np.abs(cumulative_a - cumulative_b) * widths))


def compare_four(reference, current, bins: int = DEFAULT_BINS) -> dict:
    """All four, together, because each hides something the others show.

    The cross-entropy says what today cost; the divergence says how much of that
    cost was avoidable; the index says the same with no reference order to argue
    about; the distance says how far the world went, in units somebody can act
    on. Report one and somebody will ask the question it cannot answer.

    The two binnings are the interesting part, and they differ on purpose.

    The **index** bins on the reference's own quantiles, opened at both ends.
    That is a monitor's yardstick: cut once from the reference and never moved,
    so that a stable world produces a stable number. It floors empty bins, so it
    stays finite even when today holds something yesterday never did -- which is
    exactly the signal it is throwing away, and why unseen values need their own
    check.

    The **cross-entropy and the divergence** bin on equal-width edges spanning
    both samples. Here the two samples are both in front of you, so the bins
    cover everything either one holds, and a value the reference never held falls
    in a bin whose reference share is nought. The divergence then answers
    "infinite", loudly, instead of hiding the new value inside an existing bin.

    On the real archive that happens: the second day reached speeds the first day
    never did, so the divergence on mean_speed is infinite while the index is
    2.289 and the distance is 0.46 metres per second. Three answers, all correct,
    to three different questions.

    Definition graded by the check:
        H and D: equal-width edges over both samples · J: the reference's quantile edges, opened at both ends
        (Siddiqi, 2006; Yurdakul & Naranjo, 2020). Choices: numpy's
        histogram_bin_edges over the two samples concatenated for the entropy and
        the divergence, and Lab 2's own index for the third. Slide: "Definition —
        the two binnings inside compare_four".
    Needs: numpy.histogram_bin_edges, numpy.histogram, lab_support.load_lab
    """
    lab2 = load_lab(2)

    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    # Equal-width edges over both samples, for the two measures that are
    # comparing a pair rather than watching one against a fixed past.
    edges = np.histogram_bin_edges(np.concatenate([reference, current]), bins=bins)
    reference_share = np.histogram(reference, bins=edges)[0] / len(reference)
    current_share = np.histogram(current, bins=edges)[0] / len(current)

    return {
        "cross_entropy": lab2.cross_entropy(current_share, reference_share),
        "kl_divergence": lab2.kl_divergence(current_share, reference_share),
        "population_stability_index": lab2.population_stability_index(
            reference, current, bins),
        "wasserstein": wasserstein(reference, current),
    }


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    import plotly.graph_objects as go                                # noqa: E402
    from plotly.subplots import make_subplots                        # noqa: E402
    import pandas as pd                                              # noqa: E402
    from _narrate import (narrator, show_table, save_figure,        # noqa: E402
                          reference_lab)

    # For the length of this demonstration, "Lab 2" means the shipped
    # solution rather than whatever is in labs/. compare_four() reads this
    # name at call time, so rebinding it here is enough, and nothing outside
    # this block is affected -- the check still imports the student's lab.
    load_lab = reference_lab                                        # noqa: F811
    from lab_support import SEED, reference_and_current              # noqa: E402

    BLUE, ORANGE, GREY, RED = "#2A78D6", "#E07B39", "#52514E", "#C0392B"

    say = narrator(LAB)
    say.info("Lab 3 — the measure that answers in metres per second, and the three "
             "cases where it disagrees with the divergence")

    # 1. All four on the archive, where they answer three different questions.
    reference, current = reference_and_current()
    say.info("archive slice, %d reference windows against %d current windows",
             len(reference), len(current))
    rows = []
    for feature in ("mean_speed", "mean_payload"):
        measures = compare_four(reference[feature].dropna(), current[feature].dropna())
        rows.append({"feature": feature, **{k: round(v, 4) for k, v in measures.items()}})
    show_table(pd.DataFrame(rows), "all four measures, one pair each", logger=say)
    say.info("mean speed moved %.3f metres per second — a sentence an operations "
             "manager can act on, where the index's %.3f is measured in nothing",
             rows[0]["wasserstein"], rows[0]["population_stability_index"])
    say.info("the divergence on mean speed is %s: the second day reached speeds the "
             "first day never did", rows[0]["kl_divergence"])

    # 2. The units check: shift a sample by a known amount and the distance is it.
    base = reference["mean_payload"].dropna().to_numpy()
    say.info("the same payload sample shifted by 5 kilograms: distance %.6f",
             wasserstein(base, base + 5.0))

    # 3. Case one — no overlap at all.
    rng = np.random.default_rng(SEED)
    yesterday = rng.uniform(0.0, 2.0, 400)
    today = yesterday + 5.0
    disjoint = compare_four(yesterday, today)
    say.info("generated, seed %d. No overlap: divergence %s, distance %.3f metres "
             "per second — the divergence cannot tell a gap of five from a gap of "
             "five hundred", SEED, disjoint["kl_divergence"], disjoint["wasserstein"])

    # 4. Case two — the bins relabelled, every share keeping its partner.
    centres = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    reference_shares = np.array([0.40, 0.30, 0.20, 0.07, 0.03])
    current_shares = np.array([0.03, 0.07, 0.20, 0.30, 0.40])
    order = np.argsort(np.random.default_rng(SEED).permutation(len(centres)))

    def sample_at(shares, size=2000):
        return np.repeat(centres, np.round(shares * size).astype(int))

    before = compare_four(sample_at(reference_shares), sample_at(current_shares))
    after = compare_four(sample_at(reference_shares[order]),
                         sample_at(current_shares[order]))
    say.info("bins relabelled: divergence %.3f then %.3f — unchanged, it never knew "
             "the values were ordered", before["kl_divergence"], after["kl_divergence"])
    say.info("bins relabelled: distance %.3f then %.3f metres per second — it is built "
             "on the geometry, and the geometry is what moved",
             before["wasserstein"], after["wasserstein"])

    # 5. Case three — the same law moved, against the same law squeezed, chosen so
    #    that the distance cannot tell them apart. The reference's mean and spread
    #    are the archive's own; the two changes are constructed from them.
    mean = float(reference["mean_speed"].mean())
    spread = float(reference["mean_speed"].std(ddof=1))
    factor = 0.25
    shift = (1 - factor) * spread * np.sqrt(2 / np.pi)
    draw = np.random.default_rng(SEED)
    law = draw.normal(mean, spread, 20000)
    moved = law + shift
    squeezed = mean + factor * (law - mean)
    say.info("generated from the archive's own mean %.3f and spread %.3f metres per "
             "second: moved by %.3f, or squeezed to %.2f of its spread",
             mean, spread, shift, factor)
    say.info("distance: %.3f moved, %.3f squeezed — equal by construction, to the "
             "sampling noise of twenty thousand draws",
             wasserstein(law, moved), wasserstein(law, squeezed))
    say.info("divergence at %d bins: %.3f moved, %.3f squeezed — the measure the "
             "distance cannot separate, this one can", DEFAULT_BINS,
             compare_four(law, moved)["kl_divergence"],
             compare_four(law, squeezed)["kl_divergence"])
    say.info("the slide reads the same pair in closed form on the normal law, where "
             "nothing is binned; binning into %d equal-width bins pulls both "
             "divergences down, which is what the binning costs to stay finite",
             DEFAULT_BINS)

    # The picture: one column per case, distributions above, cumulative curves below.
    figure = make_subplots(
        rows=2, cols=3, vertical_spacing=0.14, horizontal_spacing=0.08,
        subplot_titles=("no overlap", "bins relabelled", "moved, or squeezed",
                        "cumulative curves", "the same, relabelled", "cumulative curves"))

    def cumulative(sample, grid):
        return np.searchsorted(np.sort(sample), grid, side="right") / len(sample)

    grid_one = np.linspace(-0.2, 7.2, 400)
    edges = np.linspace(0.0, 7.0, 36)
    mids = (edges[:-1] + edges[1:]) / 2
    for sample, colour, name in ((yesterday, BLUE, "reference"), (today, ORANGE, "current")):
        figure.add_bar(x=mids, y=np.histogram(sample, bins=edges)[0] / len(sample),
                       marker_color=colour, name=name, row=1, col=1)
        figure.add_scatter(x=grid_one, y=cumulative(sample, grid_one), mode="lines",
                           line=dict(color=colour, width=2), showlegend=False,
                           row=2, col=1)

    for row, (a, b) in enumerate(((reference_shares, current_shares),
                                  (reference_shares[order], current_shares[order])),
                                 start=1):
        figure.add_bar(x=centres, y=a, marker_color=BLUE, width=0.38, offset=-0.4,
                       showlegend=False, row=row, col=2)
        figure.add_bar(x=centres, y=b, marker_color=ORANGE, width=0.38, offset=0.02,
                       showlegend=False, row=row, col=2)

    grid_three = np.linspace(mean - 4 * spread, mean + 4 * spread, 400)
    for sample, colour, dash in ((law, BLUE, "solid"), (moved, ORANGE, "solid"),
                                 (squeezed, GREY, "dash")):
        counts, bin_edges = np.histogram(sample, bins=60,
                                         range=(grid_three[0], grid_three[-1]))
        figure.add_scatter(x=(bin_edges[:-1] + bin_edges[1:]) / 2, y=counts / len(sample),
                           mode="lines", line=dict(color=colour, width=2, dash=dash),
                           showlegend=False, row=1, col=3)
        figure.add_scatter(x=grid_three, y=cumulative(sample, grid_three), mode="lines",
                           line=dict(color=colour, width=2, dash=dash), showlegend=False,
                           row=2, col=3)

    figure.update_xaxes(title_text="metres per second", row=2, col=1)
    figure.update_xaxes(title_text="bin centre, metres per second", row=2, col=2)
    figure.update_xaxes(title_text="window mean speed, metres per second", row=2, col=3)
    figure.update_yaxes(title_text="share", row=1, col=1)
    figure.update_yaxes(title_text="cumulative share", row=2, col=1)
    figure.update_layout(barmode="overlay",
                         title="Three cases where the divergence and the distance disagree",
                         legend=dict(orientation="h", x=0.35, y=1.13))
    save_figure(figure, "divergence_against_distance", LAB, logger=say)

    say.info("what the check grades: your distance against scipy on four cases "
             "including unequal lengths, exactly 5 on a sample shifted by 5, and "
             "compare_four's index equal to Lab 2's, its divergence infinite where "
             "the samples do not overlap, and its distance moving when the bins are "
             "relabelled while the divergence does not")
