"""Lab 1, solved — with the reasoning, not only the code.

Run it: `python3 solutions/lab_01.py` from the exercises directory. It narrates
every step and writes both of block one's pictures under out/.
"""
from __future__ import annotations

import math

import numpy as np

LAB = 1
Z_95 = 1.959963984540054


def naive_interval(successes: int, trials: int, z: float = Z_95):
    """The interval everybody writes, and the two places it fails.

    It treats the observed proportion as the centre and puts symmetric error
    bars around it. Two consequences follow immediately:

    1. At the edges it collapses. Forty out of forty gives p = 1, so p(1-p) = 0,
       so the half-width is nought: the interval is [1.0, 1.0]. Perfect
       certainty from forty observations, which is absurd and which the formula
       states without embarrassment.
    2. It can run outside [0, 1], reporting a negative probability, which tends
       to be noticed only when somebody plots it.

    Both come from the same root: it asks "what is the error around my
    estimate?" when the useful question is "which true rates could plausibly
    have produced what I saw?"

    Definition graded by the check:
        p̂ ± z·√( p̂(1−p̂)/n ), with p̂ = k/n successes in n trials
        (Brown, Cai & DasGupta, 2001; Agresti & Coull, 1998). Choices: the
        ninety-five per cent level, so z = 1.96, and the normal approximation to
        the binomial law. Slide: "Definition — the Wald interval".
    Needs: math.sqrt
    """
    if trials == 0:
        return (float("nan"), float("nan"))
    proportion = successes / trials
    half = z * math.sqrt(proportion * (1 - proportion) / trials)
    return (proportion - half, proportion + half)


def wilson_interval(successes: int, trials: int, z: float = Z_95):
    """The interval that holds up, and why the centre moves.

    Wilson (1927) inverts the question: which true rates would produce an
    observation like this one, at least five per cent of the time? Solving that
    gives a centre pulled towards one half and a width that never collapses.

    The pull is the interesting part. With forty out of forty the centre is not
    1.0 but about 0.95, and the interval has real width. That is not the formula
    hedging; it is the formula correctly declining to conclude certainty from a
    finite sample.

    It is also barely more code than the naive one, which is the strongest
    argument for using it.

    Definition graded by the check:
        ( p̂ + z²/2n ± z·√( p̂(1−p̂)/n + z²/4n² ) ) / ( 1 + z²/n )
        (Wilson, 1927; Brown, Cai & DasGupta, 2001). Choices: the same level and
        the same approximation; what changes is the question — which true rates
        could have produced what was seen. Slide: "Definition — the Wilson score
        interval".
    Needs: math.sqrt
    """
    if trials == 0:
        return (float("nan"), float("nan"))
    centre = (successes + z ** 2 / 2) / (trials + z ** 2)
    half = (z / (trials + z ** 2)) * math.sqrt(
        successes * (trials - successes) / trials + z ** 2 / 4)
    return (centre - half, centre + half)


def coverage(interval, true_rate: float, trials: int, repeats: int = 4000,
             seed: int = 20200122) -> float:
    """Measure the promise rather than believing it.

    A 95 per cent interval makes a testable claim: over many samples, it
    contains the truth 95 per cent of the time. So test it. This is a habit
    worth more than either formula — when a method makes a claim you can
    simulate, simulate it.

    Near a true rate of 0.02 with forty observations, the naive interval delivers
    about a third of what it promises. Most samples contain no successes at all,
    the interval is [0, 0], and the truth is not in it.

    Definition graded by the check:
        coverage(p, n) = (1/R)·Σ_{r=1}^{R} 1{ low_r ≤ p ≤ high_r }
        (Brown, Cai & DasGupta, 2001; Agresti & Coull, 1998). Choices: R = 4000
        samples of n = 40 at each true rate, and numpy's default_rng(20200122),
        so the answer is the same every time. Slide: "Definition — coverage, and
        what a half-width costs".
    Needs: numpy.random.default_rng, rng.binomial
    """
    rng = np.random.default_rng(seed)
    successes = rng.binomial(trials, true_rate, repeats)
    contained = 0
    for count in successes:
        low, high = interval(int(count), trials)
        if low <= true_rate <= high:
            contained += 1
    return contained / repeats


def labels_needed(half_width: float) -> int:
    """Rearranged, and the square root is the whole answer.

        half = z * sqrt(0.25 / n)   ->   n = 0.25 * (z / half)^2

    Halve the half-width and n goes up fourfold. From ±0.05 to ±0.025 is 385
    labels to 1,537 -- four times the hand-checking for twice the precision.

    That is why Module 5 buys truth in small samples and puts an interval around
    it rather than pretending to a precision nobody paid for.

    Definition graded by the check:
        n = ⌈ 0.25·(z/h)² ⌉ at the worst case p = ½
        (Brown, Cai & DasGupta, 2001). Choices: the worst case p = ½, where
        p(1−p) is largest; the normal approximation, so the count is itself an
        approximation; and rounding up, because labels come whole. Slide:
        "Definition — coverage, and what a half-width costs".
    Needs: math.ceil
    """
    if half_width <= 0:
        raise ValueError("half_width must be positive")
    return int(math.ceil(0.25 * (Z_95 / half_width) ** 2))


if __name__ == "__main__":
    import sys
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    import plotly.graph_objects as go                                # noqa: E402
    from plotly.subplots import make_subplots                        # noqa: E402
    import pandas as pd                                              # noqa: E402
    from _narrate import narrator, show_table, save_figure           # noqa: E402
    from lab_support import windowed                                 # noqa: E402

    BLUE, ORANGE, GREY, RED = "#2A78D6", "#E07B39", "#52514E", "#C0392B"

    say = narrator(LAB)
    say.info("Lab 1 — what an interval promises, what it delivers, and what it costs")

    # 1. The two worked cases, which are the whole argument in four numbers.
    for successes, trials in ((34, 40), (40, 40)):
        wald = naive_interval(successes, trials)
        wilson = wilson_interval(successes, trials)
        say.info("%d of %d hand-checked right: Wald [%.3f, %.3f] width %.3f, "
                 "Wilson [%.3f, %.3f] width %.3f",
                 successes, trials, wald[0], wald[1], wald[1] - wald[0],
                 wilson[0], wilson[1], wilson[1] - wilson[0])
    say.info("the Wald interval has no width at all on 40 of 40 — certainty from "
             "forty observations, which is the reason this lab exists")

    # 2. Coverage: the promise, measured rather than believed.
    rates = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50]
    measured = pd.DataFrame({
        "true rate": rates,
        "Wald coverage": [coverage(naive_interval, rate, 40) for rate in rates],
        "Wilson coverage": [coverage(wilson_interval, rate, 40) for rate in rates],
    })
    show_table(measured, "coverage of a nominal 0.95 interval, 40 labels, "
                         "4000 samples per rate", logger=say)
    say.info("near the edge the Wald interval delivers %.2f of the 0.95 it promises; "
             "Wilson delivers %.2f",
             measured["Wald coverage"][0], measured["Wilson coverage"][0])

    # 3. What precision costs, in hand-checks.
    for half in (0.10, 0.05, 0.025, 0.01):
        say.info("a half-width of +/-%.3f needs %d labels", half, labels_needed(half))
    say.info("halving the half-width multiplies the labels by %.1f — the square root "
             "in the formula, and the reason precision is a budget question",
             labels_needed(0.025) / labels_needed(0.05))

    # 4. Picture one: every result forty hand-checks could produce.
    counts = list(range(41))
    wald = [naive_interval(k, 40) for k in counts]
    wilson = [wilson_interval(k, 40) for k in counts]
    bands = go.Figure()
    bands.add_scatter(x=counts, y=[high for _, high in wilson], mode="lines",
                      line=dict(color=BLUE, width=2), showlegend=False)
    bands.add_scatter(x=counts, y=[low for low, _ in wilson], mode="lines",
                      name="Wilson interval", line=dict(color=BLUE, width=2),
                      fill="tonexty", fillcolor="rgba(42, 120, 214, 0.22)")
    bands.add_scatter(x=counts, y=[high for _, high in wald], mode="lines",
                      line=dict(color=GREY, width=2, dash="dot"), showlegend=False)
    bands.add_scatter(x=counts, y=[low for low, _ in wald], mode="lines",
                      name="Wald interval", line=dict(color=GREY, width=2, dash="dot"),
                      fill="tonexty", fillcolor="rgba(82, 81, 78, 0.18)")
    bands.add_scatter(x=[0, 40], y=[0.0, 1.0], mode="markers", name="Wald: no width",
                      marker=dict(color=RED, size=12, symbol="x"))
    bands.update_layout(title="Forty hand-checks: what each method reports",
                        xaxis_title="hand-checked predictions found correct, out of 40",
                        yaxis_title="interval reported for the true rate")
    save_figure(bands, "wilson_against_wald", LAB, logger=say)

    # 5. Picture two: the same choice on the archive's own windows, where the
    # rate being estimated is the share of readings a person drove.
    table = windowed()
    share = (table["manual_readings"] / table["n_readings"]).to_numpy()
    wald = [naive_interval(int(k), int(n)) for k, n
            in zip(table["manual_readings"], table["n_readings"])]
    wilson = [wilson_interval(int(k), int(n)) for k, n
              in zip(table["manual_readings"], table["n_readings"])]
    collapsed = [high - low < 1e-12 for low, high in wald]
    say.info("archive slice, %d five-minute windows over both days: the Wald interval "
             "has no width at all in %d of them", len(table), sum(collapsed))

    position = list(range(1, len(table) + 1))
    archive = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                            subplot_titles=("every window, both days",
                                            "the same windows, magnified onto the floor"))
    for row in (1, 2):
        archive.add_scatter(
            x=position, y=share, mode="markers",
            marker=dict(color=[RED if flat else GREY for flat in collapsed], size=6),
            error_y=dict(type="data", symmetric=False,
                         array=[high - value for (low, high), value in zip(wald, share)],
                         arrayminus=[value - low for (low, high), value in zip(wald, share)],
                         color=GREY, thickness=1.4, width=0),
            name="Wald (red where it has no width)", showlegend=row == 1, row=row, col=1)
        archive.add_scatter(
            x=[p + 0.3 for p in position], y=share, mode="markers",
            marker=dict(color=BLUE, size=5),
            error_y=dict(type="data", symmetric=False,
                         array=[high - value for (low, high), value in zip(wilson, share)],
                         arrayminus=[value - low for (low, high), value in zip(wilson, share)],
                         color=BLUE, thickness=1.4, width=0),
            name="Wilson", showlegend=row == 1, row=row, col=1)
    archive.update_yaxes(title_text="share of readings in manual mode",
                         range=[-0.03, 1.06], row=1, col=1)
    archive.update_yaxes(title_text="share of readings", range=[-0.002, 0.02], row=2, col=1)
    archive.update_xaxes(title_text="five-minute window, in order of time", row=2, col=1)
    archive.update_layout(title="The same two intervals, on the archive's own windows",
                          legend=dict(orientation="h", x=0.3, y=1.14))
    save_figure(archive, "wilson_on_the_archive", LAB, logger=say)

    say.info("what the check grades: your Wilson interval against an independent "
             "implementation on five cases, the Wald collapse on 40 of 40, coverage "
             "below 0.75 for Wald and above 0.88 for Wilson at a true rate of 0.02, "
             "and labels_needed at three half-widths including one never quoted here")
