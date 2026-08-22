#!/usr/bin/env python3
"""Check 1 — the interval that holds up, and the one that does not."""
import importlib.util
import sys, pathlib

VERIFY = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(VERIFY))
from _harness import run, close, not_ready                            # noqa: E402
# And straight back off again -- every copy of it, including the one Python adds
# for the script's own directory. While verify/ is importable, or while
# "statsmodels_free" sits in sys.modules, a lab can import the very
# implementation it is being graded against, and three lines of delegation pass
# this check without the student writing the formula at all. That is not a
# hypothetical: the file used to live in the exercises root, which every lab puts
# on sys.path itself, and the delegate exited 0. Moving it here was not enough on
# its own, because `python3 verify/check_01.py` puts verify/ on the path anyway.
sys.path[:] = [entry for entry in sys.path
               if pathlib.Path(entry or ".").resolve() != VERIFY]
sys.path.insert(0, str(VERIFY.parent))
try:
    import numpy as np                                                # noqa: E402
except ImportError as unready:                                        # noqa: E402
    not_ready(unready)


def load_reference():
    """The independent Wilson implementation, loaded by path under a private name.

    Loaded rather than imported so that nothing named statsmodels_free is left in
    sys.modules for a lab to find. A check that compares a student's code against
    the student's own copy of the reference proves nothing.
    """
    specification = importlib.util.spec_from_file_location(
        "_check_01_wilson_reference", VERIFY / "statsmodels_free.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.wilson_reference


wilson_reference = load_reference()


def body(lab):
    # The naive interval, against the formula.
    low, high = lab.naive_interval(34, 40)
    proportion = 34 / 40
    half = 1.959963984540054 * np.sqrt(proportion * (1 - proportion) / 40)
    close(low, proportion - half, 1e-9, "naive_interval lower bound")
    close(high, proportion + half, 1e-9, "naive_interval upper bound")

    # And its failure, which is the point of having it.
    low, high = lab.naive_interval(40, 40)
    close(high - low, 0.0, 1e-12, (
        "the naive interval on 40 of 40 should collapse to zero width — that is the "
        "flaw the lab exists to show. Yours did not, so it is not the naive formula."))

    # Wilson, against an independent implementation of the published formula.
    for successes, trials in ((34, 40), (40, 40), (0, 40), (1, 200), (150, 300)):
        expected = wilson_reference(successes, trials)
        got = lab.wilson_interval(successes, trials)
        close(got[0], expected[0], 1e-9, f"wilson_interval({successes}, {trials}) lower")
        close(got[1], expected[1], 1e-9, f"wilson_interval({successes}, {trials}) upper")

    low, high = lab.wilson_interval(40, 40)
    assert high - low > 0.02, (
        f"Wilson on 40 of 40 gave a width of {high - low:.4f}. It must not collapse: "
        "forty observations do not establish certainty, and declining to say so is "
        "the whole reason to prefer it.")
    assert high <= 1.0 + 1e-12 and low >= -1e-12, (
        f"Wilson returned [{low:.4f}, {high:.4f}] — a probability outside [0, 1]")

    # Coverage: the promise, measured.
    naive_edge = lab.coverage(lab.naive_interval, 0.02, 40)
    wilson_edge = lab.coverage(lab.wilson_interval, 0.02, 40)
    assert naive_edge < 0.75, (
        f"the naive interval covered {naive_edge:.2f} of the time at a true rate of "
        "0.02. It should be far below its promised 0.95 — most samples contain no "
        "successes at all, giving [0, 0]. Check that you build the interval from each "
        "simulated sample rather than from the true rate.")
    assert wilson_edge > 0.88, (
        f"the Wilson interval covered only {wilson_edge:.2f} at a true rate of 0.02; "
        "it should stay near 0.95")
    assert wilson_edge > naive_edge, "Wilson must beat the naive interval near the edge"

    middle = lab.coverage(lab.wilson_interval, 0.5, 40)
    assert 0.90 < middle < 1.0, (
        f"at a true rate of one half Wilson covered {middle:.2f}; expected about 0.95")

    # The price of labels. Three half-widths, and the third is deliberately not
    # one the lab mentions: with only the two advertised widths, a two-entry
    # lookup table passed this check without any arithmetic in it at all.
    close(lab.labels_needed(0.05), 385, 0, "labels_needed(0.05)")
    close(lab.labels_needed(0.025), 1537, 0, "labels_needed(0.025)")
    close(lab.labels_needed(0.037), 702, 0, (
        "labels_needed(0.037) — an unadvertised half-width, which is the point: "
        "this has to be the arithmetic rather than the two answers in the docstring"))
    ratio = lab.labels_needed(0.025) / lab.labels_needed(0.05)
    assert 3.9 < ratio < 4.1, (
        f"halving the half-width changed the count by {ratio:.2f} times; it should be "
        "about four. The square root is why bought truth gets expensive.")


run(1, "01_how_sure_are_you", "wilson_interval", body)
