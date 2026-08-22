"""Lab 4 — The verdict.

Why this lab exists: three instruments and a real question — did anything change
between 22 and 23 January, and does it matter? The answer on the thing that
matters is no, and this lab is where you learn to write that down and defend it
with a threshold you derived, a control that fires, and a stated detection
limit, which is the harder and the more common professional skill.
Where it sits: Block four — "The verdict — two inputs moved, and one measure
failed", and the definition slides "Definition — the standardised shift, and the
rule it is read against", "Definition — the positive control", "Definition — the
detection limit", "Definition — the drift verdict, and the three calls it may
make", "Definition — the classifier two-sample test", "Definition — Welch's
t-test, and the bootstrap to the reading grain" and "Definition — Cohen's d".
What the check grades: a materiality threshold derived from Lab 2's measured
floor rather than borrowed, per feature; mean speed material and the target not;
the largest index belonging to sd_speed; the degenerate column reported
unmeasured rather than nought; a positive control that fires, leaves the real
answer untouched and reports the smallest shift it can still see; one call out of
three with a reason built from your own measurements; a classifier two-sample
test with an interval on its accuracy; and a p-value that falls with the sample
size while Cohen's d, pooled by degrees of freedom, does not move.
Needs: numpy, scipy, and lab_support.load_lab to reuse Labs 1, 2 and 3.

Twenty-five minutes, and it is the fullest block of the day. If you are short of
time, `classifier_two_sample_test` is the one to leave until last: it is the
closing comparison rather than the verdict itself.

Everything so far has been an instrument. Now use them on the real question:
**did anything change between 22 and 23 January, and does it matter?**

A warning before you start, because it is the point of the lab. You are looking
for drift. You will find some. The honest verdict on the thing that actually
matters — the target — is that it did not move, and the check *expects* that.
If your verdict says the target drifted, you have made an error somewhere, not a
discovery.

Learning to write "no material change" and defend it is harder than learning to
find a change, and it is the more common professional situation.

Three things are fixed for you in `lab_support.py`, and they are printed beside
every number you report:

    vehicle      VJRD1A10224000055 only -- the one shuttle that ran on both days
    windows      five-minute tumbling windows on utc_time, at least 300 readings
    reference    22 January; current 23 January

Why the vehicle is fixed matters. The other shuttle ran on the first day only.
Pool the two on day one and compare against one on day two, and part of what you
would call drift is a vehicle going to the depot. An earlier version of this
course's own plan did exactly that and got the sign of the target's movement
wrong. The grain is the correction.

And one thing is **not** fixed for you, deliberately: the index threshold. Block
two spends six bullets on why credit scoring's 0.25 is a statement about
somebody else's sample size. So this lab does not use it. Every feature's
threshold is derived from the null you measured in Lab 2 —
`load_lab(2).index_threshold(...)` — at the bin count you are comparing at and
at the sample sizes you actually have. That number is on no slide, and it is not
the same for two columns.

--------------------------------------------------------------------------
1. verdict(reference, current, features, thresholds)

    For each feature, return a dictionary holding at minimum:

        shift_in_reference_sd        the difference in means, in units of the
                                     reference day's own standard deviation
        population_stability_index   from Lab 2 through Lab 3, or None when the
                                     reference cannot be binned
        index_measured               True, or False when the index refused
        noise_floor                  what the index reads on this column when
                                     nothing changed, from Lab 2
        index_threshold              the threshold derived from that same null
        wasserstein                  from Lab 3
        material                     True or False

    Call it material when the shift is at least MATERIAL_SHIFT_SD reference
    standard deviations OR the index was measured and is at least the threshold
    **you derived for that feature**. Two independent instruments agreeing is
    worth more than either alone, and requiring both would miss a shift only one
    of them is shaped to see.

    `thresholds` is an optional mapping from a feature name to the dictionary
    `index_threshold()` returned for it, so that a caller who has already
    derived one does not pay for it again. The null is built out of the
    reference alone, so it does not change while a sweep moves the current day —
    reusing it is a statement that the instrument did not change, not a
    shortcut. Derive whatever is not handed to you.

    One column will refuse. `human_driven` is nought in 39 of the 45 reference
    windows, so its quantile edges collapse to a single bin and Lab 2 raises
    DegenerateReference. Catch it, record the index, the floor and the threshold
    as **unmeasured**, and decide materiality on the shift alone. Do not record
    nought: nought means "did not move", and this column moved by more than a
    standard deviation.

2. positive_control(reference, current, injected_shift_sd, sizes)

    A null result nobody has shown the instrument could break is an opinion. So
    break it on purpose, and then find out how small a break it can still see.

    Add injected_shift_sd * (the reference day's standard deviation of the
    target) to every current value of the target, put that through your
    **unchanged** verdict, and return the target's row plus the size you
    injected under the key "injected_shift_sd". It must come back material.

    Then sweep. Walk every size in `sizes`, inject it the same way, and record
    whether the verdict calls it material. Report:

        detection_limit_sd           the smallest swept size from which the
                                     verdict is material at that size and at
                                     every larger swept size
        detection_limit_in_target_units   the same number times the reference
                                     standard deviation of the target, in the
                                     target's own unit (here, kilograms)
        first_material_sd            the first size at which it fires at all
        sizes, material_by_size, index_by_size   the sweep itself

    Those two sizes are not the same number, and the gap is the lesson. At this
    sample size the index is not monotone in the size of the shift: move a
    handful of windows across a quantile edge and it jumps, then falls back. The
    honest limit is the size from which the answer *stays* material, not the
    first size at which it flickers.

    A control at one size says the detector detects. A sweep says what it is
    blind to, and that sentence belongs in the report.

3. drift_verdict(evidence)

    Return (call, reason). `call` is one of "act", "watch", "no material
    change". `evidence` is what you measured, handed to you as a dictionary.

    The rule, in the order it is read:

        the target is material                     -> "act"
        its index could not be measured at all     -> "watch"
        the positive control did not fire          -> "watch"
        its index is at or below the noise floor   -> "no material change"
        anything else                              -> "watch"

    The third line is the one people leave out. A quiet detector nobody has
    tested is not evidence of quiet; it is evidence of nothing, and "watch" is
    what you owe the operator until the control fires.

    The reason is graded, not read for style. It has to be at least forty
    characters, every number in it has to be a number in the evidence you were
    handed, and it has to name at least two of the quantities it weighed. A
    sentence off a slide will fail, and that is the point of the exercise.

4. significance_is_not_size(sample_a, sample_b, readings, seed)

    The two arguments are the two days' **window** samples -- about eighty
    observations. The reading grain is 48,290 observations of the identical
    difference, and you reach it here by resampling: draw with replacement from
    each sample, in proportion to its size, until the two together hold
    `readings` observations, and test again.

    Nothing about the world changes between those two tests. Only how much of it
    we looked at. Return

        {"p_value_windows": ..., "n_windows": ...,
         "p_value_readings": ..., "n_readings": ..., "effect_size": ...}

    Use `numpy.random.default_rng(seed)` for the resampling, with the seed
    defaulted below to the course seed, 20200122. A number that changes every
    time you run it cannot be checked, quoted or defended.

    Use Cohen's d for the effect size: the difference in means over the pooled
    standard deviation. It does not move with the sample size, which is the
    entire point of reporting it.

5. classifier_two_sample_test(reference, current, features, seed)

    The required reading — Rabanser, Günnemann and Lipton (2019) — tests every
    method in this lab against one more: train a classifier to tell a reference
    row from a current row, and ask whether it can. If the two days are the same
    world, nothing can beat chance.

    Balance the two days by taking `min(len(reference), len(current))` rows from
    each, so chance is exactly one half and no accuracy has to be read against a
    majority class. Split each in half: the first half trains, the second is
    held out. Standardise every feature by the **training reference's** own mean
    and standard deviation, take the two class centroids, and assign each
    held-out row to whichever centroid is nearer. That is a linear classifier in
    six lines and it is enough to make the point.

    Then put block one's interval around the accuracy — `load_lab(1)` — and call
    the two days different only when the interval's lower bound is above chance.
    Return "accuracy", "correct", "held_out", "interval", "chance" and
    "detected".

    Do not skip the interval. Thirty-six held-out windows is a small number and
    the interval will show you exactly how small.
"""
from __future__ import annotations

import sys
import pathlib

import numpy as np
from scipy.stats import ttest_ind

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from lab_support import (CANDIDATE_FEATURES, DETECTION_SIZES,        # noqa: E402
                         DegenerateReference, NotSolved, READING_COUNT,
                         REQUIRED_FEATURES, SEED, load_lab)

LAB = 4

# The five the check grades, and why each is here rather than because it was
# available: the two window moments of the model's speed input, the target
# stand-in and its spread, and the suspected cause. lab_support states the same
# list once, so the loader, the lab and the check cannot drift apart.
#
# You are invited to add your own. lab_support.CANDIDATE_FEATURES offers
# max_speed, share_stopped, n_readings, mileage_delta and mean_battery, and
# verdict() takes any list you like:
#
#     verdict(reference, current, FEATURES + ["share_stopped"])
#
# The check grades the five required rows and prints yours beside them without
# judging them. Say in one line why each addition is there: a monitor watching
# two hundred columns is mostly watching its own arithmetic, which is what the
# twenty-questions slide is about.
FEATURES = list(REQUIRED_FEATURES)
TARGET = "mean_payload"

# The shift half of the material rule, and it is a choice of this course's
# rather than a measurement: two reference standard deviations. The index half
# is NOT a constant here and there is no MATERIAL_INDEX to read. It is derived,
# per feature, from the null Lab 2 measures.
MATERIAL_SHIFT_SD = 2.0

# The size the control injects when nobody asks for another. Comfortably above
# the measured noise floor and comfortably below MATERIAL_SHIFT_SD, so that what
# fires is the index rather than the shift rule -- which is the instrument the
# control is there to test. The sweep says what this single size cannot.
INJECTED_SHIFT_SD = 1.5


def verdict(reference, current, features=FEATURES, thresholds=None) -> dict:
    """Three measures, a threshold you derived, and a judgement, per feature.

    Definition graded by the check:
        Δ = ( mean_current − mean_reference ) / s_reference, ddof = 1
        (Glass, 1976). Choices: the reference period's own spread rather than a
        pooled one, and the sample standard deviation, ddof = 1 — the convention
        Module 5 grades as well. Slide: "Definition — the standardised shift, and
        the rule it is read against".

    And the rule the same slide states, with the index half of it derived rather
    than borrowed:
        material when |Δ| ≥ 2.0 or J ≥ threshold(B, q) derived from this feature's own null
        (Yurdakul & Naranjo, 2020). Choices: the shift bound is this course's,
        the index bound is measured per feature at the bin count in use; either
        instrument may fire; a refused index leaves the shift to decide alone.
    Needs: numpy, lab_support.load_lab, lab_support.DegenerateReference
    """
    # TODO: for each feature, derive its threshold, measure three ways, decide.
    # One column will refuse.
    raise NotSolved("verdict(reference, current, features, thresholds) still raises "
                    "instead of returning a verdict per feature")


def positive_control(reference, current,
                     injected_shift_sd: float = INJECTED_SHIFT_SD,
                     sizes=DETECTION_SIZES) -> dict:
    """Inject a shift of a known size, re-run the verdict, then sweep for the limit.

    Definition graded by the check:
        verdict( reference, current + k·s_reference ) must return material, with k stated beside the result
        (Saltelli et al., 2019). Choices: k = INJECTED_SHIFT_SD, deliberately
        below the shift threshold so that what fires is the index; a copy of the
        current frame, so the real answer is left exactly as it was. Slide:
        "Definition — the positive control".

    And the limit the sweep reports, which is the module's honest statement of
    what its instrument cannot see:
        detection limit = min{ k in the swept sizes : verdict( reference, current + j·s_reference ) is material for every swept j ≥ k }
        (Currie, 1968). Choices: the swept grid and its step, which is the
        resolution the answer is quoted to; the sustained crossing rather than
        the first one, because at this sample size the index flickers above the
        threshold before it stays there. Slide: "Definition — the detection
        limit".
    Needs: numpy, and the sizes in lab_support
    """
    # TODO: shift the target, run the unchanged verdict, then sweep every size
    # and report the smallest one the verdict still calls material.
    raise NotSolved("positive_control(reference, current, injected_shift_sd, sizes) "
                    "still raises instead of returning the verdict on an injected "
                    "shift and the detection limit swept out around it")


def drift_verdict(evidence: dict) -> tuple[str, str]:
    """One call out of three, and the reason you would defend it with.

    Definition graded by the check:
        act if the target is material; watch if its index is unmeasurable, or the control did not fire, or its index is above the floor; no material change only when the index is at or below the measured floor and the control fired
        (Saltelli et al., 2019). Choices: the order the four clauses are read
        in; that an untested instrument's silence is "watch" rather than a null;
        and that the reason must be built out of the evidence handed in. Slide:
        "Definition — the drift verdict, and the three calls it may make".
    Needs: nothing but the evidence you were handed
    """
    # TODO: read the four clauses in order, and write a reason out of the
    # numbers in `evidence` -- not out of a slide.
    raise NotSolved("drift_verdict(evidence) still raises instead of returning "
                    "(call, reason)")


def significance_is_not_size(sample_a, sample_b, readings: int = READING_COUNT,
                             seed: int = SEED) -> dict:
    """The same difference at two grains: what moves, and what does not.

    Definition graded by the check:
        t = ( m₁ − m₂ ) / √( s₁²/n₁ + s₂²/n₂ ) · the same difference resampled with replacement to n readings, seed 20200122
        (Welch, 1947; Efron, 1979). Choices: Welch rather than Student, so no
        common variance is assumed; the resampling split between the two days in
        proportion to the windows each has; the seed in the signature. Slide:
        "Definition — Welch's t-test, and the bootstrap to the reading grain".

    And the effect size, which this course grades with one pooling:
        d = ( m₁ − m₂ ) / s_pooled, s_pooled = √( ((n₁−1)s₁² + (n₂−1)s₂²) / (n₁+n₂−2) )
        (Cohen, 1988). Choices: m₁ is the second sample, so the sign points from
        the reference to today; the pooling is weighted by degrees of freedom,
        which differs from the unweighted root mean square whenever the two
        samples differ in size — and here they do. Slide: "Definition — Cohen's d".
    Needs: scipy, numpy
    """
    # TODO: a t-test at each grain, and Cohen's d.
    raise NotSolved("significance_is_not_size(sample_a, sample_b) still raises instead "
                    "of returning p-values and an effect size")


def classifier_two_sample_test(reference, current, features=FEATURES,
                               seed: int = SEED) -> dict:
    """Can anything tell the two days apart? The closing test, from the reading.

    Definition graded by the check:
        the two samples differ when the interval around a held-out classifier's accuracy lies above chance, chance = 1/2 on balanced classes
        (Rabanser, Günnemann & Lipton, 2019). Choices: the two days balanced by
        taking the smaller count from each, so chance is one half rather than a
        majority share; one half of each trains and one half is held out;
        features standardised by the training reference alone; the nearer of two
        class centroids as the rule; and Wilson's interval from Lab 1 at 95 per
        cent. Slide: "Definition — the classifier two-sample test".
    Needs: numpy, lab_support.load_lab
    """
    # TODO: balance, split, standardise, two centroids, held-out accuracy, and
    # block one's interval around it.
    raise NotSolved("classifier_two_sample_test(reference, current, features, seed) "
                    "still raises instead of returning an accuracy with an interval")


if __name__ == "__main__":
    from lab_support import reference_and_current

    reference, current = reference_and_current()
    print(f"{'feature':14}{'shift SD':>10}{'index':>12}{'threshold':>12}"
          f"{'wasserstein':>13}{'material':>10}")
    for feature, result in verdict(reference, current).items():
        index = (f"{result['population_stability_index']:.3f}"
                 if result["index_measured"] else "unmeasured")
        threshold = (f"{result['index_threshold']:.3f}"
                     if result["index_measured"] else "unmeasured")
        print(f"{feature:14}{result['shift_in_reference_sd']:+10.2f}{index:>12}"
              f"{threshold:>12}{result['wasserstein']:13.3f}"
              f"{str(result['material']):>10}")

    control = positive_control(reference, current)
    print(f"\npositive control: {control['injected_shift_sd']} SD injected into the "
          f"target -> index {control['population_stability_index']:.3f}, "
          f"material {control['material']}")
    print(f"detection limit:  {control['detection_limit_sd']} reference SD "
          f"({control['detection_limit_in_target_units']:.1f} kilograms) — below that "
          f"this instrument cannot see a shift in the target")
