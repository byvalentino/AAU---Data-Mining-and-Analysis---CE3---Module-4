"""Lab 4, solved — with the reasoning, not only the code.

Run it: `python3 solutions/lab_04.py` from the exercises directory. It derives
each feature's own threshold, narrates the verdict, sweeps the positive control
for a detection limit, writes the call and its reason, runs the classifier
two-sample test from the required reading, and writes block four's pictures
under out/.
"""
from __future__ import annotations

import sys
import pathlib

import numpy as np
from scipy.stats import ttest_ind

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from lab_support import (CANDIDATE_FEATURES, DETECTION_SIZES,        # noqa: E402
                         DegenerateReference, READING_COUNT,
                         REQUIRED_FEATURES, SEED, load_lab)

LAB = 4

# The five the check grades, named once in lab_support so that the loader, the
# lab and the check cannot drift apart: the two window moments of the model's
# speed input, the target stand-in and its spread, and the suspected cause.
# lab_support.CANDIDATE_FEATURES offers five more for anybody who wants to add
# their own; verdict() takes any list.
FEATURES = list(REQUIRED_FEATURES)
TARGET = "mean_payload"

# The shift half of the rule, and a choice rather than a measurement. There is
# deliberately no MATERIAL_INDEX beside it: the index half is derived from the
# null Lab 2 measures, per feature, at the bin count in use.
MATERIAL_SHIFT_SD = 2.0
INJECTED_SHIFT_SD = 1.5


def verdict(reference, current, features=FEATURES, thresholds=None) -> dict:
    """Three measures, a threshold you derived, and a judgement, per feature.

    What this returns, on the real archive at five bins, is worth stating plainly
    because it is the lesson:

        mean_speed      moved +2.30 reference standard deviations, index 2.289
                        against a threshold of 1.354 derived from its own null.
                        Material by both instruments. The real change.
        sd_speed        moved -0.47, index 2.975 against a threshold of 0.431 --
                        the largest index of all. Material by the index alone,
                        and the two measures rank the features differently
                        because they answer different questions.
        sd_payload      moved -0.47, index 0.422 against a threshold of 0.440.
                        NOT material -- and banking's 0.25 would have called it
                        material without ever saying how close the call was.
        human_driven    moved +1.24, and the index REFUSES: nought in 39 of 45
                        reference windows, so the bins collapse. This is the
                        column that explains the whole event -- a human drove
                        41 per cent of the second day against 9 per cent of the
                        first -- and it is the one the index cannot see.
        mean_payload    the target. Moved -0.03, index 0.081 -- below its own
                        measured noise floor of 0.105. Nothing.

    So: two inputs are material, the cause is in a column the index could not
    measure, and the thing the model predicts sits below the floor of the
    instrument watching it.

    A monitor watching inputs would have fired. A monitor watching the target
    would not. Both would have been right, and the operator's question -- do we
    need to do anything? -- is answered by the second.

    That is the ordinary case in production and it is why "no material change"
    has to be a sayable conclusion. A drift detector that only ever finds drift
    is not a detector.

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
    Needs: numpy.asarray, numpy.std, lab_support.load_lab, lab_support.DegenerateReference
    """
    two, three = load_lab(2), load_lab(3)
    thresholds = dict(thresholds or {})
    results = {}
    for feature in features:
        before = np.asarray(reference[feature].dropna(), dtype=float)
        after = np.asarray(current[feature].dropna(), dtype=float)
        spread = float(np.std(before, ddof=1))

        measures = {
            "shift_in_reference_sd": float((after.mean() - before.mean()) / spread)
                                     if spread else float("nan"),
            "wasserstein": three.wasserstein(before, after),
        }
        try:
            measures["population_stability_index"] = three.compare_four(
                before, after)["population_stability_index"]
            # The null is built out of the reference alone, so a caller sweeping
            # the current day can derive it once and hand it back -- that is a
            # statement that the instrument did not change across the sweep, not
            # a shortcut. Anything not handed over is derived here.
            derived = thresholds.get(feature)
            if derived is None:
                derived = two.index_threshold(before, after, bins=three.DEFAULT_BINS)
            measures["noise_floor"] = derived["noise_floor"]
            measures["index_threshold"] = derived["threshold"]
            measures["index_measured"] = True
        except DegenerateReference:
            # Not a crash and not a nought. The reference cannot be binned, so
            # this instrument has no reading to give and no threshold either,
            # and recording nought here would put "did not move" in a report
            # about a column that did.
            measures["population_stability_index"] = None
            measures["noise_floor"] = None
            measures["index_threshold"] = None
            measures["index_measured"] = False

        # Either instrument may fire. Requiring both would miss a change in
        # shape that leaves the mean alone, and a shift in mean that leaves the
        # binned shape alone -- and each of those happens here.
        measures["material"] = bool(
            abs(measures["shift_in_reference_sd"]) >= MATERIAL_SHIFT_SD
            or (measures["index_measured"]
                and measures["population_stability_index"]
                >= measures["index_threshold"]))
        results[feature] = measures
    return results


def positive_control(reference, current,
                     injected_shift_sd: float = INJECTED_SHIFT_SD,
                     sizes=DETECTION_SIZES) -> dict:
    """Prove the detector can detect, then find out how small a change it misses.

    "The target did not move" is an absence claim, and an absence claim from an
    instrument nobody has tested is an opinion. So put a shift of a known size
    through the *unchanged* verdict and confirm it comes back material.

    At 1.5 reference standard deviations the shift rule (2.0) does not fire, so
    what fires is the index: 8.221 against a threshold of 0.465 derived from the
    target's own null. That is the sentence the null result rests on -- the same
    code, on the same grain, detects movement when there is some.

    But a control at one size only says the detector detects *that* size. So
    sweep. Walking 0.00 to 1.50 reference standard deviations in steps of 0.05,
    the verdict is material from 0.40 upwards and stays material: 0.40 reference
    standard deviations, about 44 kilograms of mean payload per five-minute
    window, is this instrument's detection limit on this target. Below it, the
    honest thing to say is that we would not have seen it.

    Two details of the sweep are worth more than the number. First, the index
    flickers above the threshold at 0.15 and falls back at 0.20: at thirty-five
    current windows, moving a handful of them across a quantile edge moves the
    index a long way, so the limit has to be the size from which the answer
    *stays* material rather than the first size at which it fires. Second, the
    threshold is derived once and reused across the sweep, because the null is
    built from the reference alone and the reference did not change.

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
    Needs: numpy.std, frame.copy
    """
    two, three = load_lab(2), load_lab(3)
    before = np.asarray(reference[TARGET].dropna(), dtype=float)
    spread = float(np.std(before, ddof=1))

    # Derived once. The null resamples the reference against itself, so nothing
    # the sweep does to the current day can change it -- and holding it fixed is
    # what "the unchanged verdict" means.
    derived = {TARGET: two.index_threshold(
        before, np.asarray(current[TARGET].dropna(), dtype=float),
        bins=three.DEFAULT_BINS)}

    def injected(size):
        moved = current.copy()
        moved[TARGET] = moved[TARGET] + size * spread
        return verdict(reference, moved, [TARGET], derived)[TARGET]

    result = dict(injected(injected_shift_sd))
    result["injected_shift_sd"] = injected_shift_sd

    swept = [float(size) for size in sizes]
    rows = [injected(size) for size in swept]
    material = [bool(row["material"]) for row in rows]
    index = [row["population_stability_index"] for row in rows]

    # The first crossing and the sustained one. They differ, and the difference
    # is what a detection limit is for.
    first = next((size for size, fired in zip(swept, material) if fired), None)
    limit = next((size for position, size in enumerate(swept)
                  if all(material[position:])), None)

    result["sizes"] = swept
    result["material_by_size"] = material
    result["index_by_size"] = index
    result["first_material_sd"] = first
    result["detection_limit_sd"] = limit
    result["detection_limit_in_target_units"] = (None if limit is None
                                                 else float(limit * spread))
    result["target_reference_sd"] = spread
    return result


def drift_verdict(evidence: dict) -> tuple[str, str]:
    """One call out of three, and the reason you would defend it with.

    The order of the four clauses is the whole design, and the third is the one
    people leave out: an instrument nobody has tested has not said "quiet", it
    has said nothing, and until the control fires the honest call is "watch"
    rather than a null.

    Note what the reason may not contain: a number that is not in the evidence.
    Every figure quoted below is formatted out of the dictionary handed in, so
    the sentence is a report of measurements rather than a recital, and it
    changes when the measurements change. That is the property the check grades,
    and it is why a sentence copied off a slide fails it.

    Definition graded by the check:
        act if the target is material; watch if its index is unmeasurable, or the control did not fire, or its index is above the floor; no material change only when the index is at or below the measured floor and the control fired
        (Saltelli et al., 2019). Choices: the order the four clauses are read
        in; that an untested instrument's silence is "watch" rather than a null;
        and that the reason must be built out of the evidence handed in. Slide:
        "Definition — the drift verdict, and the three calls it may make".
    Needs: nothing but the evidence you were handed
    """
    index = evidence.get("target_index")
    measured = bool(evidence.get("target_index_measured"))
    shift = float(evidence.get("standardised_shift"))
    floor = evidence.get("noise_floor")
    threshold = evidence.get("index_threshold")
    control = evidence.get("control_index")
    limit = evidence.get("detection_limit")
    material_count = evidence.get("material_count")

    fired = (control is not None and threshold is not None and control >= threshold)
    material = (abs(shift) >= MATERIAL_SHIFT_SD
                or (measured and threshold is not None and index >= threshold))

    if material:
        seen = (f"its index {index:.3f} is at or above the index threshold "
                f"{threshold:.3f}" if measured and index is not None and
                threshold is not None
                else f"its standardised shift is {shift:.2f}")
        return "act", (
            f"The target itself moved: {seen}, on a standardised shift of "
            f"{shift:.2f} reference standard deviations. That is a change in the "
            f"thing the model predicts, not in an input, so watching it is not "
            f"enough; find the cause before the next release.")

    if not measured:
        return "watch", (
            f"The target's index could not be measured at all, so the only "
            f"instrument left is the standardised shift, at {shift:.2f}. An "
            f"absence nobody could measure is not a null result; keep watching "
            f"and find a second way to look at this column.")

    if not fired:
        control_text = ("no control was run" if control is None
                        else f"the control index reached only {control:.3f}")
        return "watch", (
            f"The target's index is {index:.3f}, below the index threshold "
            f"{threshold:.3f}, but {control_text} — so the detector has not been "
            f"shown to detect anything. Silence from an untested instrument is "
            f"not evidence of quiet.")

    if floor is not None and index <= floor:
        return "no material change", (
            f"The target's index is {index:.3f}, at or below its own measured "
            f"noise floor of {floor:.3f} and far below the index threshold "
            f"{threshold:.3f} derived from that same null; the standardised "
            f"shift is {shift:.2f} reference standard deviations; and the "
            f"positive control fired at {control:.3f}, so the instrument was "
            f"shown to work before the silence was believed. The "
            f"{material_count} material features are inputs, not the target, "
            f"and the detection limit is {limit:.2f} reference standard "
            f"deviations, which is what we would still have missed.")

    return "watch", (
        f"The target's index is {index:.3f}: above its measured noise floor of "
        f"{floor:.3f} but below the index threshold {threshold:.3f}, on a "
        f"standardised shift of {shift:.2f}. That is measurable movement that "
        f"is not distinguishable from noise, which is a reason to keep "
        f"measuring rather than to declare either way.")


def significance_is_not_size(sample_a, sample_b, readings: int = READING_COUNT,
                             seed: int = SEED) -> dict:
    """The same difference at two grains. Only one of these numbers is about the world.

    A p-value answers: if there were no difference at all, how often would I see
    a gap this large by chance? That answer depends on how many observations I
    took, so with enough rows every difference becomes significant. It is a
    statement about the sample.

    An effect size answers: how big is the gap, in units of the spread? That does
    not depend on how many observations I took. It is a statement about the
    world.

    Measured here on mean speed: at the window grain, eighty observations, the
    p-value is around 0.0002. Resampled to the reading grain -- 48,290
    observations of the identical difference -- it falls through the floor.
    Cohen's d does not move at all, because nothing about the difference changed.

    The resampling is seeded, and the seed is in the signature rather than hidden
    in the body, because a number that changes every time you run it cannot be
    quoted, checked or defended. The archive's own reading-grain test agrees with
    this simulation: it underflows to exactly 0.0, which is why the slide prints a
    bound rather than a zero.

    Which is why a monitor thresholded on a p-value fires constantly in
    production, where sample sizes are enormous, and tells you nothing.

    Definition graded by the check:
        t = ( m₁ − m₂ ) / √( s₁²/n₁ + s₂²/n₂ ) · the same difference resampled with replacement to n readings, seed 20200122
        (Welch, 1947; Efron, 1979). Choices: Welch rather than Student, so no
        common variance is assumed; the resampling split between the two days in
        proportion to the windows each has; the seed in the signature. Slide:
        "Definition — Welch's t-test, and the bootstrap to the reading grain".

    And the effect size, with the one pooling this course grades:
        d = ( m₁ − m₂ ) / s_pooled, s_pooled = √( ((n₁−1)s₁² + (n₂−1)s₂²) / (n₁+n₂−2) )
        (Cohen, 1988). Choices: m₁ is the second sample, so the sign points from
        the reference to today; the pooling is weighted by degrees of freedom.
        The unweighted root mean square of the two variances is a different
        number whenever the samples differ in size, and here they do — 45 windows
        against 35, 1.021 against 0.966. Slide: "Definition — Cohen's d".
    Needs: scipy.stats.ttest_ind, numpy.random.default_rng, rng.choice, numpy.var
    """
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)

    windows = ttest_ind(a, b, equal_var=False)

    # The same difference, resampled to a much larger n, split between the two
    # days in proportion to the windows each day actually has. Nothing about the
    # world changed -- only how much of it we looked at.
    rng = np.random.default_rng(seed)
    share_a = int(round(readings * len(a) / (len(a) + len(b))))
    big_a = rng.choice(a, size=share_a, replace=True)
    big_b = rng.choice(b, size=readings - share_a, replace=True)
    readings_test = ttest_ind(big_a, big_b, equal_var=False)

    # Cohen's pooling, weighted by degrees of freedom rather than the unweighted
    # root mean square of the two variances. The two agree only when the samples
    # are the same size, and these are 45 windows against 35.
    first, second = len(a), len(b)
    pooled = np.sqrt(((first - 1) * np.var(a, ddof=1) + (second - 1) * np.var(b, ddof=1))
                     / (first + second - 2))
    effect = float((b.mean() - a.mean()) / pooled) if pooled else float("nan")

    return {
        "p_value_windows": float(windows.pvalue),
        "n_windows": int(len(a) + len(b)),
        "p_value_readings": float(readings_test.pvalue),
        "n_readings": int(len(big_a) + len(big_b)),
        "effect_size": effect,
    }


def classifier_two_sample_test(reference, current, features=FEATURES,
                               seed: int = SEED) -> dict:
    """Can anything tell the two days apart? The test the required reading argues for.

    Rabanser, Günnemann and Lipton (2019) put every method in this lab against
    one more and found this one hard to beat: train a classifier to tell a
    reference row from a current row, and ask whether it can. If the two days
    are one world, nothing can beat chance.

    Measured here on the five features: 24 of 36 held-out windows, an accuracy
    of 0.667, and Wilson's interval on it is [0.503, 0.798]. The lower bound
    clears one half by three thousandths. So the classifier detects, and block
    one's interval says how thin the finding is: on 36 held-out windows the
    honest report is "distinguishable, barely, and I cannot tell you by how
    much".

    Which is the comparison this module owes the reading. Rabanser and colleagues
    run these tests on thousands of samples, where a classifier's extra power
    over four univariate marginals is real and measurable. On eighty five-minute
    windows it is not: the interval is nearly three tenths of a unit wide, and
    the answer moves with the split. The four measures in Labs 2 and 3 are not
    better -- they are cheaper, they name *which* feature moved, and at this
    sample size they are no worse.

    And on the target alone the two agree, which is the point that matters: the
    classifier scores 0.417 with an interval of [0.271, 0.578], not above
    chance, and the index scores 0.081 against a floor of 0.105. Two very
    different instruments, one answer.

    Definition graded by the check:
        the two samples differ when the interval around a held-out classifier's accuracy lies above chance, chance = 1/2 on balanced classes
        (Rabanser, Günnemann & Lipton, 2019). Choices: the two days balanced by
        taking the smaller count from each, so chance is one half rather than a
        majority share; one half of each trains and one half is held out;
        features standardised by the training reference alone; the nearer of two
        class centroids as the rule; and Wilson's interval from Lab 1 at 95 per
        cent. Slide: "Definition — the classifier two-sample test".
    Needs: numpy.random.default_rng, numpy.mean, numpy.std, lab_support.load_lab
    """
    one = load_lab(1)
    rng = np.random.default_rng(seed)

    before = np.asarray(reference[list(features)].dropna(), dtype=float)
    after = np.asarray(current[list(features)].dropna(), dtype=float)

    # Balanced on purpose: with 45 windows against 35, always answering "the
    # reference" scores 0.5625, and an accuracy read against the wrong chance
    # level is the commonest way this test is got wrong.
    size = min(len(before), len(after))
    before = before[rng.permutation(len(before))[:size]]
    after = after[rng.permutation(len(after))[:size]]

    train = size // 2
    # Standardised by the training reference alone. Standardising with both days
    # together would let the held-out rows inform the scaling, which is the
    # quiet kind of leakage: the test would then be partly about itself.
    centre = before[:train].mean(axis=0)
    scale = before[:train].std(axis=0, ddof=1)
    scale = np.where(scale > 0, scale, 1.0)

    reference_centroid = ((before[:train] - centre) / scale).mean(axis=0)
    current_centroid = ((after[:train] - centre) / scale).mean(axis=0)

    def says_current(rows):
        standardised = (rows - centre) / scale
        to_current = ((standardised - current_centroid) ** 2).sum(axis=1)
        to_reference = ((standardised - reference_centroid) ** 2).sum(axis=1)
        return to_current < to_reference

    correct = int((~says_current(before[train:])).sum()
                  + says_current(after[train:]).sum())
    held_out = int(len(before[train:]) + len(after[train:]))
    low, high = one.wilson_interval(correct, held_out)

    return {
        "accuracy": correct / held_out,
        "correct": correct,
        "held_out": held_out,
        "interval": (float(low), float(high)),
        "chance": 0.5,
        # The interval, not the point estimate. An accuracy of 0.667 on 36 rows
        # is compatible with a great deal, including very little.
        "detected": bool(low > 0.5),
        "features": list(features),
    }


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    import plotly.graph_objects as go                                # noqa: E402
    from plotly.subplots import make_subplots                        # noqa: E402
    import pandas as pd                                              # noqa: E402
    from _narrate import (narrator, show_table, save_figure,         # noqa: E402
                          reference_lab)
    from lab_support import BORROWED_INDEX, reference_and_current    # noqa: E402

    # For the length of this demonstration, "Lab 3" means the shipped solution
    # rather than whatever is in labs/. verdict() reads this name at call time,
    # so rebinding it here is enough; the check still imports the student's lab.
    load_lab = reference_lab                                         # noqa: F811

    BLUE, ORANGE, GREY, RED = "#2A78D6", "#E07B39", "#52514E", "#C0392B"

    say = narrator(LAB)
    say.info("Lab 4 — a threshold derived rather than borrowed, the verdict it "
             "produces, the detection limit behind it, and the classifier the "
             "required reading argues for")

    reference, current = reference_and_current()
    say.info("archive slice, %d reference windows on the first day against %d on the "
             "second, five-minute windows of at least 300 readings, one vehicle",
             len(reference), len(current))
    say.info("the five graded features are the model's two speed moments, the target "
             "stand-in and its spread, and the suspected cause: %s", ", ".join(FEATURES))

    # 1. The verdict, on thresholds derived from each feature's own null.
    results = verdict(reference, current)
    rows = []
    for feature, measured in results.items():
        rows.append({
            "feature": feature,
            "shift (reference s.d.)": round(measured["shift_in_reference_sd"], 3),
            "index": (round(measured["population_stability_index"], 3)
                      if measured["index_measured"] else "unmeasured"),
            "floor": (round(measured["noise_floor"], 3)
                      if measured["index_measured"] else "unmeasured"),
            "threshold": (round(measured["index_threshold"], 3)
                          if measured["index_measured"] else "unmeasured"),
            "distance (own units)": round(measured["wasserstein"], 3),
            "material": measured["material"]})
    show_table(pd.DataFrame(rows), "the verdict, feature by feature, each against a "
                                   "threshold derived from its own null", logger=say)

    material = [name for name, measured in results.items() if measured["material"]]
    say.info("%d of %d features are material by the rule |shift| >= %.1f or index >= "
             "the threshold derived for that feature: %s", len(material),
             len(FEATURES), MATERIAL_SHIFT_SD, ", ".join(material))
    borrowed = [name for name, measured in results.items()
                if abs(measured["shift_in_reference_sd"]) >= MATERIAL_SHIFT_SD
                or (measured["index_measured"]
                    and measured["population_stability_index"] >= BORROWED_INDEX)]
    say.info("with banking's borrowed %.2f in place of the derived thresholds it would "
             "be %d: %s — the extra alarm is %s, whose index %.3f sits just under its "
             "own threshold of %.3f. A borrowed threshold cannot tell you how close a "
             "call was", BORROWED_INDEX, len(borrowed), ", ".join(borrowed),
             ", ".join(sorted(set(borrowed) - set(material))),
             results["sd_payload"]["population_stability_index"],
             results["sd_payload"]["index_threshold"])
    say.info("and the target, %s, moved %+.2f reference standard deviations with an "
             "index of %.3f against its own measured noise floor of %.3f — below the "
             "floor, so the instrument cannot tell it from nothing happening",
             TARGET, results[TARGET]["shift_in_reference_sd"],
             results[TARGET]["population_stability_index"],
             results[TARGET]["noise_floor"])
    say.info("human_driven is the column that explains the whole event and the one "
             "the index cannot measure: index_measured is %s, so the shift decides "
             "alone", results["human_driven"]["index_measured"])

    # 2. The invitation: any list of features, the five graded and the rest printed.
    extra = verdict(reference, current, FEATURES + ["share_stopped", "mileage_delta"])
    say.info("the same verdict over the five plus share_stopped and mileage_delta — "
             "share_stopped moved %+.2f, mileage_delta %+.2f. Adding a column costs a "
             "name in a list; the check grades the five and prints yours beside them",
             extra["share_stopped"]["shift_in_reference_sd"],
             extra["mileage_delta"]["shift_in_reference_sd"])

    # 3. The positive control, and the sweep that turns it into a detection limit.
    control = positive_control(reference, current)
    say.info("positive control: %.1f reference standard deviations injected into the "
             "target, put through the unchanged verdict -> shift %+.2f, index %.3f "
             "against a threshold of %.3f, material %s", control["injected_shift_sd"],
             control["shift_in_reference_sd"], control["population_stability_index"],
             control["index_threshold"], control["material"])
    say.info("the sweep, %.2f to %.2f reference standard deviations in steps of %.2f: "
             "material from %.2f upwards and material at every larger size — a "
             "detection limit of %.2f reference standard deviations, %.1f kilograms of "
             "mean payload per five-minute window", control["sizes"][0],
             control["sizes"][-1], control["sizes"][1] - control["sizes"][0],
             control["detection_limit_sd"], control["detection_limit_sd"],
             control["detection_limit_in_target_units"])
    say.info("it first fires at %.2f and falls back at the next size, which is why the "
             "limit is the sustained crossing: at %d current windows a handful of them "
             "crossing a quantile edge moves the index a long way",
             control["first_material_sd"], len(current))
    untouched = verdict(reference, current)[TARGET]
    say.info("and the real answer is untouched afterwards: material %s — the control "
             "shifted a copy, not the archive", untouched["material"])

    # 4. The call, and the reason it would be defended with.
    evidence = {
        "target_index": results[TARGET]["population_stability_index"],
        "target_index_measured": results[TARGET]["index_measured"],
        "standardised_shift": results[TARGET]["shift_in_reference_sd"],
        "noise_floor": results[TARGET]["noise_floor"],
        "index_threshold": results[TARGET]["index_threshold"],
        "control_index": control["population_stability_index"],
        "detection_limit": control["detection_limit_sd"],
        "material_features": material,
        "material_count": len(material),
        "features_watched": len(FEATURES),
    }
    call, reason = drift_verdict(evidence)
    say.info("the call: %s", call)
    say.info("the reason, built out of the evidence and out of nothing else: %s", reason)

    # 5. Significance is not size.
    outcome = significance_is_not_size(reference["mean_speed"].dropna(),
                                       current["mean_speed"].dropna())
    say.info("mean speed, window grain: p = %.6g on %d observations",
             outcome["p_value_windows"], outcome["n_windows"])
    say.info("the identical difference, resampled to the reading grain: p = %.3g on "
             "%d observations — it underflows the smallest number a double can hold, "
             "so report it as a bound rather than as a nought. Nothing about the "
             "world changed", outcome["p_value_readings"], outcome["n_readings"])
    say.info("Cohen's d = %.3f, pooled by degrees of freedom, and it does not move "
             "with the sample size — that is the number about the world",
             outcome["effect_size"])

    # 6. The closing test, from the required reading.
    joint = classifier_two_sample_test(reference, current)
    say.info("classifier two-sample test on all five features: %d of %d held-out "
             "windows, accuracy %.3f, Wilson interval [%.3f, %.3f] against a chance "
             "of %.1f — detected %s, and the width of that interval is the honest "
             "part of the finding", joint["correct"], joint["held_out"],
             joint["accuracy"], joint["interval"][0], joint["interval"][1],
             joint["chance"], joint["detected"])
    alone = classifier_two_sample_test(reference, current, [TARGET])
    say.info("the same test on the target alone: accuracy %.3f, interval [%.3f, "
             "%.3f], detected %s — which is the index's answer reached by an "
             "entirely different route", alone["accuracy"], alone["interval"][0],
             alone["interval"][1], alone["detected"])

    # 7. Picture one: the five shifts, coloured by the verdict they were given.
    names = list(results)
    shifts = [results[name]["shift_in_reference_sd"] for name in names]
    bars = go.Figure(go.Bar(
        x=shifts, y=[name.replace("_", " ") for name in names], orientation="h",
        marker_color=[ORANGE if results[name]["material"] else BLUE for name in names],
        text=[f"{value:+.2f}" for value in shifts], textposition="outside",
        cliponaxis=False))
    for name, value in zip(names, shifts):
        if not results[name]["index_measured"]:
            bars.add_annotation(x=value + 0.55, y=name.replace("_", " "),
                                text="index unmeasured", showarrow=False,
                                font=dict(size=13, color=RED))
    bars.update_layout(
        title=f"{len(material)} of {len(FEATURES)} material (orange), each against a "
              "threshold derived from its own null. The target is not.",
        xaxis_title="shift from the first day, in the first day's standard deviations",
        yaxis_title="", showlegend=False)
    bars.update_yaxes(autorange="reversed")
    bars.update_xaxes(range=[min(shifts) - 0.9, max(shifts) + 1.4])
    save_figure(bars, "verdict_by_feature", LAB, logger=say)

    # 8. Picture two: the target before and after the injected shift, so that the
    #    null result and the control that backs it sit side by side.
    spread = control["target_reference_sd"]
    injected_values = (current[TARGET].dropna().to_numpy()
                       + control["injected_shift_sd"] * spread)
    edges = np.linspace(min(reference[TARGET].min(), current[TARGET].min()),
                        max(injected_values.max(), reference[TARGET].max()), 24)
    centres = (edges[:-1] + edges[1:]) / 2
    width = float(edges[1] - edges[0])
    control_figure = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.08,
        subplot_titles=(f"as measured: {results[TARGET]['shift_in_reference_sd']:+.2f} "
                        "reference s.d., not material",
                        f"with {control['injected_shift_sd']} s.d. injected: "
                        f"{control['shift_in_reference_sd']:+.2f}, material"))
    for column, after in ((1, current[TARGET].dropna().to_numpy()),
                          (2, injected_values)):
        control_figure.add_bar(
            x=centres, y=np.histogram(reference[TARGET].dropna(), bins=edges)[0],
            width=width, marker_color=BLUE, opacity=0.75, name="22 January",
            showlegend=column == 1, row=1, col=column)
        control_figure.add_bar(
            x=centres, y=np.histogram(after, bins=edges)[0], width=width,
            marker_color=ORANGE if column == 1 else RED, opacity=0.75,
            name="23 January" if column == 1 else "23 January, shifted on purpose",
            showlegend=True, row=1, col=column)
    control_figure.update_layout(
        barmode="overlay", title="The null result, and the control that makes it credible",
        legend=dict(orientation="h", x=0.15, y=1.16))
    control_figure.update_xaxes(title_text="mean payload per five-minute window (kilograms)")
    control_figure.update_yaxes(title_text="windows", col=1)
    save_figure(control_figure, "target_and_positive_control", LAB, logger=say)

    # 9. Picture three: the sweep itself, which is where the detection limit is.
    sweep = go.Figure()
    sweep.add_scatter(x=control["sizes"], y=control["index_by_size"],
                      mode="lines+markers", line=dict(color=BLUE, width=2.5),
                      name="index of the injected target")
    sweep.add_hline(y=control["index_threshold"], line=dict(color=ORANGE, dash="dash"),
                    annotation_text=f"threshold derived from the null "
                                    f"({control['index_threshold']:.3f})")
    sweep.add_hline(y=results[TARGET]["noise_floor"],
                    line=dict(color=GREY, dash="dot"),
                    annotation_text=f"measured noise floor "
                                    f"({results[TARGET]['noise_floor']:.3f})")
    sweep.add_vline(x=control["detection_limit_sd"], line=dict(color=RED, width=2),
                    annotation_text=f"detection limit "
                                    f"{control['detection_limit_sd']:.2f} s.d.")
    sweep.update_layout(
        title="How small a shift this instrument can still see",
        xaxis_title="shift injected into the target, in reference standard deviations",
        yaxis_title="symmetrised index (nats)", yaxis_type="log")
    save_figure(sweep, "detection_limit", LAB, logger=say)

    say.info("what the check grades: a threshold derived from Lab 2's null rather than "
             "borrowed, mean_speed material and the target not, the largest index "
             "belonging to sd_speed, the degenerate column reported unmeasured, a "
             "control that fires and leaves the real answer alone, a detection limit "
             "swept rather than assumed, one call out of three with a reason built "
             "from the evidence, Cohen's d pooled by degrees of freedom, and a "
             "classifier whose accuracy carries block one's interval")
