#!/usr/bin/env python3
"""Check 4 — a threshold derived rather than borrowed, the verdict it produces,
the control and the detection limit behind it, the call, and the closing test."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _harness import run, not_ready, grade_reason, explain            # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
try:
    import numpy as np                                               # noqa: E402
    from lab_support import (BORROWED_INDEX, load_lab,                # noqa: E402
                             reference_and_current)                  # noqa: E402
except ImportError as unready:                                       # noqa: E402
    not_ready(unready)

DEGENERATE = "human_driven"

# The five rows this check grades, held here rather than read off the lab: a
# student is invited to add features, and inviting additions is not the same as
# letting the five required ones be edited away. Anything else the student
# measured is printed below, unjudged.
REQUIRED = ("mean_speed", "sd_speed", "sd_payload", "human_driven", "mean_payload")

MEASURES = ("shift_in_reference_sd", "population_stability_index", "wasserstein",
            "material", "index_measured", "noise_floor", "index_threshold")

# The six situations drift_verdict has to tell apart, and the call each one
# deserves. Four of them would be answered "no material change" by anybody
# reading only the target's index, and three of those four are wrong.
VERDICT_KEY = "m4:verdict"


def body(lab):
    reference, current = reference_and_current()
    results = lab.verdict(reference, current)

    for feature in REQUIRED:
        assert feature in results, (
            f"verdict() returned nothing for '{feature}'. The five required features "
            f"are {', '.join(REQUIRED)}; you may add as many more as you can justify, "
            "but these five are the ones this check reads.")
        for key in MEASURES:
            assert key in results[feature], (
                f"verdict()['{feature}'] has no '{key}'. Every feature's row carries "
                "its three measures, the floor and the threshold it was judged "
                "against, and the judgement — a verdict that does not print what it "
                "judged by cannot be argued with.")

    # Whatever else the student chose to watch: measured the same way, printed
    # here, and judged nowhere. The only thing asked of it is that it has the
    # same shape as the rest, so that a report can be written from one table.
    chosen = [name for name in results if name not in REQUIRED]
    for feature in chosen:
        missing = [key for key in MEASURES if key not in results[feature]]
        assert not missing, (
            f"you added '{feature}' to the features and its row is missing {missing}. "
            "Added features are not graded, but they have to come back in the same "
            "shape as the five, or the verdict table cannot be printed as one table.")
    if chosen:
        print(f"  also measured, not graded: {', '.join(sorted(chosen))}")

    # ------------------------------------------------------------------
    # The threshold, derived per feature rather than quoted from a slide.
    # ------------------------------------------------------------------
    # This is the module's own headline lesson, and it is graded here as well as
    # in Lab 2 because it is here that it decides an answer.
    two, three = load_lab(2), load_lab(3)
    for feature in REQUIRED:
        row = results[feature]
        if not row["index_measured"]:
            continue
        before = np.asarray(reference[feature].dropna(), dtype=float)
        after = np.asarray(current[feature].dropna(), dtype=float)
        own = two.index_threshold(before, after, bins=three.DEFAULT_BINS)["threshold"]
        assert abs(row["index_threshold"] - own) <= 0.3 * own, explain(
            VERDICT_KEY + ":threshold",
            f"on {feature} your verdict judged against a threshold of "
            f"{row['index_threshold']:.4f}; deriving it from that feature's own null "
            f"here gives {own:.4f}",
            "The threshold is not a constant and it is not on a slide. It is a "
            "stated quantile of the null your own index produces when the reference "
            "is compared against a resample of itself, at the bin count you are "
            f"comparing at. Note what this rejects: {BORROWED_INDEX}, credit "
            "scoring's number, which is what block two spends six bullets on.")

    spread_of = {name: float(np.std(np.asarray(reference[name].dropna(), dtype=float),
                                    ddof=1)) for name in REQUIRED}

    thresholds = {name: results[name]["index_threshold"] for name in REQUIRED
                  if results[name]["index_measured"]}
    assert max(thresholds.values()) - min(thresholds.values()) > 0.2, (
        f"every feature in your verdict was judged against about the same threshold "
        f"({', '.join(f'{name} {value:.3f}' for name, value in thresholds.items())}). "
        "Two columns with different shapes and different numbers of ties do not have "
        "the same null: on this archive mean_speed's threshold is three times "
        "sd_speed's. One threshold for every column is the borrowed kind of number "
        "with a different provenance.")

    # The one input that really moved.
    speed = results["mean_speed"]
    assert speed["material"] is True, (
        f"mean_speed shifted {speed['shift_in_reference_sd']:+.2f} reference standard "
        f"deviations with an index of {speed['population_stability_index']} against a "
        f"threshold of {speed['index_threshold']}, and your verdict calls it "
        "immaterial. It is the one input that genuinely moved.")

    # And the point of the whole lab: the target did not.
    target = results[lab.TARGET]
    assert abs(target["shift_in_reference_sd"]) < 0.5, (
        f"your verdict has the target shifting {target['shift_in_reference_sd']:+.2f} "
        "reference standard deviations. On this grain — one vehicle, five-minute "
        "windows of at least 300 readings, the first day as reference — it moves by a "
        "few hundredths. Check the grain before you check the arithmetic.")
    assert target["material"] is False, (
        "your verdict calls the target's movement material. It is not, and saying so "
        "is the lab. Two inputs moved, the reason is visible in the human_driven "
        "column, and the thing the model predicts did not move. Learning to write "
        "'no material change' and defend it is the harder skill.")

    # The module's strongest sentence, and it is a measurement rather than a
    # slogan: the target's index sits below the floor its own instrument reads
    # when nothing has changed at all.
    assert target["population_stability_index"] < target["noise_floor"], (
        f"your verdict has the target's index at "
        f"{target['population_stability_index']:.4f} and its measured noise floor at "
        f"{target['noise_floor']:.4f}. On this archive the index is below the floor — "
        "the instrument cannot tell the target apart from a day on which nothing "
        "happened, which is a stronger statement than 'below the threshold' and it is "
        "the sentence this module exists to teach you to write.")

    # The two measures rank the features differently, and both are right about
    # their own question. The largest shift is mean_speed; the largest index
    # belongs to sd_speed, whose mean barely moved while its shape changed a
    # great deal. That holds at every bin count from three to seven, so it is a
    # fact about this archive rather than about the binning.
    measured_indices = {name: results[name]["population_stability_index"]
                        for name in REQUIRED if results[name]["index_measured"]}
    largest = max(measured_indices, key=measured_indices.get)
    assert largest == "sd_speed", (
        f"the largest index in your verdict belongs to {largest}. On this archive it "
        "is sd_speed — its mean moved a fifth as far as mean_speed's while its shape "
        "changed more, and the index answers a question about shape. If yours ranks "
        "them differently, check that you are binning on the reference's quantiles.")

    material = [name for name in REQUIRED if results[name]["material"]]
    assert lab.TARGET not in material, "the target must not be among the material shifts"
    assert len(material) < len(REQUIRED) - 1, (
        f"your verdict calls {len(material)} of the {len(REQUIRED)} required features "
        f"material: {material}. At least two should not be — a detector that fires on "
        "almost everything is not a detector, and on this data the target sits well "
        "inside the noise.")

    # The degenerate column, reported as unmeasured rather than as unmoved. Its
    # index is exactly 0.0 for every correct implementation of the arithmetic,
    # and 0.0 in that cell of a report reads as "did not move" about the one
    # column that explains the whole event.
    degenerate = results[DEGENERATE]
    assert degenerate["index_measured"] is False, (
        f"your verdict reports an index for {DEGENERATE} as if it had been measured. "
        "That column is nought in 39 of the 45 reference windows, so its quantile "
        "edges collapse to one bin and the index is exactly nought whatever the "
        "column did. Catch the refusal from Lab 2 and record index_measured as "
        "False.")
    assert degenerate["population_stability_index"] != 0.0, (
        f"your verdict records {DEGENERATE}'s index as "
        f"{degenerate['population_stability_index']!r}. Nought means 'did not move'; "
        "this column moved by more than a standard deviation and caused everything "
        "else in the archive. Record None, not a number the arithmetic invented.")
    assert degenerate["index_threshold"] is None and degenerate["noise_floor"] is None, (
        f"your verdict reports a floor of {degenerate['noise_floor']!r} and a "
        f"threshold of {degenerate['index_threshold']!r} for {DEGENERATE}. A column "
        "whose reference cannot be binned has no null to resample and therefore no "
        "threshold either; report both as unmeasured rather than inventing them.")
    assert abs(degenerate["shift_in_reference_sd"]) > 1.0, (
        f"{DEGENERATE} shifted {degenerate['shift_in_reference_sd']:+.2f} reference "
        "standard deviations in your measurement; it moved by more than one, and it "
        "is the cause of the speed shift. Only the index failed here, not the shift.")

    # ------------------------------------------------------------------
    # The positive control, and the detection limit swept out around it.
    # ------------------------------------------------------------------
    control = lab.positive_control(reference, current)
    for key in ("injected_shift_sd", "population_stability_index", "material",
                "sizes", "material_by_size", "detection_limit_sd",
                "detection_limit_in_target_units", "first_material_sd"):
        assert key in control, (
            f"positive_control() returned no '{key}'. A control at one size says the "
            "detector detects; the sweep says what it is blind to, and both belong in "
            "the result rather than in a comment.")
    assert control["injected_shift_sd"] > 0, (
        "positive_control() must report the size of the shift it injected; an "
        "unstated control size is an untestable claim")
    assert control["material"] is True, (
        f"you injected {control['injected_shift_sd']} reference standard deviations "
        f"into the target and your verdict still calls it immaterial, at an index of "
        f"{control['population_stability_index']}. Then the null result on the real "
        "target says nothing about the world — it only says your detector does not "
        "detect. Fix the control before you believe the verdict.")
    assert control["population_stability_index"] > control["index_threshold"], (
        f"the injected shift gave an index of {control['population_stability_index']}, "
        f"which is not above the threshold your own verdict judges by "
        f"({control['index_threshold']}). The measured no-change floor on this target "
        f"is {target['noise_floor']:.3f}, so an injected shift of this size should be "
        "far above both.")

    sizes = [float(size) for size in control["sizes"]]
    fired = [bool(value) for value in control["material_by_size"]]
    limit = control["detection_limit_sd"]
    assert len(sizes) >= 20 and len(fired) == len(sizes), (
        f"your sweep visited {len(sizes)} size(s) and recorded {len(fired)} verdict(s). "
        "A detection limit quoted off three points is quoted off nothing; sweep the "
        "grid in lab_support.DETECTION_SIZES, or one at least as fine.")
    assert abs(min(sizes)) < 1e-9 and fired[sizes.index(min(sizes))] is False, (
        "your sweep either does not start at nought, or reports the verdict as "
        "material when nothing at all was injected. Nought injected is the null, and "
        "a detector that fires there has a bin count problem rather than a "
        "sensitivity: at ten bins on this archive that is exactly what happens.")
    assert limit in sizes, (
        f"your detection limit is {limit!r}, which is not one of the sizes you swept. "
        "The limit is quoted to the resolution of the grid — claiming more precision "
        "than the step you walked is claiming a measurement you did not make.")
    position = sizes.index(limit)
    assert all(fired[position:]), (
        f"your detection limit is {limit}, but the verdict is not material at every "
        "larger size you swept. The limit is the smallest size from which the answer "
        "*stays* material: at this sample size the index flickers above the threshold "
        "and falls back, and the first crossing is a fluke of which windows landed in "
        "which bin.")
    assert not all(fired[:position]) or position == 0, (
        f"your verdict is material at every size below {limit} as well, so {limit} is "
        "not the smallest sustained crossing. Take the first position from which every "
        "later verdict is material.")
    assert control["first_material_sd"] <= limit, (
        f"you report a first firing at {control['first_material_sd']} and a detection "
        f"limit of {limit}. The limit cannot be below the first size at which the "
        "verdict fires at all.")
    assert 0 < limit < control["injected_shift_sd"], (
        f"your detection limit is {limit} reference standard deviations against a "
        f"control injected at {control['injected_shift_sd']}. If the limit is not "
        "below the control size, the control was never evidence of anything the sweep "
        "did not already say.")
    assert abs(control["detection_limit_in_target_units"]
               - limit * spread_of[lab.TARGET]) < 1e-6, (
        f"you report a limit of {limit} reference standard deviations and "
        f"{control['detection_limit_in_target_units']} in the target's own units. The "
        f"reference day's standard deviation of the target is "
        f"{spread_of[lab.TARGET]:.3f} kilograms, and the operator reads kilograms.")

    # And the sweep has to be the verdict's own answer rather than a table. Two
    # sizes are re-run here through the student's verdict and compared.
    for probe in (sizes[-1], sizes[len(sizes) // 2]):
        moved = current.copy()
        moved[lab.TARGET] = moved[lab.TARGET] + probe * spread_of[lab.TARGET]
        again = lab.verdict(reference, moved, [lab.TARGET])[lab.TARGET]["material"]
        assert bool(again) is fired[sizes.index(probe)], (
            f"at an injected {probe} reference standard deviations your sweep recorded "
            f"material={fired[sizes.index(probe)]} and re-running your own verdict "
            f"there gives {again}. The sweep has to be the verdict, run again on moved "
            "data, and not a table written beside it.")

    untouched = lab.verdict(reference, current)[lab.TARGET]
    assert untouched["material"] is False, (
        "running the positive control changed the verdict on the untouched target, so "
        "the control modified the data it was given rather than a copy of it. The "
        "control must leave the real answer exactly as it was.")

    # And the verdict itself, on data it has not seen, at a size the lab never
    # mentions. A verdict that returns the same table whatever it is handed is a
    # table rather than a verdict, and this archive's own answers are printed on
    # a slide for anyone who wants to copy them.
    injected = current.copy()
    injected[lab.TARGET] = injected[lab.TARGET] + 3.0 * spread_of[lab.TARGET]
    moved = lab.verdict(reference, injected)[lab.TARGET]
    assert moved["material"] is True and abs(moved["shift_in_reference_sd"]) > 2.0, (
        f"handed a target shifted by three reference standard deviations, your "
        f"verdict reports a shift of {moved['shift_in_reference_sd']:+.2f} and "
        f"material={moved['material']}. It has to measure what it is given rather "
        "than report what this archive happens to say.")

    # ------------------------------------------------------------------
    # The call, and the reason it would be defended with.
    # ------------------------------------------------------------------
    archive_evidence = {
        "target_index": target["population_stability_index"],
        "target_index_measured": target["index_measured"],
        "standardised_shift": target["shift_in_reference_sd"],
        "noise_floor": target["noise_floor"],
        "index_threshold": target["index_threshold"],
        "control_index": control["population_stability_index"],
        "detection_limit": limit,
        "material_features": material,
        "material_count": len(material),
        "features_watched": len(REQUIRED),
    }

    def moved_target(**changes):
        """The archive's evidence with one or two quantities replaced."""
        altered = dict(archive_evidence)
        altered.update(changes)
        return altered

    situations = [
        # the archive itself: the defended null this module says is the deliverable
        ("the archive as measured", archive_evidence, "no material change"),
        # the target moved past its own threshold
        ("the target's index above its threshold",
         moved_target(target_index=8.221, standardised_shift=1.47), "act"),
        # the shift rule alone, with an index that says nothing
        ("the target's shift past the shift rule",
         moved_target(standardised_shift=2.6), "act"),
        # nothing to measure: an absence you could not measure is not a null
        ("the target's index unmeasurable",
         moved_target(target_index=None, target_index_measured=False,
                      noise_floor=None, index_threshold=None), "watch"),
        # the instrument was never shown to work
        ("the positive control silent", moved_target(control_index=0.12), "watch"),
        # measurable movement that is not distinguishable from noise
        ("the index between the floor and the threshold",
         moved_target(target_index=0.31), "watch"),
    ]

    seen_calls, reasons = set(), {}
    for name, evidence, expected in situations:
        answer = lab.drift_verdict(evidence)
        assert isinstance(answer, tuple) and len(answer) == 2, (
            f"drift_verdict() returned {answer!r} on {name}. It returns a pair: the "
            "call, and the reason you would defend it with.")
        call, reason = answer
        assert call in ("act", "watch", "no material change"), (
            f"drift_verdict() called {call!r} on {name}. The three calls are 'act', "
            "'watch' and 'no material change', and nothing else is an instruction "
            "anybody can act on.")
        assert call == expected, explain(
            VERDICT_KEY + ":call",
            f"on {name} you called {call!r}; the defensible call is {expected!r}",
            "Read the four clauses in the order the stub gives them: a material "
            "target is 'act'; an index that could not be measured is 'watch'; a "
            "control that did not fire is 'watch', because silence from an untested "
            "instrument is not evidence; and only an index at or below the measured "
            "floor, with the control fired, earns 'no material change'.")
        seen_calls.add(call)
        reasons[name] = reason

    assert seen_calls == {"act", "watch", "no material change"}, (
        f"your drift_verdict() only ever answers {sorted(seen_calls)} across six "
        "situations that differ. A rule that cannot reach all three calls is not a "
        "rule, it is a default.")

    # The reason is graded on two of the six, because a reason that fits both
    # was built out of the evidence rather than remembered from a slide.
    grade_reason(reasons["the archive as measured"], archive_evidence, key=VERDICT_KEY)
    grade_reason(reasons["the target's index above its threshold"],
                 situations[1][1], key=VERDICT_KEY)
    assert reasons["the archive as measured"] != reasons[
        "the target's index above its threshold"], (
        "the same reason came back for a target that did not move and for one that "
        "moved past its own threshold. A reason is a report of the evidence handed "
        "in; if it does not change when the evidence does, it is a sentence rather "
        "than an argument.")

    # ------------------------------------------------------------------
    # Significance is not size.
    # ------------------------------------------------------------------
    outcome = lab.significance_is_not_size(reference["mean_speed"].dropna(),
                                           current["mean_speed"].dropna())
    for key in ("p_value_windows", "p_value_readings", "effect_size"):
        assert key in outcome, f"significance_is_not_size() returned no '{key}'"

    assert outcome["p_value_readings"] < outcome["p_value_windows"], (
        f"the p-value at the coarse grain was {outcome['p_value_windows']:.3g} and at "
        f"the fine grain {outcome['p_value_readings']:.3g}. With more observations of "
        "the same difference it must fall, which is the whole demonstration.")
    assert outcome["p_value_windows"] < 0.05, (
        "the difference in mean speed is real; the p-value at the window grain should "
        "already be small")
    assert abs(outcome["effect_size"]) > 0.5, (
        f"the effect size came out at {outcome['effect_size']:.3f}. The difference is "
        "large — that is the number that describes the world rather than the sample.")

    # And it has to be Cohen's d with the pooling the slide states: the two
    # variances weighted by their degrees of freedom. The unweighted root mean
    # square is the common shortcut and it is a different number whenever the two
    # samples differ in size, which here they do -- 45 windows against 35. A check
    # that accepted either would grade neither, and the slide would state a
    # formula nothing enforces.
    before = reference["mean_speed"].dropna().to_numpy()
    after = current["mean_speed"].dropna().to_numpy()
    n_before, n_after = len(before), len(after)
    pooled = np.sqrt(((n_before - 1) * np.var(before, ddof=1)
                      + (n_after - 1) * np.var(after, ddof=1))
                     / (n_before + n_after - 2))
    cohen = float((after.mean() - before.mean()) / pooled)
    unweighted = float((after.mean() - before.mean())
                       / np.sqrt((np.var(before, ddof=1) + np.var(after, ddof=1)) / 2))
    assert abs(outcome["effect_size"] - cohen) < 1e-9, (
        f"your effect size is {outcome['effect_size']:.4f}; Cohen's d with the pooled "
        f"standard deviation weighted by degrees of freedom is {cohen:.4f}, and the "
        f"unweighted root mean square of the two variances gives {unweighted:.4f}. "
        "The slide states the weighted pooling, so that is the one graded: "
        "s_pooled = sqrt( ((n1-1)s1^2 + (n2-1)s2^2) / (n1+n2-2) ).")

    # The same call twice. The reading grain is reached by resampling, and a
    # resampling without a fixed seed gives a different answer every run, which
    # cannot be quoted in a report or reproduced by anybody reading it.
    again = lab.significance_is_not_size(reference["mean_speed"].dropna(),
                                         current["mean_speed"].dropna())
    assert again["p_value_readings"] == outcome["p_value_readings"], (
        f"two identical calls gave {outcome['p_value_readings']:.6g} and then "
        f"{again['p_value_readings']:.6g}. Seed the resampling with the seed in the "
        "signature — an answer that changes on every run cannot be defended.")

    assert outcome["n_readings"] > 40_000, (
        f"your reading grain holds {outcome['n_readings']:,} observations; the "
        "archive holds 48,290, and the point of the demonstration is the size of "
        "that number against the eighty windows.")

    # The two days the other way round. An effect size is a signed statement
    # about a direction, so swapping the samples must flip it -- and a constant
    # returned from a dictionary cannot.
    swapped = lab.significance_is_not_size(current["mean_speed"].dropna(),
                                           reference["mean_speed"].dropna())
    assert swapped["effect_size"] == -outcome["effect_size"], (
        f"measuring the second day against the first gave an effect size of "
        f"{swapped['effect_size']:.3f}, where the first against the second gave "
        f"{outcome['effect_size']:.3f}. Swapping the two samples must flip the sign "
        "and nothing else: the effect size says which way and by how much.")

    # ------------------------------------------------------------------
    # The closing test, from the required reading.
    # ------------------------------------------------------------------
    joint = lab.classifier_two_sample_test(reference, current)
    for key in ("accuracy", "correct", "held_out", "interval", "chance", "detected"):
        assert key in joint, (
            f"classifier_two_sample_test() returned no '{key}'. An accuracy without "
            "the count it came from and the interval around it is the number this "
            "whole module exists to stop you reporting.")

    balanced = min(len(reference), len(current))
    expected_held_out = 2 * (balanced - balanced // 2)
    assert joint["held_out"] == expected_held_out, (
        f"your test held out {joint['held_out']} rows; balancing the two days to "
        f"{balanced} each and training on half of each leaves {expected_held_out}. "
        "Balance first, or chance is the majority share rather than one half.")
    assert abs(joint["chance"] - 0.5) < 1e-12, (
        f"you report a chance level of {joint['chance']}. Once the two days hold the "
        "same number of rows, a coin scores one half, and an accuracy read against "
        "anything else is being read against the wrong question.")
    assert abs(joint["accuracy"] - joint["correct"] / joint["held_out"]) < 1e-12, (
        f"you report an accuracy of {joint['accuracy']} on {joint['correct']} correct "
        f"out of {joint['held_out']}, and those do not agree.")

    low, high = joint["interval"]
    reference_interval = load_lab(1).wilson_interval(joint["correct"], joint["held_out"])
    assert (abs(low - reference_interval[0]) < 1e-9
            and abs(high - reference_interval[1]) < 1e-9), (
        f"your interval is [{low:.4f}, {high:.4f}]; Wilson's interval from Lab 1 on "
        f"{joint['correct']} of {joint['held_out']} is "
        f"[{reference_interval[0]:.4f}, {reference_interval[1]:.4f}]. Block one built "
        "that interval for exactly this: a proportion measured on a few dozen trials, "
        "near enough to the edge that the obvious interval misbehaves.")
    assert joint["detected"] is bool(low > joint["chance"]), (
        f"you report detected={joint['detected']} with an interval of [{low:.4f}, "
        f"{high:.4f}] against a chance of {joint['chance']}. The test is the interval "
        "against chance, not the point estimate against chance — that is the whole "
        "reason for putting an interval on it.")

    # A negative control for the classifier: the reference day against itself.
    # One world, so nothing should be able to beat chance with any confidence.
    half = len(reference) // 2
    itself = lab.classifier_two_sample_test(reference.iloc[:half], reference.iloc[half:])
    assert itself["detected"] is False, (
        f"splitting the reference day in two and handing your classifier both halves, "
        f"it reports detected={itself['detected']} at an accuracy of "
        f"{itself['accuracy']:.3f} with an interval of [{itself['interval'][0]:.3f}, "
        f"{itself['interval'][1]:.3f}]. Those are two samples from one day. A test "
        "that finds a difference there is finding its own split.")

    # And a positive control for it: three standard deviations into the target.
    obvious = lab.classifier_two_sample_test(reference, injected, [lab.TARGET])
    assert obvious["detected"] is True, (
        f"with three reference standard deviations injected into the target your "
        f"classifier reports detected={obvious['detected']} at an accuracy of "
        f"{obvious['accuracy']:.3f}. A test that cannot see that is not evidence "
        "about anything smaller.")


run(4, "04_the_verdict", "verdict", body,
    requires=[(1, lambda lab: lab.wilson_interval(34, 40)),
              (2, lambda lab: lab.kl_divergence([0.5, 0.5], [0.5, 0.5])),
              (3, lambda lab: lab.wasserstein([0.0, 1.0], [0.0, 1.0]))])
