# Why these solutions look like this

Every one of them runs. `python3 solutions/lab_0K.py` from the exercises
directory narrates the lab end to end — what was loaded and its shape, every
intermediate quantity with its unit, the answer, and the sentence the check
grades — and writes its figures under `out/`. `make demo` runs all four and
writes one page linking the seven figures. For the length of a demonstration
Labs 3 and 4 resolve "the lab before me" to the shipped solution rather than to
`labs/`, so the demonstration runs before anybody has written anything; the
checks do the opposite, which is what makes them checks.

## Lab 1 — simulate the promise

A 95 per cent interval makes a testable claim, so test it. Near a true rate of
0.02 with forty observations the naive interval delivers about a third of what it
promises: most samples contain no successes, the interval is [0, 0], and the
truth is not in it.

The naive formula asks "what is the error around my estimate?" Wilson asks "which
true rates could plausibly have produced what I saw?" The second is the question
you actually have, and it costs one extra line.

`labels_needed` is the arithmetic behind Module 5. Halving the half-width
quadruples the labels: ±0.05 costs 385 hand-checks, ±0.025 costs 1,537. Bought
truth gets expensive faster than anybody expects, which is why you buy a small
sample and put an interval round it.

## Lab 2 — one identity, and the zeros

The identity first, because it is what the check tests before anything else:

    cross-entropy − entropy = divergence

Three functions, one relation, and if it does not close then one of the three is
not what you think it is. The usual error is writing the cross-entropy where the
divergence belongs — the two differ by today's own entropy, which is small on
tidy examples, so nothing else notices. This course's own first draft of the
slides made exactly that error.

The consequence is the reason the divergence is what goes on a monitor: its
floor is fixed at nought, and the cross-entropy's floor is today's entropy,
which moves. A measure whose zero moves cannot tell you that nothing happened.

Then the zeros. Zeros in **P** contribute nothing, because the limit genuinely is
nought. Zeros in **Q** make the divergence infinite, and that is the correct
reply to "how surprising is something you said was impossible?"

Infinity is useless in a dashboard, so everyone bins — and the binning hides
exactly the event that produced the infinity, because a genuinely new value lands
in an existing bin and the index reports something modest. Monitor the appearance
of unseen values as a separate check. Do not expect a divergence to shout.

Bin edges come from the reference, never from today. Take today's quantiles and
the yardstick moves every time you measure, so a stable world produces a
wandering index.

And the refusal. On this archive `human_driven` is nought in 39 of the 45
reference windows, so all six requested quantiles collapse to two distinct
edges, one bin holds everything, and the index is exactly nought — for the column
that explains the whole event. Nought there means "no measurement", not "no
change", so the solution raises `DegenerateReference` instead of returning a
reassuring number. A measure that cannot fail visibly will fail invisibly.

## Lab 3 — units are what make a number actionable

"The speed distribution moved 0.46 metres per second" is a sentence someone can
act on. "The divergence is 0.031" ends the conversation.

Report all four. The cross-entropy says what today cost; the divergence says how
much of that cost was avoidable; the index says the same with no reference order
to argue about; the distance says how far the world went, in units somebody can
act on. Each hides what the others show.

`compare_four` uses two binnings on purpose. The index bins on the reference's
quantiles, because a monitor's yardstick is cut once and never moved. The
cross-entropy and the divergence bin on equal-width edges spanning both samples,
because there both samples are in front of you: a value the reference never held
then gets a bin of its own, and the divergence answers "infinite" instead of
hiding it. On the real archive that happens — the second day reached speeds the
first never did.

Two cases separate the distance from the other three, and the check runs both.
Samples with no overlap: the divergence is infinite whether the gap is five or
five hundred, and the distance is the gap. Relabelled bins: the divergence does
not move at all, because it never knew the values were ordered, and the distance
moves completely, because the geometry is what changed.

## Lab 4 — writing "no material change" and defending it

The honest verdict on the archive, at five bins:

| feature | shift | index | material |
|---|---|---|---|
| mean_speed | +2.30 SD | 2.289 | yes |
| sd_speed | −0.47 SD | 2.975 — the largest of all | yes |
| sd_payload | −0.47 SD | 0.422 | no — 0.422 against its own threshold of 0.440 |
| human_driven | +1.24 SD | **unmeasured** — one surviving bin | no |
| mean_payload — **the target** | −0.03 SD | 0.081 | **no** |

Two inputs are material, and *which* two is decided by a threshold nobody
borrowed. Lab 2's `index_threshold()` compares the reference against a resample
of itself a thousand times and takes the 0.99 quantile of what comes back: 1.354
on `mean_speed`, 0.431 on `sd_speed`, 0.440 on `sd_payload`, 0.465 on the target.
Four columns, four thresholds, one archive and one bin count. Credit scoring's
0.25 would have made three features material instead of two, the extra one being
`sd_payload` at 0.422 against its own 0.440 — a call that close is exactly what a
borrowed threshold hides.

The two measures rank the features differently — the largest index belongs to
`sd_speed` and the largest shift to `mean_speed`, and both are right about their
own question — and the column that caused everything is the one the index cannot
see.

A monitor watching inputs would have fired; a monitor watching the target would
not. Both were right, and the operator's question — must we do anything? — is
answered by the second.

That is the ordinary case in production, and it is why "no material change" must
be a sayable conclusion. A drift detector that only ever finds drift is not a
detector.

And the target's own index, 0.081, is below its own measured noise floor of
0.105. That is the sentence the module is built to produce: not "the target moved
a little" but "the instrument cannot tell this day apart from a day on which
nothing happened".

`positive_control` is what makes that conclusion worth anything. Inject 1.5
reference standard deviations into the target, re-run the **unchanged** verdict,
and the index reports 8.221 against a derived threshold of 0.465. The shift rule
does not fire at 1.5, so what fired is the index — which is the instrument the
null result depends on.

Then it sweeps, because a control at one size establishes one size. Walking 0.00
to 1.50 in steps of 0.05, the verdict is material from **0.40** upwards and stays
material: 44.3 kilograms of mean payload per five-minute window. Two details
matter more than the number. The threshold is derived once and reused across the
sweep — the null is built from the reference alone, and holding it fixed is what
"the unchanged verdict" means. And the index first fires at 0.15 and falls back
at 0.20: at thirty-five current windows a handful crossing a quantile edge moves
it a long way, so the limit is the sustained crossing rather than the first one.

`drift_verdict` turns all of that into one instruction. Four clauses in order: a
material target is "act"; an unmeasurable index is "watch"; a control that did
not fire is "watch", because silence from an untested instrument is not evidence;
and only an index at or below the measured floor, with the control fired, earns
"no material change". The third clause is the one people leave out, and the check
hands the function six situations whose right answers differ so that leaving it
out is visible.

The reason is graded as well as the call. Every number in it has to be one the
evidence contains, at least two of the quantities have to be named, and forty
characters is the floor. The point is not prose: a sentence copied off a slide
quotes numbers that are not in the evidence and fails, which is the same lesson
as the threshold in another form.

`classifier_two_sample_test` is the closing comparison the required reading asks
for. A nearest-centroid classifier on the five standardised features gets 24 of
36 held-out windows — accuracy 0.667, Wilson's interval [0.503, 0.798] against a
chance of one half. It detects, by three thousandths at the lower bound. On the
target alone it does not: 0.417, interval [0.271, 0.578], which is the index's
answer reached by a different route. Rabanser, Günnemann and Lipton's finding
stands; their conditions do not hold on eighty windows, and the honest report
says the classifier is untestable here rather than better or worse.

`verdict` takes any list of features. The five it is graded on are there for a
reason each: `mean_speed` and `sd_speed` are the two window moments of the
model's speed input, `mean_payload` and `sd_payload` are the target stand-in and
its spread, and `human_driven` is the suspected cause. Five more are measured and
offered — `max_speed`, `share_stopped`, `n_readings`, `mileage_delta`,
`mean_battery` — and the check prints whatever a student added without judging
it. Few well-chosen features beat everything available, for the arithmetic reason
on the twenty-questions slide.

`significance_is_not_size` makes the other half of the point. The same difference
at eighty observations gives p ≈ 0.0002; resampled to forty-eight thousand it
falls through the floor, and the archive's own reading-grain test underflows to
exactly 0.0 — which is why the slide prints a bound rather than a nought. Cohen's
d does not move at all — 1.021, over the pooled standard deviation weighted by
degrees of freedom, which is the pooling the slide states and the check grades;
the unweighted root mean square of the two variances would give 0.966, because
the two days hold 45 windows and 35. The resampling is seeded with the course
seed, in the signature rather than buried in the body, because an answer that
changes on every run cannot be quoted or defended. A monitor thresholded on a p-value fires
constantly in production and tells you nothing.

## The grain

Every number above is for one vehicle — the shuttle that ran on both days — in
five-minute windows holding at least 300 readings. The other shuttle ran on the
first day only. Pool the two on day one against one on day two and part of what
you would call drift is a vehicle going to the depot. An earlier version of this
course's plan did exactly that and got the sign of the target's movement wrong.

The 300-reading floor is part of the grain and not housekeeping: without it the
headline shift in mean speed reads +2.51 rather than +2.30 reference standard
deviations, so the choice is worth 0.21 of a standard deviation and belongs
beside the number.

## References

- Jensen, J. L. W. V. (1906). *Sur les fonctions convexes et les inégalités entre les valeurs moyennes.* Acta Mathematica 30, 175–193.
- Bayley, G. V. & Hammersley, J. M. (1946). *The "effective" number of independent observations in an autocorrelated time series.* Supplement to the JRSS 8(2), 184–197.
- Wilson, E. B. (1927). *Probable inference, the law of succession, and statistical inference.* JASA 22(158), 209–212.
- Jeffreys, H. (1946). *An invariant form for the prior probability in estimation problems.* Proceedings of the Royal Society A 186, 453–461.
- Welch, B. L. (1947). *The generalization of "Student's" problem when several different population variances are involved.* Biometrika 34(1/2), 28–35.
- Shannon, C. E. (1948). *A Mathematical Theory of Communication.* Bell System Technical Journal 27(3), 379–423.
- Kullback, S. & Leibler, R. A. (1951). *On Information and Sufficiency.* Annals of Mathematical Statistics 22(1), 79–86.
- Vallender, S. S. (1974). *Calculation of the Wasserstein distance between probability distributions on the line.* Theory of Probability and Its Applications 18(4), 784–786.
- Glass, G. V. (1976). *Primary, secondary, and meta-analysis of research.* Educational Researcher 5(10), 3–8.
- Efron, B. (1979). *Bootstrap methods: another look at the jackknife.* Annals of Statistics 7(1), 1–26.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed. Lawrence Erlbaum — the pooled standard deviation these solutions use.
- Lewis, E. M. (1994). *An Introduction to Credit Scoring.* Athena Press — where the 0.1 and 0.25 thresholds start.
- Agresti, A. & Coull, B. A. (1998). *Approximate is better than "exact" for interval estimation of binomial proportions.* The American Statistician 52(2), 119–126.
- Brown, L. D., Cai, T. T. & DasGupta, A. (2001). *Interval Estimation for a Binomial Proportion.* Statistical Science 16(2), 101–133.
- MacKay, D. J. C. (2003). *Information Theory, Inference, and Learning Algorithms*, §2.6 — Gibbs' inequality by name.
- Cover, T. M. & Thomas, J. A. (2006). *Elements of Information Theory*, 2nd ed., ch. 2. Wiley — Theorem 2.6.3, the information inequality.
- Siddiqi, N. (2006). *Credit Risk Scorecards*, and (2017) *Intelligent Credit Scoring*, 2nd ed. Wiley — where practitioners meet the index.
- Wasserstein, R. & Lazar, N. (2016). *The ASA Statement on p-Values.* The American Statistician 70(2), 129–133.
- Ramdas, A., García Trillos, N. & Cuturi, M. (2017). *On Wasserstein Two-Sample Testing and Related Families of Nonparametric Tests.* Entropy 19(2), 47 — the test, not the closed form.
- Peyré, G. & Cuturi, M. (2019). *Computational Optimal Transport.* Foundations and Trends in Machine Learning 11(5–6), 355–607 — Remarks 2.30 and 2.28.
- Currie, L. A. (1968). *Limits for qualitative detection and quantitative determination.* Analytical Chemistry 40(3), 586–593 — the detection limit, and the rule of never reporting "not detected" without it.
- Rabanser, S., Günnemann, S. & Lipton, Z. (2019). *Failing Loudly.* NeurIPS 32 — the classifier two-sample test, and the required reading.
- Saltelli, A. et al. (2019). *Why so many published sensitivity analyses are false.* Environmental Modelling and Software 114, 29–39 — the argument behind the positive control.
- Yurdakul, B. & Naranjo, J. (2020). *Statistical properties of the population stability index.* Journal of Risk Model Validation 14(4), 89–100 — the sampling distribution behind the noise floor.
- Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*, §6.1.2. MIT Press — the cross-entropy as a loss.
