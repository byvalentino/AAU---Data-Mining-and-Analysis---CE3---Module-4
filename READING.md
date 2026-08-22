# Module 4 — Reading, and the exam question

## Required, before Module 5

**Rabanser, S., Günnemann, S. & Lipton, Z. (2019). *Failing Loudly: An Empirical
Study of Methods for Detecting Dataset Shift*. NeurIPS 32.**
<https://arxiv.org/abs/1810.11953>

Free on arXiv. They test the methods you built today against each other on real
shifts, and the results are usefully humbling: no single measure wins, and
several widely-used ones detect very little. Read it as the answer to "which of
these three should I use?" — the answer being that the question is wrong.

## Recommended — the intervals of Block one

**Wilson, E. B. (1927). *Probable inference, the law of succession, and
statistical inference*. Journal of the American Statistical Association 22(158),
209–212.** <https://doi.org/10.1080/01621459.1927.10502953> — through the AAU
library. Four pages, and the interval you implemented in Lab 1.

**Brown, L. D., Cai, T. T. & DasGupta, A. (2001). *Interval Estimation for a
Binomial Proportion*. Statistical Science 16(2), 101–133.**
<https://doi.org/10.1214/ss/1009213286> — free. If you read one paper from this
block, read this one: it measures the coverage of every interval in common use
across the whole range of true rates, finds the standard one erratic even at
large n, and recommends Wilson's. It is the published version of Lab 1's
coverage experiment.

**Agresti, A. & Coull, B. A. (1998). *Approximate is better than "exact" for
interval estimation of binomial proportions*. The American Statistician 52(2),
119–126.** <https://doi.org/10.1080/00031305.1998.10480550> — through the AAU
library. Short, and the source of the "add two successes and two failures"
shortcut, which is Wilson's interval in disguise.

**Wasserman, L. (2004). *All of Statistics*. Springer.**
<https://link.springer.com/book/10.1007/978-0-387-21736-9> — through the AAU
library. Chapter 5 for the central limit theorem stated with its conditions,
which is rarer than it should be; Theorem 4.9 for Jensen's inequality in the
form this module uses.

**Bayley, G. V. & Hammersley, J. M. (1946). *The "effective" number of
independent observations in an autocorrelated time series*. Supplement to the
Journal of the Royal Statistical Society 8(2), 184–197.**
<https://doi.org/10.2307/2983560> — through the AAU library. Fourteen pages, and
the origin of the n(1−ρ)/(1+ρ) that turns this module's 48,290 readings into an
effective sample of about seventy.

## Recommended — surprise, and the divergence

**Shannon, C. E. (1948). *A Mathematical Theory of Communication*. Bell System
Technical Journal 27(3), 379–423.**
<https://doi.org/10.1002/j.1538-7305.1948.tb01338.x> — free. Read sections 6 and
7 for the entropy, and for the argument that fixes the form of the formula. The
rest is a founding document and worth an evening of anybody's time.

**Kullback, S. & Leibler, R. A. (1951). *On Information and Sufficiency*.
Annals of Mathematical Statistics 22(1), 79–86.**
<https://doi.org/10.1214/aoms/1177729694> Free. Where the divergence you wrote in
Lab 2 comes from — read the first pages for the definition and the asymmetry, and
stop when the measure theory starts.

**MacKay, D. J. C. (2003). *Information Theory, Inference, and Learning
Algorithms*, §2.6. Cambridge University Press.**
<https://www.inference.org.uk/itprnn/book.pdf> — free from the author. Two pages
for Gibbs' inequality **by that name**, which is the non-negativity Lab 2's check
requires; the surrounding chapter is the friendliest introduction to the entropy
in print.

**Cover, T. M. & Thomas, J. A. (2006). *Elements of Information Theory*, 2nd
ed., chapter 2. Wiley.** <https://doi.org/10.1002/047174882X> — through the AAU
library. Entropy, relative entropy and mutual information with the proofs filled
in; its Theorem 2.6.3 is the non-negativity, under the name "information
inequality". Note what it does **not** contain, because an earlier edition of
this course said otherwise: it defines no cross-entropy, and it never says
"Gibbs' inequality".

**Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction*,
§6.1.2. MIT Press.** <https://probml.github.io/pml-book/book1.html> — free from
the author. The cross-entropy as the loss you have already trained a classifier
with, and the sentence that connects it to the divergence in one line.

**Jensen, J. L. W. V. (1906). *Sur les fonctions convexes et les inégalités
entre les valeurs moyennes*. Acta Mathematica 30, 175–193.**
<https://doi.org/10.1007/BF02418571> — free, and in French. Read it for the
history rather than the technique: the inequality the module opens on, in the
paper that first stated it in this generality.

## Recommended — the index, and its borrowed thresholds

**Jeffreys, H. (1946). *An invariant form for the prior probability in
estimation problems*. Proceedings of the Royal Society A 186, 453–461.**
<https://doi.org/10.1098/rspa.1946.0056> — through the AAU library. The
symmetrised divergence you implemented, decades before credit scoring renamed it
the Population Stability Index.

**Yurdakul, B. & Naranjo, J. (2020). *Statistical properties of the population
stability index*. Journal of Risk Model Validation 14(4), 89–100.**
<https://doi.org/10.21314/JRMV.2020.227> — through the AAU library. The paper the
0.25 threshold never had: under no change the index has a known sampling
distribution that depends on both sample sizes and the bin count. Read it beside
Lab 2's `index_threshold()`, which is the same statement obtained by simulation
and is where this module's thresholds actually come from.

**Currie, L. A. (1968). *Limits for qualitative detection and quantitative
determination. Application to radiochemistry*. Analytical Chemistry 40(3),
586–593.** <https://doi.org/10.1021/ac60259a007> — through the AAU library. Where
the detection limit comes from, and the discipline this module borrows with it:
in analytical chemistry nobody reports "not detected" without saying what the
smallest detectable quantity was. Read the first three pages for the definitions
and skip the radiochemistry; the transfer to a drift monitor is exact.

**Regulation (EU) 2024/1689, the European Union Artificial Intelligence Act,
Article 15 — accuracy, robustness and cybersecurity.**
<https://eur-lex.europa.eu/eli/reg/2024/1689/oj> — free. Read Article 15(3) and
15(4) only, which is one page. They ask that accuracy metrics be *declared* in
the instructions for use, and that a high-risk system be resilient to errors,
faults or inconsistencies in the environment it operates in. The positive control
is how a team evidences that its monitor works, and a declared metric with no
measured floor beside it is not a declaration. The Annex III high-risk
obligations fall due on 2 December 2027 after the Digital Omnibus deferral.

**Lewis, E. M. (1994). *An Introduction to Credit Scoring*. Athena Press.** Out
of print; through the AAU library. Cited here because it is where the 0.1 and
0.25 rule of thumb originates, and knowing that is the difference between
quoting a convention and quoting a derivation.

**Siddiqi, N. (2006). *Credit Risk Scorecards*, and (2017) *Intelligent Credit
Scoring*, 2nd ed. Wiley.** <https://doi.org/10.1002/9781119282396> — through the
AAU library. Where practitioners actually meet the index and its thresholds.
Worth seeing the original context, because it is not yours: populations of many
thousands, scored monthly.

## Recommended — the distance

**Vallender, S. S. (1974). *Calculation of the Wasserstein distance between
probability distributions on the line*. Theory of Probability and Its
Applications 18(4), 784–786.** <https://doi.org/10.1137/1118101> — through the
AAU library. Three pages, and the closed form you implemented in Lab 3: the
integral of the absolute difference between two cumulative distribution
functions.

**Peyré, G. & Cuturi, M. (2019). *Computational Optimal Transport*. Foundations
and Trends in Machine Learning 11(5–6), 355–607.**
<https://doi.org/10.1561/2200000073> — free from the authors. Read Remark 2.30
for the one-dimensional closed form and Remark 2.28 for the sorted-samples
shortcut, then stop unless you want the whole field; the rest is a textbook.

**Kantorovich, L. V. (1942). *On the translocation of masses*. Doklady Akademii
Nauk SSSR 37(7–8), 227–229.** Reprinted in English in Management Science 5(1),
1958, 1–4, <https://doi.org/10.1287/mnsc.5.1.1>, and again in the Journal of
Mathematical Sciences (2006). Three pages, for where the name comes from.

**Ramdas, A., García Trillos, N. & Cuturi, M. (2017). *On Wasserstein
Two-Sample Testing and Related Families of Nonparametric Tests*. Entropy 19(2),
47.** <https://doi.org/10.3390/e19020047> — open access. The two-sample test built
on the distance. Read it for the testing framing rather than for the closed form:
its Proposition 1 states the quantile version, which is why this module cites
Vallender and Peyré & Cuturi for the area between the curves.

## Recommended — the tests, the sizes, and the dates

**Welch, B. L. (1947). *The generalization of "Student's" problem when several
different population variances are involved*. Biometrika 34(1/2), 28–35.**
<https://doi.org/10.1093/biomet/34.1-2.28> — through the AAU library. Eight
pages, and the test Lab 4 uses. Read the first two for the assumption that is
dropped: a common variance.

**Efron, B. (1979). *Bootstrap methods: another look at the jackknife*. Annals
of Statistics 7(1), 1–26.** <https://doi.org/10.1214/aos/1176344552> — free. The
resampling idea in its original form. Lab 4 uses it for one narrow purpose —
changing the sample size and nothing else — and the paper is the general version.

**Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*,
2nd ed. Lawrence Erlbaum.** Through the AAU library. Chapter 2 for d, its pooled
standard deviation, and Cohen's own warning that his "small, medium, large"
labels were a last resort rather than a finding — which is the same lesson as the
index's thresholds, in another field.

**Glass, G. V. (1976). *Primary, secondary, and meta-analysis of research*.
Educational Researcher 5(10), 3–8.**
<https://doi.org/10.3102/0013189X005010003> — through the AAU library. Six pages
of a presidential address, and the origin of dividing by the reference group's
standard deviation alone, which is what Lab 4 and Module 5 both do.

**Benjamini, Y. & Hochberg, Y. (1995). *Controlling the False Discovery Rate: A
Practical and Powerful Approach to Multiple Testing*. Journal of the Royal
Statistical Society B 57(1), 289–300.**
<https://doi.org/10.1111/j.2517-6161.1995.tb02031.x> — through the AAU library.
What to do when you monitor twenty features instead of one. The procedure is
four lines long and it is the right answer to the twenty-questions slide.

**Wasserstein, R. & Lazar, N. (2016). *The ASA Statement on p-Values*. The
American Statistician 70(2), 129–133.**
<https://doi.org/10.1080/00031305.2016.1154108> — free, two pages, and written
because the profession collectively needed telling.

**Page, E. S. (1954). *Continuous Inspection Schemes*. Biometrika 41(1/2),
100–115.** <https://doi.org/10.1093/biomet/41.1-2.100> — through the AAU library.
Drift with a date: the first sequential scheme for detecting the moment a process
changes, from industrial quality control, and still the ancestor of every change
detector in production.

**Truong, C., Oudre, L. & Vayatis, N. (2020). *Selective review of offline
change point detection methods*. Signal Processing 167, 107299.**
<https://doi.org/10.1016/j.sigpro.2019.107299> — free on arXiv. The modern survey:
what to reach for when the question is not "did it change" but "when".

**Saltelli, A., Aleksankina, K., Becker, W., Fennell, P., Ferretti, F.,
Holst, N., Li, S. & Wu, Q. (2019). *Why so many published sensitivity analyses
are false: a systematic review of sensitivity analysis practices*. Environmental
Modelling and Software 114, 29–39.**
<https://doi.org/10.1016/j.envsoft.2019.01.012> — free. Read it as the argument
behind Lab 4's positive control: an analysis that cannot fail has not been done,
and ranking inputs is not the same as saying how far the answer should be
trusted.

> Nothing licensed is redistributed in this repository.

## Further afield, and worth it

**Anane, E., López C., D. C., Barz, T., Sin, G., Gernaey, K. V., Neubauer, P. &
Cruz Bournazou, M. N. (2019). *Output uncertainty of dynamic growth models:
effect of uncertain parameter estimates on model reliability*. Biochemical
Engineering Journal 150, 107247.** <https://doi.org/10.1016/j.bej.2019.107247> —
through the AAU library. Read the propagation section only. Block one put an
interval on a rate you observed. This carries the chain further than the course
does: measurement noise becomes uncertainty on a fitted parameter, which becomes
uncertainty on the prediction somebody acts on. It is the worked version of the
sentence "report both, and never let the first stand in for the second".

**Raue, A., et al. (2009). *Structural and practical identifiability analysis of
partially observed dynamical models by exploiting the profile likelihood*.
Bioinformatics 25(15), 1923–1929.**
<https://doi.org/10.1093/bioinformatics/btp358> — free. Optional, and the sharpest
counter-case to everything Block one teaches. Some parameters cannot be recovered
even from perfect data — the fault is then the model, and more data never helps.
Where that holds, any interval you compute is an artefact of the optimiser rather
than a statement about the world. Worth knowing before you trust an interval you
did not derive yourself.

## The exam question for Module 4

> **A monitor reports that a feature's Population Stability Index is 0.31,
> above the usual threshold of 0.25, and recommends retraining. What do you need
> to know before agreeing, and what would you do first?**
>
> Twelve further questions, with what a good answer contains, the commonest wrong
> answer and a follow-up for each, are in `Module 4/EXAM.md`.

A strong answer asks about the sample size before anything else, because the
index's conventional thresholds come from populations of many thousands and its
noise floor on a small sample can exceed 0.25 with no change at all — measured
in this module at 0.28 on about forty observations with ten bins. It asks what
the reference is, and whether it is fixed or moving. It asks how many features
are being monitored, because twenty at five per cent produces an alarm a run by
arithmetic alone. It asks for a second measure in the variable's own units, so
the size of the shift can be judged rather than only its surprise. And crucially
it asks whether the **target** moved, because a shifted input with a stationary
target is the ordinary case and the right response to it is usually to watch
rather than retrain. The first action is to find the noise floor by simulation
and to look for a cause in the unmonitored columns — as in this archive, where
the one real shift is explained by the `mode` column. A very strong answer adds
two things this module measures: an index of exactly nought is not "no change"
but usually "no measurement", so ask how many bins survived; and before any null
result is believed, ask what the detector reports on an injected shift of a
stated size — and what the smallest shift is that it still reports, which is the
detection limit and is the number that bounds the null.
