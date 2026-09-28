"""Part 4 and Laboratory 4: tests and effect size, the alarm rule, the decision per
feature, the positive control and the detection limit, the drift decision, the
classifier two-sample test and its power."""

S = "Module 4/exercises/solutions"
LABS = "Module 4/exercises/labs"


def build(nb, explain) -> None:
    nb.md("""
    ---
    # Part 4 — Effect size, the decision per feature, and regulatory evidence

    For each feature of the bus data we decide one thing: did its distribution change
    between the reference day, 22 January 2020, and the current day, 23 January 2020,
    by more than that feature's own threshold? Three possible answers: **moved
    materially**, **did not move**, **instrument failed**. Three sub-questions, in
    order: is the change real, is it large enough to matter, and what do I do about it?

    ### The main statistical tests and their assumptions

    *Slide: "Overview - the main statistical tests and their assumptions".* Every test
    assumes independent observations, which is why the unit of analysis is the window.

    | Test | What it compares | Use when |
    |---|---|---|
    | Student's t-test | means of two groups, equal variance | two groups with similar spread |
    | Welch's t-test | means of two groups, unequal variance allowed | the default two-sample test in this course |
    | Paired t-test | mean of within-pair differences | the same units measured twice |
    | Mann-Whitney U | ranks of two groups | skewed data, outliers, ordinal data |
    | One-way ANOVA | means of three or more groups | more than two groups |
    | Chi-squared | counts per category | category frequencies, histogram bins |
    | Kolmogorov-Smirnov | largest gap between two cumulative curves | the shape changed, not only the mean |
    | Classifier two-sample test | whether a classifier can tell the samples apart | abundant, high-dimensional data |
    """)

    # --- slides 54-56: Welch, significance, effect size ------------------------------------
    nb.md("### Welch's t-test, and significance against size\n\n"
          "*Slides: \"Definition - Welch's t-test, and the bootstrap as a demonstration\", "
          "\"Statistical significance is not effect size\" and \"Effect size - the number that "
          "decides whether a change matters\". The degrees of freedom are Appendix D.*")
    explain(
        "Write Welch's statistic, and test whether the windows' mean speed differs between "
        "the two days.",
        "Welch's test compares two means without assuming a common variance "
        "[@welch1947]; its degrees of freedom come from the Welch-Satterthwaite formula.",
        "Builds the statistic in sympy and runs `scipy.stats.ttest_ind(..., equal_var=False)` "
        "on the 45 and 35 windows.",
        "p = 0.0002 on 80 windows. The test says the days differ; it says nothing about how "
        "much.")
    nb.equation("welch", '''
    m1, m2, s1, s2, n1, n2 = sp.symbols("m_1 m_2 s_1 s_2 n_1 n_2", positive=True)
    formula(sp.Eq(sp.Symbol("t"), (m1 - m2) / sp.sqrt(s1**2 / n1 + s2**2 / n2), evaluate=False))
    ''', slides=["54"])
    explain(
        "Run Welch's test on the windows' mean speed.",
        "The slide reports t = 4.08, 39.1 degrees of freedom and p = 0.0002.",
        "`scipy.stats.ttest_ind(..., equal_var=False)` on the 45 and 35 windows, checked against the slide.",
        "The two days differ; the next cells show why that says little about how much.")
    nb.code('''
    before_speed, after_speed = reference["mean_speed"].to_numpy(float), current["mean_speed"].to_numpy(float)
    welch = ttest_ind(after_speed, before_speed, equal_var=False)
    agrees("Welch t on 80 windows", welch.statistic, 4.08, 2, source="slides/measured_v3.json")
    agrees("Welch degrees of freedom", welch.df, 39.1, 1, source="slides/measured_v3.json")
    agrees("p-value on 80 windows", welch.pvalue, 0.0002, 4)
    ''')
    explain(
        "Define the significance-versus-size demonstration as the Lab 4 solution writes it, "
        "with the rest of Lab 4's constants.",
        "A p-value shrinks with the sample size as much as with the size of the change; an "
        "effect size does not [@wasserstein2016]. Resampling the same two days to more "
        "observations changes only n [@efron1979].",
        "Copies Lab 4's constants and `significance_is_not_size()` from `solutions/lab_04.py`: "
        "Welch at the window grain, then the same difference resampled to 48,290 "
        "observations with a seed in the signature, and Cohen's d pooled by degrees of "
        "freedom.",
        "The p-value falls through the floor of a double; d stays near 1.")
    nb.source(f"{S}/lab_04.py", "LAB", "FEATURES", "TARGET", "MATERIAL_SHIFT_SD",
              "INJECTED_SHIFT_SD", "significance_is_not_size",
              cite="[@welch1947; @efron1979; @cohen1988]")
    explain(
        "Compute a p-value that small without underflow.",
        "SciPy returns 0.0 below about 10⁻³⁰⁸, and \"p = 0\" is a statement no test can make. "
        "The two-sided p-value of a t-statistic is a regularised incomplete beta function, "
        "which `mpmath` evaluates at any precision.",
        "Copies `log10_two_sided_t()` from `slides/make_figs_v3.py`.",
        "The slide can print a bound, \"p < 1e-300\", and this cell says how far below it is.")
    nb.source("Module 4/slides/make_figs_v3.py", "log10_two_sided_t")
    explain(
        "Draw how the p-value and the effect size move as n grows, for the same difference.",
        "With enough data every difference is statistically significant; few are material.",
        "Two versions, both from the 45 and 35 windows of mean speed. The model holds the "
        "two days' means and spreads fixed and grows n in the same 45 : 35 proportion. The "
        "resampling draws with replacement to n observations (seed 20200122), as Lab 4 "
        "does. Both report −log₁₀ p from the t-distribution and Cohen's d.",
        "p falls by some 1,800 orders of magnitude while d stays near 1.02. The raw readings "
        "test, which is invalid anyway because consecutive readings correlate at 0.997, "
        "gives p ≈ 10⁻⁵⁹⁴.")
    nb.figure(["p_value_against_n", "effect_size_against_n"], '''
    sizes = [80, 500, 2000, 10000, READING_COUNT]
    m_ref, m_cur = before_speed.mean(), after_speed.mean()
    s_ref, s_cur = before_speed.std(ddof=1), after_speed.std(ddof=1)
    model, resampled = [], []
    for size in sizes:
        n_ref = size * len(before_speed) / (len(before_speed) + len(after_speed)); n_cur = size - n_ref
        variance = s_ref**2 / n_ref + s_cur**2 / n_cur
        df = variance**2 / ((s_ref**2 / n_ref)**2 / (n_ref - 1) + (s_cur**2 / n_cur)**2 / (n_cur - 1))
        model.append(-log10_two_sided_t((m_cur - m_ref) / math.sqrt(variance), df))
        rng = np.random.default_rng(SEED)
        share = int(round(size * len(before_speed) / (len(before_speed) + len(after_speed))))
        big_ref = before_speed if size == 80 else rng.choice(before_speed, size=share, replace=True)
        big_cur = after_speed if size == 80 else rng.choice(after_speed, size=size - share, replace=True)
        test = ttest_ind(big_cur, big_ref, equal_var=False)
        pooled = math.sqrt(((len(big_ref) - 1) * big_ref.var(ddof=1) + (len(big_cur) - 1) * big_cur.var(ddof=1))
                           / (len(big_ref) + len(big_cur) - 2))
        resampled.append((-log10_two_sided_t(test.statistic, test.df), (big_cur.mean() - big_ref.mean()) / pooled))
    labels = [f"{s:,}" for s in sizes]
    fig = go.Figure()
    fig.add_scatter(x=labels, y=model, mode="lines+markers", line=dict(color=RED, width=3),
                    name="model: the two days' means and spreads held fixed")
    fig.add_scatter(x=labels, y=[r[0] for r in resampled], mode="lines+markers",
                    line=dict(color=NAVY, width=2, dash="dash"), name="resampled with replacement, seed 20200122")
    fig.update_layout(title="The p-value falls toward 0 as n grows: −log₁₀ p (1e-100 means 10 to the power −100)",
                      xaxis_title="observations", yaxis=dict(title="−log₁₀ p", type="log"),
                      legend=dict(orientation="h", y=-0.25))
    show(fig, "p_value_against_n", height=440)
    fig = go.Figure()
    fig.add_scatter(x=labels, y=[r[1] for r in resampled], mode="lines+markers", line=dict(color=GREEN, width=3),
                    name="Cohen's d, resampled")
    fig.add_hline(y=1.02, line=dict(color=GREY, dash="dash"), annotation_text="d = 1.02, the model holds it fixed")
    fig.update_layout(title="The effect size does not move with n", xaxis_title="observations",
                      yaxis=dict(title="Cohen's d", range=[0, 1.5]))
    show(fig, "effect_size_against_n", height=380)
    agrees("-log10 p at 80 windows", model[0], 3.7, 1)
    beside("-log10 p at 500 / 2,000 / 10,000 / 48,290 (model; resampled)",
           ([round(v, 1) for v in model[1:]], [round(r[0], 1) for r in resampled[1:]]),
           [19.8, 76.5, 377.1, 1815],
           "the slide's series comes from a model this repository does not record; both recipes "
           "here give the same order of magnitude, and the same message")
    day_of_reading = pd.to_datetime(bus["utc_time"], utc=True).dt.date.astype(str)
    raw = ttest_ind(bus.loc[day_of_reading == CURRENT_DAY, "speed"], bus.loc[day_of_reading == REFERENCE_DAY, "speed"],
                    equal_var=False)
    print(f"raw readings, both days: t = {raw.statistic:.2f}, SciPy's p = {raw.pvalue}, "
          f"true -log10 p = {-log10_two_sided_t(raw.statistic, raw.df):.0f} — so the slide prints p < 1e-300")
    ''', slides=["55"], treatment="lab data: the model and the resampling, both from the real windows")
    explain(
        "Write the two effect sizes and compute them on mean speed.",
        "Cohen's d divides the difference by the pooled spread of both days [@cohen1988]; "
        "Glass's Δ divides by the reference day's spread alone, because in a monitor the "
        "reference day is the untreated group whose spread the monitor knew before the "
        "change [@glass1976]. This module's standardised shift is Glass's Δ.",
        "Displays both formulas and computes the slide's numbers from the windows.",
        "d = 1.02 spread-widths; Δ = 2.30. Whether that is an alarm is decided against a "
        "bound, not by p.")
    nb.equation("effect_sizes", '''
    m1, m2, s1, s2, n1, n2 = sp.symbols("m_1 m_2 s_1 s_2 n_1 n_2", positive=True)
    s_pooled = sp.Symbol(r"s_{\\mathrm{pooled}}")
    formula(sp.Eq(sp.Symbol("d"), (m1 - m2) / s_pooled, evaluate=False),
            sp.Eq(s_pooled, sp.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)), evaluate=False))
    formula(sp.Eq(sp.Symbol(r"\\Delta_{\\mathrm{Glass}}"),
                  (sp.Symbol(r"m_{\\mathrm{current}}") - sp.Symbol(r"m_{\\mathrm{reference}}")) / sp.Symbol(r"s_{\\mathrm{reference}}"),
                  evaluate=False))
    pooled_sd = math.sqrt(((len(before_speed) - 1) * s_ref**2 + (len(after_speed) - 1) * s_cur**2)
                          / (len(before_speed) + len(after_speed) - 2))
    for label, value, stated, places in (("mean speed, 22 January, m/s", m_ref, 0.04, 2),
                                         ("mean speed, 23 January, m/s", m_cur, 0.50, 2),
                                         ("standard deviation, 22 January", s_ref, 0.20, 2),
                                         ("standard deviation, 23 January", s_cur, 0.64, 2),
                                         ("difference in means, m/s", m_cur - m_ref, 0.46, 2),
                                         ("pooled standard deviation", pooled_sd, 0.450, 3),
                                         ("Cohen's d", (m_cur - m_ref) / pooled_sd, 1.02, 2),
                                         ("Glass's delta (the standardised shift)", (m_cur - m_ref) / s_ref, 2.30, 2)):
        agrees(label, value, stated, places)
    ''', slides=["56"])

    # --- slides 57-59: the alarm rule -----------------------------------------------------------
    nb.md("""
    ### The five features, and the alarm rule

    *Slides: "The five features and the two numbers: the names used on every slide",
    "The alarm rule - when a change in a feature is called material" and "How a
    threshold is measured - the null distribution of a feature".*

    | Name on the slides | Column in `windowed()` | One value per window | Unit |
    |---|---|---|---|
    | mean speed | `mean_speed` | average of the speed readings | m/s |
    | speed dispersion | `sd_speed` | standard deviation of the speed readings | m/s |
    | payload dispersion | `sd_payload` | standard deviation of the payload readings | kg |
    | share driven manually | `human_driven` | fraction of readings a human drove | 0 to 1 |
    | mean payload (the target) | `mean_payload` | average of the payload readings | kg |

    Two numbers per feature, each with its own threshold: the **standardised shift**
    (Glass's Δ, in reference-day standard deviations) and the **stability index**
    (PSI, 5 bins). A change is material when either reaches its threshold.

    **Which shift bound.** The deck's decision slides judge the shift against the
    fixed 2.0 reference standard deviations, and so does this notebook's headline.
    Laboratory 4 judges it twice — against 2.0 and against a bound measured the same
    way as the index threshold, the 99th percentile of the shift over 1,000
    resamples of the reference against itself — and reports both. Appendix F sets the
    two side by side; the call on the target is the same either way.
    """)
    explain(
        "Write the alarm rule.",
        "Either number is enough: requiring both would miss a change of shape that leaves "
        "the mean alone, and a shift of the mean that leaves the binned shape alone "
        "[@glass1976; @yurdakul2020].",
        "Displays the rule with the fixed shift bound.",
        "The index threshold is always measured, per feature; there is no borrowed 0.25.")
    nb.equation("alarm_rule", '''
    delta_sym, J_sym = sp.symbols(r"\\Delta J")
    threshold_sym = sp.Function(r"\\mathrm{threshold}")(sp.Symbol("B"), sp.Symbol("q"))
    material_sym = sp.Symbol(r"\\mathrm{material}")
    x_current, x_reference, s_reference = sp.symbols(
        r"\\bar{x}_{\\mathrm{current}} \\bar{x}_{\\mathrm{reference}} s_{\\mathrm{reference}}")
    formula(sp.Equivalent(material_sym, sp.Or(sp.Ge(sp.Abs(delta_sym), sp.Float(2.0, 2)), sp.Ge(J_sym, threshold_sym)),
                          evaluate=False),
            sp.Eq(delta_sym, (x_current - x_reference) / s_reference, evaluate=False), r"B = 5,\\ q = 0.99")
    ''', slides=["58"])
    explain(
        "Define the verdict as the Lab 4 solution writes it.",
        "The first deliverable of Laboratory 4: three measures per feature, a threshold "
        "derived from that feature's own null, and the judgement made twice.",
        "Copies `verdict()` from `solutions/lab_04.py`. It reuses Lab 2's `index_threshold` "
        "and Lab 3's `compare_four` and `wasserstein` through `load_lab`, catches the "
        "degenerate reference and records it as unmeasured, and returns `material` (measured "
        "shift bound) and `material_fixed` (the fixed 2.0) side by side.",
        "The next cells draw what it returns.")
    nb.source(f"{S}/lab_04.py", "verdict", cite="[@glass1976; @yurdakul2020]")
    explain(
        "Run the verdict on the five features.",
        "Every figure and decision in the rest of this part reads its answer.",
        "Calls `verdict()` and prints, per feature, the shift, the index, its threshold and both decisions.",
        "Under the fixed 2.0 two features are material, under the measured bound four; the target under neither.")
    nb.code('''
    results = verdict(reference, current)
    for feature, row in results.items():
        print(f"{feature:13} shift {row['shift_in_reference_sd']:+.2f}  index "
              f"{row['population_stability_index'] if row['index_measured'] else 'unmeasured'}  "
              f"threshold {row['index_threshold'] if row['index_measured'] else 'unmeasured'}  "
              f"material with 2.0: {row['material_fixed']}   with the measured bound "
              f"{row['shift_threshold']:.3f}: {row['material']}")
    ''')
    explain(
        "Draw the alarm rule on the archive: the shift against ±2.0, and each index against "
        "its own bound.",
        "One orange bar in either panel raises the alarm.",
        "Redraws the slide's two-panel figure (drawn in the deck by "
        "`slides/make_figs_alarm_rule.py` with matplotlib) in plotly, from `verdict()`'s "
        "numbers.",
        "Mean speed passes the shift bound; mean speed and speed dispersion pass their index "
        "bounds; the manual-driving share cannot be measured by the index.")
    nb.figure("alarm_rule", '''
    names = {"mean_speed": "Mean speed", "sd_speed": "Speed dispersion", "sd_payload": "Payload dispersion",
             "human_driven": "Share driven manually", "mean_payload": "Mean payload (target)"}
    order = list(results)[::-1]
    QUIET = "#B9BDC3"
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.2,
                        subplot_titles=("Number 1: standardised shift, in reference-day standard deviations",
                                        "Number 2: stability index, each feature against its own measured bound"))
    shifts = [results[f]["shift_in_reference_sd"] for f in order]
    fired = [abs(s) >= MATERIAL_SHIFT_SD for s in shifts]
    fig.add_bar(y=[names[f] for f in order], x=shifts, orientation="h", marker_color=[ORANGE if f else QUIET for f in fired],
                text=[f"{s:+.2f}" + ("  alarm" if f else "") for s, f in zip(shifts, fired)], textposition="outside",
                showlegend=False, row=1, col=1)
    for side in (-MATERIAL_SHIFT_SD, MATERIAL_SHIFT_SD):
        fig.add_vline(x=side, line=dict(color=NAVY, width=2), row=1, col=1)
    fig.update_xaxes(range=[-3.0, 3.4], row=1, col=1)
    for feature in order:
        row = results[feature]
        if not row["index_measured"]:
            fig.add_annotation(x=0.05, y=names[feature], xanchor="left", showarrow=False, row=2, col=1,
                               text="cannot be measured: the reference day is almost all zeros",
                               font=dict(color=GREY, size=12))
            continue
        passed = row["population_stability_index"] >= row["index_threshold"]
        fig.add_bar(y=[names[feature]], x=[row["population_stability_index"]], orientation="h",
                    marker_color=ORANGE if passed else QUIET, showlegend=False, row=2, col=1,
                    text=[f"{row['population_stability_index']:.3f} (bound {row['index_threshold']:.3f})"
                          + ("  alarm" if passed else "")], textposition="outside")
        fig.add_scatter(x=[row["index_threshold"]] * 2, y=[names[feature]] * 2, mode="markers",
                        marker=dict(symbol="line-ns", size=26, line=dict(width=3, color=NAVY)),
                        showlegend=False, row=2, col=1)
    fig.update_yaxes(categoryorder="array", categoryarray=[names[f] for f in order], row=2, col=1)
    fig.update_xaxes(range=[0, 4.75], row=2, col=1)
    fig.update_layout(title="The alarm rule on the archive: 23 January against 22 January")
    show(fig, "alarm_rule", height=640)
    for feature, shift, index, bound in (("mean_speed", 2.30, 2.289, 1.354), ("sd_speed", -0.47, 2.975, 0.431),
                                         ("sd_payload", -0.47, 0.422, 0.440), ("mean_payload", -0.03, 0.081, 0.465)):
        agrees(f"{feature}: shift", results[feature]["shift_in_reference_sd"], shift, 2)
        agrees(f"{feature}: index", results[feature]["population_stability_index"], index, 3)
        agrees(f"{feature}: index bound", results[feature]["index_threshold"], bound, 3)
    agrees("human_driven: shift", results["human_driven"]["shift_in_reference_sd"], 1.24, 2)
    ''', slides=["58"], treatment="lab data: the slide's matplotlib figure redrawn in plotly "
                                  "from verdict()")
    explain(
        "Draw where the threshold 1.354 for mean speed comes from.",
        "From the reference day itself: 1,000 comparisons of 22 January with a resample of "
        "itself, 35 windows each, 5 bins, seed 20200122. Every value is noise; the threshold "
        "is the value the noise exceeds 1% of the time [@efron1979].",
        "Rebuilds the null with the same random stream as `index_threshold()`, counts the "
        "resamples that left a bin empty, and draws the null twice: as a histogram (step 4) "
        "and sorted (step 5), with the real day marked.",
        "23 January reads 2.289, above the bound: alarm. The 21 resamples above 1.0 all left "
        "a bin empty — the index jumps when a bin empties.")
    nb.figure("null_distribution", '''
    rng = np.random.default_rng(SEED)
    edges = np.unique(np.quantile(before_speed, np.linspace(0, 1, 6)))
    edges[0], edges[-1] = -np.inf, np.inf
    null_speed, emptied = [], []
    for _ in range(NULL_RESAMPLES):
        resample = rng.choice(before_speed, size=len(after_speed), replace=True)
        null_speed.append(population_stability_index(before_speed, resample))
        emptied.append(bool((np.histogram(resample, edges)[0] == 0).any()))
    null_speed, emptied = np.array(null_speed), np.array(emptied)
    bound = float(np.quantile(null_speed, NULL_QUANTILE))
    assert abs(bound - index_threshold(before_speed, after_speed)["threshold"]) < 1e-12
    real = population_stability_index(before_speed, after_speed)
    agrees("median of the null (the noise floor)", np.median(null_speed), 0.103, 3)
    agrees("the bound, value number 990 of 1,000", np.sort(null_speed)[989], 1.354, 3)
    agrees("23 January", real, 2.289, 3)
    agrees("resamples above 1.0", (null_speed > 1.0).sum(), 21, 0)
    print(f"of which left a bin empty: {emptied[null_speed > 1.0].sum()}")
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.2,
                        subplot_titles=("Step 4. The 1,000 values, counted: the null distribution",
                                        "Step 5. The same 1,000 values, sorted: the bound is number 990"))
    fig.add_histogram(x=null_speed, xbins=dict(start=0, end=2.75, size=0.05), marker_color="#B9BDC3",
                      showlegend=False, row=1, col=1)
    fig.add_vline(x=bound, line=dict(color=NAVY, width=2), annotation_text=f"bound {bound:.3f}", row=1, col=1)
    fig.add_vline(x=real, line=dict(color=ORANGE, width=3), annotation_text=f"23 January {real:.3f}", row=1, col=1)
    fig.add_scatter(x=np.arange(1, 1001), y=np.sort(null_speed), mode="lines", fill="tozeroy",
                    line=dict(color="#B9BDC3"), showlegend=False, row=2, col=1)
    fig.add_hline(y=real, line=dict(color=ORANGE, width=3), row=2, col=1,
                  annotation_text=f"23 January: {real:.3f}, above the bound: alarm")
    fig.add_scatter(x=[990], y=[bound], mode="markers+text", marker=dict(color=NAVY, size=10), showlegend=False,
                    text=[f"value number 990 = 99th percentile = bound {bound:.3f}"], textposition="middle left",
                    row=2, col=1)
    fig.update_xaxes(title_text="stability index of one resample", range=[0, 2.75], row=1, col=1)
    fig.update_yaxes(title_text="resamples", row=1, col=1)
    fig.update_xaxes(title_text="position after sorting, smallest value first", row=2, col=1)
    fig.update_yaxes(title_text="index", range=[0, 2.75], row=2, col=1)
    fig.update_layout(title="Mean speed: the stability index when nothing has changed<br><sup>22 January "
                            "(45 windows) against 1,000 resamples of itself (35 windows each), 5 bins, "
                            "seed 20200122</sup>", margin=dict(t=110))
    show(fig, "null_distribution", height=640)
    ''', slides=["59"], treatment="lab data: the slide's matplotlib figure redrawn in plotly")

    nb.md("### What the alarm on 23 January means\n\n"
          "*Slides: \"So what? What the alarm on 23 January means and what to do next\" and "
          "\"Multiplicity: monitoring many features at a fixed level\".*\n\n"
          "Inputs moved and the target did not: **covariate shift** — the model now receives "
          "speed values outside the range it was fitted on, which is extrapolation, and "
          "extrapolation error is not visible in the training score [@shimodaira2000; "
          "@quinonero2009]. The first thing to do is compute the prediction error on "
          "23 January, per window, and compare it with 22 January's.")
    explain(
        "Put a number on the arithmetic of watching many features.",
        "Twenty tests at the 5% level give one false alarm per run on average even when "
        "nothing changed; controlling the false discovery rate is the repair that does not "
        "collapse power as features grow [@benjamini1995].",
        "Displays the family-wise false-alarm probability and evaluates it for 20 features, "
        "and the expected number of false alarms per run for 20 and 100.",
        "Alert fatigue is the predictable output of this arithmetic, not a human weakness.")
    nb.equation("multiplicity", '''
    alpha, m = sp.symbols(r"\\alpha m", positive=True)
    family = 1 - (1 - alpha)**m
    at_twenty = sp.Add(1, -sp.Pow(sp.Float(0.95, 2), 20, evaluate=False), evaluate=False)
    formula(sp.Eq(sp.Symbol(r"\\mathbb{P}(\\text{at least one false alarm})"), family, evaluate=False),
            r"\\Rightarrow", sp.Eq(at_twenty, sp.Float(round(float(family.subs({alpha: 0.05, m: 20})), 2), 2),
                                   evaluate=False))
    agrees("probability of at least one false alarm, 20 features at 5%", family.subs({alpha: 0.05, m: 20}), 0.64, 2)
    print("expected false alarms per run: 20 features ->", 20 * 0.05, "; 100 features ->", 100 * 0.05)
    ''', slides=["61"])

    # --- slides 62-65: the decision per feature --------------------------------------------------
    nb.md("### Decision per feature: two inputs moved, one instrument failed\n\n"
          "*Slides: \"Decision per feature: two inputs moved, one instrument failed\", \"The table "
          "each feature's decision is read from\", \"The target lies below its own noise floor\" "
          "and \"The target distribution did not move\".*")
    explain(
        "Draw the five shifts, coloured by the decision each feature received under the "
        "deck's rule.",
        "Mean speed moved 2.3 reference standard deviations and every statistic agrees on it; "
        "the largest index belongs to speed dispersion, whose shift is a fifth the size: two "
        "statistics, two orderings, both right about their own question.",
        "Runs `figure_shifts()` from `slides/make_figs.py`, verbatim, fed with `verdict()`'s "
        "numbers and the fixed-bound decision.",
        "2 of 5 features are material; the target is not.")
    nb.source("Module 4/slides/make_figs.py", "figure_shifts")
    explain(
        "Draw the slide's shift figure.",
        "It colours each feature by the decision it received under the deck's rule.",
        "Calls `figure_shifts()` with `verdict()`'s numbers and the fixed-bound decision.",
        "2 of 5 features are material; the target is not.")
    nb.figure("shifts", '''
    table_rows = {feature: {"shift": row["shift_in_reference_sd"], "material": row["material_fixed"],
                            "index": row["population_stability_index"] if row["index_measured"] else "unmeasured"}
                  for feature, row in results.items()}
    material_fixed = [f for f, row in results.items() if row["material_fixed"]]
    figure_shifts({"verdict_table": {"value": table_rows},
                   "verdict_summary": {"value": {"material_count": len(material_fixed)}}})
    agrees("features material under the deck's rule", len(material_fixed), 2, 0)
    ''', slides=["62"], treatment="exact: the slide's own code, fed with verdict()")
    explain(
        "Print the table each decision is read from, and the count under the borrowed 0.25.",
        "No row is judged against a number from the literature.",
        "Tabulates shift, index, floor, derived threshold and both decisions per feature; "
        "then counts how many indices would pass 0.25.",
        "Under the borrowed 0.25 the count would be 3: the extra alarm is payload dispersion, "
        "whose index 0.422 sits just under its own threshold of 0.44 — a close call the "
        "borrowed number could not have identified as close.")
    nb.code('''
    decision = pd.DataFrame({
        feature: {"shift (reference s.d.)": round(row["shift_in_reference_sd"], 2),
                  "PSI": round(row["population_stability_index"], 3) if row["index_measured"] else "unmeasured",
                  "noise floor": round(row["noise_floor"], 3) if row["index_measured"] else "unmeasured",
                  "derived threshold": round(row["index_threshold"], 3) if row["index_measured"] else "unmeasured",
                  "material (2.0)": row["material_fixed"], "material (measured bound)": row["material"]}
        for feature, row in results.items()}).T
    display(decision)
    for feature, floor in (("mean_speed", 0.103), ("sd_speed", 0.067), ("sd_payload", 0.102), ("mean_payload", 0.105)):
        agrees(f"{feature}: noise floor", results[feature]["noise_floor"], floor, 3)
    borrowed = [f for f, row in results.items() if row["index_measured"]
                and row["population_stability_index"] >= BORROWED_INDEX]
    agrees("features whose index passes the borrowed 0.25", len(borrowed), 3, 0)
    print("they are:", borrowed)
    target = results[TARGET]
    print(f"the target: index {target['population_stability_index']:.3f}, floor {target['noise_floor']:.3f}, "
          f"threshold {target['index_threshold']:.3f} — below its threshold, so no alarm; below its floor too, "
          "but the floor is the centre of the null, so that is context, not stronger evidence")
    ''')
    explain(
        "Find the cause, which sits in a column nobody was monitoring.",
        "The share of the day driven by a person rose from about a tenth to about two fifths; "
        "that is the change with a date, and it points at one event on 23 January.",
        "Computes the manual share two ways — the mean of the per-window shares, and the share "
        "of all readings — for each day.",
        "The index cannot measure this column (almost all zeros on the reference day), but its "
        "shift of +1.24 reference standard deviations says it moved.")
    nb.code('''
    window_share = {day: round(100 * float(group["human_driven"].mean()), 1) for day, group in table.groupby("day")}
    reading_share = {day: round(100 * float((bus.loc[day_of_reading == day, "mode"] == "manual").mean()), 2)
                     for day in (REFERENCE_DAY, CURRENT_DAY)}
    print("manual mode, mean of the per-window shares, per cent:", window_share)
    print("manual mode, share of all readings, per cent:        ", reading_share)
    agrees("manual share, 22 January (windows), per cent", window_share[REFERENCE_DAY], 9.1, 1)
    agrees("manual share, 23 January (windows), per cent", window_share[CURRENT_DAY], 41, 0)
    ''')
    explain(
        "Draw the target's two distributions.",
        "Inputs moved, target did not. A monitor on the inputs would have raised an alarm; a "
        "monitor on the target would have stayed silent. Both are right, and the operator's "
        "question — must anything be done? — is answered by the second.",
        "Runs `figure_target()` from `slides/make_figs.py`, verbatim.",
        "The target here is mean payload per window, a continuous surrogate observable without "
        "buying a label — not the aboard label Module 3's service predicts.")
    nb.source("Module 4/slides/make_figs.py", "figure_target")
    explain(
        "Draw the target's two distributions.",
        "The slide shows that the thing the model predicts did not move.",
        "Calls `figure_target()` with the two days' windows and the verdict's shift.",
        "The two histograms coincide: a shift of −0.03 reference standard deviations.")
    nb.figure("target", '''
    figure_target(reference, current, results)
    agrees("the target's shift", results[TARGET]["shift_in_reference_sd"], -0.03, 2)
    ''', slides=["65"], treatment="exact: the slide's own code")

    nb.md("""
    ### Covariate shift, concept drift, change points — and what sensitivity analysis is not

    *Slides: "Covariate shift, concept drift, and change points" and "Sensitivity
    analysis is not uncertainty quantification".*

    - **Covariate shift** — the input distribution moves, the target's does not. On
      the bus: mean speed +2.3 reference standard deviations, PSI 2.289 against
      1.354; the target −0.03, PSI 0.081 against 0.465 [@quinonero2009].
    - **Concept drift** — the rule linking inputs to target moves, possibly with both
      distributions unchanged. Nothing in this notebook can show it; only the
      prediction error can, which is Module 5's subject.
    - **Change point** — either kind of change, located at a date [@page1954;
      @truong2020]. The manual share jumped from 9.1% to 41% between the two days.
      Two days cannot tell a jump from a slope, so a monitor must keep its history:
      one row per day per feature.
    - **Sensitivity analysis is not uncertainty quantification** [@saltelli2019].
      The accuracy interval and the noise floor are uncertainty quantification; the
      ranking of features by index is sensitivity analysis. Report both, and never let
      the first stand in for the second.
    """)

    # --- slides 68-71: positive control and detection limit ---------------------------------------
    nb.md("### The positive control and the detection limit\n\n"
          "*Slides: \"Definition - the positive control\", \"Definition - the detection limit\", "
          "\"The detection-limit sweep\" and \"Why the sweep spikes at +16.6 kg: windows crossing "
          "a bin edge\".*")
    explain(
        "Write the positive control and the detection limit.",
        "The monitor said the target did not move. Without a control that shows it can fire, "
        "that is an opinion — the underlying discipline is argued in {@saltelli2019}; the detection limit states what it cannot see, "
        "as analytical chemistry does before reporting \"not detected\" [@currie1968].",
        "Displays both definitions.",
        "The control proves a 1.5 standard-deviation shift is seen; only a sweep says where "
        "seeing stops.")
    nb.equation("positive_control_and_limit", '''
    k, j = sp.symbols("k j", nonnegative=True)
    verdict_of = sp.Function(r"\\mathrm{verdict}")
    ref_sym, cur_sym, s_ref_sym = sp.symbols(r"\\mathrm{ref} \\mathrm{cur} s_{\\mathrm{ref}}")
    material_sym = sp.Symbol(r"\\texttt{material}")
    formula(sp.Eq(verdict_of(ref_sym, cur_sym + k * s_ref_sym), material_sym, evaluate=False), r"k \\text{ reported with the result}")
    swept = sp.Interval(0, sp.Rational(3, 2))           # the grid 0, 0.05, ..., 1.5 lies in it
    sustained = sp.Eq(verdict_of(ref_sym, cur_sym + j * s_ref_sym), material_sym, evaluate=False)
    formula(sp.Eq(sp.Symbol(r"\\mathrm{LOD}"), sp.Function(r"\\min")(sp.ConditionSet(k, sustained, swept)),
                  evaluate=False), r"\\text{for every swept } j \\ge k")
    ''', slides=["68", "69"])
    explain(
        "Define the positive control and its sweep as the Lab 4 solution writes them.",
        "The second deliverable of Laboratory 4.",
        "Copies `positive_control()` from `solutions/lab_04.py`: inject 1.5 reference "
        "standard deviations into a copy of the current day's target, run the unchanged "
        "verdict, then sweep 0 to 1.5 in steps of 0.05 and report the first size from which "
        "the verdict stays material.",
        "At k = 1.5 the shift stays under 2.0, so under the deck's rule only the index can "
        "fire — the control tests the instrument whose silence we rely on.")
    nb.source(f"{S}/lab_04.py", "positive_control", cite="[@currie1968]; the discipline behind the control: [@saltelli2019]")
    explain(
        "Run the positive control and the sweep, and check the slides' numbers.",
        "The slides report an index of 8.221 after the injection, a shift of 1.47, and a detection limit of 0.40 s.d. = 44.3 kg.",
        "Calls `positive_control()`, checks each number, then re-runs the verdict to show the archive's answer is untouched.",
        "The control fires, and the real answer is still not material.")
    nb.code('''
    control = positive_control(reference, current)
    agrees("index before the injection", results[TARGET]["population_stability_index"], 0.081, 3)
    agrees("index after injecting 1.5 s.d.", control["population_stability_index"], 8.221, 3)
    agrees("shift after injecting 1.5 s.d.", control["shift_in_reference_sd"], 1.47, 2)
    agrees("detection limit, reference s.d.", control["detection_limit_sd"], 0.40, 2)
    agrees("detection limit, kg per window", control["detection_limit_in_target_units"], 44.3, 1)
    agrees("reference spread of the target, kg", control["target_reference_sd"], 110.8, 1)
    agrees("first size at which it fires at all, s.d.", control["first_material_sd"], 0.15, 2)
    untouched = verdict(reference, current)[TARGET]
    print("the real answer after the control:", "material" if untouched["material"] else "not material",
          "— the control shifted a copy, not the archive")
    ''')
    explain(
        "Draw the sweep the detection limit is read from.",
        "One positive control says the monitor can fire; the sweep says which sizes it can and "
        "cannot see.",
        "Runs `figure_detection()` from `slides/make_figs.py`, verbatim, fed with the sweep "
        "above.",
        "0.40 reference standard deviations, 44.3 kg per window: a smaller shift is invisible "
        "to this monitor. The spike at 0.15 is not detection — it fires once and falls back.")
    nb.source("Module 4/slides/make_figs.py", "figure_detection")
    explain(
        "Draw the detection-limit sweep.",
        "The slide reads the limit off this figure.",
        "Calls `figure_detection()` with the sweep above.",
        "The limit is where the index crosses the threshold and stays above it.")
    nb.figure("detection", '''
    figure_detection({"_detection_sweep": {"sizes": control["sizes"], "index": control["index_by_size"],
                                           "threshold": control["index_threshold"],
                                           "floor": results[TARGET]["noise_floor"]},
                      "detection_limit": {"value": {"limit_sd": control["detection_limit_sd"],
                                                    "limit_kilograms": control["detection_limit_in_target_units"]}}})
    ''', slides=["70"], treatment="exact: the slide's own code, fed with positive_control()")
    explain(
        "Show why the sweep spikes at +16.6 kg and falls back at +22.2 kg, on the real windows.",
        "Adding kilograms moves the windows, not the bins: at one step a cluster of windows "
        "leaves a bin before any arrive from below, that bin is nearly empty, and the index "
        "jumps.",
        "Counts the current day's 35 windows in the reference's five fixed payload bins with "
        "no shift, +0.15 and +0.20 reference standard deviations, and draws them. The slide "
        "works this with constructed values; here are the real ones.",
        "The limit is the first size that fires and keeps firing, not the first that fires "
        "once.")
    nb.figure("bin_edge_crossing", '''
    spread_target = control["target_reference_sd"]
    fixed_edges = np.unique(np.quantile(ref_payload, np.linspace(0, 1, 6)))
    fixed_edges[0], fixed_edges[-1] = -np.inf, np.inf
    rows = []
    for size in (0.0, 0.15, 0.20):
        moved = cur_payload + size * spread_target
        rows.append((size, moved, np.histogram(moved, fixed_edges)[0],
                     population_stability_index(ref_payload, moved)))
    fig = go.Figure()
    for level, (size, moved, counts, index) in enumerate(rows):
        fig.add_scatter(x=moved, y=np.full(len(moved), -level), mode="markers", marker=dict(size=8, color=NAVY),
                        showlegend=False)
        fig.add_annotation(x=0, y=-level + 0.35, xanchor="left", showarrow=False,
                           text=f"+{size * spread_target:.1f} kg ({size:.2f} reference s.d.) → index {index:.2f}; "
                                f"windows per bin {counts.tolist()}")
    for edge in fixed_edges[1:-1]:
        fig.add_vline(x=edge, line=dict(color=GREY, dash="dash"))
    fig.update_layout(title="Adding kilograms moves the windows, not the bins (dashed: the reference's fixed edges)",
                      xaxis_title="mean payload of one window, kg", yaxis=dict(visible=False, range=[-2.6, 0.7]))
    show(fig, "bin_edge_crossing", height=420)
    agrees("shift at 0.15 s.d., kg", 0.15 * spread_target, 16.6, 1)
    agrees("shift at 0.20 s.d., kg", 0.20 * spread_target, 22.2, 1)
    beside("index at 0, +16.6 and +22.2 kg", [round(r[3], 2) for r in rows], [0.01, 0.39, 0.16],
           "the slide's worked example uses constructed payload values; the real windows spike at +16.6 kg too")
    ''', slides=["71"], treatment="lab data: the slide's constructed windows replaced by the real ones")

    # --- slides 72-73: the decision -------------------------------------------------------------------
    nb.md("### The drift decision and a defensible negative result\n\n"
          "*Slides: \"Definition - the drift decision and its three admissible calls\" and "
          "\"Writing a defensible negative result\".*")
    explain(
        "Define the decision as the Lab 4 solution writes it, and issue it on the archive.",
        "Two calls come from the data — act, or no material change; watch means the monitor "
        "has no verdict yet, because its positive control was not run or did not fire "
        "[@saltelli2019].",
        "Copies `drift_verdict()` from `solutions/lab_04.py`, builds the evidence dictionary "
        "from the measurements above, and prints the call and the reason — a reason built out "
        "of the evidence and nothing else.",
        "No material change: do not retrain. The claim carries its detection limit.")
    nb.source(f"{S}/lab_04.py", "drift_verdict", cite="[@saltelli2019]")
    explain(
        "Issue the drift decision on the archive.",
        "The call and its reason are Lab 4's third deliverable.",
        "Builds the evidence from the measurements above, with the deck's fixed-rule list of material inputs, and calls `drift_verdict()`.",
        "No material change, quoted with the detection limit.")
    nb.code('''
    # the deck's rule, the fixed 2.0, decides which inputs count as material here;
    # the lab's own demonstration below passes the measured-bound list instead
    material_fixed = [f for f, row in results.items() if row["material_fixed"]]
    evidence = {"target_index": results[TARGET]["population_stability_index"],
                "target_index_measured": results[TARGET]["index_measured"],
                "standardised_shift": results[TARGET]["shift_in_reference_sd"],
                "shift_threshold": results[TARGET]["shift_threshold"],
                "noise_floor": results[TARGET]["noise_floor"],
                "index_threshold": results[TARGET]["index_threshold"],
                "control_index": control["population_stability_index"],
                "detection_limit": control["detection_limit_sd"],
                "material_features": material_fixed, "material_count": len(material_fixed),
                "features_watched": len(FEATURES)}
    call, reason = drift_verdict(evidence)
    print("call:  ", call)
    print("reason:", reason)
    assert call == "no material change"
    print()
    print(f"The sentence the slide asks for: \\"Mean payload shows no change bigger than "
          f"{control['detection_limit_in_target_units']:.1f} kilograms per window. Its stability index is "
          f"{results[TARGET]['population_stability_index']:.3f}, against a threshold of "
          f"{results[TARGET]['index_threshold']:.3f} fixed before looking. The monitor fires on an injected shift "
          f"of {control['detection_limit_in_target_units']:.1f} kilograms per window or more. A smaller shift "
          "would not have been seen.\\"")
    ''')

    # --- slides 74-76: C2ST and power ---------------------------------------------------------------
    nb.md("### The classifier two-sample test, and its power\n\n"
          "*Slides: \"The classifier two-sample test, run on the bus data\", \"Power: how often "
          "would the classifier test have fired?\" and \"Power 0.58, drawn: the counts that fire "
          "and the counts that miss\".*")
    explain(
        "Define and run the domain-classifier test the required reading evaluates.",
        "Train a classifier to tell a reference window from a current one; if the days are "
        "one world, nothing beats chance [@friedman2004; @lopezpaz2017]. The required reading "
        "compares it with univariate tests and finds that it performs badly with few samples, "
        "and that a pretrained classifier's outputs with univariate tests (BBSDs) do best "
        "[@rabanser2019, § 4]. The lab solution's docstring below, copied verbatim, says the "
        "reading found the classifier hard to beat; read it with that correction.",
        "Copies `classifier_two_sample_test()` from `solutions/lab_04.py` — balanced days, "
        "half trains and half is held out, features standardised by the training reference, "
        "nearest centroid — and reads the accuracy through Lab 1's Wilson interval.",
        "All five features: 24 of 36, interval 0.503 to 0.798 — drift detected, just. The "
        "target alone: 15 of 36, interval containing 0.5. It says something moved; it cannot "
        "say what.")
    nb.source(f"{S}/lab_04.py", "classifier_two_sample_test", cite="[@rabanser2019]")
    explain(
        "Run the classifier test on all five features and on the target alone.",
        "The slides report 24 of 36 with [0.503, 0.798], and 15 of 36 with [0.271, 0.578].",
        "Calls the function twice and checks the slides' numbers, including the Wilson centre and half-width computed with z = 1.96 as printed.",
        "Detected, barely, on all features; not on the target.")
    nb.code('''
    joint = classifier_two_sample_test(reference, current)
    alone = classifier_two_sample_test(reference, current, [TARGET])
    agrees("all features: correct", joint["correct"], 24, 0)
    agrees("all features: held out", joint["held_out"], 36, 0)
    agrees("all features: Wilson lower", joint["interval"][0], 0.503, 3)
    agrees("all features: Wilson upper", joint["interval"][1], 0.798, 3)
    # the slide's centre and half-width, from its formula with z = 1.96 as printed
    k_ok, n_ok, z_slide = joint["correct"], joint["held_out"], 1.96
    centre = (k_ok / n_ok + z_slide**2 / (2 * n_ok)) / (1 + z_slide**2 / n_ok)
    half = z_slide * math.sqrt(k_ok * (n_ok - k_ok) / n_ok**3 + z_slide**2 / (4 * n_ok**2)) / (1 + z_slide**2 / n_ok)
    beside("Wilson centre, z = 1.96", round(centre, 3), 0.650,
           "the slide rounds 0.6506 down; slides/measured_v3.json records 0.651")
    agrees("Wilson half-width, z = 1.96", half, 0.147, 3)
    agrees("target alone: correct", alone["correct"], 15, 0)
    agrees("target alone: Wilson lower", alone["interval"][0], 0.271, 3)
    agrees("target alone: Wilson upper", alone["interval"][1], 0.578, 3)
    ''')
    explain(
        "Compute the test's power at the observed accuracy, and draw it.",
        "Rabanser and colleagues ran their tests on 10 to 10,000 samples and report that the "
        "domain classifier performs badly in the low-sample regime [@rabanser2019, § 4]. At "
        "n = 36 this test is close to a coin flip — check the sample size before adopting a "
        "method from a paper.",
        "Finds the smallest count k whose Wilson interval lies above 0.5, then adds the "
        "binomial probabilities of every count from there up, at p = 24/36; repeats at "
        "n = 100 and n = 1,000.",
        "Power 0.58 at n = 36: the peak of the bars sits on the firing line. Below power 0.8 the "
        "test's silence proves nothing.")
    nb.figure("power", '''
    # the slide's stated inputs: p = 0.667, the observed accuracy rounded, and z = 1.96
    p_observed, z_slide = 0.667, 1.96
    power = {}
    for n_held in (36, 100, 1000):
        k_fire = min(k for k in range(n_held + 1) if wilson_interval(k, n_held, z_slide)[0] > 0.5)
        power[n_held] = (k_fire, float(stats.binom.sf(k_fire - 1, n_held, p_observed)))
    for n_held, stated_k, stated_power in ((36, 24, 0.58), (100, 60, 0.94), (1000, 531, 1.00)):
        agrees(f"n = {n_held}: fire when k >=", power[n_held][0], stated_k, 0)
        agrees(f"n = {n_held}: power", power[n_held][1], stated_power, 2)
    beside("lower end at 23 of 36", round(wilson_interval(23, 36, z_slide)[0], 4), 0.475,
           "0.4757 — the slide truncates it; either way it is below 0.5, so k = 23 does not fire")
    agrees("lower end at 24 of 36", wilson_interval(24, 36, z_slide)[0], 0.503, 3)
    for k, stated in ((24, 0.140), (25, 0.135), (26, 0.114), (27, 0.085), (28, 0.054), (29, 0.030)):
        agrees(f"P({k}) at n = 36", stats.binom.pmf(k, 36, p_observed), stated, 3)
    print(f"with the unrounded p = 24/36 and z = {Z_95:.5f}: power at n = 100 is "
          f"{stats.binom.sf(power[100][0] - 1, 100, 24 / 36):.3f} — the rounding moves the second decimal")
    ks = np.arange(12, 37)
    fig = go.Figure(go.Bar(x=ks, y=stats.binom.pmf(ks, 36, p_observed),
                           marker_color=[GREEN if k >= power[36][0] else "#B9BDC3" for k in ks]))
    fig.add_vline(x=power[36][0] - 0.5, line=dict(color=NAVY, width=2))
    fig.add_annotation(x=power[36][0] + 5, y=0.13, showarrow=False,
                       text=f"fires: k ≥ {power[36][0]}, adds to {power[36][1]:.2f}")
    fig.add_annotation(x=power[36][0] - 7, y=0.13, showarrow=False,
                       text=f"silent, adds to {1 - power[36][1]:.2f}")
    fig.update_layout(title="Probability of each count k, n = 36, true accuracy 0.667",
                      xaxis_title="k, held-out windows labelled correctly", yaxis_title="probability")
    show(fig, "power", height=420)
    ''', slides=["76"], treatment="exact: the binomial law at the observed accuracy")

    # --- the laboratory -------------------------------------------------------------------------------
    nb.md("""
    ## Laboratory 4 — the decision

    *Slide: "Laboratory 4 - the decision".* Five deliverables, twenty-five minutes; no
    index threshold is supplied, and the correct answer on the target is that nothing
    material changed. The check verifies that mean speed is material and the target is
    not; that the target's index falls below its own measured floor; that your control
    fires; that the sweep re-runs your own decision rather than reading a stored
    table; and that the degenerate column is reported as unmeasured rather than as
    unmoved.
    """)
    nb.statement(f"{LABS}/04_the_verdict.py")
    nb.md("""
    ### The solution, step by step

    The five deliverables were defined above: `verdict`, `positive_control`,
    `drift_verdict`, `significance_is_not_size` and `classifier_two_sample_test`.
    First the stub's own demonstration, then the solution's.
    """)
    explain(
        "Run the stub's own `__main__` block against the solved functions.",
        "It is what a student sees when the lab file is complete.",
        "The lines below are the demonstration at the end of `labs/04_the_verdict.py`, verbatim; its "
        "`from lab_support import …` resolves to this notebook's cells.",
        "The printed values are the ones the lab's check expects.")
    nb.step(f"{LABS}/04_the_verdict.py", 0)
    explain(
        "Start the narrator the solution's demonstration prints with.",
        "The demonstration below reports every step through `say.info`, as it does in the terminal.",
        "Sets the lab number and opens the narrator, as the first lines of the solution's "
        "`__main__` block do.",
        "The numbered steps that follow run unchanged.")
    nb.code('''
    LAB = 4
    say = narrator(LAB)
    say.info("Lab 4 — a threshold derived rather than borrowed, the verdict it produces, the detection "
             "limit behind it, and the classifier the required reading argues for")
    say.info("archive slice, %d reference windows on the first day against %d on the second, five-minute "
             "windows of at least 300 readings, one vehicle", len(reference), len(current))
    ''')
    for number, (goal, so_what) in enumerate([
            ("The verdict, on thresholds derived from each feature's own null.",
             "four features are material with the measured shift bound, two with the fixed 2.0; "
             "the target is material under neither."),
            ("The invitation: any list of features, the five graded and the rest printed.",
             "adding a column costs a name in a list."),
            ("The positive control, and the sweep that turns it into a detection limit.",
             "the control fires and leaves the real answer untouched; the limit is 0.40 s.d."),
            ("The call, and the reason it would be defended with.",
             "no material change, quoted with the detection limit."),
            ("Significance is not size.",
             "the p-value underflows at the reading grain; Cohen's d stays at 1.02."),
            ("The closing test, the domain classifier the required reading evaluates.",
             "distinguishable, barely — and not on the target alone."),
            ("Picture one: the five shifts, coloured by the verdict with the measured bound.",
             "the lab's version colours by the measured bound, so four bars are orange; the "
             "slide's version above colours by the fixed 2.0."),
            ("Picture two: the target before and after the injected shift.",
             "the null result and the control that backs it, side by side."),
            ("Picture three: the sweep itself.",
             "where the detection limit is read."),
    ], start=1):
        explain(f"Step {number} of the solution's demonstration: {goal[0].lower() + goal[1:]}",
                "It is the reference solution's own demonstration; running it here shows the "
                "functions above at work on the lab data, exactly as `make demo` does.",
                f"Runs step {number} of the demonstration block of `solutions/lab_04.py`, verbatim; "
                "`say.info` prints each finding with the seconds elapsed since the lab began.",
                so_what[0].upper() + so_what[1:])
        nb.step(f"{S}/lab_04.py", number)
