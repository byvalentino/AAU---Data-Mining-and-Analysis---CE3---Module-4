"""Part 1, second half, and Laboratory 1: coverage, labels, the archive's windows,
the small-sample exercise, and the Lab 1 solution run step by step."""

S = "Module 4/exercises/solutions"
LABS = "Module 4/exercises/labs"


def build(nb, explain) -> None:
    explain(
        "Define the two intervals as the Lab 1 solution writes them.",
        "These two functions are the first two deliverables of Laboratory 1, and every "
        "interval in the rest of the module — including the one on the classifier test in "
        "Part 4 — is built with them.",
        "Copies `naive_interval` (the Wald interval) and `wilson_interval` from "
        "`solutions/lab_01.py`, with their docstrings.",
        "Two functions of the same inputs; the rest of Part 1 measures where they differ.")
    nb.source(f"{S}/lab_01.py", "LAB", "Z_95", "naive_interval", "wilson_interval",
              cite="[@wilson1927; @brown2001; @agresti1998]")

    explain(
        "Measure the Wald interval's real coverage at every true rate from 0.01 to 0.50.",
        "The slide's chart shows how far below its promised 95% the Wald interval falls "
        "at n = 40, and that it falls furthest near 0, where a monitor's rare failures live "
        "[@brown2001].",
        "Computes the coverage exactly rather than by simulation: for each true rate p it "
        "adds the binomial probability of every count k whose Wald interval contains p.",
        "33% at p = 0.01. The coverage also saw-tooths as p moves — the \"erratic\" "
        "behaviour Brown, Cai and DasGupta describe.")
    nb.figure("wald_exact_coverage", '''
    grid = [round(0.01 * i, 2) for i in range(1, 51)]
    exact = [100 * sum(stats.binom.pmf(k, 40, p) for k in range(41)
                       if naive_interval(k, 40)[0] <= p <= naive_interval(k, 40)[1])
             for p in grid]
    fig = go.Figure()
    fig.add_scatter(x=grid, y=exact, mode="lines+markers", line=dict(color=RED, width=2.5),
                    name="Wald: actual coverage")
    fig.add_hline(y=95, line=dict(color=NAVY, dash="dash"), annotation_text="nominal 95%")
    fig.update_layout(title="Wald coverage, n = 40 (exact binomial computation)",
                      xaxis_title="true rate p", yaxis_title="coverage, per cent",
                      yaxis_range=[0, 100], showlegend=False)
    show(fig, "wald_exact_coverage", height=440)
    agrees("Wald coverage at p = 0.01, per cent", exact[0], 33.1, 1)
    agrees("Wald coverage at p = 0.02, per cent", exact[1], 55.3, 1)
    ''', slides=["13"], treatment="exact: the binomial law, no simulation")

    explain(
        "Draw both intervals' endpoints for every count k out of 40.",
        "The Wilson slide's chart shows the two sets of endpoints: Wald's dip below 0 near "
        "the lower boundary and pinch to a point at both ends; Wilson's do neither "
        "[@wilson1927].",
        "Evaluates both functions at k = 0, …, 40 and draws the four endpoint curves.",
        "At k = 40 Wald reports [1, 1]; Wilson reports [0.912, 1].")
    nb.figure("interval_endpoints", '''
    ks = np.arange(41)
    wald = np.array([naive_interval(int(k), 40) for k in ks])
    wilson = np.array([wilson_interval(int(k), 40) for k in ks])
    fig = go.Figure()
    for column, name in ((1, "Wilson upper"), (0, "Wilson lower")):
        fig.add_scatter(x=ks, y=wilson[:, column], mode="lines", name=name,
                        line=dict(color=GREEN, width=3))
    for column, name in ((1, "Wald upper"), (0, "Wald lower")):
        fig.add_scatter(x=ks, y=wald[:, column], mode="lines", name=name,
                        line=dict(color=NAVY, width=2, dash="dash"))
    fig.add_hline(y=0, line=dict(color=GREY, width=1))
    fig.add_hline(y=1, line=dict(color=GREY, width=1))
    fig.update_layout(title="Interval endpoints, n = 40. Green: Wilson. Dashed navy: Wald, "
                            "which leaves [0, 1]",
                      xaxis_title="k, predictions found correct", yaxis_title="interval endpoint")
    show(fig, "interval_endpoints", height=460)
    agrees("Wald lower end at k = 1", wald[1, 0], -0.023, 3)
    agrees("Wilson upper end at k = 0", wilson[0, 1], 0.088, 3)
    ''', slides=["14"], treatment="exact: closed form")

    explain(
        "Work the slide's two cases: 34 of 40 correct, and 40 of 40.",
        "Wald and Wilson agree when it does not matter and differ exactly when it does. "
        "The slide also explains why Wilson is not centred on 0.85: a true accuracy of 0.78 "
        "produces 34 of 40 more easily than one of 0.92 does.",
        "Computes both intervals for both cases, the Wilson centre, and the two binomial "
        "probabilities behind the asymmetry, and draws the four intervals as bars.",
        "Report the interval, not the single number; with 40 of 40 the single number would "
        "claim a perfect model.")
    nb.figure("two_cases", '''
    cases = {"Case 1 · Wald": naive_interval(34, 40), "Case 1 · Wilson": wilson_interval(34, 40),
             "Case 2 · Wald": naive_interval(40, 40), "Case 2 · Wilson": wilson_interval(40, 40)}
    fig = go.Figure(go.Bar(y=list(cases), x=[high - low for low, high in cases.values()],
                           base=[low for low, _ in cases.values()], orientation="h",
                           marker_color=[NAVY, GREEN, NAVY, GREEN],
                           text=[f"[{low:.3f}, {high:.3f}]" for low, high in cases.values()],
                           textposition="outside"))
    fig.update_layout(title="95% intervals for the two cases, n = 40",
                      xaxis_title="accuracy", xaxis_range=[0.6, 1.08])
    show(fig, "two_cases", height=380)
    low, high = naive_interval(34, 40)
    agrees("Wald 34/40, lower", low, 0.739, 3); agrees("Wald 34/40, upper", high, 0.961, 3)
    agrees("Wald half-width at 34/40", (high - low) / 2, 0.111, 3)
    low, high = wilson_interval(34, 40)
    agrees("Wilson 34/40, lower", low, 0.709, 3); agrees("Wilson 34/40, upper", high, 0.929, 3)
    agrees("Wilson centre at 34/40", (low + high) / 2, 0.819, 3)
    agrees("Wald 40/40, lower", naive_interval(40, 40)[0], 1, 3)
    agrees("Wilson 40/40, lower", wilson_interval(40, 40)[0], 0.912, 3)
    print(f"P(34 of 40 | p = 0.78) = {stats.binom.pmf(34, 40, 0.78):.3f}   "
          f"P(34 of 40 | p = 0.92) = {stats.binom.pmf(34, 40, 0.92):.3f}")
    ''', slides=["15"], treatment="exact: closed form")

    nb.md("### Both intervals across the whole outcome space\n\n"
          "*Slide: \"Both intervals across the whole outcome space\".* The figure on this slide "
          "is drawn by `slides/make_figs.py`; the next cells run that script's own drawing "
          "function, fed with the Lab 1 functions above.")
    explain(
        "Bring in the deck's figure helpers, so the slide's figures can be drawn by the "
        "slide's own code.",
        "Six figures in this deck come from `slides/make_figs.py`. Running its drawing "
        "functions verbatim is the surest way to reproduce them.",
        "Copies the palette, the layout and `axis()` from `make_figs.py`; `save()` is the "
        "one change — there it writes `figures/<name>.png`, here it applies the same "
        "layout and shows the figure.",
        "From here on a figure labelled *the slide's own code* is the deck's figure, "
        "recomputed.")
    nb.source("Module 4/slides/make_figs.py", "TARGET", "GRID", "LAYOUT", "axis")
    explain(
        "Show make_figs.py's figures in place.",
        "make_figs.py's `save()` writes `figures/<name>.png` for the deck.",
        "Applies make_figs.py's own layout and shows the figure instead of writing it.",
        "Figures drawn by the slide's own functions appear here unchanged.")
    nb.code('''
    def save(fig, name, width=1100, height=620):
        """make_figs.py's save(), shown in place instead of written to figures/."""
        settings = dict(LAYOUT)
        if fig.layout.margin.t is not None:      # a figure that asked for its own room
            settings.pop("margin")
        fig.update_layout(**settings)
        display(Image(fig.to_image(format="png", width=width, height=height, scale=1)))
        return name
    ''')
    explain(
        "Draw every result forty hand-checks could produce, with both intervals.",
        "Was 34 of 40 a lucky example? The figure answers for every count at once.",
        "Runs `figure_wilson_bands()` from `make_figs.py`, verbatim.",
        "The Wald band pinches to a point at both ends and crosses below zero; Wilson's never "
        "does. In the middle they are indistinguishable, which is why the defect survives: "
        "it is invisible on the cases people inspect.")
    nb.source("Module 4/slides/make_figs.py", "figure_wilson_bands")
    explain(
        "Draw the slide's band figure.",
        "It answers for every possible count from 0 to 40 at once.",
        "Calls `figure_wilson_bands()`, defined above, which uses the Lab 1 intervals.",
        "Wald pinches to a point at both ends and dips below zero; Wilson does neither.")
    nb.figure("wilson_bands", "figure_wilson_bands()", slides=["16"],
              treatment="exact: the slide's own code")

    nb.md("### Coverage: the operational definition of a confidence level\n\n"
          "*Slides: \"Coverage: the operational definition of a confidence level\" and "
          "\"Measured coverage against the nominal level\".*")
    explain(
        "Write the definition of coverage.",
        "\"95%\" is a promise about the procedure, and a promise you can simulate you should "
        "simulate [@brown2001].",
        "Builds the average of an indicator over R simulated samples in sympy.",
        "The indicator is 1 when the interval built *from sample r* contains the true p.")
    nb.equation("coverage", '''
    r, R = sp.symbols("r R", positive=True, integer=True)
    p, n = sp.symbols("p n", positive=True)
    lower, upper = sp.IndexedBase(r"\\ell")[r], sp.IndexedBase("u")[r]
    indicator = sp.Function(r"\\mathbf{1}")(sp.And(sp.Le(lower, p), sp.Le(p, upper)))
    formula(sp.Eq(sp.Function("coverage")(p, n), sp.Sum(indicator, (r, 1, R)) / R, evaluate=False))
    ''', slides=["17"])
    explain(
        "Define the coverage experiment as the Lab 1 solution writes it.",
        "It is the third deliverable of Laboratory 1, and the habit it teaches — simulate "
        "the claim — is worth more than either interval.",
        "Copies `coverage()` from `solutions/lab_01.py`: draw `repeats` binomial counts at "
        "the true rate, build the interval from each, count how often it contains the truth.",
        "Near a true rate of 0.02 with forty observations the Wald interval delivers a "
        "fraction of what it promises.")
    nb.source(f"{S}/lab_01.py", "coverage", cite="[@brown2001; @agresti1998]")
    explain(
        "Draw 25 Wald intervals at a true rate of 0.05, and show which ones miss.",
        "The slide's picture makes the coverage definition concrete: each bar is one "
        "simulated sample's interval, and the line is the truth.",
        "Draws 25 binomial counts at p = 0.05, n = 40 with the course seed, builds each Wald "
        "interval (cut at 0 for drawing), and colours the misses red. It also computes the "
        "exact miss rate over all samples.",
        "The misses are the samples with no successes: their interval is [0, 0], which "
        "cannot contain 0.05.")
    nb.figure("twenty_five_intervals", '''
    rng = np.random.default_rng(SEED)
    ks = rng.binomial(40, 0.05, 25)
    lows, highs = [], []
    for k in ks:
        low, high = naive_interval(int(k), 40)
        lows.append(max(low, 0.0)); highs.append(high)
    misses = [not (low <= 0.05 <= high) for low, high in zip(lows, highs)]
    fig = go.Figure(go.Bar(x=list(range(1, 26)), y=[h - l for l, h in zip(lows, highs)],
                           base=lows, marker_color=[RED if miss else NAVY for miss in misses],
                           name="Wald interval of one simulated sample"))
    missed = [i + 1 for i, miss in enumerate(misses) if miss]
    fig.add_scatter(x=missed, y=[0.0] * len(missed), mode="markers", marker=dict(color=RED, size=14,
                    symbol="x"), name="interval [0, 0]: no width, misses the truth")
    fig.add_hline(y=0.05, line=dict(color=NAVY, width=3), annotation_text="truth, p = 0.05")
    fig.update_layout(title=f"25 Wald intervals, n = 40, true p = 0.05: {sum(misses)} of 25 miss",
                      xaxis_title="simulated sample", yaxis_title="interval",
                      yaxis_range=[-0.01, 0.3])
    show(fig, "twenty_five_intervals", height=420)
    agrees("intervals that miss, of 25", sum(misses), 3, 0)
    exact_miss = 1 - sum(stats.binom.pmf(k, 40, 0.05) for k in range(41)
                         if naive_interval(k, 40)[0] <= 0.05 <= naive_interval(k, 40)[1])
    agrees("share of all Wald intervals that miss at p = 0.05, per cent", 100 * exact_miss, 13, 0)
    ''', slides=["17"], treatment="exact: the same seeded simulation")
    explain(
        "Measure both intervals' coverage across true rates from 0.01 to 0.5.",
        "This is the figure that settles the question textbook argument does not: 4,000 "
        "samples of 40 at each of 30 rates, both intervals built from each sample "
        "[@brown2001].",
        "Runs `figure_coverage()` from `make_figs.py`, verbatim. It draws the samples from "
        "one random stream across all rates, so its numbers differ in the third decimal "
        "from Lab 1's `coverage()`, which restarts the stream at each rate.",
        "Worst case: Wald 0.33 at p = 0.01, where 67% of samples hold no success; Wilson "
        "never below 0.932. Wilson is mildly conservative in places — the right direction "
        "for an interval in a compliance report.")
    nb.source("Module 4/slides/make_figs.py", "figure_coverage")
    explain(
        "Run the slide's coverage experiment and check its headline numbers.",
        "The slide reports a worst-case coverage of 0.33 for Wald and 0.932 for Wilson.",
        "Calls `figure_coverage()`, checks its numbers against the slide, and prints Lab 1's `coverage()` at p = 0.01 beside them.",
        "Both headline numbers reproduce.")
    nb.figure("coverage", '''
    measured = figure_coverage()
    agrees("worst Wald coverage", measured["naive_worst"], 0.33, 2)
    agrees("worst Wilson coverage", measured["wilson_worst"], 0.932, 3)
    agrees("share of samples with no success at p = 0.01",
           measured["no_success_share_at_worst"], 0.67, 2)
    print("Lab 1's coverage() at p = 0.01:", coverage(naive_interval, 0.01, 40),
          "(Wald)", coverage(wilson_interval, 0.01, 40), "(Wilson)")
    ''', slides=["18"], treatment="exact: the slide's own code")

    nb.md("### Sample size determination: the price of precision\n\n"
          "*Slide: \"Sample size determination: the price of precision\".*")
    explain(
        "Invert the half-width for the number of labels.",
        "The operator's question is not \"what is the accuracy\" but \"how many predictions "
        "must be checked\".",
        "Solves h = z √(p(1 − p)/n) for n with sympy, at the least favourable p = ½.",
        "The square root in the standard error becomes a square in the cost: halve the "
        "half-width and the labels quadruple.")
    nb.equation("labels_needed", '''
    h, z, n, p = sp.symbols("h z n p", positive=True)
    solved = sp.solve(sp.Eq(h, z * sp.sqrt(p * (1 - p) / n)), n)[0]
    at_half = sp.simplify(solved.subs(p, sp.Rational(1, 2)))
    formula(sp.Eq(n, sp.ceiling(at_half), evaluate=False),
            sp.latex(sp.ceiling(sp.Mul(sp.Rational(1, 4), sp.Pow(sp.Mul(sp.Float(1.96, 3), 1 / h, evaluate=False), 2,
                                                                   evaluate=False), evaluate=False), evaluate=False))
            + r"\\quad \\text{at}\\ p = \\tfrac{1}{2}")
    ''', slides=["19"])
    explain(
        "Define the label count as the Lab 1 solution writes it.",
        "It is Lab 1's fourth deliverable: the formula above, as a function.",
        "Copies `labels_needed()` from `solutions/lab_01.py`.",
        "The next cell prices four precisions with it.")
    nb.source(f"{S}/lab_01.py", "labels_needed", cite="[@brown2001]")
    explain(
        "Price four precisions in hand-checks.",
        "A stakeholder requesting ±0.01 is requesting ten thousand manual adjudications, "
        "whether or not they know it.",
        "Calls `labels_needed()` at four half-widths and draws the bars.",
        "From ±0.10 to ±0.01 costs 99 times more.")
    nb.figure("labels_needed", '''
    widths = [0.01, 0.02, 0.05, 0.10]
    labels = [labels_needed(w) for w in widths]
    fig = go.Figure(go.Bar(x=[f"±{w:.2f}" for w in widths], y=labels, marker_color=NAVY,
                           text=[f"{v:,}" for v in labels], textposition="outside"))
    fig.update_layout(title="Labels needed, 95% interval, by half-width",
                      xaxis_title="half-width", yaxis_title="labels", yaxis_range=[0, 11000])
    show(fig, "labels_needed", height=400)
    for w, stated in zip(widths, (9604, 2401, 385, 97)):
        agrees(f"labels for ±{w}", labels_needed(w), stated, 0)
    agrees("cost ratio, ±0.01 against ±0.10", labels_needed(0.01) / labels_needed(0.10), 99, 0)
    ''', slides=["19"], treatment="exact: closed form")

    nb.md("### The same choice on the archive's own windows\n\n"
          "*Slide: \"The same choice on the archive's own windows\".*")
    explain(
        "Put both intervals on a proportion the bus logs itself: the share of each window "
        "driven by hand.",
        "Is the zero-width failure a textbook curiosity? On this data it is the normal case.",
        "For each of the 80 windows, k = readings flagged manual and n = readings in the "
        "window; builds both intervals and draws the Wilson width per window, marking where "
        "the Wald width is zero.",
        "71 of 80 windows sit at 0 or 1, and Wald reports certainty from a few hundred "
        "readings on every one of them. At 0 of 600 Wilson admits that about 3.8 manual "
        "readings could have been missed.")
    nb.figure("window_widths", '''
    windows = pd.concat([reference, current]).reset_index(drop=True)
    pairs = list(zip(windows["manual_readings"].astype(int), windows["n_readings"].astype(int)))
    wilson_width = [wilson_interval(k, n)[1] - wilson_interval(k, n)[0] for k, n in pairs]
    wald_zero = np.array([k == 0 or k == n for k, n in pairs])
    position = np.arange(1, len(windows) + 1)
    fig = go.Figure()
    fig.add_bar(x=position, y=wilson_width, marker_color=GREEN, name="Wilson interval width")
    fig.add_scatter(x=position[wald_zero], y=np.zeros(wald_zero.sum()), mode="markers",
                    marker=dict(color=RED, size=6), name="Wald interval width = 0")
    fig.add_vline(x=len(reference) + 0.5, line=dict(color=GREY, dash="dash"))
    fig.update_layout(title="Share driven manually, per window: Wilson's width, and where "
                            "Wald has none",
                      xaxis_title="five-minute window, in order of time "
                                  "(22 January, then 23 January)",
                      yaxis_title="interval width", legend=dict(orientation="h", y=1.08))
    show(fig, "window_widths", height=440)
    share = windows["human_driven"]
    agrees("windows at 0 or 1", ((share == 0) | (share == 1)).sum(), 71, 0)
    agrees("reference windows with no manual reading", (reference["human_driven"] == 0).sum(), 39, 0)
    agrees("windows driven manually throughout", (share == 1).sum(), 15, 0)
    agrees("windows with a share strictly between 0 and 1", ((share > 0) & (share < 1)).sum(), 9, 0)
    beside("of those, windows with a share between 0.1 and 0.9",
           int(((share > 0.1) & (share < 0.9)).sum()), 9,
           "the slide calls all nine 'between 0.1 and 0.9'; two sit just outside that range: "
           + ", ".join(f"{v:.3f}" for v in share[(share > 0) & (share < 1)
                                                  & ~((share > 0.1) & (share < 0.9))]))
    agrees("Wilson upper end at 0 of 600", wilson_interval(0, 600)[1], 0.0064, 4)
    agrees("the same, in readings", wilson_interval(0, 600)[1] * 600, 3.8, 1)
    # the windows whose share is between a tenth and nine tenths (make_figs.py's recipe)
    middling = [(k, n) for k, n in pairs if 0.10 <= k / n <= 0.90]
    wald_widths = [naive_interval(k, n)[1] - naive_interval(k, n)[0] for k, n in middling]
    width_gaps = [abs((wilson_interval(k, n)[1] - wilson_interval(k, n)[0]) - w)
                  for (k, n), w in zip(middling, wald_widths)]
    agrees("median Wald width in those windows", np.median(wald_widths), 0.078, 3)
    agrees("median Wald-Wilson width difference there", np.median(width_gaps), 0.0002, 4)
    ''', slides=["20"], treatment="lab data")

    nb.md("""
    ### Exercise: what can you conclude from these numbers?

    *Slides: "Exercise: what can you conclude from these numbers?" and "Reporting
    discipline for small samples".*

    A published study {@servizi2023} asked passengers on the same autonomous buses to
    validate a system's count of their boardings and alightings; the system did not
    exist yet, and an experimenter played it (a Wizard-of-Oz study). 18 people rode and
    were filmed, so the true count is known for each (Sec. IV-A); 14 replied to the
    validation request (Sec. III-A). The paper reports that more than 40% of the
    validations contained at least one error, and that six errors remained after the
    users' correction (Sec. IV-A). The slide's 6 of 14 is inferred from those two
    statements, not printed in the paper: more than 40% of 14 replies is at least 6, and
    six errors in all can sit in at most 6 replies.

    **Your task, as the slide sets it.** 1. Write down what k and n are. 2. Compute
    the interval the 40% carries, with the Wilson score interval. 3. Say what the low
    end and the high end would mean for a real system. 4. Decide what the study can and
    cannot claim. Take five minutes before reading the answer below.
    """)
    explain(
        "Draw the counts the study reports.",
        "The exercise hinges on which number is the denominator.",
        "Three bars: rode the bus, replied, reply had an error.",
        "The red bar is the numerator of the \"40%\"; the denominator is the 14 who replied.")
    nb.figure("study_counts", '''
    fig = go.Figure(go.Bar(x=["Rode the bus", "Replied", "Reply had an error"], y=[18, 14, 6],
                           marker_color=[GREY, NAVY, RED], text=[18, 14, 6],
                           textposition="outside"))
    fig.update_layout(title="People counted in the study (Servizi et al., 2023, Secs. III-A and IV-A)",
                      yaxis_title="people", yaxis_range=[0, 21])
    show(fig, "study_counts", height=380)
    ''', slides=["21"], treatment="exact: the counts the study publishes")
    explain(
        "Answer the exercise with the Lab 1 functions.",
        "From 14 people, 40% means anywhere between about one in five and two in three — and "
        "those are different systems.",
        "k = 6, n = 14: Wilson and Wald intervals; the same share with 30 and with 100 "
        "people; and, for the error *count* (6 errors over 14 validations), the exact "
        "Poisson interval from the chi-squared quantiles [@garwood1936].",
        "At 21% one passenger in five errs and user validation is a workable label source; "
        "at 67% two in three err and it is not. Fourteen people cannot separate the cases, "
        "so report the interval and say no single number is supported.")
    nb.code('''
    agrees("k/n", 6 / 14, 0.43, 2)
    low, high = wilson_interval(6, 14)
    agrees("Wilson 6/14, lower", low, 0.21, 2); agrees("Wilson 6/14, upper", high, 0.67, 2)
    low, high = naive_interval(6, 14)
    agrees("Wald 6/14, lower", low, 0.17, 2); agrees("Wald 6/14, upper", high, 0.69, 2)
    low, high = wilson_interval(13, 30)
    agrees("same share, 30 people, lower", low, 0.27, 2); agrees("upper", high, 0.61, 2)
    low, high = wilson_interval(43, 100)
    agrees("same share, 100 people, lower", low, 0.34, 2); agrees("upper", high, 0.53, 2)
    # the exact (Garwood) interval for a Poisson count of 6, divided by 14 validations
    agrees("errors per validation, Poisson interval, lower",
           stats.chi2.ppf(0.025, 2 * 6) / 2 / 14, 0.16, 2)
    agrees("errors per validation, Poisson interval, upper",
           stats.chi2.ppf(0.975, 2 * 7) / 2 / 14, 0.93, 2)
    ''')

    # --- the laboratory ----------------------------------------------------------------
    nb.md("""
    ## Laboratory 1 — interval estimation and coverage

    *Slide: "Laboratory 1 - interval estimation and coverage".* Four functions,
    twenty-five minutes. The check validates your Wilson interval against an
    independent implementation of the published formula; requires your Wald interval
    to collapse to zero width at k = n; measures your coverage near the lower
    boundary; and checks the label count at a half-width the file never quotes.
    """)
    nb.statement(f"{LABS}/01_how_sure_are_you.py")
    nb.md("""
    ### The solution, step by step

    The four functions were defined above, where the deck introduces each concept:
    `naive_interval`, `wilson_interval`, `coverage` and `labels_needed`. First, the
    lab file's own demonstration, run on the solved functions.
    """)
    explain(
        "Run the stub's own `__main__` block against the solved functions.",
        "It is what a student sees when the file is complete.",
        "The lines below are the stub's demonstration, verbatim.",
        "34 of 40 and 40 of 40 both ways, the coverage near the edge, and one label count.")
    nb.step(f"{LABS}/01_how_sure_are_you.py", 0)
    explain(
        "Show the commonest error the slide names, so it can be recognised.",
        "\"Constructing the interval from the population rate rather than from each "
        "simulated sample, which forces coverage to 1.0.\"",
        "Builds each interval around the *true* rate instead of the sample's count.",
        "Coverage 1.0 whatever the method — the check names this error explicitly.")
    nb.code('''
    def coverage_done_wrong(interval, true_rate, trials, repeats=4000):
        low, high = interval(round(true_rate * trials), trials)   # from the truth, not a sample
        return float(np.mean([low <= true_rate <= high] * repeats))

    print("the error — Wald at p = 0.5:", coverage_done_wrong(naive_interval, 0.5, 40))
    print("the right way — Wald at p = 0.5:", coverage(naive_interval, 0.5, 40))
    ''')
    explain(
        "Start the narrator the solution's demonstration prints with.",
        "The demonstration below reports every step through `say.info`, as it does in the terminal.",
        "Sets the lab number and opens the narrator, as the first lines of the solution's "
        "`__main__` block do.",
        "The numbered steps that follow run unchanged.")
    nb.code('''
    LAB = 1
    say = narrator(LAB)
    say.info("Lab 1 — what an interval promises, what it delivers, and what it costs")
    ''')
    for number, (goal, so_what) in enumerate([
            ("The two worked cases, which are the whole argument in four numbers.",
             "the Wald interval has no width at all on 40 of 40."),
            ("Coverage at six true rates, both intervals.",
             "near the edge Wald delivers about a third of the 0.95 it promises."),
            ("What precision costs, in hand-checks.",
             "halving the half-width multiplies the labels by four."),
            ("Picture one: every result forty hand-checks could produce.",
             "the lab's own version of the slide's band figure."),
            ("Picture two: the same choice on the archive's own windows.",
             "Wald has no width in most windows; Wilson always does."),
    ], start=1):
        explain(f"Step {number} of the solution's demonstration: {goal[0].lower() + goal[1:]}",
                "It is the reference solution's own demonstration; running it here shows the "
                "functions above at work on the lab data, exactly as `make demo` does.",
                f"Runs step {number} of the demonstration block of `solutions/lab_01.py`, verbatim; "
                "`say.info` prints each finding with the seconds elapsed since the lab began.",
                so_what[0].upper() + so_what[1:])
        nb.step(f"{S}/lab_01.py", number)
