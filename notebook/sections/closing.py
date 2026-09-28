"""The closing slides, the practice questions with their answers, and Appendices A to F."""


def build(nb, explain) -> None:
    nb.md("""
    ---
    # After the decision

    *Slides: "Responses to a detected shift, in increasing order of cost", "Five
    failure modes of production drift monitors", "Regulatory context - Regulation (EU)
    2024/1689, the Artificial Intelligence Act", "Mapping today's measurements onto
    the compliance record", "The bias thread: distribution shift" and "Summary of
    capabilities".*

    **Responses, cheapest first.** 1. Record and continue — the right response on
    23 January, where inputs alarmed and the target did not. 2. Investigate the cause
    — here one column away, the manual-driving share. 3. Retrain — expensive, and
    useless on a target that has not moved. 4. Redesign — rare, and the only honest
    response when the cause is structural.

    **Five failure modes.** The reference is yesterday, so a slow decline is invisible;
    only inputs are monitored, so concept drift passes unseen; the threshold is
    borrowed (at ten bins 0.587 of no-change comparisons exceed 0.25); the alarm is a
    p-value, which at production volumes fires permanently; and too many questions are
    asked — twenty features at 5% give one false alarm per run.

    ### What the law asks of a monitor

    Regulation (EU) 2024/1689, the Artificial Intelligence Act [@aiact2024]:

    - **Article 15(3)** — the levels of accuracy and the relevant accuracy metrics of a
      high-risk system shall be declared in the instructions for use. Declared, to
      whoever operates the system — not measured internally and filed.
    - **Article 15(4)** — the system shall be as resilient as possible regarding
      errors, faults or inconsistencies that may occur within it or in the environment
      in which it operates. A drifting input distribution is that environment moving.
    - **Article 12** — automatic recording of events over the system's lifetime. A
      monitor that keeps only its latest reading cannot date a change.
    - **Article 14** — human oversight: a person must be able to interpret the output
      correctly. A call of act, watch or no material change, with a measured reason, is
      that output.
    - **Article 72** — a documented post-market monitoring system, which is what
      Module 5 builds.

    Two sentences worth saying plainly. The positive control is how you evidence that
    a monitor works; and a declared metric without a measured floor is not a
    declaration. The deck adds that the Digital Omnibus deferred the Annex III
    high-risk obligations to 2 December 2027 — a deadline for evidence, not for
    intentions. *This is the text of the Regulation, not legal advice.*
    """)
    explain(
        "Assemble the compliance record from this notebook's own measurements.",
        "The four quantities measured today, written down where someone outside the team can "
        "read them, are the Article 15 declaration. The work is the writing, not the "
        "measuring.",
        "Builds the slide's mapping table from the variables computed above.",
        "Every value in the right-hand column was produced by a cell in this notebook.")
    nb.code('''
    record = pd.DataFrame([
        ("Declared accuracy metric", "15(3)", "purchased rate with a Wilson interval",
         f"±0.05 costs {labels_needed(0.05)} labels"),
        ("Metric is interpretable", "15(3)", "measured noise floor of the statistic",
         f"{results[TARGET]['noise_floor']:.3f} at five bins"),
        ("Evidence the monitor functions", "15(4)", "positive control at a stated size",
         f"fires at 1.5 s.d., index {control['population_stability_index']:.3f}"),
        ("Stated sensitivity of the claim", "15(4)", "detection limit from a sweep",
         f"{control['detection_limit_sd']:.1f} s.d. = {control['detection_limit_in_target_units']:.1f} kg per window"),
        ("Events recorded over the lifetime", "12", "retained index series with dates",
         "change dated to 23 January"),
        ("Output a person can act on", "14", "decision plus measured justification", f"{call}; do not retrain"),
    ], columns=["Regulatory requirement", "Article", "Artefact produced today", "Value on this archive"])
    display(record)
    ''')
    nb.md("""
    **The bias thread.** A model that was fair on the training population can become
    unfair when the population moves; shift is a risk to measure continuously, not a
    defect found once. It was measured three ways today — how improbable, how far, how
    certain — and on this archive the shift is in the inputs while the model's world,
    the target, is intact.

    **What you can now do.** Construct an interval around a purchased rate that keeps
    its coverage at the boundary, and price a precision in adjudications; quantify how
    improbable the current period is and say when the answer is infinite; quantify
    displacement in units an operator can act on; report an effect size instead of a
    p-value; measure your own noise floor and turn it into a threshold with a stated
    false-alarm rate; sweep an injected shift to find what your instrument cannot see;
    and write a decision of no material change and defend it. Required reading:
    {@rabanser2019}.
    """)

    # --- practice ----------------------------------------------------------------------------
    nb.md("""
    ---
    ## Practice

    1. **Does the verdict survive a different grain?** Recompute the target's shift at
       one minute and at fifteen minutes. Does the target ever become material? What
       does that tell you about quoting a shift without its grain?
    2. **Where is the noise floor for the Wasserstein distance?** Resample the reference
       against itself and find the distribution of distances. Is the target's 20.9
       kilograms inside it?
    3. **How small a shift would the control still catch?** Repeat the positive control
       at 1.0, 0.5 and 0.25 standard deviations. At what size does the detector stop
       firing, and what does that say about what your null result established?

    Answers below — try first.
    """)
    explain(
        "Work the practice questions.",
        "Each answer is a measurement you can make with the functions defined above.",
        "An empty cell for your own code.",
        "Compare with the answers below.")
    nb.code("# Your workings here.")
    nb.md("### Answers")
    explain(
        "Answer the three practice questions with the lab functions above.",
        "Each answer is a measurement, so it is computed rather than asserted.",
        "1: regroups the readings at 1, 5 and 15 minutes, keeping windows with at least half "
        "their readings at two a second. 2: 500 resamples of the reference against itself, "
        "the distance each time. 3: the unchanged verdict on the target shifted by four "
        "sizes, with the index threshold derived once.",
        "The verdict is stable across grains though the numbers are not; the target's "
        "distance sits inside its own noise; and the control keeps firing down to the "
        "detection limit the sweep reported, not below it.")
    nb.code('''
    # 1. The verdict across grains -- the numbers move, which is why the grain is printed.
    one = bus.assign(_t=pd.to_datetime(bus["utc_time"], utc=True))
    for window, floor_reads in (("1min", 60), ("5min", 300), ("15min", 900)):
        grouped = one.assign(w=one["_t"].dt.floor(window)).groupby("w").agg(
            mean_payload=("payload", "mean"), readings=("speed", "size")).reset_index()
        grouped = grouped[grouped["readings"] >= floor_reads]
        grouped["day"] = grouped["w"].dt.date.astype(str)
        a = grouped.loc[grouped["day"] == REFERENCE_DAY, "mean_payload"]
        b = grouped.loc[grouped["day"] == CURRENT_DAY, "mean_payload"]
        print(f"{window:>6}: {len(a):3} vs {len(b):3} windows, target shift {(b.mean() - a.mean()) / a.std(ddof=1):+.2f} s.d.")

    # 2. The distance's own null: the reference against resamples of itself.
    rng = np.random.default_rng(SEED)
    distances = [wasserstein(ref_payload, rng.choice(ref_payload, size=len(current), replace=True))
                 for _ in range(500)]
    observed = wasserstein(ref_payload, cur_payload)
    print(f"\\nnull distances: median {np.median(distances):.1f} kg, 95th percentile "
          f"{np.quantile(distances, 0.95):.1f} kg; the target's distance {observed:.1f} kg is "
          f"{'inside the noise' if observed < np.quantile(distances, 0.95) else 'outside it'}")

    # 3. The control at smaller sizes, through the unchanged verdict.
    held = {TARGET: index_threshold(ref_payload, cur_payload, bins=5)}
    print()
    for size in (1.5, 1.0, 0.5, 0.25):
        moved = current.copy()
        moved[TARGET] = moved[TARGET] + size * control["target_reference_sd"]
        row = verdict(reference, moved, [TARGET], held)[TARGET]
        print(f"injected {size:>4} s.d. -> index {row['population_stability_index']:7.3f}   "
              f"material (2.0 rule) {row['material_fixed']}")
    print(f"the sustained limit, off the sweep: {control['detection_limit_sd']} s.d. = "
          f"{control['detection_limit_in_target_units']:.1f} kg per window")
    ''')

    # --- appendices ----------------------------------------------------------------------------------
    nb.md("---\n# Appendices\n\nThe deck's six appendices, each worked in code.")
    nb.md("### Appendix A — non-negativity of the divergence, in three lines")
    explain(
        "Show the three-line proof, and test it on many random pairs.",
        "The information inequality follows from Jensen's inequality applied to the convex "
        "function −log [@cover2006, Theorem 2.6.3].",
        "Displays the chain, then draws 10,000 random pairs of five-bin distributions "
        "(Dirichlet, seed 20200122) and records the smallest divergence found.",
        "Never negative; zero only when the two distributions coincide.")
    nb.equation("appendix_a", '''
    P_, Q_ = sp.symbols("P Q", positive=True)
    i, B = sp.symbols("i B", positive=True, integer=True)
    E_P, Q_of = sp.Function(r"\\mathbb{E}_P"), sp.Function("Q")
    formula(sp.Eq(sp.Symbol(r"D_{\\mathrm{KL}}(P \\| Q)"), E_P(-sp.log(Q_ / P_)), evaluate=False),
            sp.Ge(E_P(-sp.log(Q_ / P_)), -sp.log(E_P(Q_ / P_)), evaluate=False))
    formula(sp.Eq(-sp.log(E_P(Q_ / P_)), -sp.log(sp.Sum(Q_of(i), (i, 1, B))), evaluate=False),
            sp.Eq(sp.Sum(Q_of(i), (i, 1, B)), 1, evaluate=False),
            sp.Eq(-sp.log(sp.Integer(1), evaluate=False), 0, evaluate=False))
    rng = np.random.default_rng(SEED)
    smallest = min(kl_divergence(rng.dirichlet(np.ones(5)), rng.dirichlet(np.ones(5))) for _ in range(10000))
    print(f"smallest divergence over 10,000 random pairs: {smallest:.5f};  D(P || P) = "
          f"{kl_divergence([0.2, 0.3, 0.5], [0.2, 0.3, 0.5])}")
    ''', slides=["91"])

    nb.md("### Appendix B — five instruments on one target, and five blind spots")
    explain(
        "Put every instrument's reading on the target side by side.",
        "The agreement of instruments with different blind spots is the finding; no single "
        "number is.",
        "Tabulates the shift, the index with its floor and threshold, the distance and the "
        "classifier test on mean payload.",
        "Four instruments, four blind spots, one answer: the target did not move.")
    nb.code('''
    agrees("standardised shift", results[TARGET]["shift_in_reference_sd"], -0.03, 2)
    agrees("index", results[TARGET]["population_stability_index"], 0.081, 3)
    agrees("Wasserstein distance, kg", results[TARGET]["wasserstein"], 20.85, 2)
    agrees("classifier accuracy on the target alone", alone["accuracy"], 0.417, 3)
    ''')

    nb.md("### Appendix C — why the symmetrised index is not a distance")
    explain(
        "Find three distributions for which the index breaks the triangle inequality.",
        "Symmetry is one of three metric axioms; the index satisfies it and fails the triangle "
        "inequality, so averaging indices or thresholding their sum is arithmetic without a "
        "referent.",
        "Computes J between P = [0.9, 0.1], Q = [0.5, 0.5] and R = [0.1, 0.9] with no floor, "
        "and the Wasserstein distance between the same three, placed on the points 0 and 1.",
        "J(P, R) = 3.52 exceeds J(P, Q) + J(Q, R) = 1.76; the distance obeys the triangle "
        "inequality.")
    nb.code('''
    def J(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return float(np.sum((a - b) * np.log(a / b)))

    P3, Q3, R3 = [0.9, 0.1], [0.5, 0.5], [0.1, 0.9]
    agrees("J(P, R)", J(P3, R3), 3.52, 2, source="slides/measured_v3.json")
    agrees("J(P, Q)", J(P3, Q3), 0.88, 2, source="slides/measured_v3.json")
    agrees("J(Q, R)", J(Q3, R3), 0.88, 2, source="slides/measured_v3.json")
    print("triangle inequality holds for J:", J(P3, R3) <= J(P3, Q3) + J(Q3, R3))
    w = lambda a, b: stats.wasserstein_distance([0, 1], [0, 1], a, b)
    print(f"Wasserstein on the same three: {w(P3, R3):.2f} <= {w(P3, Q3):.2f} + {w(Q3, R3):.2f}:",
          w(P3, R3) <= w(P3, Q3) + w(Q3, R3) + 1e-12)
    ''')

    nb.md("### Appendix D — degrees of freedom and the Welch-Satterthwaite formula")
    explain(
        "Write the formula and evaluate it on mean speed.",
        "Welch's test keeps the two variances separate, so no whole number of degrees of "
        "freedom is exact; the Welch-Satterthwaite formula chooses the one that makes the "
        "combined variance behave approximately like a chi-squared variable "
        "[@satterthwaite1946; @welch1947].",
        "Builds the formula in sympy, evaluates it on the 45 and 35 windows, checks it "
        "against SciPy's, and checks the bounds the slide states.",
        "37.4 on the slide is an illustration; on this pair of days the value is 39.1, "
        "between 34 and 78 as it must be.")
    nb.equation("appendix_d", '''
    s1, s2, n1, n2 = sp.symbols("s_1 s_2 n_1 n_2", positive=True)
    ws = (s1**2 / n1 + s2**2 / n2)**2 / ((s1**2 / n1)**2 / (n1 - 1) + (s2**2 / n2)**2 / (n2 - 1))
    formula(sp.Eq(sp.Symbol("df"), ws, evaluate=False))
    value = float(ws.subs({s1: s_ref, s2: s_cur, n1: len(before_speed), n2: len(after_speed)}))
    agrees("Welch-Satterthwaite df on mean speed", value, 39.1, 1, source="slides/measured_v3.json")
    assert abs(value - welch.df) < 1e-9
    print(f"bounds: min(n1 - 1, n2 - 1) = {min(len(before_speed), len(after_speed)) - 1} <= {value:.1f} "
          f"<= n1 + n2 - 2 = {len(before_speed) + len(after_speed) - 2}")
    ''', slides=["94"])

    nb.md("### Appendix E — the chi-squared distribution and the mean squared error")
    explain(
        "Show by simulation the three facts the appendix states.",
        "A sum of k squared standard normals is chi-squared with k degrees of freedom, mean k "
        "and variance 2k; dividing the sum of squared errors by its degrees of freedom makes "
        "the mean squared error unbiased [@casella2002, ch. 5].",
        "Draws 100,000 sums of four squared normals; then 100,000 samples of five normal "
        "errors with σ² = 1, and compares SSE/(n − 1) with SSE/n.",
        "Mean 4 and variance 8; the (n − 1) divisor averages σ², the n divisor falls short by "
        "a fifth.")
    nb.equation("appendix_e", '''
    i, k = sp.symbols("i k", positive=True, integer=True)
    Z = sp.IndexedBase("Z")
    chi2 = sp.Function(r"\\chi^2")
    SSE, sigma, df, MSE = sp.symbols(r"\\mathrm{SSE} \\sigma \\mathrm{df} \\mathrm{MSE}", positive=True)
    formula(sp.Sum(Z[i]**2, (i, 1, k)), r"\\sim", chi2(k))
    formula(SSE / sigma**2, r"\\sim", chi2(df), sp.Eq(MSE, SSE / df, evaluate=False),
            sp.Eq(sp.Function(r"\\mathbb{E}")(MSE), sigma**2, evaluate=False))
    rng = np.random.default_rng(SEED)
    sums = (rng.standard_normal((100000, 4))**2).sum(axis=1)
    print(f"sum of 4 squared normals: mean {sums.mean():.3f} (k = 4), variance {sums.var():.3f} (2k = 8)")
    samples = rng.standard_normal((100000, 5))
    sse = ((samples - samples.mean(axis=1, keepdims=True))**2).sum(axis=1)
    print(f"SSE/(n - 1) averages {np.mean(sse / 4):.3f}; SSE/n averages {np.mean(sse / 5):.3f}; sigma^2 = 1")
    ''', slides=["95"])

    nb.md("### Appendix F — the shift bound: two options, and what the labs show")
    explain(
        "Set the fixed and the measured shift bound side by side on all five features.",
        "The deck's decision per feature uses the fixed 2.0; Laboratory 4 reports both. The "
        "measured bound is the 99th percentile of the absolute shift over 1,000 resamples of "
        "the reference against itself — the same procedure as the index threshold, so its "
        "false-alarm rate is 1% per feature [@glass1976; @efron1979].",
        "Tabulates each feature's shift, its measured bound and both decisions, and the "
        "control's shift against the target's measured bound.",
        "Two options, the same call on the target: no material change. The measured bound "
        "adds payload dispersion and the manual-driving share, whose index cannot be computed "
        "and which the fixed 2.0 does not flag.")
    nb.code('''
    both = pd.DataFrame({feature: {"shift": round(row["shift_in_reference_sd"], 2),
                                   "measured bound": round(row["shift_threshold"], 3),
                                   "material, fixed 2.0": row["material_fixed"],
                                   "material, measured bound": row["material"]}
                         for feature, row in results.items()}).T
    display(both)
    bounds = [row["shift_threshold"] for row in results.values()]
    agrees("smallest measured bound", min(bounds), 0.39, 2)
    agrees("largest measured bound", max(bounds), 0.46, 2)
    agrees("measured bound on the target", results[TARGET]["shift_threshold"], 0.43, 2)
    agrees("material under the fixed 2.0", sum(r["material_fixed"] for r in results.values()), 2, 0)
    agrees("material under the measured bound", sum(r["material"] for r in results.values()), 4, 0)
    agrees("control's shift, against the measured bound on the target", control["shift_in_reference_sd"], 1.47, 2)
    ''')
