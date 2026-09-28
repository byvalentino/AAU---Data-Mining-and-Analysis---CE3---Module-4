"""Part 3 and Laboratory 3: the Wasserstein-1 distance, the two binning policies,
the three cases that separate the statistics, and the four statistics compared."""

S = "Module 4/exercises/solutions"
LABS = "Module 4/exercises/labs"


def build(nb, explain) -> None:
    nb.md("""
    ---
    # Part 3 — Displacement in the units of the variable

    **The question of this part:** by how much has it changed, in units I can act on?
    """)

    nb.md("### A dimensionless statistic is not an actionable one\n\n"
          "*Slide: \"A dimensionless statistic is not an actionable one\".*")
    explain(
        "Measure how far the windows' mean speed moved between the two days, in metres per "
        "second, and draw it as the area between the two cumulative curves.",
        "Report an index of 0.03 to an operations manager and the honest reply is \"is that "
        "a lot?\" The Wasserstein distance answers in the variable's own units: in one "
        "dimension it is the area between the two cumulative distribution functions "
        "[@vallender1974; @peyre2019, Remark 2.30].",
        "Evaluates both days' empirical cumulative curves of window mean speed on a grid and "
        "shades the area between them. The slide's chart is drawn on constructed values "
        "(3 to 10 m/s, 40 windows); this one uses the real 45 and 35 windows.",
        "0.46 m/s: the current day sits to the right, faster. That is a sentence an operator "
        "can act on.")
    nb.figure("cdf_area", '''
    ref_speed = reference["mean_speed"].to_numpy(float)
    cur_speed = current["mean_speed"].to_numpy(float)
    grid = np.linspace(-0.30, 2.00, 461)
    f_ref = np.searchsorted(np.sort(ref_speed), grid, side="right") / len(ref_speed)
    f_cur = np.searchsorted(np.sort(cur_speed), grid, side="right") / len(cur_speed)
    area = float(np.sum(np.abs(f_ref - f_cur)[:-1] * np.diff(grid)))
    fig = go.Figure()
    fig.add_scatter(x=grid, y=np.minimum(f_ref, f_cur), mode="lines", line=dict(width=0),
                    showlegend=False, line_shape="hv")
    fig.add_scatter(x=grid, y=np.maximum(f_ref, f_cur), mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor="rgba(46,139,87,0.35)", line_shape="hv",
                    name=f"area between the curves = {area:.2f} m/s")
    fig.add_scatter(x=grid, y=f_ref, mode="lines", line=dict(color=GREY, width=3), line_shape="hv",
                    name=f"reference day, {len(ref_speed)} windows")
    fig.add_scatter(x=grid, y=f_cur, mode="lines", line=dict(color=NAVY, width=3), line_shape="hv",
                    name=f"current day, {len(cur_speed)} windows")
    fig.update_layout(title="The distance is an area: window mean speed, 22 against 23 January",
                      xaxis_title="mean speed per window, m/s",
                      yaxis_title="share of the day's windows at or below", legend=dict(x=0.55, y=0.2))
    show(fig, "cdf_area", height=460)
    agrees("area between the two cumulative curves, m/s", area, 0.46, 2)
    ''', slides=["43"], treatment="lab data: the slide's constructed curves replaced by the "
                                  "real 45 and 35 windows")

    nb.md("### The Wasserstein-1 distance\n\n*Slide: \"Definition - the Wasserstein-1 distance\".*")
    explain(
        "Write the distance.",
        "The least total effort needed to move one distribution's mass until it matches the "
        "other; on the line it equals the area between the cumulative curves, and the "
        "average gap between the quantile functions [@vallender1974; @kantorovich1942; "
        "@peyre2019, Remarks 2.28 and 2.30].",
        "Builds both integrals in sympy.",
        "No bins and no logarithm: it is a true distance, in the variable's own units.")
    nb.equation("wasserstein", '''
    x, u = sp.symbols("x u", real=True)
    P_, Q_ = sp.symbols("P Q")
    F_P, F_Q = sp.Function("F_P"), sp.Function("F_Q")
    Finv_P, Finv_Q = sp.Function("F_P^{-1}"), sp.Function("F_Q^{-1}")
    formula(sp.Eq(sp.Function("W_1")(P_, Q_), sp.Integral(sp.Abs(F_P(x) - F_Q(x)), (x, -sp.oo, sp.oo)),
                  evaluate=False),
            sp.Eq(sp.Integral(sp.Abs(F_P(x) - F_Q(x)), (x, -sp.oo, sp.oo)),
                  sp.Integral(sp.Abs(Finv_P(u) - Finv_Q(u)), (u, 0, 1)), evaluate=False))
    ''', slides=["44"])
    explain(
        "Define the distance as the Lab 3 solution writes it.",
        "The first deliverable of Laboratory 3.",
        "Copies `wasserstein()` from `solutions/lab_03.py`: every value either sample holds "
        "is a breakpoint; on each strip between two breakpoints, add the gap between the two "
        "cumulative shares times the strip's width. It handles samples of different lengths.",
        "The same integral as the picture, taken along the value axis.")
    nb.source(f"{S}/lab_03.py", "LAB", "DEFAULT_BINS", "wasserstein",
              cite="[@vallender1974; @peyre2019]")
    explain(
        "Draw the slide's rank-pairing picture on the real windows.",
        "For two sorted samples of equal size the distance is the mean absolute difference of "
        "the order statistics: join the slowest to the slowest, the second to the second, and "
        "average the gaps [@peyre2019, Remark 2.28].",
        "Takes six evenly spaced quantiles of each day's window mean speed — six windows a "
        "day, to keep it readable — pairs them by rank and averages the gaps. Then checks the "
        "sorted-pairs formula against `wasserstein()` and SciPy on equal-size samples.",
        "The mean gap of six pairs approximates the 0.46 m/s of the full 45 and 35 windows; on "
        "equal-size samples the two formulas are the same number.")
    nb.figure("rank_pairing", '''
    levels = (np.arange(6) + 0.5) / 6
    six_ref = np.quantile(ref_speed, levels)
    six_cur = np.quantile(cur_speed, levels)
    gaps = np.abs(six_cur - six_ref)
    fig = go.Figure()
    for rank, (r_value, c_value, gap) in enumerate(zip(six_ref, six_cur, gaps)):
        fig.add_scatter(x=[r_value, c_value], y=[0, 1], mode="lines", line=dict(color=GREEN, width=2),
                        showlegend=False)
        height = 0.2 + 0.12 * rank                   # staggered so close pairs do not overprint
        fig.add_annotation(x=r_value + height * (c_value - r_value), y=height, text=f"{gap:.2f}",
                           showarrow=False, font=dict(color=GREEN), bgcolor="white")
    fig.add_scatter(x=six_ref, y=np.zeros(6), mode="markers", marker=dict(color=GREY, size=14),
                    name="reference day, 6 quantiles")
    fig.add_scatter(x=six_cur, y=np.ones(6), mode="markers", marker=dict(color=NAVY, size=14),
                    name="current day, 6 quantiles")
    fig.update_layout(title=f"Pair by rank, average the gaps: {gaps.mean():.2f} m/s "
                            f"(all 45 and 35 windows: {wasserstein(ref_speed, cur_speed):.2f} m/s)",
                      xaxis_title="mean speed per window, m/s", yaxis=dict(visible=False),
                      legend=dict(orientation="h", y=-0.3))
    show(fig, "rank_pairing", height=380)
    rng = np.random.default_rng(SEED)
    a_equal, b_equal = rng.choice(ref_speed, 30, replace=False), rng.choice(cur_speed, 30, replace=False)
    sorted_pairs = float(np.mean(np.abs(np.sort(a_equal) - np.sort(b_equal))))
    print(f"equal sizes, 30 each: sorted pairs {sorted_pairs:.6f}, wasserstein() "
          f"{wasserstein(a_equal, b_equal):.6f}, SciPy {stats.wasserstein_distance(a_equal, b_equal):.6f}")
    ''', slides=["44"], treatment="lab data: the slide's six constructed windows replaced by six "
                                  "quantiles of each real day")

    nb.md("### Wasserstein in the monitor, and why the closed form is cheap\n\n"
          "*Slides: \"Wasserstein in the monitor: what it removes, what it does not\" and "
          "\"Computing the distance: why the closed form is cheap\".*")
    explain(
        "Check the slide's code sketch, and the claim that a novel value costs in proportion "
        "to how far away it is.",
        "On the line the cheapest transport plan matches sorted to sorted, which removes the "
        "optimisation; and because the distance never divides by a reference probability, "
        "an unseen value costs its probability times its distance, never infinity.",
        "Runs the slide's four-line sketch `wasserstein1` against Lab 3's function and SciPy "
        "on the real windows; then adds one window at growing distances above the reference "
        "maximum and measures the distance each time.",
        "The three implementations agree; the extra cost grows linearly with the distance "
        "of the novel value.")
    nb.code('''
    def wasserstein1(a, b):                       # the slide's sketch, as printed
        grid = np.sort(np.concatenate([a, b]))      # all breakpoints
        Fa = np.searchsorted(np.sort(a), grid, 'right') / a.size
        Fb = np.searchsorted(np.sort(b), grid, 'right') / b.size
        return float(np.sum(np.abs(Fa[:-1] - Fb[:-1]) * np.diff(grid)))

    print(f"slide sketch {wasserstein1(ref_speed, cur_speed):.10f}   Lab 3 {wasserstein(ref_speed, cur_speed):.10f}"
          f"   SciPy {stats.wasserstein_distance(ref_speed, cur_speed):.10f}")
    base_distance = wasserstein(ref_payload, cur_payload)
    for beyond in (0, 100, 200, 400):
        with_novel = np.append(cur_payload, ref_payload.max() + beyond)
        print(f"one window {beyond:>3} kg above the reference maximum: distance "
              f"{wasserstein(ref_payload, with_novel):.2f} kg")
    ''')

    nb.md("### Four statistics, one pair of days, two sets of bin edges\n\n"
          "*Slide: \"Four statistics, one pair of days, two sets of bin edges\".*")
    explain(
        "Show the two binning policies on the target, with one novel value added to the "
        "current day.",
        "Entropy and the divergence describe one pair, so their edges may be cut from both "
        "days pooled; the index is a yardstick reused for months, so its edges come from the "
        "reference alone and are never recut [@siddiqi2006; @yurdakul2020].",
        "Adds one window 200 kg heavier than anything the reference held. Top: five "
        "equal-width bins spanning both days — the novel value sits alone in the last bin, so "
        "the reference share there is 0 and the divergence is infinite. Bottom: the "
        "reference's quintile edges, outer bins open — the index stays finite and quiet.",
        "Pick the policy by the job, and never compare an index computed under one policy "
        "with one computed under the other.")
    nb.figure("two_edge_policies", '''
    novel = ref_payload.max() + 200.0
    today = np.append(cur_payload, novel)
    pooled_edges = np.histogram_bin_edges(np.concatenate([ref_payload, today]), bins=5)
    p_pool = np.histogram(today, pooled_edges)[0] / len(today)
    q_pool = np.histogram(ref_payload, pooled_edges)[0] / len(ref_payload)
    assert q_pool[-1] == 0 and kl_divergence(p_pool, q_pool) == float("inf")
    assert np.isfinite(population_stability_index(ref_payload, today))
    quantile_edges = np.quantile(ref_payload, np.linspace(0, 1, 6))
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.22,
                        subplot_titles=(f"equal-width edges over both days: KL = {kl_divergence(p_pool, q_pool)}",
                                        f"reference quantile edges, outer bins open: PSI = "
                                        f"{population_stability_index(ref_payload, today):.3f}, finite"))
    for row, edges in ((1, pooled_edges), (2, quantile_edges[1:-1])):
        fig.add_scatter(x=ref_payload, y=np.full(len(ref_payload), 1.0), mode="markers",
                        marker=dict(color=BLUE, size=9), name="reference day", showlegend=row == 1,
                        row=row, col=1)
        fig.add_scatter(x=cur_payload, y=np.full(len(cur_payload), 0.0), mode="markers",
                        marker=dict(color=ORANGE, size=9), name="current day", showlegend=row == 1,
                        row=row, col=1)
        fig.add_scatter(x=[novel], y=[0.0], mode="markers", marker=dict(color=RED, size=15, symbol="diamond"),
                        name="current day, one novel value", showlegend=row == 1, row=row, col=1)
        fig.update_yaxes(visible=False, range=[-0.6, 1.6], row=row, col=1)
        for edge in edges:                      # drawn after the traces, so the subplot is not empty
            fig.add_vline(x=edge, line=dict(color=GREY, dash="dash"), row=row, col=1)
    fig.update_xaxes(title_text="mean payload per five-minute window, kg", row=2, col=1)
    fig.update_layout(title="Same two days, two edge policies", legend=dict(orientation="h", y=-0.15))
    show(fig, "two_edge_policies", height=520)
    print("pooled equal-width shares, reference:", q_pool.round(3), " current:", p_pool.round(3))
    ''', slides=["47"], treatment="lab data: the slide's schematic drawn with the real windows "
                                  "and one added value")
    explain(
        "Define the function that returns all four statistics on one pair.",
        "The second deliverable of Laboratory 3: the comparison is the examinable part, "
        "because each statistic hides something the others show.",
        "Copies `compare_four()` from `solutions/lab_03.py`. It bins twice on purpose — "
        "equal-width over both samples for cross-entropy and divergence, the reference's "
        "quantiles for the index (reused from Lab 2 through `load_lab(2)`) — and calls "
        "`wasserstein()` unbinned.",
        "On mean speed the divergence is infinite, the index 2.289 and the distance "
        "0.46 m/s: three answers, all correct, to three different questions.")
    nb.source(f"{S}/lab_03.py", "compare_four", cite="[@siddiqi2006; @yurdakul2020]")

    nb.md("### Three cases that separate the statistics\n\n"
          "*Slide: \"Three cases that separate the statistics, all of which occur\".*")
    explain(
        "Compute the three contrasts the slide reports, in closed form where the slide does.",
        "Disjoint supports: divergence infinite, distance finite. Bins relabelled: "
        "divergence blind, distance sees it. A shift against a squeeze of the same "
        "distance: distance blind, divergence sees it.",
        "Case 1: two uniform samples 5 m/s apart (seed 20200122). Case 2: five shares, "
        "relabelled by a seeded permutation; distances with SciPy on the weighted bin "
        "centres. Case 3: a normal law with the reference day's own mean and spread of "
        "window mean speed, moved by Δ or squeezed to a quarter of its spread, Δ chosen so "
        "the two distances are equal; the divergences in closed form, checked by numerical "
        "integration.",
        "These numbers fill the titles of the slide's figure, drawn next by the slide's "
        "own code.")
    nb.code('''
    rng = np.random.default_rng(SEED)
    low = rng.uniform(0.0, 2.0, 400)
    high = low + 5.0
    centres = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    q_shares = np.array([0.40, 0.30, 0.20, 0.07, 0.03])
    p_shares = np.array([0.03, 0.07, 0.20, 0.30, 0.40])
    order = np.argsort(np.random.default_rng(SEED).permutation(5))
    mean, spread = float(ref_speed.mean()), float(ref_speed.std(ddof=1))
    factor = 0.25
    move = (1 - factor) * spread * math.sqrt(2 / math.pi)
    kl_moved = move**2 / (2 * spread**2)                           # KL(N(m+Δ,s²) || N(m,s²))
    kl_squeezed = math.log(1 / factor) + (factor**2 - 1) / 2       # KL(N(m,(fs)²) || N(m,s²))

    def numerical_kl(p_law, q_law):
        """KL(p || q) by numerical integration, on log densities so the tails cannot underflow."""
        from scipy.integrate import quad
        return quad(lambda v: math.exp(p_law.logpdf(v)) * (p_law.logpdf(v) - q_law.logpdf(v)),
                    mean - 12 * spread, mean + 12 * spread, limit=400)[0]
    reference_law = stats.norm(mean, spread)
    print(f"closed form against numerical integration: moved {kl_moved:.4f} / "
          f"{numerical_kl(stats.norm(mean + move, spread), reference_law):.4f}, squeezed "
          f"{kl_squeezed:.4f} / {numerical_kl(stats.norm(mean, factor * spread), reference_law):.4f}")
    contrast = {
        "disjoint": {"wasserstein": round(wasserstein(low, high), 1)},
        "shuffled": {"divergence_before": round(kl_divergence(p_shares, q_shares), 3),
                     "divergence_after": round(kl_divergence(p_shares[order], q_shares[order]), 3),
                     "wasserstein_before": round(stats.wasserstein_distance(centres, centres, q_shares, p_shares), 2),
                     "wasserstein_after": round(stats.wasserstein_distance(centres, centres, q_shares[order],
                                                                           p_shares[order]), 2)},
        "shift_vs_spread": {"reference_mean": mean, "reference_sd": spread, "spread_factor": factor,
                            "shift_ms": move, "wasserstein_shift": round(move, 2),
                            "divergence_shift": round(kl_moved, 3), "divergence_spread": round(kl_squeezed, 3)}}
    agrees("case 1: distance, m/s", contrast["disjoint"]["wasserstein"], 5, 0)
    agrees("case 2: divergence before", contrast["shuffled"]["divergence_before"], 1.293, 3)
    agrees("case 2: divergence after relabelling", contrast["shuffled"]["divergence_after"], 1.293, 3)
    agrees("case 2: distance before, m/s", contrast["shuffled"]["wasserstein_before"], 1.94, 2)
    agrees("case 2: distance after, m/s", contrast["shuffled"]["wasserstein_after"], 0.97, 2)
    agrees("case 3: centre displaced by, m/s", move, 0.119, 3)
    agrees("case 3: divergence, moved", kl_moved, 0.179, 3)
    agrees("case 3: divergence, squeezed", kl_squeezed, 0.918, 3)
    ''')
    explain(
        "Copy the slide's own drawing function for the three cases.",
        "The slide's figure is drawn by `make_figs.py`; the numbers it prints are the ones computed above.",
        "Copies `figure_measures()` from `slides/make_figs.py`, verbatim.",
        "The next cell draws it.")
    nb.source("Module 4/slides/make_figs.py", "figure_measures")
    explain(
        "Draw the three cases.",
        "One figure shows where the divergence and the distance disagree.",
        "Calls `figure_measures()` with the numbers above.",
        "Left: no overlap; middle: bins relabelled; right: moved or squeezed.")
    nb.figure("kl_vs_wasserstein", 'figure_measures({"measure_contrast": {"value": contrast}})',
              slides=["48"], treatment="exact: the slide's own code, fed with the numbers above")

    nb.md("""
    ### The four statistics compared, and a selection rule

    *Slides: "The four statistics compared" and "A selection rule".*

    | Property | Cross-entropy | KL divergence | Symmetrised index | Wasserstein-1 |
    |---|---|---|---|---|
    | What it tells you | cost in nats of scoring today with yesterday's model | how much of that cost is waste | the same waste counted both ways | how far the values moved, in the variable's unit |
    | Symmetric | no | no | yes | yes |
    | Metric | no | no | no (no triangle inequality) | yes |
    | Minimum | current entropy, which moves | 0 | 0 | 0 |
    | Units | nats | nats | nats | the variable's own |
    | Unseen value | infinite | infinite | infinite (finite once floored) | finite, proportional to the gap |
    | Sensitive to ordering | no | no | no | yes |
    | Requires binning | yes, if continuous | yes, if continuous | yes, if continuous | no |
    | Use it for | training a model | drift from one chosen reference | one dashboard threshold for many columns | telling an operator how much moved |

    None of the four decides whether a change matters: that needs a threshold and a
    decision, which is Part 4. Report at least two, so that the discussion moves from
    the instrument to the world.
    """)

    # --- the laboratory -----------------------------------------------------------------------
    nb.md("""
    ## Laboratory 3 — distance, and one function for all four statistics

    *Slide: "Laboratory 3 - distance, and one function for all four statistics".* Two
    functions, twenty-five minutes. Your distance is compared with SciPy on four cases,
    including samples of different lengths; displacing a sample by exactly five must
    give five; two samples with no overlap must give an infinite divergence and a
    finite distance; permuting a distribution's bins must leave the divergence
    unchanged and move the distance. Lab 2's index is reused, not reimplemented.
    """)
    nb.statement(f"{LABS}/03_how_far.py")
    nb.md("""
    ### The solution, step by step

    `wasserstein` and `compare_four` were defined above. First the stub's own
    demonstration, then the checks' two signature tests, then the solution's
    demonstration.
    """)
    explain(
        "Run the stub's own `__main__` block against the solved functions.",
        "It is what a student sees when the lab file is complete.",
        "The lines below are the demonstration at the end of `labs/03_how_far.py`, verbatim; its "
        "`from lab_support import …` resolves to this notebook's cells.",
        "The printed values are the ones the lab's check expects.")
    nb.step(f"{LABS}/03_how_far.py", 0)
    explain(
        "Run the check's units test and its SciPy comparison on unequal lengths.",
        "The whole purpose of the distance is that its answer is in the variable's units.",
        "Shifts a real sample by exactly 5 kg, and compares four pairs of unequal length "
        "with SciPy.",
        "Exactly 5, and agreement with the reference implementation to machine precision.")
    nb.code('''
    print("shifted by five:", round(wasserstein(ref_payload, ref_payload + 5.0), 12))
    for a_sample, b_sample in ((ref_speed, cur_speed), (ref_payload, cur_payload),
                               (ref_speed[:7], cur_speed), (np.array([0.0, 1.0]), np.array([0.5]))):
        assert abs(wasserstein(a_sample, b_sample) - stats.wasserstein_distance(a_sample, b_sample)) < 1e-12
    print("agrees with scipy.stats.wasserstein_distance on four pairs of unequal length")
    ''')
    explain(
        "Start the narrator the solution's demonstration prints with.",
        "The demonstration below reports every step through `say.info`, as it does in the terminal.",
        "Sets the lab number and opens the narrator, as the first lines of the solution's "
        "`__main__` block do.",
        "The numbered steps that follow run unchanged.")
    nb.code('''
    LAB = 3
    say = narrator(LAB)
    say.info("Lab 3 — the measure that answers in metres per second, and the three cases where it "
             "disagrees with the divergence")
    ''')
    for number, (goal, so_what) in enumerate([
            ("All four statistics on the archive, where they answer three different questions.",
             "mean speed moved 0.46 m/s; its divergence is infinite; its index is 2.289."),
            ("The units check: shift a sample by a known amount and the distance is it.",
             "the distance comes out in kilograms."),
            ("Case one — no overlap at all.",
             "the divergence cannot tell a gap of five from a gap of five hundred."),
            ("Case two — the bins relabelled, every share keeping its partner.",
             "the divergence never knew the values were ordered; the distance did."),
            ("Case three — the same law moved, against the same law squeezed. The lab's "
             "version bins the divergence into five equal-width bins, so its numbers are lower "
             "than the slide's closed form above; the lab's figure closes the step.",
             "the distance cannot separate the two; the divergence can."),
    ], start=1):
        explain(f"Step {number} of the solution's demonstration: {goal[0].lower() + goal[1:]}",
                "It is the reference solution's own demonstration; running it here shows the "
                "functions above at work on the lab data, exactly as `make demo` does.",
                f"Runs step {number} of the demonstration block of `solutions/lab_03.py`, verbatim; "
                "`say.info` prints each finding with the seconds elapsed since the lab began.",
                so_what[0].upper() + so_what[1:])
        nb.step(f"{S}/lab_03.py", number)
