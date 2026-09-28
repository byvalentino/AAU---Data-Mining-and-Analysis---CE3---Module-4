"""Part 2 and Laboratory 2: Jensen's inequality, surprisal, entropy, cross-entropy,
the Kullback-Leibler divergence, the symmetrised index, its noise floor and the
threshold derived from it."""

S = "Module 4/exercises/solutions"
LABS = "Module 4/exercises/labs"


def build(nb, explain) -> None:
    nb.md("""
    ---
    # Part 2 — Measures of distribution shift

    **The question of this part:** has the input data changed since the model was
    trained, and how would I measure that?

    The part opens with an inequality, because the same inequality proves that the
    divergence at its centre is never negative.
    """)

    # --- slide 25: the journey --------------------------------------------------------
    nb.md("### Averaging before a nonlinearity is not averaging after it\n\n"
          "*Slide: \"Motivation: averaging before a nonlinearity is not averaging after it\".*")
    explain(
        "Work the slide's journey: 30 km at 20 km/h, then 30 km at 60 km/h.",
        "Average the speeds first and you get 40 km/h; compute the journey and you get "
        "30 km/h. Journey time is a convex function of speed, t = 30/v, and averaging the "
        "input before a convex function is not averaging after it.",
        "Computes both answers, then draws t = 30/v with the chord between the two legs, "
        "the average of the two times and the time at the average speed.",
        "The first answer is wrong by a third. The gap between the chord and the curve is "
        "the Jensen gap, and it is the reason the divergence below cannot be negative.")
    nb.figure("journey", '''
    LEG_KM, FIRST_KMH, SECOND_KMH = 30.0, 20.0, 60.0
    mean_of_speeds = (FIRST_KMH + SECOND_KMH) / 2
    hours = LEG_KM / FIRST_KMH + LEG_KM / SECOND_KMH
    journey_speed = 2 * LEG_KM / hours
    agrees("average of the two speeds, km/h", mean_of_speeds, 40, 0)
    agrees("speed of the journey, km/h", journey_speed, 30, 0)

    v = np.linspace(15, 70, 300)
    fig = go.Figure()
    fig.add_scatter(x=v, y=LEG_KM / v, mode="lines", line=dict(color=NAVY, width=3),
                    name="journey time t = 30 / v")
    fig.add_scatter(x=[FIRST_KMH, SECOND_KMH], y=[LEG_KM / FIRST_KMH, LEG_KM / SECOND_KMH],
                    mode="lines+markers", line=dict(color=GREY, dash="dash"),
                    name="straight line between the two legs")
    fig.add_scatter(x=[mean_of_speeds], y=[hours / 2], mode="markers",
                    marker=dict(color=RED, size=13),
                    name=f"average of the two times: {hours / 2:.2f} h")
    fig.add_scatter(x=[mean_of_speeds], y=[LEG_KM / mean_of_speeds], mode="markers",
                    marker=dict(color=GREEN, size=13),
                    name=f"time at the average speed: {LEG_KM / mean_of_speeds:.2f} h")
    fig.update_layout(title="Averaging speeds before the convex function understates the time",
                      xaxis_title="speed, km/h", yaxis_title="hours per 30 km leg")
    show(fig, "journey", height=440)
    ''', slides=["25"], treatment="exact: the slide's worked example")

    nb.md("### Jensen's inequality\n\n*Slide: \"Definition - Jensen's inequality\".*")
    explain(
        "State Jensen's inequality and check it on the slide's example.",
        "For a convex function the average of the function is at least the function of the "
        "average [@jensen1906; @wasserman2004, Theorem 4.9]; the two sides are equal when the "
        "function is a straight line over the values taken, or the variable never varies.",
        "Writes the inequality, then takes f(x) = x² and X equal to 0.5 or 3.5 with "
        "probability ½ each, and computes both sides symbolically.",
        "E[f(X)] = 6.25 against f(E[X]) = 4: the gap is (3.5 − 0.5)²/4 = 2.25, the variance "
        "of X — for f(x) = x² the Jensen gap is exactly the variance.")
    nb.equation("jensen", '''
    x, a, b, X = sp.symbols("x a b X", real=True)
    f, expectation = sp.Function("f"), sp.Function(r"\\mathbb{E}")
    formula(sp.Le(f(expectation(X)), expectation(f(X))), r"\\text{for convex } f")
    square = x**2
    two_point = {a: sp.Rational(1, 2), b: sp.Rational(7, 2)}
    mean_of_f = ((square.subs(x, a) + square.subs(x, b)) / 2).subs(two_point)
    f_of_mean = square.subs(x, (a + b) / 2).subs(two_point)
    print("E[f(X)] =", mean_of_f, "=", float(mean_of_f), "   f(E[X]) =", f_of_mean, "=", float(f_of_mean))
    print("the gap, symbolically:",
          sp.factor((square.subs(x, a) + square.subs(x, b)) / 2 - square.subs(x, (a + b) / 2)))
    ''', slides=["26"])
    explain(
        "Draw the slide's picture of the inequality.",
        "The chord between two points of a convex curve lies above the curve: that is the inequality, drawn.",
        "Plots f(x) = x², the chord from 0.5 to 3.5, and the two averages computed above.",
        "E[f(X)] = 6.25 sits above f(E[X]) = 4.")
    nb.figure("jensen_chord", '''
    grid = np.linspace(0, 4, 200)
    fig = go.Figure()
    fig.add_scatter(x=grid, y=grid**2, mode="lines", line=dict(color=NAVY, width=3),
                    name="convex f(x) = x²")
    fig.add_scatter(x=[0.5, 3.5], y=[0.25, 12.25], mode="lines+markers",
                    line=dict(color=GREY, dash="dash"), name="chord between x = 0.5 and x = 3.5")
    fig.add_scatter(x=[2], y=[6.25], mode="markers", marker=dict(color=RED, size=13),
                    name="E[f(X)]: average of the function, 6.25")
    fig.add_scatter(x=[2], y=[4], mode="markers", marker=dict(color=GREEN, size=13),
                    name="f(E[X]): function of the average, 4")
    fig.update_layout(title="The chord lies above a convex curve: E[f(X)] ≥ f(E[X])",
                      xaxis_title="x", yaxis_title="f(x)")
    show(fig, "jensen_chord", height=440)
    ''', slides=["26"], treatment="exact: the slide's worked example")

    # --- slide 27: capacity ----------------------------------------------------------------
    nb.md("### Cost-sensitive capacity under uncertain demand\n\n"
          "*Slide: \"Application: cost-sensitive capacity under uncertain demand\".*")
    explain(
        "Price one departure with 12 seats when demand is Poisson with mean 10.",
        "Plan at the mean and nobody is left standing, so the cost looks like 0. Average "
        "the cost over the days instead: the shortfall cost is convex in demand, so by "
        "Jensen the cost of the average day understates the average cost of the days.",
        "Computes, from the Poisson law, the probability that more than 12 turn up, the "
        "expected number left standing and its cost at 4 units each; then the expected cost "
        "of every capacity from 8 to 16 seats, with 1 unit per empty seat.",
        "2.12 units per departure is the inequality in money. The cheapest capacity is the "
        "quantile set by the cost ratio 4/(4 + 1) = 0.8: 13 seats, not 12 "
        "[@arrow1951, § 3].")
    nb.equation("capacity_quantile", '''
    c_u, c_o = sp.symbols("c_u c_o", positive=True)
    F_inv = sp.Function("F^{-1}")
    level = c_u / (c_u + c_o)
    formula(sp.Eq(sp.Symbol("q^{*}"), F_inv(level), evaluate=False),
            sp.Eq(F_inv(level.subs({c_u: 4, c_o: 1})), int(stats.poisson(10).ppf(0.8)), evaluate=False),
            r"\\text{seats}")
    ''', slides=["27"])
    explain(
        "Compute the capacity example's numbers and draw the demand.",
        "The slide's 20.8%, 0.53, 2.12 and 13 seats all follow from the Poisson law.",
        "Evaluates them with `scipy.stats.poisson`, prices every capacity from 8 to 16 seats, and draws the demand with the overflow in red.",
        "The cheapest capacity is 13 seats, the 80th percentile.")
    nb.figure("departure_demand", '''
    demand = stats.poisson(10)
    counts = np.arange(0, 25)
    left_standing = sum(max(d - 12, 0) * demand.pmf(d) for d in range(200))
    agrees("share of departures with more than 12 passengers, per cent", 100 * demand.sf(12), 20.8, 1)
    agrees("passengers left standing, on average", left_standing, 0.53, 2)
    agrees("expected cost of those left standing, units", 4 * left_standing, 2.12, 2)
    agrees("80th percentile of demand, seats", demand.ppf(0.8), 13, 0)
    cost = {seats: sum((4 * max(d - seats, 0) + max(seats - d, 0)) * demand.pmf(d)
                       for d in range(200)) for seats in range(8, 17)}
    print("expected cost per departure by capacity:",
          {seats: round(float(value), 2) for seats, value in cost.items()})
    agrees("cheapest capacity, seats", min(cost, key=cost.get), 13, 0)

    fig = go.Figure(go.Bar(x=counts, y=100 * demand.pmf(counts),
                           marker_color=[RED if c > 12 else NAVY for c in counts]))
    fig.add_vline(x=12.5, line=dict(color=GREY, dash="dash"), annotation_text="12 seats")
    fig.update_layout(title=f"Passengers per departure, mean 10: more than 12 turn up on "
                            f"{100 * demand.sf(12):.1f}% of departures",
                      xaxis_title="passengers turning up", yaxis_title="per cent of departures")
    show(fig, "departure_demand", height=420)
    ''', slides=["27"], treatment="exact: the Poisson law")

    # --- slide 28: surprisal ---------------------------------------------------------------
    nb.md("### Surprisal, and the two ways of averaging it\n\n"
          "*Slide: \"Surprisal, and the two ways of averaging it\".*")
    explain(
        "Draw the surprisal −ln p of an outcome of probability p.",
        "An outcome with probability p carries −log p of surprise; the logarithm makes the "
        "surprises of independent events add [@shannon1948]. With natural logarithms the "
        "unit is the nat; 1 nat = 1.44 bits.",
        "Plots −ln p and marks a certainty, a coin flip and a rare event.",
        "A coin flip is 0.69 nats; a one-in-twenty event 3.0 nats. Every number in this part "
        "is in nats.")
    nb.figure("surprisal", '''
    p = np.linspace(0.01, 1, 200)
    fig = go.Figure()
    fig.add_scatter(x=p, y=-np.log(p), mode="lines", line=dict(color=NAVY, width=3),
                    name="surprisal −ln p")
    for value, label, colour in ((1.0, "certain, 0 nats", GREEN), (0.5, "coin flip, 0.69 nats", ORANGE),
                                 (0.05, "rare, 3.0 nats", RED)):
        fig.add_scatter(x=[value], y=[-np.log(value)], mode="markers+text",
                        text=[f"p = {value}: {label}"], textposition="top right",
                        marker=dict(color=colour, size=12), showlegend=False)
    fig.update_layout(title="Surprisal of an outcome of probability p", xaxis_title="probability p",
                      yaxis_title="nats", showlegend=False)
    show(fig, "surprisal", height=420)
    agrees("surprisal of a coin flip, nats", -np.log(0.5), 0.69, 2)
    agrees("surprisal at p = 0.05, nats", -np.log(0.05), 3.0, 1)
    agrees("bits per nat", 1 / np.log(2), 1.44, 2)
    ''', slides=["28"], treatment="exact: closed form")

    # --- slide 29: entropy and cross-entropy -------------------------------------------------
    nb.md("### Entropy and cross-entropy\n\n*Slide: \"Definition - entropy and cross-entropy\".*")
    explain(
        "Write the two averages of surprisal.",
        "Both average over today's data P. Entropy puts log P inside: today judged by today. "
        "Cross-entropy puts log Q inside, where Q is the training-day reference: today judged "
        "by yesterday [@shannon1948; @murphy2022, § 6.1.2].",
        "Builds both sums in sympy over B bins.",
        "They differ only in which distribution is inside the logarithm.")
    nb.equation("entropy_cross_entropy", '''
    i, B = sp.symbols("i B", integer=True, positive=True)
    P, Q = sp.Function("P"), sp.Function("Q")
    formula(sp.Eq(sp.Symbol("H(P)"), -sp.Sum(P(i) * sp.log(P(i)), (i, 1, B)), evaluate=False),
            sp.Eq(sp.Symbol("H(P, Q)"), -sp.Sum(P(i) * sp.log(Q(i)), (i, 1, B)), evaluate=False))
    ''', slides=["29"])
    explain(
        "Define entropy and cross-entropy as the Lab 2 solution writes them.",
        "Two of the five deliverables of Laboratory 2.",
        "Copies the module's bin-count choice and the two functions from "
        "`solutions/lab_02.py`. Bins where P is zero contribute nothing, because "
        "p log p → 0 as p → 0; a zero in Q where P has mass makes the cross-entropy infinite.",
        "The same convention runs through the divergence and the index below.")
    nb.source(f"{S}/lab_02.py", "LAB", "DEFAULT_BINS", "entropy", "cross_entropy",
              cite="[@shannon1948; @murphy2022]")
    explain(
        "Measure both on the bus: the speed distribution of 23 January (P) judged by that of "
        "22 January (Q).",
        "The slide's chart uses constructed shares over speed bins in km/h. Here the shares "
        "are real: every speed reading of each day, in km/h, in five equal-width bins "
        "spanning both days — the binning the deck prescribes for entropy and divergence "
        "when two samples are compared once.",
        "Computes P and Q, then −ln P and −ln Q per bin, and both averages with the Lab 2 "
        "functions. The slide's own constructed example is computed beside it.",
        "Entropy averages the navy line, cross-entropy the red dashed line, both weighted by "
        "P; the gap between them is the divergence.")
    nb.code('''
    day_of = pd.to_datetime(bus["utc_time"], utc=True).dt.date.astype(str)
    speed_kmh = {day: bus.loc[day_of == day, "speed"].to_numpy(float) * 3.6
                 for day in (REFERENCE_DAY, CURRENT_DAY)}
    speed_edges = np.histogram_bin_edges(np.concatenate(list(speed_kmh.values())), bins=5)
    P_speed = np.histogram(speed_kmh[CURRENT_DAY], speed_edges)[0] / len(speed_kmh[CURRENT_DAY])
    Q_speed = np.histogram(speed_kmh[REFERENCE_DAY], speed_edges)[0] / len(speed_kmh[REFERENCE_DAY])
    speed_bins = [f"{low:.0f} to {high:.0f}" for low, high in zip(speed_edges[:-1], speed_edges[1:])]
    print(pd.DataFrame({"speed bin, km/h": speed_bins, "P: 23 January": P_speed.round(3),
                        "Q: 22 January": Q_speed.round(3)}).to_string(index=False))
    print(f"H(P) = {entropy(P_speed):.3f} nats   H(P, Q) = {cross_entropy(P_speed, Q_speed):.3f} nats")
    print("readings per day:", {day: len(values) for day, values in speed_kmh.items()},
          "— negative speeds are reversing, and they are real")
    # the slide's constructed example, computed with the same functions
    P_slide, Q_slide = np.array([.1, .2, .4, .2, .1]), np.array([.3, .3, .2, .1, .1])
    agrees("slide's constructed example: H(P)", entropy(P_slide), 1.47, 2)
    agrees("slide's constructed example: H(P, Q)", cross_entropy(P_slide, Q_slide), 1.70, 2)
    ''')
    explain(
        "Draw the two days' shares and the two surprisal curves.",
        "The slide's chart shows which surprisal each average weights by P.",
        "Bars for P and Q per speed bin; lines for −ln P and −ln Q on a second axis.",
        "Where Q is small and P is not, −ln Q is large: that is where the cross-entropy exceeds the entropy.")
    nb.figure("entropy_cross_entropy", '''
    fig = go.Figure()
    fig.add_bar(x=speed_bins, y=P_speed, name="P: 23 January (weights)", marker_color=GREEN)
    fig.add_bar(x=speed_bins, y=Q_speed, name="Q: 22 January", marker_color=GREY)
    fig.add_scatter(x=speed_bins, y=-np.log(P_speed), mode="lines+markers", yaxis="y2",
                    line=dict(color=NAVY, width=3), name="−ln P: entropy averages this")
    fig.add_scatter(x=speed_bins, y=-np.log(Q_speed), mode="lines+markers", yaxis="y2",
                    line=dict(color=RED, width=3, dash="dash"),
                    name="−ln Q: cross-entropy averages this")
    fig.update_layout(title=f"Bus speed, all readings: H(P) = {entropy(P_speed):.3f} nats, "
                            f"H(P, Q) = {cross_entropy(P_speed, Q_speed):.3f} nats",
                      xaxis_title="speed bin, km/h", yaxis=dict(title="share of readings"),
                      yaxis2=dict(title="surprisal, nats", overlaying="y", side="right"),
                      barmode="group", legend=dict(orientation="h", y=-0.25))
    show(fig, "entropy_cross_entropy", height=480)
    ''', slides=["29"], treatment="lab data: the slide's constructed shares replaced by both "
                                  "days' speed readings")

    # --- slide 30: non-negativity -----------------------------------------------------------
    nb.md("### The gap between P and Q is non-negative, by Jensen's inequality\n\n"
          "*Slide: \"The gap between P and Q is non-negative, by Jensen's inequality\". "
          "The full three-line derivation is Appendix A.*")
    explain(
        "Write the chain that makes the gap non-negative.",
        "Cross-entropy minus entropy is the expectation under P of −log(Q/P); −log is "
        "convex, so Jensen bounds it below by −log of the expectation of Q/P, which is "
        "−log 1 = 0 [@cover2006, Theorem 2.6.3; @mackay2003, § 2.6].",
        "Displays the chain, then checks on the bus's P and Q that the average of −ln(Q/P) "
        "under P is at least −ln of the average of Q/P.",
        "Believing the wrong distribution never costs less than believing the right one.")
    nb.equation("kl_nonnegative", '''
    P_, Q_ = sp.symbols("P Q", positive=True)
    H, E_P = sp.Function("H"), sp.Function(r"\\mathbb{E}_P")
    formula(sp.Eq(H(P_, Q_) - H(P_), E_P(-sp.log(Q_ / P_)), evaluate=False),
            sp.Ge(E_P(-sp.log(Q_ / P_)), -sp.log(E_P(Q_ / P_)), evaluate=False),
            sp.Eq(-sp.log(E_P(Q_ / P_)), -sp.log(sp.Integer(1), evaluate=False), evaluate=False),
            sp.Eq(-sp.log(sp.Integer(1), evaluate=False), 0, evaluate=False))
    live = P_speed > 0
    left = float(np.sum(P_speed[live] * -np.log(Q_speed[live] / P_speed[live])))
    right = float(-np.log(np.sum(P_speed[live] * Q_speed[live] / P_speed[live])))
    print(f"on the bus: E_P[-ln(Q/P)] = {left:.3f}  >=  -ln E_P[Q/P] = {right:.3f}")
    ''', slides=["30"])
    explain(
        "Draw the convex function whose average is the divergence.",
        "The slide places each bin on the curve −ln(Q/P).",
        "Plots −ln(Q/P), the five speed bins at their ratios, their P-weighted average and the point at ratio 1.",
        "The average of the points — the divergence — lies above the curve's value at the average ratio, zero.")
    nb.figure("kl_convex", '''
    ratio = Q_speed / P_speed
    grid = np.linspace(0.2, 50, 500)
    divergence_here = float(np.sum(P_speed * -np.log(ratio)))
    fig = go.Figure()
    fig.add_scatter(x=grid, y=-np.log(grid), mode="lines", line=dict(color=NAVY, width=3),
                    name="−ln(Q/P): convex")
    fig.add_scatter(x=ratio, y=-np.log(ratio), mode="markers+text",
                    marker=dict(color=GREY, size=11, line=dict(color=NAVY, width=1)),
                    text=speed_bins, textposition="top center",
                    name="the five speed bins (km/h), at their ratio Q/P")
    fig.add_scatter(x=[1], y=[divergence_here], mode="markers+text", marker=dict(color=RED, size=14),
                    text=[f"KL = {divergence_here:.3f}"], textposition="middle right",
                    name=f"average of the points, weighted by P: KL = {divergence_here:.3f}")
    fig.add_scatter(x=[1], y=[0], mode="markers", marker=dict(color=GREEN, size=14),
                    name="point at the average ratio, 1: −ln 1 = 0")
    fig.update_layout(title="The divergence is the P-weighted average of a convex function, "
                            "above its value at the average",
                      xaxis=dict(title="ratio Q/P"),
                      yaxis_title="−ln(Q/P), nats", legend=dict(orientation="h", y=-0.25))
    show(fig, "kl_convex", height=480)
    ''', slides=["30"], treatment="lab data: the slide's constructed bins replaced by the "
                                  "bus's speed bins")

    # --- slide 31: KL -----------------------------------------------------------------------
    nb.md("### The Kullback-Leibler divergence\n\n"
          "*Slide: \"Definition - the Kullback-Leibler divergence\".*")
    explain(
        "Write the divergence and its identity.",
        "The extra surprise from still believing the reference Q when the data follow P. It "
        "has a fixed floor: zero exactly when P and Q are the same, positive otherwise "
        "[@kullback1951; @mackay2003, § 2.6].",
        "Builds the sum in sympy, and checks symbolically that each of its terms equals the "
        "cross-entropy term minus the entropy term.",
        "A statistic with a fixed floor can be compared against a threshold. The "
        "cross-entropy cannot: its floor is H(P), which moves.")
    nb.equation("kl_divergence", '''
    i, B = sp.symbols("i B", integer=True, positive=True)
    P, Q = sp.Function("P"), sp.Function("Q")
    formula(sp.Eq(sp.Symbol(r"D_{\\mathrm{KL}}(P \\| Q)"),
                  sp.Sum(P(i) * sp.log(P(i) / Q(i)), (i, 1, B)), evaluate=False),
            r"= H(P,Q) - H(P) \\quad [\\text{nats}]")
    p_i, q_i = sp.symbols("p_i q_i", positive=True)
    term = p_i * sp.log(p_i / q_i) - ((-p_i * sp.log(q_i)) - (-p_i * sp.log(p_i)))
    print("one term of D, minus (the H(P, Q) term minus the H(P) term):",
          sp.simplify(sp.expand_log(term)))
    ''', slides=["31"])
    explain(
        "Define the divergence as the Lab 2 solution writes it.",
        "It is Lab 2's third deliverable; its docstring states the definition the check grades.",
        "Copies `kl_divergence()` from `solutions/lab_02.py`.",
        "The next figure uses it bin by bin.")
    nb.source(f"{S}/lab_02.py", "kl_divergence", cite="[@kullback1951; @mackay2003; @jensen1906]")
    explain(
        "Draw the divergence bin by bin on the bus's speed.",
        "The slide colours each bin's term: negative where today is rarer than the "
        "reference, positive where it is commoner, and the total never negative.",
        "Computes P(i) ln(P(i)/Q(i)) per bin and the sum with `kl_divergence`, and checks "
        "the identity with the two functions above to ten decimals.",
        "Individual terms can be negative; their sum cannot.")
    nb.figure("kl_terms", '''
    terms = P_speed * np.log(P_speed / Q_speed)
    total = kl_divergence(P_speed, Q_speed)
    assert abs(total - (cross_entropy(P_speed, Q_speed) - entropy(P_speed))) < 1e-10
    colours = [RED if t < 0 else GREEN for t in terms] + [NAVY]
    fig = go.Figure(go.Bar(x=speed_bins + ["KL"], y=list(terms) + [total], marker_color=colours,
                           text=[f"{t:+.3f}" for t in terms] + [f"{total:.3f}"],
                           textposition="outside"))
    fig.update_layout(title="P(i) ln(P(i)/Q(i)) per speed bin, and their sum: never negative",
                      xaxis_title="speed bin, km/h", yaxis_title="nats")
    show(fig, "kl_terms", height=420)
    print(f"KL(23 January || 22 January) on speed = {total:.3f} nats")
    agrees("slide's constructed example: KL (the slide's chart prints 0.225)",
           kl_divergence(P_slide, Q_slide), 0.225, 3)
    ''', slides=["31"], treatment="lab data: the bus's speed bins")
    explain(
        "Show where the divergence breaks, on the bus.",
        "If today holds a value the reference never held, Q(i) = 0 where P(i) > 0 and that "
        "term is infinite. Mathematically right — an event the reference called impossible is "
        "infinitely surprising — and useless on a dashboard.",
        "Bins the windows' mean speed over both days in five equal-width bins: 23 January "
        "reached window speeds 22 January never did.",
        "One empty reference bin makes the whole number infinite. Two repairs follow: a small "
        "floor under every share, and the symmetrised index.")
    nb.code('''
    pooled = np.histogram_bin_edges(np.concatenate([reference["mean_speed"], current["mean_speed"]]),
                                    bins=5)
    P_window = np.histogram(current["mean_speed"], pooled)[0] / len(current)
    Q_window = np.histogram(reference["mean_speed"], pooled)[0] / len(reference)
    print("window mean speed, P (23 January):", P_window.round(3))
    print("window mean speed, Q (22 January):", Q_window.round(3))
    print("KL(P || Q) =", kl_divergence(P_window, Q_window))
    print("the lab's own example, a value the reference called impossible:",
          kl_divergence([0.5, 0.5], [1.0, 0.0]))
    ''')

    # --- slide 32: training loss vs monitoring statistic --------------------------------------
    nb.md("### Why the training loss is not the monitoring statistic\n\n"
          "*Slide: \"Why the training loss is not the monitoring statistic\".*")
    explain(
        "Split two cross-entropies into their entropy floor and their divergence.",
        "Minimising the cross-entropy over a model's parameters minimises the divergence, "
        "because the entropy term does not depend on the model — which is why it is the "
        "right *training* loss [@murphy2022, § 6.2.5]. As a *monitor* it is wrong: its "
        "minimum is the current period's own entropy, which moves.",
        "Stacks entropy and divergence for the slide's worked pair ([0.9, 0.1] against "
        "[0.5, 0.5]) and for the bus's speed pair.",
        "The totals differ mainly because the grey floor differs; only the divergence has a "
        "floor fixed at zero.")
    nb.figure("cross_entropy_split", '''
    today, yesterday = np.array([0.9, 0.1]), np.array([0.5, 0.5])
    pairs = {"worked pair [0.9, 0.1] vs [0.5, 0.5]": (today, yesterday),
             "bus speed, 23 vs 22 January": (P_speed, Q_speed)}
    fig = go.Figure()
    fig.add_bar(x=list(pairs), y=[entropy(p) for p, q in pairs.values()],
                name="entropy H(P): the floor", marker_color=GREY)
    fig.add_bar(x=list(pairs), y=[kl_divergence(p, q) for p, q in pairs.values()],
                name="divergence KL: floor fixed at zero", marker_color=GREEN,
                text=[f"total {cross_entropy(p, q):.3f}" for p, q in pairs.values()],
                textposition="outside")
    fig.update_layout(barmode="stack", title="Cross-entropy = entropy + divergence, in nats",
                      yaxis_title="nats")
    show(fig, "cross_entropy_split", height=440)
    agrees("worked pair: entropy", entropy(today), 0.325, 3)
    agrees("worked pair: cross-entropy", cross_entropy(today, yesterday), 0.693, 3)
    agrees("worked pair: divergence", kl_divergence(today, yesterday), 0.368, 3)
    beside("bus speed pair (entropy, divergence, total)",
           (round(entropy(P_speed), 3), round(kl_divergence(P_speed, Q_speed), 3),
            round(cross_entropy(P_speed, Q_speed), 3)), (1.47, 0.23, 1.70),
           "the slide's bus pair is its constructed five-bin example; these are the real "
           "speed readings")
    ''', slides=["32"], treatment="exact for the worked pair; lab data for the bus pair")

    # --- slide 33: asymmetry --------------------------------------------------------------------
    nb.md("### The divergence is asymmetric\n\n"
          "*Slide: \"Worked example: the divergence is asymmetric\".*")
    explain(
        "Compute the divergence both ways on the slide's two-state example.",
        "Which day is the reference is a modelling decision and must be stated.",
        "Per-state terms of D(current‖reference) and D(reference‖current), and their sums.",
        "0.368 one way, 0.511 the other: the reverse is larger because the reference puts 0.5 "
        "where today puts 0.1, and log(0.5/0.1) is large.")
    nb.figure("asymmetry", '''
    current_states, reference_states = np.array([0.9, 0.1]), np.array([0.5, 0.5])
    forward = current_states * np.log(current_states / reference_states)
    reverse = reference_states * np.log(reference_states / current_states)
    fig = go.Figure()
    fig.add_bar(x=["State 1", "State 2"], y=forward, marker_color=NAVY,
                name=f"forward: D(current || reference) = {forward.sum():.3f}",
                text=[f"{v:+.3f}" for v in forward], textposition="outside")
    fig.add_bar(x=["State 1", "State 2"], y=reverse, marker_color=ORANGE,
                name=f"reverse: D(reference || current) = {reverse.sum():.3f}",
                text=[f"{v:+.3f}" for v in reverse], textposition="outside")
    fig.update_layout(barmode="group", title="Per-state terms of the two divergences, in nats",
                      yaxis_title="nats", legend=dict(orientation="h", y=1.12))
    show(fig, "asymmetry", height=420)
    for label, value, stated in (
            ("forward, state 1", forward[0], 0.529), ("forward, state 2", forward[1], -0.161),
            ("reverse, state 1", reverse[0], -0.294), ("reverse, state 2", reverse[1], 0.805),
            ("D(current || reference)", kl_divergence(current_states, reference_states), 0.368),
            ("D(reference || current)", kl_divergence(reference_states, current_states), 0.511)):
        agrees(label, value, stated, 3)
    ''', slides=["33"], treatment="exact: the slide's worked example")

    # --- slide 34: discretisation policy ---------------------------------------------------------
    nb.md("### Discretisation policy, and the data contract it requires\n\n"
          "*Slide: \"Discretisation policy, and the data contract it requires\".*")
    explain(
        "Show, on the target's real windows, what binning hides.",
        "Binning removes the infinity but also hides a genuinely new value inside an existing "
        "bin; only a separate check on the raw values — is it within the reference range? — "
        "catches a value outside it.",
        "Cuts 22 January's 45 mean-payload windows into five bins at their own quantiles and "
        "draws them, with one novel value placed inside a bin and one outside the reference "
        "range. The next cells measure what the index and a range check each report.",
        "The slide's schematic, drawn with the real windows.")
    nb.figure("binning_policy", '''
    ref_payload = reference[TARGET].to_numpy(float)
    cur_payload = current[TARGET].to_numpy(float)
    inner = np.quantile(ref_payload, [0.2, 0.4, 0.6, 0.8])
    ordered = np.sort(ref_payload)
    widest = int(np.argmax(np.diff(ordered)))
    novel_inside = float((ordered[widest] + ordered[widest + 1]) / 2)
    out_of_range = float(ref_payload.max() + 100.0)
    fig = go.Figure()
    for edge in inner:
        fig.add_vline(x=edge, line=dict(color=GREY, dash="dash"))
    fig.add_scatter(x=ref_payload, y=np.zeros(len(ref_payload)), mode="markers",
                    marker=dict(color=BLUE, size=10), name="22 January windows (reference)")
    fig.add_scatter(x=[novel_inside], y=[0], mode="markers",
                    marker=dict(color=ORANGE, size=16, symbol="diamond"),
                    name=f"a novel value inside a bin ({novel_inside:.0f} kg)")
    fig.add_scatter(x=[out_of_range], y=[0], mode="markers",
                    marker=dict(color=RED, size=16, symbol="diamond"),
                    name=f"a value outside the reference range ({out_of_range:.0f} kg)")
    fig.update_layout(title="Five bins cut at the reference's own quantiles (dashed), outer bins open",
                      xaxis_title="mean payload per five-minute window, kg", yaxis=dict(visible=False),
                      legend=dict(orientation="h", y=-0.35))
    show(fig, "binning_policy", height=380)
    ''', slides=["34"], treatment="lab data: the slide's schematic drawn with the real windows")
    explain(
        "Define the symmetrised index as the Lab 2 solution writes it.",
        "Three decisions quietly decide its answer: the edges come from the reference, a "
        "small floor goes under empty bins, and a reference that cannot be binned is refused.",
        "Copies `population_stability_index()` from `solutions/lab_02.py`.",
        "The next cell uses it on the two added values and on the degenerate column.")
    nb.source(f"{S}/lab_02.py", "population_stability_index",
              cite="[@jeffreys1946; @yurdakul2020; @siddiqi2006]")
    explain(
        "Measure what the index and the range check report for the two added values, and "
        "show the degenerate reference.",
        "The slide's three claims: a novel value inside a bin barely moves a binned statistic; "
        "an out-of-range value needs a schema check; and a reference that is almost all "
        "zeros cannot be binned at all.",
        "Index of 23 January against 22 January, then with each value added; a range check "
        "against the reference's observed range; and the quantile edges of the manual-driving "
        "share, which collapse.",
        "An index of exactly zero on a column that moved is not evidence of no change; it is "
        "evidence of no measurement, and Lab 2's function refuses to report it.")
    nb.code('''
    base = population_stability_index(ref_payload, cur_payload)
    inside = population_stability_index(ref_payload, np.append(cur_payload, novel_inside))
    outside = population_stability_index(ref_payload, np.append(cur_payload, out_of_range))
    low, high = ref_payload.min(), ref_payload.max()
    print(f"index, 23 January as measured             {base:.3f}")
    print(f"index, plus one novel value inside a bin  {inside:.3f}")
    print(f"index, plus one value outside the range   {outside:.3f}")
    print(f"range check against the reference, [{low:.1f}, {high:.1f}] kg: "
          f"{novel_inside:.1f} kg {'passes' if low <= novel_inside <= high else 'FAILS'}, "
          f"{out_of_range:.1f} kg {'passes' if low <= out_of_range <= high else 'FAILS'}")
    manual = reference["human_driven"].to_numpy(float)
    agrees("reference windows with no manual reading", (manual == 0).sum(), 39, 0)
    print("quantile edges of the manual-driving share:",
          np.unique(np.quantile(manual, np.linspace(0, 1, 6))))
    try:
        population_stability_index(manual, current["human_driven"].to_numpy(float))
    except DegenerateReference as refused:
        print("the index refuses:", refused)
    ''')

    # --- slides 35-36: PSI worked --------------------------------------------------------------
    nb.md("### The symmetrised index (Population Stability Index)\n\n"
          "*Slides: \"Definition - the symmetrised index (Population Stability Index)\" and "
          "\"PSI worked through: 75 windows, 20 bins, one constant\".*")
    explain(
        "Write the index and prove its identity.",
        "Adding the divergence in both directions removes the choice of reference. Credit "
        "scoring calls the sum the Population Stability Index [@jeffreys1946; @yurdakul2020].",
        "Builds J(P, Q) in sympy and checks, term by term, that (P − Q) log(P/Q) equals "
        "P log(P/Q) + Q log(Q/P).",
        "Symmetric by construction — but, as Appendix C shows, not a distance.")
    nb.equation("psi", '''
    i, B = sp.symbols("i B", integer=True, positive=True)
    P, Q = sp.Function("P"), sp.Function("Q")
    formula(sp.Eq(sp.Symbol("J(P, Q)"), sp.Sum((P(i) - Q(i)) * sp.log(P(i) / Q(i)), (i, 1, B)),
                  evaluate=False),
            r"= D_{\\mathrm{KL}}(P \\| Q) + D_{\\mathrm{KL}}(Q \\| P)")
    p_i, q_i = sp.symbols("p_i q_i", positive=True)
    difference = (p_i - q_i) * sp.log(p_i / q_i) - (p_i * sp.log(p_i / q_i) + q_i * sp.log(q_i / p_i))
    print("term of J minus the two divergence terms, simplified:",
          sp.simplify(sp.expand_log(difference)))
    ''', slides=["35"])
    explain(
        "Work the index through on the target, the way the slide does: a current day drawn at "
        "random from the reference, so that nothing has changed.",
        "The index is never zero even when nothing changed, and at many bins most of what it "
        "reports is the constant put under empty bins.",
        "Draws 35 windows with replacement from the 45 reference windows of mean payload "
        "(seed 20200122 — the first resample of the null below), bins at the reference's "
        "quantiles, and shows each bin's shares and term at 5 and at 20 bins.",
        "At 5 bins the index of a day on which nothing changed is small; at 20 bins each empty "
        "current bin adds a term set by the constant 10⁻⁶, not by any kilogram measured.")
    nb.code('''
    rng = np.random.default_rng(SEED)
    unchanged_day = rng.choice(ref_payload, size=len(current), replace=True)

    def worked(bins):
        """The index of the unchanged day, bin by bin (the recipe of population_stability_index)."""
        edges = np.unique(np.quantile(ref_payload, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        q_count = np.histogram(ref_payload, bins=edges)[0]
        p_count = np.histogram(unchanged_day, bins=edges)[0]
        q = np.clip(q_count / len(ref_payload), PSI_EPSILON, None)
        p = np.clip(p_count / len(unchanged_day), PSI_EPSILON, None)
        return q, p, (p - q) * np.log(p / q), p_count == 0

    for bins in (5, 20):
        q, p, terms, empty = worked(bins)
        assert abs(terms.sum() - population_stability_index(ref_payload, unchanged_day, bins)) < 1e-12
        print(f"{bins} bins: index {terms.sum():.3f}; empty current bins {empty.sum()}, adding "
              f"{terms[empty].sum():.3f} ({terms[empty].sum() / terms.sum():.3f} of the total); "
              f"filled bins add {terms[~empty].sum():.3f}")
    q, p, terms, empty = worked(20)
    agrees("term of one empty bin at 20 bins, reference share 2/45", terms[empty][0], 0.476, 3,
           source="slides/measured_v3.json")
    beside("index at 5 and at 20 bins, and empty bins at 20",
           (round(float(worked(5)[2].sum()), 3), round(float(worked(20)[2].sum()), 3),
            int(worked(20)[3].sum())),
           (0.14, 1.87, 3),
           "the slides work a constructed example with 40 reference windows (share 2/40 = 0.05, "
           "0.54 per empty bin); here the real 45 windows and the first resample of the null")
    ''')
    explain(
        "Draw the worked index at 5 and at 20 bins.",
        "The two slides show the shares and each bin's term, and how empty bins dominate at 20 bins.",
        "Uses `worked()` above: shares and terms at 5 bins, then the terms at 20 bins with the empty bins in red.",
        "At 20 bins most of the index is the constant under the empty bins.")
    nb.figure(["psi_shares", "psi_terms_20_bins"], '''
    q, p, terms, _ = worked(5)
    labels = [f"i = {k}" for k in range(1, 6)]
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.18,
                        subplot_titles=("the shares: reference, and a day drawn from it",
                                        "the term each range adds"))
    fig.add_bar(x=labels, y=q, name="reference, 45 windows", marker_color=GREY, row=1, col=1)
    fig.add_bar(x=labels, y=p, name="current, 35 windows drawn from the reference",
                marker_color=NAVY, row=1, col=1)
    fig.add_bar(x=labels, y=terms, marker_color=GREEN, showlegend=False,
                text=[f"{t:.3f}" for t in terms], textposition="outside", row=2, col=1)
    fig.update_yaxes(range=[0, max(terms) * 1.35], row=2, col=1)
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=-0.12),
                      title=f"Five bins: index {terms.sum():.3f} although nothing changed")
    show(fig, "psi_shares", height=560)
    q, p, terms, empty = worked(20)
    fig = go.Figure(go.Bar(x=list(range(1, 21)), y=terms,
                           marker_color=[RED if e else GREEN for e in empty],
                           text=[f"{t:.2f}" if e else "" for t, e in zip(terms, empty)],
                           textposition="outside"))
    fig.update_layout(title=f"Twenty bins, the same two days: index {terms.sum():.2f}, "
                            f"{terms[empty].sum() / terms.sum():.0%} of it from {empty.sum()} "
                            "empty bins (red)",
                      xaxis_title="bin", yaxis_title="term, nats")
    show(fig, "psi_terms_20_bins", height=420)
    ''', slides=["35", "36"], treatment="lab data: the slides' 40-window worked example replaced "
                                        "by the 45 real windows and the first null resample")

    # --- slides 37-40: null, floor, threshold -----------------------------------------------------
    nb.md("### Calibrating the index: the threshold comes from the reference alone\n\n"
          "*Slides: \"Calibrating the index: the threshold comes from the reference alone\", "
          "\"Definition - the noise floor of the index\", \"Definition - the materiality "
          "threshold, derived from the null\" and \"Why the conventional threshold of 0.25 does "
          "not transfer\".*")
    explain(
        "Write the noise floor and the threshold.",
        "Under no change, the index scaled by the harmonic factor is approximately "
        "chi-squared with B − 1 degrees of freedom, so any threshold depends on both sample "
        "sizes and on B [@yurdakul2020]. The floor is the median of the index over resamples "
        "of the reference against itself; the threshold is an upper quantile of the same "
        "null [@efron1979].",
        "Displays the three expressions.",
        "q is a false-alarm rate chosen and written down: q = 0.99 means one alarm in a "
        "hundred comparisons in which nothing changed.")
    nb.equation("floor_and_threshold", '''
    n, m, B, q, R = sp.symbols("n m B q R", positive=True)
    J = sp.Function("J")
    ref, resample = sp.Symbol(r"\\mathrm{ref}"), sp.Symbol(r"\\mathrm{ref}^{(r)}")
    formula(sp.Pow(1 / n + 1 / m, -1) * J(ref, resample), r"\\ \\dot\\sim\\ ", sp.Function(r"\\chi^2")(B - 1))
    formula(sp.Eq(sp.Function(r"\\mathrm{floor}")(B), sp.Function(r"\\mathrm{median}_{r \\le R}")(J(ref, resample)),
                  evaluate=False))
    formula(sp.Eq(sp.Function(r"\\mathrm{threshold}")(B, q), sp.Function("Q_q")(J(ref, resample)), evaluate=False),
            r"\\text{over } r = 1, \\dots, R, \\text{ at } B \\text{ bins}")
    ''', slides=["38", "39"])
    explain(
        "Define the null, the floor and the threshold as the Lab 2 solution writes them.",
        "The last deliverable of Laboratory 2, and the function that stops this module "
        "borrowing a number.",
        "Copies `index_threshold()` from `solutions/lab_02.py`: resample the reference "
        "against itself R times at the current sample size, and return the median and the "
        "q-quantile with every choice that produced them.",
        "Lab 4 judges every feature against the threshold this function derives for it.")
    nb.source(f"{S}/lab_02.py", "index_threshold", cite="[@yurdakul2020; @efron1979]")
    explain(
        "Draw the 1,000 values of the index on the target when nothing changed.",
        "This is the null distribution: 45 reference windows against 35 resampled from "
        "them, 5 bins, seed 20200122.",
        "Rebuilds the null with the same random stream `index_threshold()` uses, draws its "
        "histogram, and colours the bars above the threshold.",
        "Most values sit between 0 and 0.2; a real day landing in the red tail is called a "
        "change, and one such call in a hundred is a false alarm.")
    nb.figure("psi_null", '''
    rng = np.random.default_rng(SEED)
    null_payload = np.array([population_stability_index(
        ref_payload, rng.choice(ref_payload, size=len(current), replace=True), 5)
        for _ in range(NULL_RESAMPLES)])
    derived = index_threshold(ref_payload, cur_payload, bins=5)
    assert abs(np.quantile(null_payload, NULL_QUANTILE) - derived["threshold"]) < 1e-12
    agrees("noise floor, median of the null", derived["noise_floor"], 0.105, 3)
    agrees("threshold, 0.99 quantile of the null", derived["threshold"], 0.465, 3)
    edges = np.arange(0, 0.651, 0.05)
    counts = np.histogram(np.clip(null_payload, 0, 0.6499), bins=edges)[0]
    fig = go.Figure(go.Bar(x=[f"{e:.2f}" for e in edges[:-1]], y=counts,
                           marker_color=[RED if e >= derived["threshold"] else GREY for e in edges[:-1]]))
    fig.update_layout(title=f"PSI when nothing changed, runs out of {NULL_RESAMPLES:,} "
                            f"(threshold {derived['threshold']:.3f}; the last bar holds "
                            "everything above 0.60)",
                      xaxis_title="index, left edge of the bar", yaxis_title="runs")
    show(fig, "psi_null", height=420)
    print("runs per bar:", counts.tolist())
    beside("runs per bar of the null, first twelve bars", counts[:12].tolist(),
           [204, 290, 212, 116, 83, 42, 20, 11, 11, 4, 2, 1],
           "the slide's bars come from another run of the null that this repository does not "
           "record (they sum to 996); the two numbers the slide reads off it, the median 0.105 "
           "and the threshold 0.465, agree")
    ''', slides=["37"], treatment="lab data")
    explain(
        "Measure the floor, the threshold and the false-alarm rate of the borrowed 0.25 at "
        "several bin counts.",
        "The conventional thresholds — below 0.1 no change, above 0.25 a material shift — "
        "originate with Lewis and reach practitioners through Siddiqi, and neither derives "
        "them [@lewis1994; @siddiqi2006]. At forty windows a day the floor alone already "
        "exceeds 0.25 at ten bins.",
        "Runs `index_threshold()` at 3, 5, 10 and 20 bins, counts the no-change comparisons "
        "above 0.25, and asks at each bin count whether the unchanged target would fire. It "
        "also compares the mean of the scaled null with B − 1, the chi-squared prediction.",
        "Borrowing 0.25 buys a 10% false-alarm rate at five bins, 59% at ten, 99% at twenty. "
        "Every threshold in this module is instead the 0.99 quantile of the measured null. "
        "The chi-squared approximation is only rough at 45 against 35 windows, and it breaks "
        "at twenty bins, where the floor constant under empty bins dominates — one more "
        "reason the threshold is measured rather than read from a table.")
    nb.code('''
    by_bins = {}
    for bins in (3, 5, 10, 20):
        rng = np.random.default_rng(SEED)
        null = np.array([population_stability_index(
            ref_payload, rng.choice(ref_payload, size=len(current), replace=True), bins)
            for _ in range(NULL_RESAMPLES)])
        derived = index_threshold(ref_payload, cur_payload, bins=bins)
        scaled = null / (1 / len(ref_payload) + 1 / len(current))
        target = population_stability_index(ref_payload, cur_payload, bins)
        by_bins[bins] = {"floor": derived["noise_floor"], "threshold": derived["threshold"],
                         "share above 0.25": float((null > BORROWED_INDEX).mean()),
                         "target index": target, "target fires": target >= derived["threshold"],
                         "scaled null, mean": round(float(scaled.mean()), 1), "B - 1": bins - 1}
    print(pd.DataFrame(by_bins).T.to_string())
    for bins, stated in ((3, 0.039), (5, 0.105), (10, 0.28), (20, 2.057)):
        agrees(f"noise floor at {bins} bins", by_bins[bins]["floor"], stated, 3)
    for bins, stated in ((3, 0.256), (5, 0.465), (10, 2.297)):
        agrees(f"threshold at {bins} bins", by_bins[bins]["threshold"], stated, 3)
    for bins, stated in ((5, 0.101), (10, 0.587), (20, 0.993)):
        agrees(f"share of no-change comparisons above 0.25 at {bins} bins",
               by_bins[bins]["share above 0.25"], stated, 3)
    ''')
    explain(
        "Draw the floor, the threshold, and what the borrowed 0.25 costs.",
        "Each of the three slides charts one of these against the bin count.",
        "Three bar charts from the `by_bins` table above.",
        "At 10 and 20 bins even the floor exceeds 0.25; the derived threshold keeps false alarms at 1%.")
    nb.figure(["noise_floor_by_bins", "threshold_by_bins", "borrowed_false_alarms"], '''
    labels = [f"{bins} bins" for bins in by_bins]
    fig = go.Figure(go.Bar(x=labels, y=[row["floor"] for row in by_bins.values()],
                           marker_color=[GREY if bins <= 5 else RED for bins in by_bins],
                           text=[f"{row['floor']:.3f}" for row in by_bins.values()],
                           textposition="outside"))
    fig.add_hline(y=BORROWED_INDEX, line=dict(color=GREEN, dash="dash"),
                  annotation_text="0.25, the borrowed threshold")
    fig.update_layout(title="The noise floor by bin count: at 10 and 20 bins an unchanged day "
                            "already reads above 0.25",
                      yaxis_title="median index over 1,000 no-change comparisons")
    show(fig, "noise_floor_by_bins", height=420)
    beside("20-bin noise floor, as the slide's chart draws it", round(by_bins[20]["floor"], 3), 0.6,
           "the slide's bar is cut at 0.6 to fit the chart, as its caption says; its text gives "
           "2.057, which this notebook reproduces")

    three = [3, 5, 10]
    fig = go.Figure()
    fig.add_bar(x=[f"{b} bins" for b in three], y=[by_bins[b]["floor"] for b in three],
                name="noise floor, median", marker_color=GREY)
    fig.add_bar(x=[f"{b} bins" for b in three], y=[by_bins[b]["threshold"] for b in three],
                name="threshold, q = 0.99", marker_color=NAVY,
                text=[f"{by_bins[b]['threshold']:.3f}" for b in three], textposition="outside")
    fig.add_hline(y=BORROWED_INDEX, line=dict(color=GREEN, dash="dash"),
                  annotation_text="0.25 borrowed")
    fig.update_layout(barmode="group", title="Floor and threshold by bin count, from the same null",
                      yaxis_title="index")
    show(fig, "threshold_by_bins", height=420)

    alarm_bins = [5, 10, 20]
    fig = go.Figure(go.Bar(x=[f"{b} bins" for b in alarm_bins],
                           y=[by_bins[b]["share above 0.25"] for b in alarm_bins],
                           marker_color=[GREY, "#D98CA3", "#D98CA3"],
                           text=[f"{by_bins[b]['share above 0.25']:.1%}" for b in alarm_bins],
                           textposition="outside"))
    fig.update_layout(title="What 0.25 costs: share of 1,000 no-change comparisons above it",
                      yaxis_title="false-alarm rate", yaxis_range=[0, 1.1])
    show(fig, "borrowed_false_alarms", height=400)
    ''', slides=["38", "39", "40"], treatment="lab data")

    # --- the laboratory -----------------------------------------------------------------------
    nb.md("""
    ## Laboratory 2 — divergence, index, and a threshold you derived

    *Slide: "Laboratory 2 - divergence, index, and a threshold you derived".* Five
    functions, twenty-five minutes; the decomposition identity is the check that
    teaches. Your cross-entropy minus your entropy must equal your divergence to ten
    decimal places on five pairs; the divergence is compared with SciPy and must be
    asymmetric; the index must be built with edges from the reference; and the
    threshold must vary with the bin count, differ between two columns, reproduce
    on a second run, and fall where the check's own null puts it. A submission that
    writes 0.25 fails, which is the point of the part.
    """)
    nb.statement(f"{LABS}/02_how_surprising.py")
    nb.md("""
    ### The solution, step by step

    The five functions were defined above: `entropy`, `cross_entropy`,
    `kl_divergence`, `population_stability_index` and `index_threshold`. The slide's
    sketch of the threshold, `threshold(ref, n_cur, B=5, q=0.99, R=1000, seed)`, is the
    same computation as `index_threshold()`, which also returns the floor and every
    choice it was made under. First the stub's own demonstration.
    """)
    explain(
        "Run the stub's own `__main__` block against the solved functions.",
        "It is what a student sees when the lab file is complete.",
        "The lines below are the demonstration at the end of `labs/02_how_surprising.py`, verbatim; its "
        "`from lab_support import …` resolves to this notebook's cells.",
        "The printed values are the ones the lab's check expects.")
    nb.step(f"{LABS}/02_how_surprising.py", 0)
    explain(
        "Check the decomposition identity, and the functions against SciPy, as the check does.",
        "If cross-entropy minus entropy is not the divergence to ten decimals, one of the "
        "three is not the object you think it is.",
        "Five pairs of distributions; `scipy.stats.entropy` gives the entropy with one "
        "argument and the divergence with two.",
        "All three agree with the reference implementation.")
    nb.code('''
    pairs = [([0.9, 0.1], [0.5, 0.5]), ([0.5, 0.5], [0.9, 0.1]), (P_speed, Q_speed),
             ([0.2, 0.3, 0.5], [0.3, 0.3, 0.4]), ([0.1, 0.2, 0.4, 0.2, 0.1], [0.3, 0.3, 0.2, 0.1, 0.1])]
    for p, q in pairs:
        assert abs(cross_entropy(p, q) - entropy(p) - kl_divergence(p, q)) < 1e-10
        assert abs(entropy(p) - stats.entropy(p)) < 1e-10
        assert abs(kl_divergence(p, q) - stats.entropy(p, q)) < 1e-10
    print("identity and SciPy agree on all five pairs")
    ''')
    explain(
        "Start the narrator the solution's demonstration prints with.",
        "The demonstration below reports every step through `say.info`, as it does in the terminal.",
        "Sets the lab number and opens the narrator, as the first lines of the solution's "
        "`__main__` block do.",
        "The numbered steps that follow run unchanged.")
    nb.code('''
    LAB = 2
    say = narrator(LAB)
    say.info("Lab 2 — two averages of surprise, the gap between them, and the floor under the gap")
    ''')
    for number, (goal, so_what) in enumerate([
            ("The worked pair, in nats, with the identity closed out loud.",
             "the gap between the two averages is the divergence, and it is asymmetric."),
            ("The archive, where the index has to decide something.",
             "mean speed moves; the manual-driving share is refused rather than reported as 0."),
            ("The noise floor, and the threshold derived from it.",
             "the target's own index sits below its floor; the borrowed 0.25 would fire on one "
             "unchanged day in ten."),
    ], start=1):
        explain(f"Step {number} of the solution's demonstration: {goal[0].lower() + goal[1:]}",
                "It is the reference solution's own demonstration; running it here shows the "
                "functions above at work on the lab data, exactly as `make demo` does.",
                f"Runs step {number} of the demonstration block of `solutions/lab_02.py`, verbatim; "
                "`say.info` prints each finding with the seconds elapsed since the lab began.",
                so_what[0].upper() + so_what[1:])
        nb.step(f"{S}/lab_02.py", number)
