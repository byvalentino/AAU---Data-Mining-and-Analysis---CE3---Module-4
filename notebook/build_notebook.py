#!/usr/bin/env python3
"""Build Module 4's demonstration notebook, and execute it.

    python "Module 4/notebook/build_notebook.py"            build and run
    python "Module 4/notebook/build_notebook.py" --no-run   build only

Rewritten 28 September 2026. The notebook follows the deck, `slides/Module4.pptx`,
part by part, and holds itself to three things:

  every image on a shown slide is drawn here by Python -- on the lab data where
      the slide shows data, from the same simulation or closed form where it
      shows one, and labelled "illustrative" where the slide's values were
      constructed and the lab data cannot stand in (slides/figure_map is
      notebook/figure_map.json);
  every lab exercise is stated as its stub states it, and its solution runs step
      by step, every function's code visible -- copied verbatim from
      exercises/solutions/ and checked against it by
      tools/check_notebook_sources.py;
  every number the slides print is recomputed, and agrees() stops the run if
      the deck and the code disagree.

Data: `exercises/data/bus_slice.csv.gz`, the extract every Module 4 lab reads --
shuttle VJRD1A10224000055, 22 and 23 January 2020, 48,290 readings, no personal
data. Executed from `Module 4/exercises`, so the relative paths resolve exactly
as the labs' do. Needs the lab requirements plus notebook/requirements.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from notebook_kit import Notebook, execute  # noqa: E402

OUTPUT = HERE / "Module4_demonstration.ipynb"
EXERCISES = HERE.parent / "exercises"
S = "Module 4/exercises/solutions"
LABS = "Module 4/exercises/labs"
SUPPORT = "Module 4/exercises/lab_support.py"
NARRATE = "Module 4/exercises/_narrate.py"

nb = Notebook(4, HERE / "references.json")

# The shape of every explanation cell above a code cell.
def explain(goal: str, why: str, what: str, so_what: str, extra: str = "") -> None:
    text = (f"**Goal.** {goal}\n\n**Why.** {why}\n\n**What the code does.** {what}\n\n"
            f"**So what.** {so_what}")
    nb.md(text + (f"\n\n{extra}" if extra else ""))


# =============================================================================
# Front matter and set-up
# =============================================================================

def front_matter() -> None:
    nb.md("""
    # Module 4 — Detecting distribution shift

    **Data Mining and Analysis (course code CE3) · Aalborg University, Copenhagen**

    *Quantifying uncertainty, divergence and effect size after the labels stop.*

    A model is in production and today's data have no labels. The deck asks four
    questions about that situation and answers each with a number computed on our
    own data. This notebook is the deck's companion: it follows the same four parts
    in the same order, draws every figure on the shown slides (the title photograph and one text-only diagram excepted), states every laboratory
    exercise as the lab file states it, and runs the reference solution step by
    step with all of its code on the page.

    | Part | The question | Laboratory |
    |---|---|---|
    | 1 | How good is my model today, when nobody has labelled today's data? | Lab 1 — interval estimation and coverage |
    | 2 | Has the input data changed since the model was trained? | Lab 2 — divergence, index, and a threshold you derived |
    | 3 | By how much has it changed, in units I can act on? | Lab 3 — distance, and one function for all four statistics |
    | 4 | Is the change real, does it matter, and what do I do about it? | Lab 4 — the decision |

    **How to read it.** Every code cell has a short note above it: the *goal*, *why*
    it is done, *what the code does*, and *so what* — what the result lets you say.
    Cells that begin `# Source: … verbatim` are copied from the laboratory files
    and are checked against them, so the code you read is the code the labs run.
    Cells that draw a figure name the slide they reproduce. Formulas are written
    in Python with `sympy` and displayed as LaTeX.

    **How to run it.** From `Module 4/exercises`, after `bash setup.sh` and
    `pip install -r ../notebook/requirements.txt`. The whole notebook runs in under
    one minute.

    **Data.** `exercises/data/bus_slice.csv.gz` — the extract every Module 4 lab
    reads: shuttle VJRD1A10224000055 on 22 and 23 January 2020, 48,290 readings of
    vehicle telemetry, no personal data. The slides' source lines name
    `data/bus.csv`, the full archive; for this vehicle the two hold the same rows,
    so the numbers below are the slides' numbers, recomputed; where one differs, the notebook says so and why, and the closing section lists every such case.
    """)


def setup() -> None:
    nb.md("## Set-up")
    explain(
        "Load the libraries, and define the three small tools every later cell uses.",
        "A notebook that claims to reproduce a deck needs a way to show a figure, a "
        "formula and a number side by side with what the slide printed.",
        "`show()` renders a plotly figure to a static image embedded in the notebook, so "
        "it displays anywhere, including offline. `formula()` displays a sympy expression "
        "as LaTeX. `agrees()` prints a number the deck states beside the same number "
        "computed here and stops the run when they differ; `beside()` prints two numbers "
        "that are expected to differ, with the reason.",
        "If the deck and the code ever disagree, this notebook fails to run rather than "
        "quietly showing a different number.")
    nb.code('''
    import math
    import warnings
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import sympy as sp
    from scipy import stats
    from scipy.stats import ttest_ind
    from IPython.display import Image, Math, display

    warnings.filterwarnings("ignore", category=FutureWarning)
    pd.set_option("display.width", 120)

    BLUE, ORANGE, GREY, RED, GREEN, NAVY = ("#2A78D6", "#E07B39", "#52514E",
                                           "#C0392B", "#2E8B57", "#1F2A5A")


    def show(fig, name, width=1000, height=560):
        """Draw a plotly figure as a static image inside the notebook."""
        fig.update_layout(template="plotly_white", width=width, height=height)
        if fig.layout.margin.t is None:          # unless the figure asked for its own room
            fig.update_layout(margin=dict(l=70, r=30, t=70, b=60))
        display(Image(fig.to_image(format="png", width=width, height=height, scale=1)))


    def formula(*parts):
        """Display sympy expressions (or LaTeX strings) side by side."""
        pieces = [p if isinstance(p, str) else sp.latex(p, order="none") for p in parts]
        display(Math(r"\\qquad ".join(pieces)))


    def pm(centre, half):
        """LaTeX for 'centre plus or minus half', from two sympy expressions."""
        return sp.latex(centre, order="none") + r" \\pm " + sp.latex(half, order="none")


    def agrees(what, computed, stated, places, source="deck"):
        """A number the deck (or, where named, the lab's measurement record) states,
        recomputed here. Stops the run on a mismatch."""
        mine = round(float(computed), places)
        if abs(mine - float(stated)) > 0.5 * 10 ** -places + 1e-12:
            raise AssertionError(f"{what}: {source} says {stated}, the code gives {mine}")
        print(f"  {what:<62} {source} {stated:<10} computed {int(mine) if places == 0 else mine}")


    DIFFERENCES = []   # every beside() call, gathered for the closing table


    def beside(what, computed, stated, why):
        """Two numbers that are expected to differ, printed with the reason."""
        DIFFERENCES.append((what, str(computed), str(stated), why))
        print(f"  {what}: computed here {computed}, stated {stated} — {why}")
    ''')

    nb.md("""
    ### The laboratory machinery, in full

    The labs share one support file, `exercises/lab_support.py`. It fixes the grain
    — which vehicle, which windows, which day is the reference — and the constants
    every lab reads, so that a number and the choices it was made under are never
    separated. Nothing here is imported from it: the next cells *are* it.
    """)
    explain(
        "Let `lab_support.py`'s path constants resolve inside a notebook.",
        "The file finds its data relative to its own location (`__file__`), which a "
        "notebook does not have.",
        "Points `__file__` at `lab_support.py` in the working directory, which is "
        "`Module 4/exercises` when this notebook runs.",
        "The next cell can then be copied from the file unchanged.")
    nb.code('''
    import pathlib
    __file__ = str(Path.cwd() / "lab_support.py")
    assert Path(__file__).exists(), "run this notebook from Module 4/exercises"
    ''')
    explain(
        "Define the grain, the constants and the data loaders the four labs share.",
        "Every number in this module depends on choices — vehicle, window length, "
        "reference day, bin count, threshold quantile, seed. Standing rule: print them "
        "beside the number.",
        "Copies `lab_support.py` verbatim, except `load_lab`, which reads the students' "
        "unsolved files and is replaced two cells below.",
        "`windowed()` turns 48,290 readings into one row per five-minute window; every "
        "`n` in this module counts those windows.")
    nb.source(SUPPORT, "HERE", "BUS_SLICE", "VEHICLE", "WINDOW", "REFERENCE_DAY",
              "CURRENT_DAY", "MINIMUM_READINGS", "READING_COUNT", "SEED", "PSI_EPSILON",
              "MINIMUM_EDGES", "NULL_RESAMPLES", "NULL_QUANTILE", "DETECTION_SIZES",
              "shift_threshold", "BORROWED_INDEX", "NotSolved", "EnvironmentNotReady",
              "DegenerateReference", "load_bus", "REQUIRED_FEATURES",
              "CANDIDATE_FEATURES", "STOPPED_BELOW", "windowed", "reference_and_current",
              cite="[@yurdakul2020; @efron1979]")
    explain(
        "Give the solutions the narration helpers they print with.",
        "Each solution's demonstration tells its story through `narrator`, `show_table` "
        "and `save_figure`; running the demonstration here needs the same three.",
        "Copies `narrator` and `show_table` verbatim from `exercises/_narrate.py`. "
        "`save_figure` is the one change: in the terminal it writes files under `out/`; "
        "here it applies the same layout and shows the figure in place.",
        "The demonstration steps further down run unchanged.")
    nb.source(NARRATE, "_START", "_Elapsed", "narrator", "show_table")
    explain(
        "Show the lab's figures in place.",
        "In the terminal `save_figure` writes files under `out/`; in a notebook the figure belongs on the page.",
        "Applies the same layout as `_narrate.save_figure` and displays the figure instead of writing it.",
        "The demonstration steps below call it unchanged.")
    nb.code('''
    def save_figure(fig, name, lab, logger=None, width=1000, height=560):
        """The notebook's save_figure: the same layout as exercises/_narrate.py,
        shown in place instead of written to out/lab_0K_<name>.html."""
        show(fig, f"lab_{lab:02d}_{name}", width=width, height=height)
    ''')

    nb.md("""
    ### The data and the unit of analysis

    *Slides: "Position in the course" and "Experimental protocol: the unit of analysis".*
    """)
    explain(
        "Load the telemetry and cut it into the unit every statistic in this module counts.",
        "Readings half a second apart are nearly copies of each other, so 48,290 "
        "readings are not 48,290 observations. The deck's unit is one five-minute window "
        "of one bus, dropping windows with fewer than 300 readings.",
        "Reads the slice, builds the window table with `windowed()`, and splits it into "
        "the reference day (22 January) and the current day (23 January).",
        "45 windows against 35: every interval, test and threshold below works on these "
        "80 numbers per feature, and says so.")
    nb.code('''
    bus = load_bus()
    table = windowed(bus)
    reference, current = reference_and_current()
    print(f"readings in the slice          {len(bus):,}  (vehicle {VEHICLE})")
    print(f"window                         {WINDOW}, at least {MINIMUM_READINGS} readings")
    print(f"reference day {REFERENCE_DAY}   {len(reference)} windows")
    print(f"current day   {CURRENT_DAY}   {len(current)} windows")
    agrees("reference windows", len(reference), 45, 0)
    agrees("current windows", len(current), 35, 0)
    agrees("readings in the slice", len(bus), 48290, 0)
    table.head()
    ''')
    explain(
        "Make the solutions' `load_lab(n)`, and any `from lab_support import …` inside a lab "
        "file, resolve to the code defined in this notebook.",
        "Lab 3 reuses Lab 2's functions and Lab 4 reuses all three, through `load_lab(n)`; the "
        "file version reads the students' unsolved stubs, which raise `NotSolved`. A stub's own "
        "demonstration also imports from `lab_support`, which here is the cells above, not the file.",
        "`load_lab(n)` returns a view of this notebook's definitions, so every solved function is "
        "found by name as soon as its cell has run; a module named `lab_support` is registered "
        "that answers from the same definitions.",
        "The solutions' code runs here exactly as written, and nothing is imported from the lab "
        "files on disk.")
    nb.code('''
    import sys
    import types


    class _Defined:
        """The solved labs: whatever this notebook has defined, looked up by name."""
        def __getattr__(self, name):
            try:
                return globals()[name]
            except KeyError:
                raise AttributeError(name) from None


    def load_lab(number):
        """The notebook's load_lab: the functions defined in this notebook, not labs/0N_*.py."""
        return _Defined()


    _cells = types.ModuleType("lab_support", "lab_support.py, as defined in this notebook's cells")
    _cells.__getattr__ = _Defined().__getattr__
    sys.modules["lab_support"] = _cells
    ''')


# =============================================================================
# Part 1 — estimator uncertainty under label scarcity
# =============================================================================

def part_1() -> None:
    nb.md("""
    ---
    # Part 1 — Estimator uncertainty under label scarcity

    **The question of this part:** how good is my model today, when nobody has
    labelled today's data?

    Module 2 measured the label coverage: complete on the reference day, absent on
    the current day. So accuracy can no longer be computed. It can only be *bought*:
    somebody checks a random sample of predictions by hand. Out of `n` predictions
    checked, `k` are found correct, and `p̂ = k/n` is the observed accuracy. The true
    accuracy `p` is unknown; an interval says which values of it are still plausible.

    *Slides: "The thread: from the bus archive to k correct out of n checked" and
    "The post-deployment evaluation problem". The thread slide's five-step diagram is
    text in boxes and carries no data; it is not redrawn.*
    """)

    # --- slide 10: the standard normal ------------------------------------------
    nb.md("### What a confidence interval is, and where 1.96 comes from\n\n"
          "*Slide: \"Reminder: what a confidence interval is, and where 1.96 comes from\".*")
    explain(
        "Draw the standard normal curve with its middle 95% shaded.",
        "\"95% confident\" is a promise about a procedure: repeat the sampling and 95% of "
        "the intervals contain the truth. The 1.96 is where the middle 95% of a standard "
        "normal ends, and the central limit theorem is what makes a sample proportion "
        "approximately normal [@wasserman2004, ch. 5].",
        "Evaluates the normal density with `scipy.stats.norm` and computes the cut points "
        "as quantiles, rather than typing 1.96.",
        "The z that every interval below uses is a quantile, and it changes with the level: "
        "1.645 at 90%, 2.576 at 99%.")
    nb.figure("standard_normal", '''
    x = np.linspace(-3.5, 3.5, 701)
    z95 = stats.norm.ppf(0.975)
    fig = go.Figure()
    middle = np.abs(x) <= z95
    fig.add_scatter(x=x[middle], y=stats.norm.pdf(x[middle]), fill="tozeroy", mode="none",
                    fillcolor="rgba(46,139,87,0.35)", name="middle 95%")
    fig.add_scatter(x=x, y=stats.norm.pdf(x), mode="lines", line=dict(color=NAVY, width=3),
                    name="standard normal")
    for cut in (-z95, z95):
        fig.add_vline(x=cut, line=dict(color=GREY, dash="dash"),
                      annotation_text=f"{cut:+.2f}", annotation_position="top")
    fig.add_annotation(x=0, y=0.15, text="95%", showarrow=False, font=dict(size=22))
    for side in (-1, 1):
        fig.add_annotation(x=side * 2.7, y=0.04, text="2.5%", showarrow=False)
    fig.update_layout(title="The standard normal: the middle 95% lies between −1.96 and +1.96",
                      xaxis_title="z", yaxis_title="density")
    show(fig, "standard_normal", height=460)
    agrees("z at 95%", z95, 1.96, 2)
    agrees("z at 90%", stats.norm.ppf(0.95), 1.645, 3)
    agrees("z at 99%", stats.norm.ppf(0.995), 2.576, 3)
    ''', slides=["10"], treatment="exact: closed form")

    # --- slide 11: sampling distribution and the CLT ------------------------------
    nb.md("### The sampling distribution and the central limit theorem\n\n"
          "*Slide: \"The sampling distribution and the central limit theorem\".*")
    explain(
        "Write the central limit theorem the slide states.",
        "An interval around an average is possible because the average has its own "
        "distribution, which becomes normal as the sample grows [@wasserman2004, ch. 5].",
        "Builds the limit and the 95% statement as sympy expressions and displays them.",
        "The same expression tells you what the theorem needs: independent observations "
        "of finite variance. The next figure shows the first condition failing on our data.")
    nb.equation("clt", '''
    mu, sigma, n, z = sp.symbols(r"\\mu \\sigma n z", positive=True)
    Xbar = sp.Symbol(r"\\bar{X}_n")
    normal, probability = sp.Function(r"\\mathcal{N}"), sp.Function(r"\\mathbb{P}")
    formula(Xbar, r"\\longrightarrow", normal(mu, sigma**2 / n))
    formula(probability(sp.Le(sp.Abs(Xbar - mu), z * sigma / sp.sqrt(n))), r"\\approx 0.95",
            sp.Eq(z, sp.Float(1.96, 3), evaluate=False))
    ''', slides=["11"])
    explain(
        "Show the theorem at work on a skewed variable from the bus: the payload readings of "
        "the reference day.",
        "The slide draws a skewed population and the means of samples of 5 and of 30. Here "
        "the population is real: 22 January's payload readings, which pile up near empty "
        "and trail off towards full.",
        "Draws 4,000 samples of 5 and of 30 readings *independently, with replacement* "
        "(seed 20200122), averages each, and overlays the normal curve the theorem predicts: "
        "mean μ and standard deviation σ/√n.",
        "At n = 5 the means are still skewed; at n = 30 they are close to normal. Drawing "
        "independently is the assumption the theorem needs — the readings as recorded are "
        "not independent, which is the next slide.")
    nb.figure("clt_on_payload", '''
    population = bus.loc[pd.to_datetime(bus["utc_time"], utc=True).dt.date.astype(str)
                         == REFERENCE_DAY, "payload"].to_numpy(float)
    mu, sigma = population.mean(), population.std(ddof=0)
    rng = np.random.default_rng(SEED)
    draws = rng.choice(population, size=(4000, 30), replace=True)
    panels = [("the population: every reading of 22 January", population, None),
              ("means of samples of n = 5", draws[:, :5].mean(axis=1), 5),
              ("means of samples of n = 30", draws.mean(axis=1), 30)]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[title for title, _, _ in panels],
                        horizontal_spacing=0.07)
    for column, (title, values, size) in enumerate(panels, start=1):
        fig.add_histogram(x=values, histnorm="probability density", nbinsx=30,
                          marker_color=GREY if size is None else BLUE, opacity=0.8,
                          showlegend=False, row=1, col=column)
        if size:
            grid = np.linspace(values.min(), values.max(), 200)
            fig.add_scatter(x=grid, y=stats.norm.pdf(grid, mu, sigma / math.sqrt(size)),
                            mode="lines", line=dict(color=ORANGE, width=2.5),
                            name="normal curve the theorem predicts", showlegend=column == 2,
                            row=1, col=column)
        fig.update_xaxes(title_text="kilograms", row=1, col=column)
    fig.update_layout(title="The central limit theorem on real payload readings "
                            "(skewness of the population "
                            f"{stats.skew(population):.2f})",
                      legend=dict(orientation="h", y=-0.2))
    show(fig, "clt_on_payload", height=430)
    print(f"population: {len(population):,} readings, mean {mu:.1f} kg, sd {sigma:.1f} kg, "
          f"skewness {stats.skew(population):.2f}")
    for size in (5, 30):
        means = draws[:, :size].mean(axis=1)
        print(f"means of n = {size:>2}: skewness {stats.skew(means):.2f}, sd {means.std():.1f} kg "
              f"against sigma/sqrt(n) = {sigma / math.sqrt(size):.1f} kg")
    ''', slides=["11"], treatment="lab data: the slide's constructed skewed population "
                                  "replaced by 22 January's payload readings")

    # --- slide 12: effective sample size -------------------------------------------
    nb.md("### Effective sample size under autocorrelation\n\n"
          "*Slide: \"Effective sample size under autocorrelation\".*")
    explain(
        "Measure how much independent information 48,290 readings hold.",
        "Every interval divides by the square root of a sample size. If consecutive "
        "readings nearly repeat each other, that size overstates the information and every "
        "interval is too narrow [@bayley1946].",
        "Sorts the readings by time, measures the lag-1 autocorrelation ρ of speed, and "
        "applies the first-order approximation n_eff ≈ n(1 − ρ)/(1 + ρ).",
        "About 73 effective observations — the same order as the 80 windows, reached from "
        "the opposite direction. That is why the unit of analysis is the window.")
    nb.code('''
    in_order = bus.assign(_t=pd.to_datetime(bus["utc_time"], utc=True)).sort_values("_t")
    rho = float(in_order["speed"].autocorr(1))
    n_readings = len(bus)
    n_effective = n_readings * (1 - round(rho, 3)) / (1 + round(rho, 3))
    agrees("lag-1 autocorrelation of speed, in time order", rho, 0.997, 3)
    agrees("effective sample, n(1 - rho)/(1 + rho)", n_effective, 73, 0)
    agrees("share of information kept, (1 - rho)/(1 + rho), per cent",
           100 * (1 - 0.997) / (1 + 0.997), 0.15, 2)
    ''')
    explain(
        "Derive the factor (1 − ρ)/(1 + ρ) rather than quote it.",
        "The slide states the approximation; the paper it cites treats the general and the "
        "continuous-time case [@bayley1946, § 10]. Deriving the discrete first-order case "
        "here removes any doubt about where the factor comes from.",
        "For a series whose correlation at lag h is ρ^h, the variance of the mean of n "
        "values is (σ²/n²) Σᵢ Σⱼ ρ^|i−j|: n pairs on the diagonal and 2(n − h) pairs at each "
        "lag h. The effective sample is the size an independent sample would need for the "
        "same variance, n² / Σᵢ Σⱼ ρ^|i−j|. For large n the double sum is n(1 + 2 Σ ρ^h), and "
        "sympy sums the geometric series.",
        "The exact finite-n value and the approximation agree to within one observation at "
        "n = 48,290.")
    nb.equation("effective_sample", '''
    rho_, n_, h = sp.symbols(r"\\rho n h", positive=True)
    geometric = sp.summation(rho_**h, (h, 1, sp.oo))      # rho/(1 - rho), for rho < 1
    if isinstance(geometric, sp.Piecewise):
        geometric = geometric.args[0][0]
    n_eff = sp.simplify(n_ / (1 + 2 * geometric))
    formula(sp.Eq(sp.Symbol(r"n_{\\mathrm{eff}}"), n_eff, evaluate=False),
            r"\\rho = 0.997,\\ n = 48{,}290 \\;\\Rightarrow\\; n_{\\mathrm{eff}} \\approx "
            + f"{float(n_eff.subs({rho_: 0.997, n_: 48290})):.0f}")
    lags = np.arange(1, 48290)
    exact = 48290**2 / (48290 + 2 * np.sum((48290 - lags) * 0.997**lags))
    print(f"exact at n = 48,290: {exact:.1f}   first-order approximation: "
          f"{float(n_eff.subs({rho_: 0.997, n_: 48290})):.1f}")
    ''', slides=["12"])
    explain(
        "Draw the share of information kept against ρ, with the bus marked on it.",
        "The slide's chart shows how fast the factor collapses as ρ approaches 1.",
        "Plots 100 (1 − ρ)/(1 + ρ) for ρ from 0 to 1 and marks the measured ρ.",
        "At ρ = 0.997 only 0.15% of the nominal readings count.")
    nb.figure("information_kept", '''
    grid = np.linspace(0, 0.999, 400)
    fig = go.Figure()
    fig.add_scatter(x=grid, y=100 * (1 - grid) / (1 + grid), mode="lines",
                    line=dict(color=NAVY, width=3), name="(1 − ρ)/(1 + ρ)")
    fig.add_scatter(x=[rho], y=[100 * (1 - rho) / (1 + rho)], mode="markers+text",
                    marker=dict(color=RED, size=12), textposition="top left",
                    text=[f"bus speed: ρ = {rho:.3f}, {100 * (1 - rho) / (1 + rho):.2f}% kept, "
                          f"{n_readings:,} → {n_effective:.0f}"], name="measured")
    fig.update_layout(title="Share of information kept, (1 − ρ)/(1 + ρ)",
                      xaxis_title="lag-1 autocorrelation ρ", yaxis_title="per cent kept",
                      showlegend=False)
    show(fig, "information_kept", height=440)
    ''', slides=["12"], treatment="exact: closed form, with the measured ρ")

    # --- slides 13-15: Wald, Wilson ---------------------------------------------------
    nb.md("### The Wald interval, and the Wilson score interval\n\n"
          "*Slides: \"The Wald interval: definition and failure modes\", \"The Wilson score "
          "interval: definition\" and \"Worked comparison: 34 of 40, and 40 of 40\".*")
    explain(
        "Write the two interval formulas the slides define.",
        "The Wald interval puts an error bar on the estimate; the Wilson interval keeps "
        "every true rate a score test would not reject [@wilson1927; @brown2001; @agresti1998].",
        "Builds each formula in sympy, then checks symbolically that the slide's Wilson "
        "formula equals the multiplied-out form the Lab 1 solution codes "
        "(centre = (k + z²/2)/(n + z²)).",
        "The formula on the slide and the code in the solution are provably the same "
        "function; the difference simplifies to zero.")
    nb.equation("wald_and_wilson", '''
    k, n, z = sp.symbols("k n z", positive=True)
    p_hat = sp.Symbol(r"\\hat{p}")
    display(Math(r"\\mathrm{CI}_{\\mathrm{Wald}} = "
                 + pm(p_hat, z * sp.sqrt(p_hat * (1 - p_hat) / n))
                 + r",\\qquad \\hat{p} = k/n"))
    wilson_centre_slide = (p_hat + z**2 / (2 * n)) / (1 + z**2 / n)
    wilson_half_slide = z * sp.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    display(Math(r"\\mathrm{CI}_{\\mathrm{Wilson}} = \\frac{"
                 + pm(p_hat + z**2 / (2 * n), z * sp.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)))
                 + r"}{1 + \\frac{z^2}{n}}"))
    # The Lab 1 solution codes the same interval multiplied out by n:
    centre_code = (k + z**2 / 2) / (n + z**2)
    half_code = z / (n + z**2) * sp.sqrt(k * (n - k) / n + z**2 / 4)
    print("slide centre - code centre, simplified:",
          sp.simplify(wilson_centre_slide.subs(p_hat, k / n) - centre_code))
    print("slide half-width² - code half-width², simplified:",
          sp.simplify(wilson_half_slide.subs(p_hat, k / n)**2 - half_code**2))
    ''', slides=["13", "14"])


# =============================================================================

def main() -> int:
    front_matter()
    setup()
    part_1()
    from sections import lab1, part2, part3, part4, closing
    lab1.build(nb, explain)
    part2.build(nb, explain)
    part3.build(nb, explain)
    part4.build(nb, explain)
    closing.build(nb, explain)
    nb.md("""
    ---
    ## Where this notebook and the slides differ, and why

    The slides are the master and are not changed. Where a number computed here does
    not match the number a slide prints, the notebook printed both at that point, with
    the reason. The table gathers every one of them, as they were printed in this run.

    Differences in wording, which the table cannot hold:

    - **The shift bound.** "The alarm rule" and "How a threshold is measured" describe a
      measured shift threshold and call a textbook 2 standard deviations refused, while
      the decision slides judge the shift against the fixed 2.0. The notebook leads with
      the fixed 2.0, prints the measured bound beside it, and sets the two side by side in
      Appendix F, as the deck's own Appendix F does.
    - **Constructed values under bus labels.** The entropy and divergence charts (speed
      bins in km/h, "22/23 Jan"), the cumulative-curve chart ("reference day, 40
      windows", 3 to 10 m/s) and the bin-edge diagram (900 to 1,400 kg) draw constructed
      values; the notebook redraws each on the real windows and says so where it does.
    """)
    explain(
        "Gather every number this notebook printed beside a slide's number.",
        "A reader should find every difference between the notebook and the slides in one "
        "place, with its reason, without searching the notebook.",
        "Tabulates what each `beside()` call recorded during this run: the quantity, the value "
        "computed here, the value on the slide or in the archive, and why they differ.",
        "Every row is explained where it first appears; none of them is a change to the slides.")
    nb.code('''
    with pd.option_context("display.max_colwidth", None):     # the reasons, in full
        display(pd.DataFrame(DIFFERENCES, columns=["quantity", "computed here",
                                                   "the slide or the archive", "why they differ"]))
    ''')
    nb.write(OUTPUT)
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    if "--no-run" not in sys.argv:
        execute(OUTPUT, EXERCISES)
        print(f"executed {OUTPUT.relative_to(ROOT)} in {EXERCISES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
