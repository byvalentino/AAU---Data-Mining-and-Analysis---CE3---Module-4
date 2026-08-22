"""What every Module 4 lab needs: the unsolved marker and the real telemetry.

Module 4 is the one module that needs no model and no generated data. It works
on the archive itself -- shuttle VJRD1A10224000055 on 22 and 23 January 2020,
which identifies nobody -- because the question is whether the second day is
different from the first, and that question is about the data.
"""
from __future__ import annotations

import pathlib

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
BUS_SLICE = HERE / "data" / "bus_slice.csv.gz"

# The grain, fixed here so that every lab and every check uses the same one, and
# so that it can be printed beside any number computed from it.
VEHICLE = "VJRD1A10224000055"
WINDOW = "5min"
REFERENCE_DAY, CURRENT_DAY = "2020-01-22", "2020-01-23"
MINIMUM_READINGS = 300

# How many readings the committed slice holds, both days together. Lab 4 needs it
# to say what the reading grain would cost, and a number that decides an answer
# belongs beside the code that uses it rather than inside it.
READING_COUNT = 48_290

# One seed for the whole module, so two runs of a lab agree and a check can
# reproduce what a student saw.
SEED = 20200122

# The floor put under an empty bin's share before the logarithm sees it. Without
# it a single empty bin makes the index infinite. With it the index at twenty
# bins is mostly this constant rather than the data, which is why it is named
# here instead of being typed into five separate files.
PSI_EPSILON = 1e-6

# An index needs at least two bins to compare anything, so at least three edges.
# Below that there is nothing to measure and the honest answer is to say so.
MINIMUM_EDGES = 3

# How the materiality threshold is derived rather than borrowed. Three choices,
# named here so that every number computed from them can be printed beside them,
# which is standing rule 2:
#
#   NULL_RESAMPLES   how many times the reference is compared against a resample
#                    of itself to build the null distribution of the index. One
#                    thousand is enough for its upper tail to settle at this
#                    sample size and cheap enough to run inside a check.
#   NULL_QUANTILE    which point of that null distribution becomes the threshold.
#                    0.99 means "a value the index passes once in a hundred
#                    comparisons in which nothing at all has changed" -- one
#                    false alarm in a hundred, per feature, which with five
#                    features is one every twenty runs. That is a stated error
#                    rate; 0.25 out of a credit-scoring handbook is not.
#   SEED             so that two people deriving the threshold get the same one.
NULL_RESAMPLES = 1000
NULL_QUANTILE = 0.99

# The injected sizes the detection-limit sweep walks, in reference standard
# deviations: nought to 1.50 in steps of 0.05. Two choices, both of which decide
# the answer that comes out. The step is the resolution the limit is quoted to --
# a limit of 0.40 means 0.40 rather than 0.4123, and saying otherwise would claim
# a precision the grid does not have. The top of the grid is the size the fixed
# control injects, so the sweep ends exactly where the single-size control sits
# and shows how far above the limit that size was.
DETECTION_SIZES = tuple(round(0.05 * step, 2) for step in range(0, 31))

# The threshold credit scoring hands out, kept in one place so that the module
# can measure what it costs rather than use it. NOTHING in this module grades
# against it, and Lab 2's check refuses a threshold that sits near it: it is on
# the slides as the thing not to do.
BORROWED_INDEX = 0.25


class NotSolved(Exception):
    """A lab stub raises this. The check turns it into exit code 2.

    It is not an error. It means "you have not written this yet", which is a
    different state from "you wrote it and it is wrong", and the checks say so.
    """


class EnvironmentNotReady(Exception):
    """The tools or the data this module needs are missing. The checks exit 3.

    A third state, separate from the other two, because "this machine is not set
    up" is not "your code is wrong". A student told the second while the first is
    true goes hunting for a bug that was never there. Every check imports it from
    here, so the modules share one class rather than several with the same name.
    """


class DegenerateReference(Exception):
    """The reference cannot be binned, so the index has no value to report.

    Raised rather than returning nought. On this archive the human_driven column
    is nought in 39 of the 45 reference windows, so its quantile edges collapse
    to a single bin, both shares are 1.0 and the index is exactly nought for a
    column that moved by 1.24 reference standard deviations. An index of exactly
    nought for a column that plainly moved is not "no change", it is "no
    measurement", and code that cannot tell the two apart will report the first
    when it means the second.
    """


def load_lab(number: int):
    """Import another lab by its number, so one lab can build on the last.

    Lab 3 uses Lab 2's four functions; Lab 4 uses Lab 3's. Importing by number
    rather than by module name means the same line works whether the file holds
    your own attempt or the shipped solution.

        from lab_support import load_lab
        index = load_lab(2).population_stability_index
    """
    import importlib.util

    matches = sorted((HERE / "labs").glob(f"{number:02d}_*.py"))
    if not matches:
        raise FileNotFoundError(f"no lab {number:02d} in {HERE / 'labs'}")
    specification = importlib.util.spec_from_file_location(f"lab{number:02d}", matches[0])
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def load_bus() -> pd.DataFrame:
    """Real vehicle telemetry, both days, one shuttle. 48,290 readings.

    This is the whole population Module 4's grain uses -- the same rows the
    slides are measured from -- so a number computed here and a number on a
    slide are computed from identical data.
    """
    if not BUS_SLICE.exists():
        raise EnvironmentNotReady(
            f"{BUS_SLICE} is missing, so there is nothing to measure. It ships "
            "with the repository; re-clone or re-run `bash setup.sh`.")
    return pd.read_csv(BUS_SLICE, low_memory=False)


# The five columns Lab 4 is graded on, and why each is there rather than because
# it was available:
#
#   mean_speed, sd_speed     the two window moments of the model's speed input
#   mean_payload             the target stand-in -- continuous, label-free, and
#                            measurable every window; it is NOT the aboard label
#                            Module 3's service predicts
#   sd_payload               the target's spread, because a mean can sit still
#                            while the shape underneath it moves
#   human_driven             the suspected cause, the share of readings a person
#                            drove
REQUIRED_FEATURES = ["mean_speed", "sd_speed", "sd_payload", "human_driven",
                     "mean_payload"]

# And five more the same window offers, for students who want to add their own.
# They are measured here so that adding one costs a name in a list rather than a
# rewrite of the loader. Nothing grades them; Lab 4's check prints them unjudged.
CANDIDATE_FEATURES = ["max_speed", "share_stopped", "n_readings", "mileage_delta",
                      "mean_battery"]

# What counts as stopped, in metres per second. A choice, named here because it
# decides share_stopped: the sensor reports small non-zero speeds at a standstill.
STOPPED_BELOW = 0.1


def windowed(bus: pd.DataFrame = None) -> pd.DataFrame:
    """One row per five-minute window: the five graded features, and five more.

    Why this vehicle only: the other shuttle ran on the first day and not the
    second. Pool both on day one and compare against one on day two, and part of
    what you would call drift is a vehicle going to the depot. An earlier version
    of this course's plan did exactly that and got the sign of the target's
    movement wrong.

    Why at least 300 readings: a window holding a handful of readings is the edge
    of the day rather than a window, and the two days would not be comparable.

    REQUIRED_FEATURES are what Lab 4 is graded on and why each is there is in the
    comment above. CANDIDATE_FEATURES are the invitation: pass any list you like
    to verdict(), say in one line why each column is in it, and the check will
    measure the five and print yours beside them without judging them.
    """
    bus = load_bus() if bus is None else bus
    one = bus[bus["vehicle_id"] == VEHICLE].copy()
    one["_t"] = pd.to_datetime(one["utc_time"], utc=True)
    one["window"] = one["_t"].dt.floor(WINDOW)

    table = one.groupby("window").agg(
        mean_speed=("speed", "mean"),
        sd_speed=("speed", "std"),
        sd_payload=("payload", "std"),
        human_driven=("mode", lambda values: float((values == "manual").mean())),
        mean_payload=("payload", "mean"),
        # The count behind human_driven, kept because Lab 1's second picture is
        # drawn on it: k manual readings out of n is the bought-truth arithmetic,
        # measured rather than imagined.
        manual_readings=("mode", lambda values: int((values == "manual").sum())),
        # From here down: offered, not graded.
        max_speed=("speed", "max"),
        share_stopped=("speed", lambda values: float((values.abs() < STOPPED_BELOW).mean())),
        n_readings=("speed", "size"),
        mileage_delta=("mileage", lambda values: float(values.max() - values.min())),
        mean_battery=("battery_level", "mean"),
    ).reset_index()
    table["day"] = table["window"].dt.date.astype(str)
    return table[table["n_readings"] >= MINIMUM_READINGS].reset_index(drop=True)


def reference_and_current():
    """The two days, as (reference, current) frames of windows."""
    table = windowed()
    return (table[table["day"] == REFERENCE_DAY].reset_index(drop=True),
            table[table["day"] == CURRENT_DAY].reset_index(drop=True))
