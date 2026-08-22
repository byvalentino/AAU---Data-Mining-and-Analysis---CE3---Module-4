"""An independent implementation of Wilson's interval, for the check to compare against.

Deliberately not imported by the labs. A check that compares a student's code
against the student's own code proves nothing, and the published formula is short
enough to write twice (Wilson, 1927).

It lives in verify/ rather than in the exercises root, and that is not tidiness.
labs/01_how_sure_are_you.py puts the exercises root on sys.path, so from the root
this file was importable from inside the lab -- and a three-line delegate to it
made check 1 exit 0 without the student writing the formula at all. A reference
implementation a student can import is not a reference implementation.
"""
from __future__ import annotations

import math

Z_95 = 1.959963984540054


def wilson_reference(successes: int, trials: int, z: float = Z_95):
    denominator = trials + z * z
    centre = (successes + z * z / 2) / denominator
    half = (z / denominator) * math.sqrt(
        successes * (trials - successes) / trials + z * z / 4)
    return (centre - half, centre + half)
