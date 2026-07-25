# 05_inputs.py
# The original AIM introduces time-dependent external inputs B1(t) and B2(t),
# but it does not specify their exact temporal form. Here, we use a rectangular
# negative input as a simple controlled approximation. During a fixed interval,
# the external event tilts the energy landscape towards higher negative affect.
# Outside this interval, the input is zero.
#
# Mathematically:
# B2(t) = amplitude, if start <= t < start + duration
# B2(t) = 0, otherwise
# %%
import sys
from pathlib import Path

# This notebook checks that the rectangular negative input works correctly
# before we use it inside the Euler-Maruyama simulation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

# We import the reusable input function instead of writing it again.
from aim.inputs import rectangular_negative_input

#test to see if the function works
print("Before:", rectangular_negative_input(1, 4, 2, 3))
print("Start:", rectangular_negative_input(2, 4, 2, 3))
print("During:", rectangular_negative_input(4, 4, 2, 3))
print("End:", rectangular_negative_input(5, 4, 2, 3))

