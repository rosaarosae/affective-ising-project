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
def rectangular_negative_input(t, amplitude, start, duration):
    """Return the intensity of a rectangular negative input at time t."""
    end=start + duration
    if start <= t < end:
        return amplitude
    else:
        return 0.0
#test to see if the function works
print("Before:", rectangular_negative_input(1, 4, 2, 3))
print("Start:", rectangular_negative_input(2, 4, 2, 3))
print("During:", rectangular_negative_input(4, 4, 2, 3))
print("End:", rectangular_negative_input(5, 4, 2, 3))


