"""Reusable external-input functions for the Affective Ising Model."""


def rectangular_negative_input(t, amplitude, start, duration):
    """Return the intensity of a rectangular negative input at time t."""
    end=start + duration
    if start <= t < end:
        return amplitude
    else:
        return 0.0
