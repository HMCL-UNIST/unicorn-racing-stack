"""Warm-start utilities for rolling-horizon SQP.

Given previous cycle's solution d_{k-1}(s) defined on s-grid S_{k-1}, and the
ego having advanced by Δs = v_ego * dt during the solve-tick, we want an
initial guess for cycle k defined on a (possibly different) s-grid S_k.

Implementation is a simple linear interpolation with s-axis shift:
    d_init^k(s) = d_{k-1}(s + Δs)
clamped at the edges. If the previous solution is None, the caller should use
the analytic fallback (constant apex guess) from the original SQP spliner.
"""

from __future__ import annotations

import numpy as np


def shift_solution(prev_d: np.ndarray,
                   prev_s: np.ndarray,
                   new_s: np.ndarray,
                   delta_s_shift: float) -> np.ndarray:
    """Return prev solution interpolated onto new_s, shifted by +delta_s_shift.

    Args:
        prev_d: previous solution values at prev_s (shape N_prev).
        prev_s: arc-length grid of previous solution (monotonic, shape N_prev).
        new_s:  new arc-length grid (monotonic, shape N_new).
        delta_s_shift: how far forward the ego has moved since last solve.

    Returns:
        np.ndarray of shape new_s.shape — prev solution sampled at new_s+shift,
        clamped via numpy.interp behavior (boundary values held constant).
    """
    if prev_d is None or prev_s is None or prev_d.size == 0:
        raise ValueError("shift_solution called with empty prev solution")
    query = new_s + delta_s_shift
    return np.interp(query, prev_s, prev_d)


def analytic_fallback(n_knots: int, apex: float) -> np.ndarray:
    """Analytic initial guess used when no previous solution exists.

    Mirrors the original SQP planner behavior (sqp_planner):
    constant apex across all knots.
    """
    return np.full(n_knots, apex, dtype=float)
