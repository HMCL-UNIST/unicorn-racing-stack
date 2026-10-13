"""
Frenet wrap-around utilities for closed-loop tracks.

All functions handle the circular nature of Frenet s-coordinate.
Import: from track_3d_validator import circular_s_dist, signed_s_dist, unwrap_s, in_s_range
"""
import numpy as np


def circular_s_dist(s_a, s_b, max_s):
    """Unsigned shortest distance between two s values on a closed loop.

    Args:
        s_a, s_b: scalar or array of s values in [0, max_s)
        max_s: track length (raceline_length)

    Returns:
        Distance(s), always >= 0, <= max_s/2.
    """
    diff = np.abs(np.asarray(s_a) - np.asarray(s_b))
    return np.minimum(diff, max_s - diff)


def signed_s_dist(s_from, s_to, max_s):
    """Signed shortest distance from s_from to s_to (forward = positive).

    Returns positive if s_to is ahead of s_from by the shorter path,
    negative if behind. Handles wrap-around.

    Args:
        s_from, s_to: scalar or array of s values in [0, max_s)
        max_s: track length

    Returns:
        Signed distance in (-max_s/2, max_s/2].
    """
    diff = (np.asarray(s_to) - np.asarray(s_from)) % max_s
    return np.where(diff <= max_s / 2, diff, diff - max_s)


def unwrap_s(s_arr, max_s):
    """Unwrap an s-array across track wraparound so it becomes monotonic.

    Example: [84.5, 85.2, 0.3, 1.0, 2.1] (max_s=85.83)
        →    [84.5, 85.2, 85.93, 86.63, 87.73]
    Use before np.diff / np.gradient on s values.

    Args:
        s_arr: 1D array of s values
        max_s: track length

    Returns:
        Monotonically increasing array (same shape).
    """
    s_arr = np.asarray(s_arr)
    if len(s_arr) < 2:
        return s_arr.copy()
    ds = np.diff(s_arr)
    ds = np.where(ds < -max_s / 2, ds + max_s, ds)
    ds = np.where(ds > max_s / 2, ds - max_s, ds)
    return np.concatenate([[float(s_arr[0])], s_arr[0] + np.cumsum(ds)])


def in_s_range(s, s_start, s_end, max_s):
    """Check if s lies within [s_start, s_end] on a closed loop (handles wrap).

    If s_start > s_end, the range wraps around max_s (e.g., s_start=85, s_end=1
    means [85, max_s) ∪ [0, 1]).

    Args:
        s: scalar or array to test
        s_start, s_end: range bounds (scalars)
        max_s: track length

    Returns:
        bool or bool array of same shape as s.
    """
    s = np.asarray(s)
    if s_start <= s_end:
        return (s >= s_start) & (s <= s_end)
    else:
        return (s >= s_start) | (s <= s_end)


def s_forward_add(s, ds, max_s):
    """Advance s by ds (can be negative) with wrap-around.

    Args:
        s: current s (scalar or array)
        ds: increment (scalar or array, same shape)
        max_s: track length

    Returns:
        (s + ds) modulo max_s, always in [0, max_s).
    """
    return (np.asarray(s) + np.asarray(ds)) % max_s
