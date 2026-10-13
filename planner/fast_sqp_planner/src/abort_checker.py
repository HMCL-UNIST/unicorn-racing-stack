"""Abort logic for rolling-horizon overtake planner.

Two abort sources:

  SAFETY:   opponent observation lies outside the GP covariance envelope.
            → fall back to TRAILING (avoid blind committing to a stale plan).

  PERFORMANCE: the overtake-time integral along the refined (d, v) is slower
            than just trailing the opponent along GP_v by more than T_margin.
            → fall back to TRAILING (don't get stuck in a slow path).

Chattering is suppressed with consecutive-cycles hysteresis + cooldown window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import rospy


class AbortReason(Enum):
    NONE = 'none'
    SAFETY = 'safety'
    PERFORMANCE = 'performance'


@dataclass
class AbortConfig:
    performance_margin_s: float = 0.1    # T_margin
    consecutive_cycles: int = 3          # N cycles before actual abort trips
    cooldown_s: float = 2.0              # re-entry lockout after abort
    safety_sigma_multiplier: float = 3.0 # observation outside mu±k*sigma -> abort


@dataclass
class AbortState:
    safety_streak: int = 0
    performance_streak: int = 0
    cooldown_until: Optional[rospy.Time] = None
    last_reason: AbortReason = AbortReason.NONE


class AbortChecker:
    def __init__(self, config: AbortConfig):
        self.config = config
        self.state = AbortState()

    def in_cooldown(self, now: rospy.Time) -> bool:
        return (self.state.cooldown_until is not None
                and now < self.state.cooldown_until)

    # ---- individual checks ----------------------------------------------
    def _check_safety(self,
                      opp_obs_d: Optional[float],
                      opp_obs_s: Optional[float],
                      gp_d_at_s: Optional[float],
                      gp_d_var_at_s: Optional[float]) -> bool:
        """True -> safety violation detected this cycle."""
        if None in (opp_obs_d, opp_obs_s, gp_d_at_s, gp_d_var_at_s):
            return False
        if gp_d_var_at_s <= 0.0:
            return False
        sigma = np.sqrt(gp_d_var_at_s)
        return abs(opp_obs_d - gp_d_at_s) > self.config.safety_sigma_multiplier * sigma

    def _check_performance(self,
                           s_grid: np.ndarray,
                           v_grid: np.ndarray,
                           gp_v_at_s: np.ndarray) -> bool:
        """True -> performance abort triggered this cycle.

        Args:
            s_grid: arc-lengths over the RoC section of the refined path.
            v_grid: refined speeds along s_grid (must be strictly positive).
            gp_v_at_s: GP-estimated opponent speed sampled at s_grid.
        """
        if s_grid.size < 2 or v_grid.size != s_grid.size:
            return False
        ds = np.diff(s_grid)
        v_ot_mid = 0.5 * (v_grid[:-1] + v_grid[1:])
        v_opp_mid = 0.5 * (gp_v_at_s[:-1] + gp_v_at_s[1:])
        v_ot_mid = np.maximum(v_ot_mid, 1e-3)
        v_opp_mid = np.maximum(v_opp_mid, 1e-3)
        t_ot = float(np.sum(ds / v_ot_mid))
        t_trail = float(np.sum(ds / v_opp_mid))
        return t_ot > (t_trail + self.config.performance_margin_s)

    # ---- cycle entry ----------------------------------------------------
    def step(self,
             now: rospy.Time,
             opp_obs_d: Optional[float],
             opp_obs_s: Optional[float],
             gp_d_at_s: Optional[float],
             gp_d_var_at_s: Optional[float],
             s_grid: np.ndarray,
             v_grid: np.ndarray,
             gp_v_at_s: np.ndarray) -> AbortReason:
        """Advance streak counters and decide whether to abort this cycle."""
        if self.in_cooldown(now):
            return AbortReason.NONE

        safety_hit = self._check_safety(opp_obs_d, opp_obs_s,
                                        gp_d_at_s, gp_d_var_at_s)
        perf_hit = False if safety_hit else self._check_performance(
            s_grid, v_grid, gp_v_at_s)

        self.state.safety_streak = (self.state.safety_streak + 1
                                    if safety_hit else 0)
        self.state.performance_streak = (self.state.performance_streak + 1
                                         if perf_hit else 0)

        reason = AbortReason.NONE
        if self.state.safety_streak >= self.config.consecutive_cycles:
            reason = AbortReason.SAFETY
        elif self.state.performance_streak >= self.config.consecutive_cycles:
            reason = AbortReason.PERFORMANCE

        if reason != AbortReason.NONE:
            self.state.cooldown_until = now + rospy.Duration(self.config.cooldown_s)
            self.state.safety_streak = 0
            self.state.performance_streak = 0
            self.state.last_reason = reason
        return reason

    def reset(self) -> None:
        self.state = AbortState()
