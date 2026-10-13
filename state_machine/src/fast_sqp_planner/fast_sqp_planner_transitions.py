from __future__ import annotations

import os
from typing import TYPE_CHECKING, Tuple

import rospkg as _rospkg
import sys as _sys
_sm_src = os.path.join(_rospkg.RosPack().get_path('state_machine'), 'src')
if _sm_src not in _sys.path:
    _sys.path.insert(0, _sm_src)

from states_types import StateType
import rospy

if TYPE_CHECKING:
    from state_machine_node import StateMachine


def _has_obs_ahead(sm) -> bool:
    """Check if any obstacle is ahead within interest_horizon_m."""
    for obs in sm.cur_obstacles_in_interest:
        gap = (obs.s_center - sm.cur_s) % sm.track_length
        if gap < sm.track_length / 2 and gap < sm.interest_horizon_m:
            return True
    return False


def _has_obs_nearby(sm) -> bool:
    """Check if any obstacle is nearby (ahead, beside, or just behind) within 3m s-distance."""
    for obs in sm.cur_obstacles_in_interest:
        gap_fwd = (obs.s_center - sm.cur_s) % sm.track_length
        gap_bwd = (sm.cur_s - obs.s_center) % sm.track_length
        gap = min(gap_fwd, gap_bwd)
        if gap < 3.0:
            return True
    return False


# obstacle is blocking ahead (not beside/behind)
def _has_obs_blocking_ahead(sm, min_s_gap: float = 0.5) -> bool:
    """True only if obstacle center is > min_s_gap ahead in s-direction."""
    for obs in sm.cur_obstacles_in_interest:
        gap = (obs.s_center - sm.cur_s) % sm.track_length
        if gap < sm.track_length / 2 and gap < sm.interest_horizon_m and gap > min_s_gap:
            return True
    return False


def GlobalTrackingTransition(state_machine) -> Tuple[StateType, StateType]:
    """GB_TRACK state transitions."""
    ot_sector = state_machine._check_ot_sector()
    has_near = _has_obs_ahead(state_machine)
    n_obs = len(state_machine.cur_obstacles_in_interest)
    rospy.logwarn_throttle(2.0, f"[TRANS] GB: ot={ot_sector} near={has_near} n_obs={n_obs}")

    # ot_flag=true + obstacle close enough → OVERTAKE
    if ot_sector and has_near:
        min_gap = float('inf')
        for obs in state_machine.cur_obstacles_in_interest:
            gap = (obs.s_center - state_machine.cur_s) % state_machine.track_length
            if gap < state_machine.track_length / 2:
                min_gap = min(min_gap, gap)
        engage_dist = max(state_machine.cur_vs * state_machine.ot_engage_ttc_s,
                          state_machine.ot_engage_min_m)
        if min_gap < engage_dist:
            return StateType.OVERTAKE, StateType.OVERTAKE
        # obstacle detected but far → stay GB_TRACK
        return StateType.GB_TRACK, StateType.GB_TRACK

    # ot_sector off + obstacle very close → TRAILING (GB)
    if has_near and state_machine._check_enemy_in_front():
        return StateType.TRAILING, StateType.GB_TRACK

    # no obstacle but off raceline → RECOVERY (quintic re-join)
    if not has_near:
        close_to_gb = state_machine._check_close_to_raceline(
            state_machine._get_adaptive_close_threshold())
        if not close_to_gb:
            return StateType.RECOVERY, StateType.RECOVERY

    return StateType.GB_TRACK, StateType.GB_TRACK


def OvertakingTransition(state_machine) -> Tuple[StateType, StateType]:
    """OVERTAKE state transitions."""
    has_ahead = _has_obs_ahead(state_machine)
    has_nearby = _has_obs_nearby(state_machine)
    ot_sector = state_machine._check_ot_sector()
    path_collision = state_machine._check_ot_path_collision()
    # too_close = state_machine._opponent_dist < state_machine._opponent_dist_threshold
    # lateral margin replaces raw euclidean too_close
    ot_margin = state_machine._get_ot_path_min_margin()
    too_close = ot_margin < state_machine.ot_margin_safe_m

    # # obstacle ahead + path collision or too close → TRAILING
    # if has_ahead and (path_collision or too_close):
    #     return StateType.TRAILING, StateType.OVERTAKE
    # only trigger trailing when obstacle is blocking ahead (not beside)
    has_blocking = _has_obs_blocking_ahead(state_machine, state_machine.ot_blocking_s_gap_m)
    if has_blocking and (path_collision or too_close):
        return StateType.TRAILING, StateType.OVERTAKE

    # ot_flag turned off → back to GB
    if not ot_sector:
        return StateType.GB_TRACK, StateType.GB_TRACK

    # obstacle ahead or nearby → stay OVERTAKE, reset clear counter
    if has_ahead or has_nearby:
        state_machine.overtaking_ttl_count = 0
        return StateType.OVERTAKE, StateType.OVERTAKE

    # no obstacle — count consecutive clear ticks
    state_machine.overtaking_ttl_count += 1
    if state_machine.overtaking_ttl_count < state_machine.overtaking_ttl_count_threshold:
        return StateType.OVERTAKE, StateType.OVERTAKE

    # sustained clear → GB_TRACK (on raceline) or RECOVERY (off raceline)
    state_machine.overtaking_ttl_count = 0
    close_to_gb = state_machine._check_close_to_raceline(
        state_machine._get_adaptive_close_threshold())
    if close_to_gb:
        return StateType.GB_TRACK, StateType.GB_TRACK

    # obstacle gone but off raceline → RECOVERY (quintic re-join)
    return StateType.RECOVERY, StateType.RECOVERY


def TrailingTransition(state_machine) -> Tuple[StateType, StateType]:
    """TRAILING state transitions."""
    has_ahead = _has_obs_ahead(state_machine)
    has_nearby = _has_obs_nearby(state_machine)
    ot_sector = state_machine._check_ot_sector()

    # no obstacle nearby → GB_TRACK (on raceline) or RECOVERY (off raceline)
    if not has_nearby:
        close_to_gb = state_machine._check_close_to_raceline(
            state_machine._get_adaptive_close_threshold())
        if close_to_gb:
            return StateType.GB_TRACK, StateType.GB_TRACK
        # off raceline → RECOVERY (quintic re-join)
        return StateType.RECOVERY, StateType.RECOVERY

    # obstacle ahead + path clear → back to OVERTAKE
    if has_ahead and ot_sector and state_machine._ot_fast_sqp_wpnts is not None:
        path_collision = state_machine._check_ot_path_collision()
        # too_close = state_machine._opponent_dist < state_machine._opponent_dist_threshold
        # lateral margin replaces raw euclidean too_close
        ot_margin = state_machine._get_ot_path_min_margin()
        too_close = ot_margin < state_machine.ot_margin_safe_m
        if not path_collision and not too_close:
            return StateType.OVERTAKE, StateType.OVERTAKE

    # stay trailing
    if ot_sector and state_machine._ot_fast_sqp_wpnts is not None:
        return StateType.TRAILING, StateType.OVERTAKE
    return StateType.TRAILING, StateType.GB_TRACK


def StartTransition(state_machine) -> Tuple[StateType, StateType]:
    start_free = state_machine._check_free_cartesian(state_machine.cur_start_wpnts)
    on_spline = state_machine._check_on_spline(state_machine.cur_start_wpnts)
    if start_free and on_spline:
        return StateType.START, StateType.START
    else:
        state_machine.cur_start_wpnts.is_init = False
        return GlobalTrackingTransition(state_machine)
