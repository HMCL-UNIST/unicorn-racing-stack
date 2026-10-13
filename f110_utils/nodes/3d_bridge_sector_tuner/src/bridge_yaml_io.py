#!/usr/bin/env python3
# Shared yaml IO for <map>/bridge_sectors.yaml.
# Single source of truth: positions (start/end) + per-bridge tuning
# (pre_m, post_m, force_center, chi_max_rad) + global enable flag.
# Used by:
# - gg_tuner_node.py            (load/save on apply, init rqt cfg)
# - bridge_slicing.py           (save positions, preserve tuning)
# - bridge_sector_server.py     (read positions for RViz markers)
# - gen_global_racing_line.py   (read per-bridge tuning at NLP setup)
#
#      Schema:
#        n_bridges: <int>
#        bridge_constraints_enable: <bool>
#        Bridge0:
#          start: <int>        # smoothed-CSV row index (inclusive)
#          end:   <int>
#          pre_m:        <float>
#          post_m:       <float>
#          force_center: <bool>
#          chi_max_rad:  <float>   # <0 disables heading clamp
#        Bridge1: { ... }
#
#      Defaults reproduce legacy behavior (no extension, no centerline pin,
#      heading clamp at 0 = strictly parallel — matches the cfg default).
import os
import yaml


# per-bridge tuning defaults — matches GGTuner.cfg Bridge group.
DEFAULT_PRE_M        = 0.0
DEFAULT_POST_M       = 0.0
DEFAULT_FORCE_CENTER = False
DEFAULT_CHI_MAX_RAD  = 0.0       # cfg default also 0 ⇒ parallel by default
DEFAULT_CONSTRAINTS_ENABLE = True


def yaml_path(map_dir):
    return os.path.join(map_dir, 'bridge_sectors.yaml')


def load_bridges(map_dir):
    """Return (constraints_enable, [bridge_dict, ...]).
    Missing or malformed yaml ⇒ (default_enable, []).
    Each bridge_dict is a fully-populated dict (defaults filled in).
    """
    path = yaml_path(map_dir)
    if not os.path.exists(path):
        return DEFAULT_CONSTRAINTS_ENABLE, []
    try:
        with open(path, 'r') as f:
            doc = yaml.safe_load(f) or {}
    except Exception:
        return DEFAULT_CONSTRAINTS_ENABLE, []

    enable = bool(doc.get('bridge_constraints_enable',
                          DEFAULT_CONSTRAINTS_ENABLE))
    n = int(doc.get('n_bridges', 0))
    bridges = []
    for i in range(n):
        b = doc.get(f'Bridge{i}')
        if not isinstance(b, dict):
            continue
        try:
            s_i = int(b['start'])
            e_i = int(b['end'])
        except (KeyError, ValueError, TypeError):
            continue
        if e_i < s_i:
            s_i, e_i = e_i, s_i
        bridges.append({
            'start': s_i,
            'end':   e_i,
            'pre_m':        float(b.get('pre_m',        DEFAULT_PRE_M)),
            'post_m':       float(b.get('post_m',       DEFAULT_POST_M)),
            'force_center': bool (b.get('force_center', DEFAULT_FORCE_CENTER)),
            'chi_max_rad':  float(b.get('chi_max_rad',  DEFAULT_CHI_MAX_RAD)),
        })
    return enable, bridges


def save_bridges(map_dir, bridges, constraints_enable=None):
    """Write yaml. `bridges` is a list of dicts with at least start/end —
    other fields default-filled if missing. constraints_enable=None keeps
    the previously-stored value (preserve user's master toggle on partial
    saves like position-only edits from bridge_slicing).
    """
    if constraints_enable is None:
        # preserve existing enable flag on position-only saves.
        prev_enable, _ = load_bridges(map_dir)
        constraints_enable = prev_enable
    doc = {
        'n_bridges': len(bridges),
        'bridge_constraints_enable': bool(constraints_enable),
    }
    for i, b in enumerate(bridges):
        doc[f'Bridge{i}'] = {
            'start': int(b['start']),
            'end':   int(b['end']),
            'pre_m':        float(b.get('pre_m',        DEFAULT_PRE_M)),
            'post_m':       float(b.get('post_m',       DEFAULT_POST_M)),
            'force_center': bool (b.get('force_center', DEFAULT_FORCE_CENTER)),
            'chi_max_rad':  float(b.get('chi_max_rad',  DEFAULT_CHI_MAX_RAD)),
        }
    with open(yaml_path(map_dir), 'w') as f:
        yaml.dump(doc, f, sort_keys=False)


def save_positions_preserving_tuning(map_dir, positions):
    """Save new bridge positions while keeping any existing per-bridge
    tuning. `positions` is [(start, end), ...]. If yaml already had a
    Bridge{i} for the same index, its tuning is preserved; otherwise the
    new bridge gets defaults. Used by bridge_slicing.py "Done & Save".
    """
    prev_enable, prev_bridges = load_bridges(map_dir)
    new_bridges = []
    for i, (s, e) in enumerate(positions):
        if i < len(prev_bridges):
            # keep tuning from the i-th slot if present.
            b = dict(prev_bridges[i])
            b['start'], b['end'] = int(s), int(e)
        else:
            b = {'start': int(s), 'end': int(e)}
        new_bridges.append(b)
    save_bridges(map_dir, new_bridges, constraints_enable=prev_enable)


def apply_global_tuning(map_dir, pre_m, post_m, force_center, chi_max_rad,
                       constraints_enable):
    """Overwrite every bridge's tuning with the same global values. Used
    by gg_tuner_node when the user changes rqt sliders and clicks apply.
    Bridge positions are preserved untouched.
    """
    prev_enable, prev_bridges = load_bridges(map_dir)
    for b in prev_bridges:
        b['pre_m']        = float(pre_m)
        b['post_m']       = float(post_m)
        b['force_center'] = bool(force_center)
        b['chi_max_rad']  = float(chi_max_rad)
    save_bridges(map_dir, prev_bridges,
                 constraints_enable=bool(constraints_enable))
