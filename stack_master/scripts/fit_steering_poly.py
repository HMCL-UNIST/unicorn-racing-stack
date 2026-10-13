#!/usr/bin/env python3
# servo calibration fitter.
# Reads servo_calib_delta_*.bag files produced by record_steering_bags.py, extracts
# (delta_actual, target_raw_servo) pairs, and fits a polynomial
#   servo = a0 + a1*delta + a2*delta^2 + a3*delta^3
# using body-frame twist for the main analysis (R = |v| / |omega|, delta = atan(L/R))
# and a position-based circle fit as a cross-check.
#
# Writes steering_servo_poly_coeffs into vesc.yaml in place while preserving
# formatting/comments (ruamel.yaml preferred, PyYAML fallback).

import argparse
import glob
import math
import os
import re
import sys
import numpy as np

import yaml as _pyyaml   # only used for reading existing gain/offset values

try:
    import rosbag
except Exception as e:
    print("rosbag import failed. Run inside ROS environment. err=%s" % e, file=sys.stderr)
    sys.exit(2)


NAV_TOPIC   = "/vesc/high_level/ackermann_cmd_mux/input/nav_1"
ODOM_TOPIC  = "/car_state/odom"
SERVO_TOPIC = "/vesc/commands/servo/position"

# match both base name and trial-numbered variants:
#   servo_calib_delta_+0.20.bag       (trial 0, original)
#   servo_calib_delta_+0.20_t1.bag    (trial 1)
BAG_NAME_RE = re.compile(r"servo_calib_delta_([+-]?\d+(?:\.\d+)?)(?:_t\d+)?\.bag$")

# steady-state filter thresholds
MIN_SAMPLES      = 20
MAX_OMEGA_CV     = 0.15   # std(|omega|)/|mean| < 15%
MAX_V_CV         = 0.15
MIN_ABS_OMEGA    = 0.05   # rad/s — below this, the bicycle-model delta is noisy
MIN_ABS_V        = 0.2    # m/s


def _parse_delta_from_name(path):
    m = BAG_NAME_RE.search(os.path.basename(path))
    if not m:
        return None
    return float(m.group(1))


def _quat_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _analyse_bag(bag_path, wheelbase):
    # GLIL base_odom publishes twist.angular.z = 0 always, so we compute
    # yaw rate from pose quaternion differentiation instead.
    timestamps = []
    yaws = []
    speeds = []
    xs, ys = [], []
    servos_out = []

    bag = rosbag.Bag(bag_path, "r")
    try:
        for topic, msg, t in bag.read_messages(topics=[ODOM_TOPIC, SERVO_TOPIC]):
            if topic == ODOM_TOPIC:
                timestamps.append(t.to_sec())
                yaws.append(_quat_to_yaw(msg.pose.pose.orientation))
                speeds.append(math.hypot(msg.twist.twist.linear.x,
                                         msg.twist.twist.linear.y))
                xs.append(msg.pose.pose.position.x)
                ys.append(msg.pose.pose.position.y)
            elif topic == SERVO_TOPIC:
                servos_out.append(msg.data)
    except Exception as e:
        print("Warning: error reading %s: %s (using %d odom samples so far)"
              % (os.path.basename(bag_path), e, len(timestamps)))
    bag.close()

    result = {"bag": os.path.basename(bag_path), "n_odom": len(timestamps), "ok": False}

    if len(timestamps) < MIN_SAMPLES:
        result["reason"] = "too few odom samples (%d)" % len(timestamps)
        return result

    ts = np.asarray(timestamps)
    yaws_arr = np.unwrap(np.asarray(yaws))
    speeds_arr = np.asarray(speeds)
    xs = np.asarray(xs); ys = np.asarray(ys)

    # trim front 1.0s and rear 0.5s to discard transient / deceleration slip.
    # Only keep the stable middle portion for analysis.
    TRIM_FRONT = 1.0   # seconds
    TRIM_REAR  = 0.5   # seconds
    t0 = ts[0] + TRIM_FRONT
    t1 = ts[-1] - TRIM_REAR
    mask = (ts >= t0) & (ts <= t1)
    if np.sum(mask) < MIN_SAMPLES:
        # fallback: use all data if trimming leaves too few samples
        mask = np.ones(len(ts), dtype=bool)
    ts = ts[mask]; yaws_arr = yaws_arr[mask]; speeds_arr = speeds_arr[mask]
    xs = xs[mask]; ys = ys[mask]

    # yaw rate from pose differentiation (total angle / total time — robust to noise)
    total_t = ts[-1] - ts[0]
    if total_t < 0.5:
        result["reason"] = "recording too short (%.2fs)" % total_t
        return result

    w_mean = float((yaws_arr[-1] - yaws_arr[0]) / total_t)
    v_mean = float(np.mean(speeds_arr))
    v_std  = float(np.std(speeds_arr))

    result.update(dict(v_mean=v_mean, v_std=v_std, w_mean=w_mean, w_std=0.0))

    if abs(v_mean) < MIN_ABS_V or abs(w_mean) < MIN_ABS_OMEGA:
        result["reason"] = "|v| or |omega| below minimum (v=%.3f, w=%.3f)" % (v_mean, w_mean)
        return result

    if (v_std / max(abs(v_mean), 1e-6)) > MAX_V_CV:
        result["reason"] = "v not steady (cv=%.3f)" % (v_std / abs(v_mean))
        return result

    # main: bicycle-model equivalent steer angle
    R_twist = v_mean / w_mean
    delta_twist = float(np.arctan(wheelbase / R_twist))  # sign follows sign(w_mean)

    # cross-check: Kasa algebraic circle fit on (x,y)
    delta_pose = None
    R_pose = None
    if len(xs) >= MIN_SAMPLES:
        try:
            A = np.column_stack([2 * xs, 2 * ys, np.ones_like(xs)])
            b = xs * xs + ys * ys
            c, *_ = np.linalg.lstsq(A, b, rcond=None)
            xc, yc, k = c
            R_pose = float(np.sqrt(max(k + xc * xc + yc * yc, 0.0)))
            if R_pose > 1e-3:
                delta_pose = float(np.arctan(wheelbase / R_pose)) * np.sign(w_mean)
        except Exception:
            pass

    # actual raw servo observed on /vesc/commands/servo/position
    servo_actual = float(np.median(np.asarray(servos_out))) if len(servos_out) else float("nan")

    result.update(dict(
        ok=True,
        R_twist=R_twist,
        delta_twist=delta_twist,
        R_pose=R_pose,
        delta_pose=delta_pose,
        servo_actual=servo_actual,
    ))
    return result


def _fit_poly(deltas_actual, servos_target, degree):
    # np.polyfit returns highest-degree first; we want lowest first (a0 + a1*d + ...)
    deltas_actual = np.asarray(deltas_actual, dtype=float)
    servos_target = np.asarray(servos_target, dtype=float)
    order = np.argsort(deltas_actual)
    deltas_actual = deltas_actual[order]
    servos_target = servos_target[order]
    coeffs_hi = np.polyfit(deltas_actual, servos_target, degree)
    coeffs_lo = coeffs_hi[::-1].tolist()
    # residuals
    pred = np.polyval(coeffs_hi, deltas_actual)
    rms = float(np.sqrt(np.mean((pred - servos_target) ** 2)))
    return coeffs_lo, rms, deltas_actual, servos_target


def _save_mapping_plot(coeffs, d_measured, s_measured, offset, gain, png_path,
                       servo_clip_min=0.0, servo_clip_max=1.0):
    """Plot nonlinear poly vs linear map vs measured data points and save as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot.", file=sys.stderr)
        return

    d_dense = np.linspace(-0.7, 0.7, 300)

    # poly curve (before and after clip for visual comparison)
    coeffs_hi = np.array(coeffs[::-1])  # highest-degree first for np.polyval
    s_poly_raw = np.polyval(coeffs_hi, d_dense)
    s_poly = np.clip(s_poly_raw, servo_clip_min, servo_clip_max)

    # linear map
    s_linear = gain * d_dense + offset

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- left: servo = f(delta) ---
    ax = axes[0]
    # servo clip region (shaded)
    ax.axhspan(servo_clip_max, 1.1, color="tab:red", alpha=0.08, label="Servo clip region")
    ax.axhspan(-0.1, servo_clip_min, color="tab:red", alpha=0.08)
    ax.axhline(servo_clip_max, color="tab:red", linestyle=":", linewidth=1, alpha=0.6)
    ax.axhline(servo_clip_min, color="tab:red", linestyle=":", linewidth=1, alpha=0.6)

    ax.plot(np.rad2deg(d_dense), s_linear, "--", color="gray", linewidth=1.5, label="Linear (gain=%.3f, off=%.3f)" % (gain, offset))
    ax.plot(np.rad2deg(d_dense), s_poly, "-", color="tab:blue", linewidth=2, label="Poly fit (a0..a%d)" % (len(coeffs) - 1))
    # show the unclipped poly as faint dashed where it exceeds clip
    clip_mask = (s_poly_raw < servo_clip_min) | (s_poly_raw > servo_clip_max)
    if np.any(clip_mask):
        ax.plot(np.rad2deg(d_dense[clip_mask]), s_poly_raw[clip_mask], ":", color="tab:blue", linewidth=1, alpha=0.4)
    ax.scatter(np.rad2deg(np.array(d_measured)), np.array(s_measured), c="tab:red", s=40, zorder=5, label="Measured (%d pts)" % len(d_measured))
    ax.set_xlabel("Steering angle [deg]")
    ax.set_ylabel("Servo command [0-1]")
    ax.set_title("Steering angle -> Servo mapping")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-40, 40])
    ax.set_ylim([-0.05, 1.05])

    # --- right: error (poly - linear) ---
    ax2 = axes[1]
    err = s_poly - (gain * d_dense + offset)
    ax2.plot(np.rad2deg(d_dense), err, "-", color="tab:orange", linewidth=2)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Steering angle [deg]")
    ax2.set_ylabel("Servo difference (poly - linear)")
    ax2.set_title("Nonlinearity: deviation from linear map")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-40, 40])

    fig.suptitle("Nonlinear Servo Calibration  |  coeffs=%s" %
                 [round(c, 4) for c in coeffs], fontsize=9)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print("Plot saved -> %s" % png_path)


_COEFFS_KEY = "steering_servo_poly_coeffs"
_FLAG_KEY   = "enable_nonlinear_servo_gain"
_ANCHOR_KEY = "steering_angle_to_servo_offset"

_KEY_LINE_RE = lambda key: re.compile(r"^(\s*)" + re.escape(key) + r"\s*:")

def _format_coeffs_line(coeffs):
    return "{}: [{}]".format(
        _COEFFS_KEY,
        ", ".join("{:.6f}".format(float(c)) for c in coeffs),
    )

def _update_vesc_yaml(yaml_path, coeffs):
    # Text-based in-place update so comments and file layout survive intact.
    # Rules:
    #   1. If enable_nonlinear_servo_gain exists anywhere (commented or not), leave it alone.
    #      Otherwise insert `enable_nonlinear_servo_gain: false` right after
    #      steering_angle_to_servo_offset.
    #   2. Update steering_servo_poly_coeffs in place if present; otherwise insert it
    #      directly below the flag (which itself sits below the linear map).
    with open(yaml_path, "r") as f:
        lines = f.read().splitlines()

    def find_uncommented(key):
        pat = _KEY_LINE_RE(key)
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            if pat.match(line):
                return i
        return -1

    anchor_idx = find_uncommented(_ANCHOR_KEY)
    if anchor_idx < 0:
        # Anchor missing — append at EOF as a fallback.
        anchor_idx = len(lines) - 1

    flag_idx = find_uncommented(_FLAG_KEY)
    if flag_idx < 0:
        insert_at = anchor_idx + 1
        block = [
            "activate the nonlinear servo mapping below",
            "{}: false".format(_FLAG_KEY),
        ]
        # match anchor indent for the comment/key lines
        anchor_indent = re.match(r"^(\s*)", lines[anchor_idx]).group(1)
        block = [anchor_indent + "# " + block[0], anchor_indent + block[1]]
        lines[insert_at:insert_at] = block
        flag_idx = insert_at + 1  # index of the inserted flag line

    coeffs_idx = find_uncommented(_COEFFS_KEY)
    flag_indent = re.match(r"^(\s*)", lines[flag_idx]).group(1)
    new_coeffs_line = flag_indent + _format_coeffs_line(coeffs)
    if coeffs_idx >= 0:
        lines[coeffs_idx] = new_coeffs_line
    else:
        insert_at = flag_idx + 1
        comment = flag_indent + "# servo = a0 + a1*d + a2*d^2 + ... (auto-written by fit_steering_poly.py)"
        lines[insert_at:insert_at] = [comment, new_coeffs_line]

    with open(yaml_path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")


def fit_from_bags(bag_dir, vesc_yaml, wheelbase, poly_degree):
    bag_paths = sorted(glob.glob(os.path.join(bag_dir, "servo_calib_delta_*.bag")))
    if not bag_paths:
        print("No servo_calib_delta_*.bag found in %s" % bag_dir, file=sys.stderr)
        return 1

    rows = []
    for bp in bag_paths:
        target_delta = _parse_delta_from_name(bp)
        r = _analyse_bag(bp, wheelbase)
        r["target_delta"] = target_delta
        rows.append(r)

    # report
    print("\n=== per-bag analysis ===")
    print("{:>32s} {:>8s} {:>9s} {:>9s} {:>9s} {:>10s} {:>10s}  reason".format(
        "bag", "N", "v_mean", "w_mean", "R_twist", "d_twist", "d_pose"))
    for r in rows:
        if r["ok"]:
            print("{:>32s} {:>8d} {:>9.3f} {:>9.3f} {:>9.3f} {:>10.4f} {:>10}".format(
                r["bag"], r["n_odom"], r["v_mean"], r["w_mean"], r["R_twist"],
                r["delta_twist"],
                "{:.4f}".format(r["delta_pose"]) if r["delta_pose"] is not None else "  n/a  "))
        else:
            print("{:>32s} {:>8d}  SKIPPED  ({})".format(
                r["bag"], r["n_odom"], r.get("reason", "?")))

    # data pairs: target raw servo vs measured delta (twist-based)
    xs, ys = [], []
    for r in rows:
        if not r["ok"]:
            continue
        if r["target_delta"] is None:
            continue
        # target raw servo using the linear map the recorder used
        # (it's stored in the bag via NAV_TOPIC steering_angle; median is robust)
        # but simpler/equivalent: re-derive from target_delta + current linear map
        # because recorder built delta_cmd = target_delta exactly.
        xs.append(r["delta_twist"])
        # target servo was built by recorder from its delta_target via the SAME linear map
        # we will now re-read from vesc.yaml:
        # (handled below by caller supplying gain/offset via CLI or here via yaml read)
    # We need the linear map coefficients to recover target_servo. Read them from the yaml.
    with open(vesc_yaml, "r") as f:
        vdata = _pyyaml.safe_load(f) or {}
    gain   = float(vdata["steering_angle_to_servo_gain"])
    offset = float(vdata["steering_angle_to_servo_offset"])

    deltas_measured = []
    servos_target   = []
    for r in rows:
        if not r["ok"] or r["target_delta"] is None:
            continue
        deltas_measured.append(r["delta_twist"])
        servos_target.append(offset + gain * r["target_delta"])

    if len(deltas_measured) < max(poly_degree + 1, 5):
        print("Not enough valid points (%d) for degree-%d fit." %
              (len(deltas_measured), poly_degree), file=sys.stderr)
        return 1

    coeffs, rms, d_sorted, s_sorted = _fit_poly(deltas_measured, servos_target, poly_degree)
    print("\n=== fit result ===")
    print("N points         : %d" % len(deltas_measured))
    print("poly degree      : %d" % poly_degree)
    print("coeffs (a0..aN)  : %s" % ["%.6f" % c for c in coeffs])
    print("residual rms     : %.5f (servo units)" % rms)
    print("linear reference : offset=%.4f gain=%.4f (a0/a1 should be close when linkage ~linear)"
          % (offset, gain))

    # symmetry check
    pos = [(d, s) for d, s in zip(d_sorted, s_sorted) if d > 0.05]
    neg = [(d, s) for d, s in zip(d_sorted, s_sorted) if d < -0.05]
    if pos and neg:
        asym = float(np.mean([p[1] - offset for p in pos]) +
                     np.mean([n[1] - offset for n in neg]))
        print("symmetry |+|-||  : %.4f (0 ~ perfect, larger = biased linkage)" % asym)

    _update_vesc_yaml(vesc_yaml, coeffs)
    print("\n%s updated. steering_servo_poly_coeffs set. Enable with\n"
          "  enable_nonlinear_servo_gain: true\nin the same file when ready." % vesc_yaml)

    # save visualization of the nonlinear mapping
    servo_min = float(vdata.get("vesc_driver", {}).get("servo_min", 0.0))
    servo_max = float(vdata.get("vesc_driver", {}).get("servo_max", 1.0))
    png_path = os.path.join(os.path.dirname(vesc_yaml), "nonlinear_servo_mapping.png")
    _save_mapping_plot(coeffs, d_sorted, s_sorted, offset, gain, png_path,
                       servo_clip_min=servo_min, servo_clip_max=servo_max)

    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag-dir", required=True)
    ap.add_argument("--vesc-yaml", required=True)
    ap.add_argument("--wheelbase", type=float, default=0.33)
    ap.add_argument("--poly-degree", type=int, default=3)
    args = ap.parse_args()
    rc = fit_from_bags(args.bag_dir, args.vesc_yaml, args.wheelbase, args.poly_degree)
    sys.exit(rc)


if __name__ == "__main__":
    main()
