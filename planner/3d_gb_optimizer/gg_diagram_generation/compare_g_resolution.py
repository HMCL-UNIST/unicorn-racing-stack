#!/usr/bin/env python3
"""
Subsample 5 of the existing 17 g points,
reconstruct the 17 by linear interpolation and compare against the original.
Checks the interpolation error only, without regenerating.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os

base = os.path.join(os.path.dirname(__file__),
                    '..', 'global_line', 'data', 'gg_diagrams',
                    'rc_car_10th_fin', 'vehicle_frame')

# --- load 17-point data ---
g17 = np.load(os.path.join(base, 'g_list.npy'))       # (17,)
v_list = np.load(os.path.join(base, 'v_list.npy'))     # (15,)
ax_max_17 = np.load(os.path.join(base, 'ax_max.npy'))  # (15,17)
ax_min_17 = np.load(os.path.join(base, 'ax_min.npy'))
ay_max_17 = np.load(os.path.join(base, 'ay_max.npy'))
gg_exp_17 = np.load(os.path.join(base, 'gg_exponent.npy'))

# --- subsample to 5 points (uniform from 17) ---
idx5 = np.linspace(0, len(g17)-1, 5, dtype=int)  # [0, 4, 8, 12, 16]
g5 = g17[idx5]

print(f"g17: {g17}")
print(f"g5 (subsampled): {g5}")
print(f"g5 indices: {idx5}")
print()

# --- interpolate back to 17 points ---
fields = {
    'ax_max': ax_max_17,
    'ax_min': ax_min_17,
    'ay_max': ay_max_17,
    'gg_exponent': gg_exp_17,
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (name, data_17) in enumerate(fields.items()):
    ax = axes[idx]
    data_5 = data_17[:, idx5]  # (15, 5)

    # interpolate each velocity row: 5 -> 17
    data_interp = np.zeros_like(data_17)
    for vi in range(len(v_list)):
        f = interp1d(g5, data_5[vi, :], kind='linear', fill_value='extrapolate')
        data_interp[vi, :] = f(g17)

    # error
    abs_err = np.abs(data_17 - data_interp)
    rel_err = np.where(np.abs(data_17) > 1e-6,
                       abs_err / np.abs(data_17) * 100, 0)

    print(f"=== {name} ===")
    print(f"  abs error: max={abs_err.max():.4f}, mean={abs_err.mean():.4f}")
    print(f"  rel error: max={rel_err.max():.2f}%, mean={rel_err.mean():.2f}%")
    print()

    # plot a few velocity slices
    v_indices = [0, len(v_list)//4, len(v_list)//2, 3*len(v_list)//4, -1]
    for vi in v_indices:
        color = ax.plot(g17, data_17[vi, :], '-o', markersize=3,
                        label=f'v={v_list[vi]:.1f} (17pt)')[0].get_color()
        ax.plot(g17, data_interp[vi, :], '--x', markersize=5, color=color,
                alpha=0.7, label=f'v={v_list[vi]:.1f} (5pt interp)')
        # mark the 5 sample points
        ax.plot(g5, data_5[vi, :], 's', markersize=8, color=color,
                alpha=0.4, zorder=5)

    ax.set_xlabel('g_tilde [m/s²]')
    ax.set_ylabel(name)
    ax.set_title(f'{name}  (max err: {abs_err.max():.3f}, {rel_err.max():.1f}%)')
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

plt.suptitle('17-point (orig) vs 5-point (linear interp) GG Diagrams\n'
             'rc_car_10th_fin / vehicle_frame', fontsize=13)
plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), 'compare_g17_vs_g5.png')
plt.savefig(out_path, dpi=150)
print(f"Plot saved: {out_path}")
plt.show()
