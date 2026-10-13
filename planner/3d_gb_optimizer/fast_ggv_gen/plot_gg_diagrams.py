import os
import sys
import argparse
from matplotlib import pyplot as plt
import numpy as np
import casadi as ca

# add argparse for vehicle_name (was hardcoded to 'rc_car_10th')
parser = argparse.ArgumentParser(description='Plot GG diagrams')
parser.add_argument('--vehicle_name', type=str, default='rc_car_10th', help='Vehicle name (default: rc_car_10th)')
args, _ = parser.parse_known_args()
vehicle_name = args.vehicle_name
frame = 'velocity'
# paths
dir_path = os.path.dirname(os.path.abspath(__file__))
# read from fast_ggv_gen/output/ (never touch original gg_diagrams)
gg_path = os.path.join(dir_path, 'output', vehicle_name, frame+'_frame')
sys.path.append(os.path.join(dir_path, '..', 'global_line', 'src'))

from ggManager import GGManager

# casadi interpolator
gg_handler = GGManager(
    gg_path=gg_path,
    gg_margin=0.0,
)

v_test = np.linspace(gg_handler.V_list[1], gg_handler.V_max, 5)
g_test = np.linspace(0.0, gg_handler.g_max, 100)
alpha_test = np.linspace(-np.pi, np.pi, 300)
g = 9.81
# g = 0.5 * 9.81
# g = 2.0 * 9.81

fig_polar = plt.figure('Polar coordinates')
ax_polar = fig_polar.add_subplot(projection='3d')
ax_polar.set_title(f'Polar coordinates')

fig_form = plt.figure('Diamond shpaed')
ax_diamond = fig_form.add_subplot()
ax_diamond.set_title(f'Diamond shaped underapproximations')

X_test, Y_test = np.meshgrid(g_test, alpha_test, indexing='ij')
for V in v_test:
    Z_test = np.zeros_like(X_test)
    for g_i, gt in enumerate(g_test):
        for alpha_i, alpha in enumerate(alpha_test):
            Z_test[g_i, alpha_i] = gg_handler.rho_interpolator_no_margin([V, gt, alpha])
    ax_polar.plot_surface(Y_test, X_test, Z_test)

    gg_exponent = float(gg_handler.gg_exponent_interpolator(ca.vertcat(V, g)))
    ax_max = float(gg_handler.ax_max_interpolator(ca.vertcat(V, g)))
    ax_min = float(gg_handler.ax_min_interpolator(ca.vertcat(V, g)))
    ay_max = float(gg_handler.ay_max_interpolator(ca.vertcat(V, g)))
    # Polar plot
    rho_test = gg_handler.rho_interpolator_no_margin(np.array([V * np.ones_like(alpha_test), g * np.ones_like(alpha_test), alpha_test])).full().squeeze()

    # Form shape
    tmp = ax_diamond.plot(np.cos(alpha_test) * rho_test, np.sin(alpha_test) * rho_test, label=f'$V$={V}, $\\tilde{{g}}$={g}', alpha=0.3)
    ay_array = np.linspace(-ay_max, ay_max, 200)
    ax_array = np.zeros_like(ay_array)
    for i, ay in enumerate(ay_array):
        ax_array[i] = - ax_min * np.power(
            1.0 - np.power(np.abs(ay) / ay_max, gg_exponent),
            1.0 / gg_exponent,
        )
    ax_diamond.plot(ay_array, np.minimum(ax_array, ax_max), color=tmp[0].get_color())
    ax_diamond.plot(ay_array, -ax_array, color=tmp[0].get_color())

ax_polar.set_xlabel(r'$\alpha$')
ax_polar.set_ylabel(r'$\tilde{g}$')
ax_polar.set_zlabel(r'$\rho$')

ax_diamond.set_xlabel(r'$\tilde{a}_\mathrm{y}$')
ax_diamond.set_ylabel(r'$\tilde{a}_\mathrm{x}$')
ax_diamond.legend()
ax_diamond.set_aspect('equal')

# multi-gtilde subplots — diamond shape at various g_tilde values
g_sweep = np.linspace(max(float(gg_handler.g_list.min()), 1e-3), gg_handler.g_max, 6)
n_sub = len(g_sweep)
ncols = 3
nrows = int(np.ceil(n_sub / ncols))
fig_multi, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows),
                               num='Diamond shaped (multi gtilde)')
axes = np.atleast_1d(axes).flatten()
for idx, g_val in enumerate(g_sweep):
    ax = axes[idx]
    ax.set_title(f'$\\tilde{{g}}$={g_val:.2f}')
    for V in v_test:
        gg_exponent = float(gg_handler.gg_exponent_interpolator(ca.vertcat(V, g_val)))
        ax_max_v = float(gg_handler.ax_max_interpolator(ca.vertcat(V, g_val)))
        ax_min_v = float(gg_handler.ax_min_interpolator(ca.vertcat(V, g_val)))
        ay_max_v = float(gg_handler.ay_max_interpolator(ca.vertcat(V, g_val)))
        rho_v = gg_handler.rho_interpolator_no_margin(
            np.array([V * np.ones_like(alpha_test),
                      g_val * np.ones_like(alpha_test),
                      alpha_test])).full().squeeze()
        tmp = ax.plot(np.cos(alpha_test) * rho_v, np.sin(alpha_test) * rho_v,
                      label=f'$V$={V:.1f}', alpha=0.3)
        ay_arr = np.linspace(-ay_max_v, ay_max_v, 200)
        ax_arr = -ax_min_v * np.power(
            np.clip(1.0 - np.power(np.abs(ay_arr) / ay_max_v, gg_exponent), 0.0, None),
            1.0 / gg_exponent)
        ax.plot(ay_arr, np.minimum(ax_arr, ax_max_v), color=tmp[0].get_color())
        ax.plot(ay_arr, -ax_arr, color=tmp[0].get_color())
    ax.set_xlabel(r'$\tilde{a}_\mathrm{y}$')
    ax.set_ylabel(r'$\tilde{a}_\mathrm{x}$')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=8)
for k in range(n_sub, len(axes)):
    axes[k].axis('off')
fig_multi.tight_layout()

# save to file as well for headless environments
out_dir = os.path.join(dir_path, 'output', vehicle_name)
fig_polar.savefig(os.path.join(out_dir, 'gg_polar.png'), dpi=150, bbox_inches='tight')
fig_form.savefig(os.path.join(out_dir, 'gg_diamond.png'), dpi=150, bbox_inches='tight')
# save multi-gtilde figure
fig_multi.savefig(os.path.join(out_dir, 'gg_diamond_multi_g.png'), dpi=150, bbox_inches='tight')
print(f'[plot] Saved to {out_dir}/gg_polar.png, gg_diamond.png, gg_diamond_multi_g.png')
plt.show()

# EOF
