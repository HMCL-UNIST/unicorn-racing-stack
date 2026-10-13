#!/usr/bin/env python3
"""Interactive centerline regeneration + smoothing.

Loads a raw track CSV (left_bound, right_bound, center), recomputes the
centerline as midpoint(L, R), and lets the user pick a cyclic savgol
smoothing window via a slider. Save & close writes back to the CSV.
"""
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.signal import savgol_filter


def load(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        fns = reader.fieldnames
        rows = list(reader)
    L = np.array([[float(r['left_bound_x']), float(r['left_bound_y']), float(r['left_bound_z'])] for r in rows])
    R = np.array([[float(r['right_bound_x']), float(r['right_bound_y']), float(r['right_bound_z'])] for r in rows])
    return rows, fns, L, R


def solve_t_warmstart(L, R, t0, n_iter=15):
    """Iteratively reweighted LSQ to minimize ~max(perp_err).
    Builds the same linear system as solve_t but reweights each perp row by
    (|residual_i| + eps)^p so that the largest residuals get progressively
    larger weight (Chebyshev/L∞ approximation). Warm-started from t0."""
    from scipy.sparse import coo_matrix, eye, diags, vstack
    from scipy.sparse.linalg import lsmr
    n = len(L)
    d = R - L
    rh = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)

    # curvature LSQ block (constant)
    a = np.roll(L, -1, 0) - 2*L + np.roll(L, 1, 0)
    rows_, cols_, vals_ = [], [], []
    for i in range(n):
        ip = (i + 1) % n; im = (i - 1) % n
        for axis in range(2):
            r = 2*i + axis
            rows_ += [r, r, r]; cols_ += [im, i, ip]
            vals_ += [d[im, axis], -2*d[i, axis], d[ip, axis]]
    Mc = coo_matrix((vals_, (rows_, cols_)), shape=(2*n, n)).tocsr()
    bc = a.flatten()

    # perpendicular block (constant)
    rows_p, cols_p, vals_p = [], [], []
    bp = np.zeros(n)
    Lp = np.roll(L, -1, 0); Lm = np.roll(L, 1, 0)
    for i in range(n):
        ip = (i + 1) % n; im = (i - 1) % n
        bp[i] = (Lp[i] - Lm[i]) @ rh[i]
        rows_p += [i, i]; cols_p += [ip, im]
        vals_p += [d[ip] @ rh[i], -d[im] @ rh[i]]
    Mp = coo_matrix((vals_p, (rows_p, cols_p)), shape=(n, n)).tocsr()

    t = t0.copy()
    # Initialize weights from initial residuals so warm start matters
    r0 = np.abs(Mp @ t + bp)
    w = (r0 / (r0.max() + 1e-12)) ** 2 + 1e-3
    base_perp = 1e3
    best_t = t.copy()
    best_max = r0.max()
    for it in range(n_iter):
        W = diags(np.sqrt(w * base_perp))
        A = vstack([Mc, W @ Mp]).tocsr()
        rhs = np.concatenate([-bc, -(W @ bp)])
        # Warm start: shift variables → solve for delta
        # A (t0 + dt) ≈ rhs  →  A dt = rhs - A t
        delta_rhs = rhs - A @ t
        sol = lsmr(A, delta_rhs, atol=1e-9, btol=1e-9, maxiter=200)
        dt = sol[0]
        t_new = t + dt
        r_perp = np.abs(Mp @ t_new + bp)
        if r_perp.max() < best_max:
            best_max = r_perp.max()
            best_t = t_new.copy()
        # Update weights: emphasize current largest
        w = (r_perp / (r_perp.max() + 1e-12)) ** 2 + 1e-3
        t = t_new
    return np.clip(best_t, 0.0, 1.0), float(best_max), n_iter


def solve_t_constrained(L, R):
    """Equality-constrained QP:
       minimize  Σ||C[i+1]-2C[i]+C[i-1]||²    (smoothness)
       subject to  (C[i+1]-C[i-1])·rib_hat[i] = 0  ∀ i  (tangent⊥rib, hard)
    where C[i] = L[i] + t[i]*(R[i]-L[i]).
    Solved via the KKT linear system.
    """
    from scipy.sparse import coo_matrix, eye, vstack, hstack, bmat
    from scipy.sparse.linalg import spsolve
    n = len(L)
    d = R - L
    rh = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)

    # --- curvature: residual_x[i], residual_y[i] linear in t  (2n rows, n cols) ---
    a = np.roll(L, -1, 0) - 2*L + np.roll(L, 1, 0)
    rows_, cols_, vals_ = [], [], []
    for i in range(n):
        ip = (i + 1) % n; im = (i - 1) % n
        for axis in range(2):
            r = 2*i + axis
            rows_ += [r, r, r]; cols_ += [im, i, ip]
            vals_ += [d[im, axis], -2*d[i, axis], d[ip, axis]]
    Mc = coo_matrix((vals_, (rows_, cols_)), shape=(2*n, n)).tocsr()
    bc = a.flatten()                       # so curvature_resid = Mc t + bc
    H = (Mc.T @ Mc).tocsr()                # n x n
    g = (Mc.T @ bc).flatten()              # n   (gradient of 0.5 ||Mc t + bc||² is H t + g)

    # --- perpendicular constraint:  Ap t + bp = 0   (n rows, n cols) ---
    rows_p, cols_p, vals_p = [], [], []
    bp = np.zeros(n)
    Lp = np.roll(L, -1, 0); Lm = np.roll(L, 1, 0)
    for i in range(n):
        ip = (i + 1) % n; im = (i - 1) % n
        bp[i] = (Lp[i] - Lm[i]) @ rh[i]
        rows_p += [i, i]; cols_p += [ip, im]
        vals_p += [d[ip] @ rh[i], -d[im] @ rh[i]]
    Ap = coo_matrix((vals_p, (rows_p, cols_p)), shape=(n, n)).tocsr()

    # tiny ridge on H for invertibility (n_unconstrained_dofs may exist)
    H_reg = H + 1e-8 * eye(n, format='csr')

    # KKT  [H  A^T] [t]   [-g ]
    #      [A   0 ] [λ] = [-bp]
    Z = coo_matrix((n, n)).tocsr()
    K = bmat([[H_reg, Ap.T], [Ap, Z]], format='csr')
    rhs = np.concatenate([-g, -bp])
    sol = spsolve(K, rhs)
    t = sol[:n]
    return np.clip(t, 0.0, 1.0)


def solve_t(L, R, lam_mid, w_perp):
    """Find t[i] in [0,1] for C[i]=L[i]+t[i]*(R[i]-L[i]) minimizing:
       w_curv * Σ||C[i+1]-2C[i]+C[i-1]||²            (smoothness)
       + w_perp * Σ((C[i+1]-C[i-1])·rib_hat[i])²    (tangent⊥rib)
       + lam_mid * Σ(t[i]-0.5)²                       (midpoint reg)
    All linear in t — closed-form sparse LSQ."""
    n = len(L)
    d = R - L
    rib_norm = np.linalg.norm(d, axis=1) + 1e-9
    rh = d / rib_norm[:, None]   # rib unit vector

    # --- curvature residuals (2n) ---
    a = np.roll(L, -1, 0) - 2*L + np.roll(L, 1, 0)
    rows_, cols_, vals_ = [], [], []
    for i in range(n):
        ip = (i + 1) % n
        im = (i - 1) % n
        for axis in range(2):
            r = 2*i + axis
            rows_ += [r, r, r]
            cols_ += [im, i, ip]
            vals_ += [d[im, axis], -2*d[i, axis], d[ip, axis]]
    from scipy.sparse import coo_matrix, eye, vstack
    M_curv = coo_matrix((vals_, (rows_, cols_)), shape=(2*n, n)).tocsr()
    b_curv = a.flatten()

    # --- perpendicularity residuals (n) ---
    # r_perp[i] = (L[i+1]-L[i-1])·rh[i] + t[i+1]*(d[i+1]·rh[i]) - t[i-1]*(d[i-1]·rh[i])
    rows_p, cols_p, vals_p = [], [], []
    b_perp = np.zeros(n)
    Lroll_p = np.roll(L, -1, 0)
    Lroll_m = np.roll(L, 1, 0)
    for i in range(n):
        ip = (i + 1) % n
        im = (i - 1) % n
        b_perp[i] = (Lroll_p[i] - Lroll_m[i]) @ rh[i]
        rows_p += [i, i]
        cols_p += [ip, im]
        vals_p += [d[ip] @ rh[i], -d[im] @ rh[i]]
    M_perp = coo_matrix((vals_p, (rows_p, cols_p)), shape=(n, n)).tocsr()

    # --- stack ---
    sw_perp = np.sqrt(max(w_perp, 0.0))
    A = vstack([M_curv, sw_perp * M_perp, np.sqrt(lam_mid) * eye(n)]).tocsr()
    rhs = np.concatenate([-b_curv, -sw_perp * b_perp,
                          np.sqrt(lam_mid) * 0.5 * np.ones(n)])
    from scipy.sparse.linalg import lsqr
    t = lsqr(A, rhs)[0]
    return np.clip(t, 0.0, 1.0)


def cyclic_savgol(arr, win, poly=2):
    n = len(arr)
    if win < 3 or win >= n:
        return arr.copy()
    if win % 2 == 0:
        win += 1
    pad = win
    out = arr.copy()
    for d in range(arr.shape[1]):
        padded = np.concatenate([arr[-pad:, d], arr[:, d], arr[:pad, d]])
        sm = savgol_filter(padded, win, poly)
        out[:, d] = sm[pad:pad + n]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', help='Raw track CSV')
    args = ap.parse_args()

    rows, fns, L, R = load(args.csv)
    # Parametrize C[i] = L[i] + t[i] * (R[i] - L[i]); t=0.5 → midpoint.
    # Smoothing t (scalar per index) keeps C exactly on the L-R line.
    t0 = np.full(len(L), 0.5)
    C0 = L + t0[:, None] * (R - L)
    state = {'C': C0.copy(), 'win': 1, 't': t0.copy()}

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.subplots_adjust(bottom=0.18)

    ax.plot(L[:, 0], L[:, 1], 'g-', lw=1.0, label='left')
    ax.plot(R[:, 0], R[:, 1], 'r-', lw=1.0, label='right')
    rib_x = np.empty(3 * len(L))
    rib_y = np.empty(3 * len(L))
    rib_x[0::3] = L[:, 0]; rib_x[1::3] = R[:, 0]; rib_x[2::3] = np.nan
    rib_y[0::3] = L[:, 1]; rib_y[1::3] = R[:, 1]; rib_y[2::3] = np.nan
    ax.plot(rib_x, rib_y, color='gray', lw=0.3, alpha=0.4)
    raw_line, = ax.plot(C0[:, 0], C0[:, 1], 'b:', lw=0.8, alpha=0.4, label='midpoint')
    sm_line, = ax.plot(C0[:, 0], C0[:, 1], 'b-', lw=1.0, alpha=0.5,
                       label='smoothed center')
    sm_pts = ax.scatter(C0[:, 0], C0[:, 1], s=18, c=np.zeros(len(C0)),
                         cmap='RdYlGn_r', vmin=0, vmax=30, zorder=5,
                         label='center (color=⊥err°)')
    cbar = plt.colorbar(sm_pts, ax=ax, label='|tangent⊥rib| err [deg]')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=9)
    fig.suptitle(
        '[Step 2.9] Centerline Regen — C[i]=L[i]+t[i]·(R[i]-L[i]) constrained to L-R segment. '
        'Slider tunes smoothness vs ⊥-rib. Auto Optimize: IRLS warm-start QP.',
        fontsize=10)
    ax.set_title(f'{len(L)} pts')
    ax.grid(True, alpha=0.2)

    ax_slider = fig.add_axes([0.15, 0.08, 0.6, 0.025])
    s_lam = Slider(ax_slider, 'Midpoint reg (log10)',
                   -4.0, 4.0, valinit=-1.0, valfmt='%.1f')
    ax_slider2 = fig.add_axes([0.15, 0.05, 0.6, 0.022])
    s_perp = Slider(ax_slider2, 'Tangent⊥Rib (log10)',
                    -2.0, 6.0, valinit=2.0, valfmt='%.1f')

    def update(val=None):
        lam = 10.0 ** s_lam.val
        wp = 10.0 ** s_perp.val
        state['win'] = (lam, wp)
        t = solve_t(L[:, :2], R[:, :2], lam, wp)
        state['t'] = t
        Cxy = L[:, :2] + t[:, None] * (R[:, :2] - L[:, :2])
        Cz  = L[:, 2]  + t        * (R[:, 2]  - L[:, 2])
        state['C'] = np.column_stack([Cxy, Cz])
        sm_line.set_data(state['C'][:, 0], state['C'][:, 1])
        sm_pts.set_offsets(state['C'][:, :2])
        # color update — use perp error per point (computed below)
        # Live collinearity stats
        Cn = state['C'][:, :2]
        rib = R[:, :2] - L[:, :2]
        rn = np.linalg.norm(rib, axis=1) + 1e-9
        nx_r, ny_r = -rib[:, 1] / rn, rib[:, 0] / rn
        toc = Cn - L[:, :2]
        coll = np.abs(toc[:, 0] * nx_r + toc[:, 1] * ny_r)
        # Tangent⊥rib stat
        Cn = state['C'][:, :2]
        fwd = np.roll(Cn, -1, 0) - np.roll(Cn, 1, 0)
        fn = np.linalg.norm(fwd, axis=1) + 1e-9
        rib_v = R[:, :2] - L[:, :2]
        rn2 = np.linalg.norm(rib_v, axis=1) + 1e-9
        cos_tr = (fwd[:, 0]*rib_v[:, 0] + fwd[:, 1]*rib_v[:, 1]) / (fn * rn2)
        perp_deg = np.degrees(np.abs(np.arcsin(np.clip(cos_tr, -1, 1))))
        sm_pts.set_array(perp_deg)
        ax.set_title(
            f'λ_mid=1e{s_lam.val:.1f}  w_perp=1e{s_perp.val:.1f}\n'
            f'⊥err: mean={perp_deg.mean():.2f}° max={perp_deg.max():.2f}°   '
            f'collinearity: mean={coll.mean():.2e}m max={coll.max():.2e}m')
        fig.canvas.draw_idle()
    s_lam.on_changed(update)
    s_perp.on_changed(update)

    ax_opt = fig.add_axes([0.62, 0.12, 0.15, 0.045])
    opt_btn = Button(ax_opt, 'Auto Optimize')

    def on_opt(event):
        t0 = state.get('t', np.full(len(L), 0.5))
        print(f"Auto optimize (warm start from current t, mean={t0.mean():.3f}) ...")
        t, fval, nit = solve_t_warmstart(L[:, :2], R[:, :2], t0)
        print(f"  done. f={fval:.4e}, iters={nit}")
        Cxy = L[:, :2] + t[:, None] * (R[:, :2] - L[:, :2])
        Cz = L[:, 2] + t * (R[:, 2] - L[:, 2])
        state['C'] = np.column_stack([Cxy, Cz])
        state['t'] = t
        # Manually trigger plot refresh (skip slider callback)
        sm_line.set_data(state['C'][:, 0], state['C'][:, 1])
        sm_pts.set_offsets(state['C'][:, :2])
        Cn = state['C'][:, :2]
        fwd = np.roll(Cn, -1, 0) - np.roll(Cn, 1, 0)
        fn = np.linalg.norm(fwd, axis=1) + 1e-9
        rib_v = R[:, :2] - L[:, :2]
        rn2 = np.linalg.norm(rib_v, axis=1) + 1e-9
        cos_tr = (fwd[:, 0]*rib_v[:, 0] + fwd[:, 1]*rib_v[:, 1]) / (fn * rn2)
        perp_deg = np.degrees(np.abs(np.arcsin(np.clip(cos_tr, -1, 1))))
        sm_pts.set_array(perp_deg)
        ax.set_title(f'Auto-optimized (warm start, L-BFGS-B) — '
                     f'⊥err mean={perp_deg.mean():.3f}° max={perp_deg.max():.3f}°')
        fig.canvas.draw_idle()
        print(f"Auto optimize: perp_err mean={perp_deg.mean():.3f}° max={perp_deg.max():.3f}°")
    opt_btn.on_clicked(on_opt)
    fig._opt_btn = opt_btn

    ax_save = fig.add_axes([0.8, 0.05, 0.15, 0.05])
    save_btn = Button(ax_save, 'Save & Close')

    def on_save(event):
        # Only center is modified; L/R untouched. C is on segment L[i]-R[i].
        C = state['C'].copy()
        C[-1] = C[0]
        for i, r in enumerate(rows):
            r['center_x'] = f"{C[i, 0]:.6f}"
            r['center_y'] = f"{C[i, 1]:.6f}"
            r['center_z'] = f"{C[i, 2]:.6f}"
        with open(args.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fns)
            w.writeheader()
            w.writerows(rows)
        print(f"Saved centerline (win={state['win']}) -> {args.csv}")
        plt.close(fig)
    save_btn.on_clicked(on_save)
    fig._save_btn = save_btn

    update()
    plt.show()


if __name__ == '__main__':
    main()
