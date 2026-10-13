"""Generate 3D track data from 3D track bounds.

v2: Tangent is derived from boundary perpendicular direction instead of
    np.diff(centerline). This guarantees tangent ⊥ normal and avoids
    distortion on curves.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

# --- config ---
track_name_raw = 'localization_bridge_2_bounds_3d.csv'
processing_method = '3d_track_bounds_to_3d'
track_name_processed = 'localization_bridge_2_3d.csv'

step_size = 0.2  # in meter
visualize = False
ignore_banking = False

# paths
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, '..', 'data')
raw_data_path = os.path.join(data_path, 'raw_track_data')
out_data_path = os.path.join(data_path, '3d_track_data')
os.makedirs(out_data_path, exist_ok=True)
sys.path.append(os.path.join(dir_path, '..', 'src'))

from track3D import Track3D


def generate_3d_from_3d_track_bounds_v2(path, out_path, ignore_banking=False, visualize=False):
    """Generate 3D track data with boundary-derived tangent."""
    print('Generating 3d track file from 3d track bounds (v2: boundary-perpendicular tangent) ...')

    # Import track bounds
    df = pd.read_csv(path, sep=',')
    right_bounds = np.stack([
        df['right_bound_x'].to_numpy(),
        df['right_bound_y'].to_numpy(),
        df['right_bound_z'].to_numpy()
    ]).T
    left_bounds = np.stack([
        df['left_bound_x'].to_numpy(),
        df['left_bound_y'].to_numpy(),
        df['left_bound_z'].to_numpy()
    ]).T

    # Centerline = midpoint of boundaries
    center = (right_bounds + left_bounds) / 2.0

    # Track widths
    w_tr_left = np.linalg.norm(left_bounds - center, axis=1)
    w_tr_right = -np.linalg.norm(right_bounds - center, axis=1)

    # Normal vector (points to left bound)
    normal = (left_bounds - center) / w_tr_left[:, np.newaxis]

    # === v2: Tangent from boundary perpendicular ===
    # Boundary direction: left - right
    bnd_dir = left_bounds - right_bounds
    bnd_dir = bnd_dir / np.linalg.norm(bnd_dir, axis=1)[:, np.newaxis]

    # Tangent = perpendicular to boundary direction in the road plane
    # For 2D: rotate bnd_dir by -90 degrees (so tangent points forward)
    # For 3D: cross product of normal and orthogonal, but simpler:
    #   tangent = cross(bnd_dir, [0,0,1]) for mostly-flat tracks
    #   then project out any normal component to ensure orthogonality

    # Use centerline diff to determine forward direction sign
    center_diff = np.diff(center, axis=0)
    center_diff = np.append(center_diff, center_diff[0][np.newaxis], axis=0)

    # Tangent candidate: perpendicular to boundary in XY, then fix sign
    tang_candidate = np.zeros_like(bnd_dir)
    tang_candidate[:, 0] = -bnd_dir[:, 1]  # rotate 90 CW in XY
    tang_candidate[:, 1] = bnd_dir[:, 0]
    tang_candidate[:, 2] = 0.0

    # Fix sign: tangent should point in same direction as centerline progression
    dot_sign = np.sum(tang_candidate * center_diff, axis=1)
    flip = dot_sign < 0
    tang_candidate[flip] *= -1

    # Normalize
    tangent = tang_candidate / np.linalg.norm(tang_candidate, axis=1)[:, np.newaxis]

    # Orthogonal = cross(tangent, normal)
    ortho = np.cross(tangent, normal, axis=1)
    ortho = ortho / np.linalg.norm(ortho, axis=1)[:, np.newaxis]

    # Verify perpendicularity
    tdotn = np.sum(tangent * normal, axis=1)
    print(f'  Tangent·Normal: max|t·n|={np.abs(tdotn).max():.6f}, mean={np.abs(tdotn).mean():.6f}')

    # Rotation matrices -> euler angles
    rot_mat = np.stack([tangent, normal, ortho], axis=1)
    euler_angles = -Rotation.from_matrix(rot_mat).as_euler('zyx')

    if ignore_banking:
        euler_angles[:, 1:] = 0.0
        center[:, 2] = 0.0

    # s-coordinate
    s_m = np.cumsum(np.sqrt(np.sum(np.square(np.diff(center, axis=0)), axis=1)))
    s_m = np.insert(s_m, 0, 0.0)

    # Angular velocities
    ds = np.diff(s_m)
    deuler = np.diff(np.unwrap(euler_angles, axis=0), axis=0) / np.array([ds, ds, ds]).T
    deuler = np.vstack([deuler, deuler[0]])

    # Use Track3D's jacobian for omega calculation
    track_handler = Track3D()
    omega = np.zeros((len(euler_angles), 3))
    for i in range(len(euler_angles)):
        Jac = track_handler.get_jacobian_J(euler_angles[i, 1], euler_angles[i, 2])
        d = np.array([deuler[i, 2], deuler[i, 1], deuler[i, 0]])
        omega[i] = Jac.dot(d)

    if visualize:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(3, 1, figsize=(12, 8))
        axs[0].plot(s_m, np.degrees(euler_angles[:, 0]), label='theta')
        axs[0].plot(s_m, np.degrees(euler_angles[:, 1]), label='mu')
        axs[0].plot(s_m, np.degrees(euler_angles[:, 2]), label='phi')
        axs[0].legend(); axs[0].grid(); axs[0].set_ylabel('deg')
        axs[1].plot(s_m, np.degrees(deuler[:, 0]), label='dtheta/ds')
        axs[1].plot(s_m, np.degrees(deuler[:, 1]), label='dmu/ds')
        axs[1].plot(s_m, np.degrees(deuler[:, 2]), label='dphi/ds')
        axs[1].legend(); axs[1].grid(); axs[1].set_ylabel('deg/m')
        axs[2].plot(s_m, np.degrees(omega[:, 0]), label='Omega_x')
        axs[2].plot(s_m, np.degrees(omega[:, 1]), label='Omega_y')
        axs[2].plot(s_m, np.degrees(omega[:, 2]), label='Omega_z')
        axs[2].legend(); axs[2].grid(); axs[2].set_ylabel('deg/m')
        plt.tight_layout(); plt.show()

    # Save
    out = pd.DataFrame()
    out['s_m'] = s_m
    out['x_m'] = center[:, 0]
    out['y_m'] = center[:, 1]
    out['z_m'] = center[:, 2]
    out['theta_rad'] = euler_angles[:, 0]
    out['mu_rad'] = euler_angles[:, 1]
    out['phi_rad'] = euler_angles[:, 2]
    out['dtheta_radpm'] = deuler[:, 0]
    out['dmu_radpm'] = deuler[:, 1]
    out['dphi_radpm'] = deuler[:, 2]
    out['w_tr_right_m'] = w_tr_right
    out['w_tr_left_m'] = w_tr_left
    out['omega_x_radpm'] = omega[:, 0]
    out['omega_y_radpm'] = omega[:, 1]
    out['omega_z_radpm'] = omega[:, 2]
    out.to_csv(out_path, sep=',', index=False, float_format='%.6f')
    print(f'  Saved to {out_path}')


if __name__ == '__main__':
    generate_3d_from_3d_track_bounds_v2(
        path=os.path.join(raw_data_path, track_name_raw),
        out_path=os.path.join(out_data_path, track_name_processed),
        ignore_banking=ignore_banking,
        visualize=visualize,
    )
