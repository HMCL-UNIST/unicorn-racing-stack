import os
import sys
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection

params = {
    'track_name': 'experiment_3d_2_3d_smoothed_waypoint1.csv',
    'raceline_name': 'experiment_3d_2_3d_smoothed_waypoint1_timeoptimal.csv',
    'vehicle_name': 'rc_car_10th',
    'save_fig': True,
}

# paths
dir_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(dir_path, '..', 'data')
track_path = os.path.join(data_path, 'smoothed_track_data', params['track_name'])
racing_line_path = os.path.join(data_path, 'global_racing_lines', params['raceline_name'])
figure_path = os.path.join(data_path, 'figure')
sys.path.append(os.path.join(dir_path, '..', 'src'))

from track3D import Track3D


def plot_racing_line_full(track_path, raceline_path):
    # load track
    track = Track3D(path=track_path)
    normal_vector = track.get_normal_vector_numpy(
        theta=track.theta, mu=track.mu, phi=track.phi
    )
    left, right = track.get_track_bounds()

    # load raceline
    df = pd.read_csv(raceline_path, sep=',')
    s = df['s_opt'].to_numpy()
    v = df['v_opt'].to_numpy()
    n = df['n_opt'].to_numpy()
    ax_opt = df['ax_opt'].to_numpy()
    ay_opt = df['ay_opt'].to_numpy()
    laptime = df['laptime'].iloc[0]

    # racing line in cartesian
    rl_x = track.x + normal_vector[0] * n
    rl_y = track.y + normal_vector[1] * n
    rl_z = track.z + normal_vector[2] * n

    # extract track name for title
    track_name = os.path.splitext(params['track_name'])[0].replace('_3d_smoothed', '')

    # figure
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f'{track_name} — RC Car Racing Line (Laptime: {laptime:.2f}s)', fontsize=14)

    # --- 2D Racing Line (color=velocity) ---
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_title('2D: Racing Line (color=velocity)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.plot(left[0], left[1], 'k', linewidth=0.8, label='track bounds')
    ax1.plot(right[0], right[1], 'k', linewidth=0.8)
    ax1.plot(track.x, track.y, '--', color='gray', linewidth=0.5, label='centerline')

    # color-coded racing line by velocity
    points = np.array([rl_x, rl_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = Normalize(vmin=v.min(), vmax=v.max())
    lc = LineCollection(segments, cmap='RdYlGn', norm=norm)
    lc.set_array(v[:-1])
    lc.set_linewidth(2.5)
    ax1.add_collection(lc)
    cbar = fig.colorbar(lc, ax=ax1, pad=0.02)
    cbar.set_label('V [m/s]')
    ax1.scatter(rl_x[0], rl_y[0], color='blue', s=30, zorder=5, label='start')
    ax1.legend(fontsize=7, loc='best')
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')

    # --- 3D Racing Line ---
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    ax2.set_title('3D: Racing Line')
    ax2.plot(left[0], left[1], left[2], 'k', linewidth=0.5, alpha=0.5)
    ax2.plot(right[0], right[1], right[2], 'k', linewidth=0.5, alpha=0.5)

    # color-coded 3D racing line
    points_3d = np.array([rl_x, rl_y, rl_z]).T.reshape(-1, 1, 3)
    segments_3d = np.concatenate([points_3d[:-1], points_3d[1:]], axis=1)
    lc3d = Line3DCollection(segments_3d, cmap='RdYlGn', norm=norm)
    lc3d.set_array(v[:-1])
    lc3d.set_linewidth(2)
    ax2.add_collection3d(lc3d)
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.set_zlabel('z [m]')
    ax2.set_box_aspect((np.ptp(left[0]), np.ptp(left[1]), np.ptp(left[2])))

    # --- Velocity Profile ---
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title('Velocity Profile')
    ax3.fill_between(s, v, alpha=0.3, color='green')
    ax3.plot(s, v, color='green', linewidth=1.2)
    ax3.set_xlabel('s [m]')
    ax3.set_ylabel('V [m/s]')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([s[0], s[-1]])
    ax3.set_ylim(bottom=0)

    # --- Acceleration Profile ---
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title('Acceleration Profile')
    ax4.plot(s, ax_opt, label=r'$a_x$ [lon]', color='red', linewidth=1.2)
    ax4.plot(s, ay_opt, label=r'$a_y$ [lat]', color='blue', linewidth=1.2)
    ax4.axhline(y=0, color='k', linewidth=0.5)
    ax4.set_xlabel('s [m]')
    ax4.set_ylabel('a [m/s²]')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([s[0], s[-1]])

    plt.tight_layout()

    if params['save_fig']:
        os.makedirs(figure_path, exist_ok=True)
        save_name = track_name + '_racing_line_full.png'
        fig.savefig(os.path.join(figure_path, save_name), dpi=150, bbox_inches='tight')
        print(f'Saved: {os.path.join(figure_path, save_name)}')

    plt.show()


if __name__ == '__main__':
    plot_racing_line_full(track_path, racing_line_path)
