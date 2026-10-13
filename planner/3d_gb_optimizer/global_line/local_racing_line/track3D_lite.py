# Track3D lightweight version — pandas-free (uses numpy for CSV reading)
# Original: ../src/track3D.py
# Only includes methods needed by local_raceline_mux_node:
# __init__, lock_track_data, sn2cartesian, calc_apparent_accelerations,
# get_track_bounds, get_rotation_matrix_numpy, get_normal_vector_numpy,
# get_normal_vector_casadi, get_jacobian_J

import numpy as np
import casadi as ca

g_earth = 9.81


class Track3D:

    def __init__(self, path=None):
        self.__path = path

        if self.__path:
            # numpy.genfromtxt instead of pd.read_csv
            raw = np.genfromtxt(path, delimiter=',', names=True)
            self.__column_names = raw.dtype.names
            self.__data = {name: raw[name] for name in self.__column_names}
            self.track_locked = True
        else:
            self.__data = None
            self.track_locked = False

    @property
    def track_locked(self):
        return self.__track_locked

    @track_locked.setter
    def track_locked(self, value):
        if value:
            self.lock_track_data()
        self.__track_locked = value

    def lock_track_data(self):
        # discretized points of track spine
        self.s = self.__data['s_m']
        self.ds = np.mean(np.diff(self.s))
        self.x = self.__data['x_m']
        self.y = self.__data['y_m']
        self.z = self.__data['z_m']
        self.theta = self.__data['theta_rad']
        self.mu = self.__data['mu_rad']
        self.phi = self.__data['phi_rad']
        self.w_tr_right = self.__data['w_tr_right_m']
        self.w_tr_left = self.__data['w_tr_left_m']
        self.Omega_x = self.__data['omega_x_radpm']
        self.Omega_y = self.__data['omega_y_radpm']
        self.Omega_z = self.__data['omega_z_radpm']

        # derivatives of omega with finite differencing
        self.dOmega_x = np.diff(self.Omega_x) / self.ds
        self.dOmega_x = np.append(self.dOmega_x, self.dOmega_x[0])
        self.dOmega_y = np.diff(self.Omega_y) / self.ds
        self.dOmega_y = np.append(self.dOmega_y, self.dOmega_y[0])
        self.dOmega_z = np.diff(self.Omega_z) / self.ds
        self.dOmega_z = np.append(self.dOmega_z, self.dOmega_z[0])

        # track spine interpolator
        def concatenate_arr(arr):
            return np.concatenate((arr, arr[1:], arr[1:]))  # 2 track lengths
        s_augmented = np.concatenate((self.s, self.s[-1] + self.s[1:], 2 * self.s[-1] + self.s[1:]))

        # casadi interpolator instances
        self.x_interpolator = ca.interpolant('x', 'linear', [s_augmented], concatenate_arr(self.x))
        self.y_interpolator = ca.interpolant('y', 'linear', [s_augmented], concatenate_arr(self.y))
        self.z_interpolator = ca.interpolant('z', 'linear', [s_augmented], concatenate_arr(self.z))
        self.theta_interpolator = ca.interpolant('theta', 'linear', [s_augmented], concatenate_arr(self.theta))
        self.mu_interpolator = ca.interpolant('mu', 'linear', [s_augmented], concatenate_arr(self.mu))
        self.phi_interpolator = ca.interpolant('phi', 'linear', [s_augmented], concatenate_arr(self.phi))
        self.w_tr_right_interpolator = ca.interpolant('w_tr_right', 'linear', [s_augmented], concatenate_arr(self.w_tr_right))
        self.w_tr_left_interpolator = ca.interpolant('w_tr_left', 'linear', [s_augmented], concatenate_arr(self.w_tr_left))
        self.Omega_x_interpolator = ca.interpolant('omega_x', 'linear', [s_augmented], concatenate_arr(self.Omega_x))
        self.Omega_y_interpolator = ca.interpolant('omega_y', 'linear', [s_augmented], concatenate_arr(self.Omega_y))
        self.Omega_z_interpolator = ca.interpolant('omega_z', 'linear', [s_augmented], concatenate_arr(self.Omega_z))
        self.dOmega_x_interpolator = ca.interpolant('domega_x', 'linear', [s_augmented], concatenate_arr(self.dOmega_x))
        self.dOmega_y_interpolator = ca.interpolant('domega_y', 'linear', [s_augmented], concatenate_arr(self.dOmega_y))
        self.dOmega_z_interpolator = ca.interpolant('domega_z', 'linear', [s_augmented], concatenate_arr(self.dOmega_z))

    def sn2cartesian(self, s, n, normal_vector_factor: float = 1.0):
        if not self.track_locked:
            raise RuntimeError('Cannot transform. Track is not locked.')
        # convert casadi DM to numpy before stacking (DM arrays break np.array)
        euler_p = np.array([
            np.array(self.theta_interpolator(s)).flatten(),
            np.array(self.mu_interpolator(s)).flatten(),
            np.array(self.phi_interpolator(s)).flatten(),
        ])
        ref_p = np.array([
            np.array(self.x_interpolator(s)).flatten(),
            np.array(self.y_interpolator(s)).flatten(),
            np.array(self.z_interpolator(s)).flatten(),
        ]).transpose()

        return ref_p + (self.get_normal_vector_numpy(*euler_p) * normal_vector_factor * n).transpose()

    def calc_apparent_accelerations(
            self, V, n, chi, ax, ay, s, h,
            neglect_w_omega_y: bool = True, neglect_w_omega_x: bool = True, neglect_euler: bool = True,
            neglect_centrifugal: bool = True, neglect_w_dot: bool = False, neglect_V_omega: bool = False,
    ):
        if not self.track_locked:
            raise RuntimeError('Cannot calculate apparent accelerations. Track is not locked.')

        mu = self.mu_interpolator(s)
        phi = self.phi_interpolator(s)
        Omega_x = self.Omega_x_interpolator(s)
        dOmega_x = self.dOmega_x_interpolator(s)
        Omega_y = self.Omega_y_interpolator(s)
        dOmega_y = self.dOmega_y_interpolator(s)
        Omega_z = self.Omega_z_interpolator(s)
        dOmega_z = self.dOmega_z_interpolator(s)

        s_dot = (V * ca.cos(chi)) / (1.0 - n * Omega_z)
        w = n * Omega_x * s_dot

        V_dot = ax
        if not neglect_w_omega_y:
            V_dot += w * (Omega_x * ca.sin(chi) - Omega_y * ca.cos(chi)) * s_dot

        n_dot = V * ca.sin(chi)

        chi_dot = ay / V - Omega_z * s_dot
        if not neglect_w_omega_x:
            chi_dot += w * (Omega_x * ca.cos(chi) + Omega_y * ca.sin(chi)) * s_dot / V

        s_ddot = ((V_dot * ca.cos(chi) - V * ca.sin(chi) * chi_dot) * (1.0 - n * Omega_z) - (V * ca.cos(chi)) * (- n_dot * Omega_z - n * dOmega_z * s_dot)) / (1.0 + 2.0 * n * Omega_z + n ** 2 * Omega_z ** 2)

        omega_x_dot = 0.0
        omega_y_dot = 0.0
        if not neglect_euler:
            omega_x_dot = (dOmega_x * s_dot * ca.cos(chi) - Omega_x * ca.sin(chi) * chi_dot + dOmega_y * s_dot * ca.sin(chi) + Omega_y * ca.cos(chi) * chi_dot) * s_dot + (Omega_x * ca.cos(chi) + Omega_y * ca.sin(chi)) * s_ddot
            omega_y_dot = (-dOmega_x * s_dot * ca.sin(chi) - Omega_x * ca.cos(chi) * chi_dot + dOmega_y * s_dot * ca.cos(chi) - Omega_y * ca.sin(chi) * chi_dot) * s_dot + (- Omega_x * ca.sin(chi) + Omega_y * ca.cos(chi)) * s_ddot

        omega_x = 0.0
        omega_y = 0.0
        omega_z = 0.0
        if not neglect_centrifugal:
            omega_x = (Omega_x * ca.cos(chi) + Omega_y * ca.sin(chi)) * s_dot
            omega_y = (- Omega_x * ca.sin(chi) + Omega_y * ca.cos(chi)) * s_dot
            omega_z = Omega_z * s_dot + chi_dot

        w_dot = 0.0
        if not neglect_w_dot:
            w_dot = n_dot * Omega_x * s_dot + n * dOmega_x * s_dot ** 2 + n * Omega_x * s_ddot

        V_omega = 0.0
        if not neglect_V_omega:
            V_omega = (- Omega_x * ca.sin(chi) + Omega_y * ca.cos(chi)) * s_dot * V

        ax_tilde = ax + omega_y_dot * h - omega_z * omega_x * h + g_earth * (- ca.sin(mu) * ca.cos(chi) + ca.cos(mu) * ca.sin(phi) * ca.sin(chi))
        ay_tilde = ay + omega_x_dot * h + omega_z * omega_y * h + g_earth * (ca.sin(mu) * ca.sin(chi) + ca.cos(mu) * ca.sin(phi) * ca.cos(chi))
        g_tilde = ca.fmax(w_dot - V_omega + (omega_x ** 2 - omega_y ** 2) * h + g_earth * ca.cos(mu) * ca.cos(phi), 0.0)

        return ax_tilde, ay_tilde, g_tilde

    def get_track_bounds(self, margin=0.0):
        normal_vector = self.get_normal_vector_numpy(self.theta, self.mu, self.phi)
        left = np.array([self.x + normal_vector[0] * (self.w_tr_left + margin),
                         self.y + normal_vector[1] * (self.w_tr_left + margin),
                         self.z + normal_vector[2] * (self.w_tr_left + margin)])
        right = np.array([self.x + normal_vector[0] * (self.w_tr_right - margin),
                          self.y + normal_vector[1] * (self.w_tr_right - margin),
                          self.z + normal_vector[2] * (self.w_tr_right - margin)])
        return left, right

    @staticmethod
    def get_rotation_matrix_numpy(theta, mu, phi):
        return np.array([
            [np.cos(theta) * np.cos(mu), np.cos(theta) * np.sin(mu) * np.sin(phi) - np.sin(theta) * np.cos(phi), np.cos(theta) * np.sin(mu) * np.cos(phi) + np.sin(theta) * np.sin(phi)],
            [np.sin(theta) * np.cos(mu), np.sin(theta) * np.sin(mu) * np.sin(phi) + np.cos(theta) * np.cos(phi), np.sin(theta) * np.sin(mu) * np.cos(phi) - np.cos(theta) * np.sin(phi)],
            [- np.sin(mu), np.cos(mu) * np.sin(phi), np.cos(mu) * np.cos(phi)]
        ]).squeeze()

    @staticmethod
    def get_normal_vector_numpy(theta, mu, phi):
        return Track3D.get_rotation_matrix_numpy(theta, mu, phi)[:, 1]

    @staticmethod
    def get_normal_vector_casadi(theta, mu, phi):
        return ca.vertcat(
            ca.cos(theta) * ca.sin(mu) * ca.sin(phi) - ca.sin(theta) * ca.cos(phi),
            ca.sin(theta) * ca.sin(mu) * ca.sin(phi) + ca.cos(theta) * ca.cos(phi),
            ca.cos(mu) * ca.sin(phi)
        )

    @staticmethod
    def get_jacobian_J(mu, phi):
        return np.array([
            [1, 0, -np.sin(mu)],
            [0, np.cos(phi), np.cos(mu) * np.sin(phi)],
            [0, -np.sin(phi), np.cos(mu) * np.cos(phi)]
        ])

# EOF
