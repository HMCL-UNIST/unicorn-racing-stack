# Parametric version — solver built once, N as parameter (identical results, much faster)
from casadi import *
from tqdm import tqdm


# Probe HSL ma27; fall back to MUMPS if libhsl.so is not loadable.
def _select_linear_solver():
    try:
        _x = MX.sym('x')
        _probe = nlpsol('hsl_probe', 'ipopt',
                        {'x': _x, 'f': (_x - 1.0) ** 2},
                        {'ipopt.linear_solver': 'ma27',
                         'ipopt.print_level': 0, 'print_time': 0})
        _probe(x0=0.0)
        if _probe.stats().get('success', False):
            return 'ma27'
    except Exception:
        pass
    return 'mumps'

_LINEAR_SOLVER = _select_linear_solver()


def calc_max_slip_map(tire_params: dict, debug_plots: bool = False, n_points: int = 200, tol: float = 1e-6):
    N_list = np.linspace(0.5, tire_params["N_max"], n_points)
    N_list_res = []
    kappa_max_list = []
    lambda_max_list = []
    F_x_list = []
    F_y_list = []

    # --- Decision variables ---
    kappa = MX.sym("kappa")
    lambdaa = MX.sym("lambda")
    F_x = MX.sym("F_x")
    F_y = MX.sym("F_y")
    x = vertcat(kappa, lambdaa, F_x, F_y)

    # --- Parameter: normal load N ---
    p_N = MX.sym("p_N")

    # --- Tire model (parametric in N) ---
    df_z = (p_N - tire_params["N_0"]) / tire_params["N_0"]

    sigma_x = kappa / (1 + kappa)
    sigma_y = tan(lambdaa) / (1 + kappa)
    sigma = sqrt(sigma_x**2 + sigma_y**2 + 1e-6)

    K_x = p_N * tire_params["p_Kx_1"] * exp(tire_params["p_Kx_3"] * df_z)
    D_x = (tire_params["p_Dx_1"] + tire_params["p_Dx_2"] * df_z) * tire_params["lambda_mu_x"]
    B_x = K_x / (tire_params["p_Cx_1"] * D_x * p_N)

    K_y = (
        tire_params["N_0"]
        * tire_params["p_Ky_1"]
        * sin(2 * arctan(p_N / (tire_params["p_Ky_2"] * tire_params["N_0"])))
    )
    D_y = (tire_params["p_Dy_1"] + tire_params["p_Dy_2"] * df_z) * tire_params["lambda_mu_y"]
    B_y = K_y / (tire_params["p_Cy_1"] * D_y * p_N)

    # --- Constraints (Pacejka magic formula) ---
    g_con = [
        F_x - p_N * sigma_x / sigma * D_x * sin(
            tire_params["p_Cx_1"] * arctan(B_x * sigma - tire_params["p_Ex_1"] * (B_x * sigma - arctan(B_x * sigma)))
        ),
        F_y - p_N * sigma_y / sigma * D_y * sin(
            tire_params["p_Cy_1"] * arctan(B_y * sigma - tire_params["p_Ey_1"] * (B_y * sigma - arctan(B_y * sigma)))
        ),
    ]

    lbx = vertcat(0, 0, 0, -np.inf)
    ubx = vertcat(tire_params["kappa_max"], tire_params["lambda_max"], np.inf, 0)
    lbg_vec = vertcat(0, 0)
    ubg_vec = vertcat(0, 0)

    # --- Build solvers ONCE (parametric in p_N) ---
    opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.tol": tol,
            "ipopt.max_iter": 200 if tol > 1e-5 else 500,
            "ipopt.linear_solver": _LINEAR_SOLVER}  # ma27 if HSL available, else mumps
    nlp_x = {"x": x, "f": -F_x, "g": vertcat(*g_con), "p": p_N}
    nlp_y = {"x": x, "f": F_y, "g": vertcat(*g_con), "p": p_N}
    solver_x = nlpsol("solver_x", "ipopt", nlp_x, opts)
    solver_y = nlpsol("solver_y", "ipopt", nlp_y, opts)

    # use same fixed initial guess as original (no warm start)
    x0_x = [0.1, 0, tire_params["N_0"], 0]
    x0_y = [0, 0.1, 0, -tire_params["N_0"]]

    print(f"Calculating maximum slips for normal tire loads:\n")
    for i, N in tqdm(enumerate(N_list), total=len(N_list)):
        x_opt_x = solver_x(x0=x0_x, lbx=lbx, ubx=ubx, lbg=lbg_vec, ubg=ubg_vec, p=N)
        x_opt_y = solver_y(x0=x0_y, lbx=lbx, ubx=ubx, lbg=lbg_vec, ubg=ubg_vec, p=N)

        if solver_x.stats()["success"] and solver_y.stats()["success"]:
            N_list_res.append(N)
            kappa_max_list.append(float(x_opt_x["x"][0]))
            F_x_list.append(float(x_opt_x["x"][2]))
            lambda_max_list.append(float(x_opt_y["x"][1]))
            F_y_list.append(float(x_opt_y["x"][3]))

    if debug_plots:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(nrows=2)
        ax[0].set_title("Normal force dependent maximal slip")
        ax[0].plot(N_list_res, kappa_max_list, label=rf"$\kappa$ (longitudinal)", marker='o')
        ax[0].plot(N_list_res, lambda_max_list, label=rf"$\lambda$ (lateral)", marker='o')
        ax[0].legend()
        ax[1].set_title("Tire forces")
        ax[1].plot(N_list_res, np.abs(F_x_list), label=rf"$F_x$", marker='o')
        ax[1].plot(N_list_res, np.abs(F_y_list), label=rf"$F_y$", marker='o')
        ax[1].legend()
        plt.show()

    return N_list_res, kappa_max_list, lambda_max_list

# EOF
