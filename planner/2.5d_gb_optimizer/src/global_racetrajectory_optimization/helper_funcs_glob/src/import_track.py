import numpy as np


def import_track(file_path: str,
                 imp_opts: dict,
                 width_veh: float) -> tuple:
    """
    Created by:
    Alexander Heilmeier
    Modified by:
    Thomas Herrmann

    modified for 2.5D — z as 5th column so spline_approximation resamples it together

    Documentation:
    This function includes the algorithm part connected to the import of the track.

    Inputs:
    file_path:      file path of track.csv containing [x_m,y_m,w_tr_right_m,w_tr_left_m]
                    or 15-column 3D track CSV (s_m,x_m,y_m,z_m,theta,mu,phi,dtheta,dmu,dphi,w_tr_r,w_tr_l,omega_x,omega_y,omega_z)
    imp_opts:       import options showing if a new starting point should be set or if the direction should be reversed
    width_veh:      vehicle width required to check against track width

    Outputs:
    reftrack_imp:   imported track [x_m, y_m, w_tr_right_m, w_tr_left_m] (4-col) or
                    [x_m, y_m, w_tr_right_m, w_tr_left_m, z_m] (5-col if 3D)
    """

    # load data from csv file
    # use genfromtxt to handle CSV with or without header row
    csv_data_temp = np.genfromtxt(file_path, comments='#', delimiter=',')
    if np.isnan(csv_data_temp[0, 0]):
        csv_data_temp = csv_data_temp[1:]

    has_z = False  # flag for 3D track

    # get coords and track widths out of array
    if np.shape(csv_data_temp)[1] == 3:
        refline_ = csv_data_temp[:, 0:2]
        w_tr_r = csv_data_temp[:, 2] / 2
        w_tr_l = w_tr_r
        z_ = None

    elif np.shape(csv_data_temp)[1] == 4:
        refline_ = csv_data_temp[:, 0:2]
        w_tr_r = csv_data_temp[:, 2]
        w_tr_l = csv_data_temp[:, 3]
        z_ = None

    elif np.shape(csv_data_temp)[1] == 5:
        refline_ = csv_data_temp[:, 0:2]
        w_tr_r = csv_data_temp[:, 3]
        w_tr_l = csv_data_temp[:, 4]
        z_ = csv_data_temp[:, 2]  # z as 5th col
        has_z = True

    # 15-column 3D track CSV (Stage 2/3 format from TUMRT pipeline)
    elif np.shape(csv_data_temp)[1] == 15:
        refline_ = csv_data_temp[:, 1:3]   # x_m, y_m
        # TUMRT convention has w_tr_right negative, TUMFTM expects positive
        w_tr_r = np.abs(csv_data_temp[:, 10])
        w_tr_l = np.abs(csv_data_temp[:, 11])
        z_ = csv_data_temp[:, 3]  # z_m
        has_z = True

    else:
        raise IOError("Track file cannot be read!")

    refline_ = np.tile(refline_, (imp_opts["num_laps"], 1))
    w_tr_r = np.tile(w_tr_r, imp_opts["num_laps"])
    w_tr_l = np.tile(w_tr_l, imp_opts["num_laps"])
    if has_z:
        z_ = np.tile(z_, imp_opts["num_laps"])

    # assemble — 5-col [x, y, w_tr_r, w_tr_l, z] if 3D, else 4-col
    if has_z:
        reftrack_imp = np.column_stack((refline_, w_tr_r, w_tr_l, z_))
    else:
        reftrack_imp = np.column_stack((refline_, w_tr_r, w_tr_l))

    # remove closing duplicate if first==last (spline_approximation expects unclosed input)
    if np.allclose(reftrack_imp[0, :2], reftrack_imp[-1, :2], atol=1e-6):
        reftrack_imp = reftrack_imp[:-1]

    # check if imported centerline should be flipped, i.e. reverse direction
    if imp_opts["flip_imp_track"]:
        reftrack_imp = np.flipud(reftrack_imp)

    # check if imported centerline should be reordered for a new starting point
    if imp_opts["set_new_start"]:
        ind_start = np.argmin(np.power(reftrack_imp[:, 0] - imp_opts["new_start"][0], 2)
                              + np.power(reftrack_imp[:, 1] - imp_opts["new_start"][1], 2))
        reftrack_imp = np.roll(reftrack_imp, reftrack_imp.shape[0] - ind_start, axis=0)

    # check minimum track width for vehicle width plus a small safety margin
    w_tr_min = np.amin(reftrack_imp[:, 2] + reftrack_imp[:, 3])

    if w_tr_min < width_veh + 0.5:
        print("WARNING: Minimum track width %.2fm is close to or smaller than vehicle width!" % np.amin(w_tr_min))

    return reftrack_imp


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
