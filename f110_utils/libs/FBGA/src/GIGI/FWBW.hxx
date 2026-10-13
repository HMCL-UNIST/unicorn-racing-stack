/*
(***********************************************************************************)
(*                                                                                 *)
(* The GIGI project                                                               *)
(*                                                                                 *)
(* Copyright (c) 2025, Mattia Piazza                                               *)
(*                                                                                 *)
(*    Mattia Piazza                                                                *)
(*    Department of Industrial Engineering                                         *)
(*    University of Trento                                                         *)
(*    e-mail: mattia.piazza@unitn.it                                               *)
(*                                                                                 *)
(***********************************************************************************)
*/

#ifndef FWBW_HH
#define FWBW_HH

#include "brentdekker.hxx"
#include "segment.hxx"
#include "types.hxx"
#include <functional>

namespace GG
{

// using gg_range_max_min = struct gg_range_max_min
// {
//   std::function<real(real)> min = nullptr;
//   std::function<real(real)> max = nullptr;
// };
// add g (apparent gravity) argument for 3D track support
// add slope argument for 3D ggv slope-aware lookup (slope is propagated to the FWBW lambda call sites)
using gg_range_max_min = struct gg_range_max_min
{
  std::function<real(real, real, real)> min = nullptr;   // (v, g, slope)
  std::function<real(real, real, real)> max = nullptr;   // (v, g, slope)
};

using solver_params = struct solver_params
{
  real tolerance        = STD_TOL;
  int max_iter          = STD_MAX_ITER;
  std::string verbosity = STD_VERBOSE;
};

class FWBW
{
private:
  /* data */
  // std::function<real(real, real)> gg_Upper = nullptr; // Upper bound function
  // std::function<real(real, real)> gg_Lower = nullptr; // Lower bound function
  // add g argument to upper/lower bound functions for 3D track
  // add slope argument for 3D ggv slope-aware lookup
  std::function<real(real, real, real, real)> gg_Upper = nullptr; // Upper bound function (ay, v, g, slope)
  std::function<real(real, real, real, real)> gg_Lower = nullptr; // Lower bound function (ay, v, g, slope)
  // gg_exponent function for Vmax mu correction with exact diamond constraint
  // add slope argument for 3D ggv slope-aware lookup
  std::function<real(real, real, real)> gg_Exp = nullptr;  // (v, g, slope) -> exponent p
  gg_range_max_min gg_range;     // Range of the curvature (struct with min and max functions)
  brentdekker BD;                // Solver
  solver_params solver_p;        // Solver parameters
  std::vector<segment> Segments; // Vector of segments
  std::vector<real> Vmax_vec;    // Vector of maximum reachable velocities
  std::vector<real> S_vec;       // Vector of abscissas
  std::vector<real> K_vec;       // Vector of curvatures
  // apparent gravity per node (3D track support)
  std::vector<real> G_vec;       // Vector of apparent gravity values
  // slope angle per node for ax_tilde = ax - g*sin(mu) correction
  std::vector<real> MU_vec;      // Vector of slope angles [rad]
  // slope rate per node for g_tilde Vmax constraint (v < sqrt(g*cos(mu)/dmu_ds))
  std::vector<real> DMU_vec;     // Vector of dmu/ds [rad/m]
  real v_I{0.0};                 // Initial velocity
  // 2.5d_vel_planner slope_correction (1=physics, higher=more conservative)
  //   Multiplies g*sin(mu) ax_gravity in Vmax/FW/BW. Set via set_slope_corr().
  real slope_corr_{1.0};
  std::vector<int> dump_seg_id;  // Vector of segments with problems for debug
public:
  // constructors
  // FWBW(
  //   const std::function<real(real, real)> &gg_Upper,
  //   const std::function<real(real, real)> &gg_Lower,
  //   const gg_range_max_min &gg_range
  // );
  // extend gg_Upper/gg_Lower signatures with g argument
  // add gg_Exp for Vmax mu correction
  // extend lambda signatures with slope argument for 3D ggv slope-aware lookup
  FWBW(
    const std::function<real(real, real, real, real)> &gg_Upper,
    const std::function<real(real, real, real, real)> &gg_Lower,
    const gg_range_max_min &gg_range,
    const std::function<real(real, real, real)> &gg_Exp = nullptr
  );
  // main methods
  // core Forward-Backward method
  // real compute(std::vector<real> const &SS, std::vector<real> const &KK, real v0);
  // add G_vec input (apparent gravity per node)
  // add MU_vec + DMU_vec input for 3D slope correction
  real compute(
    std::vector<real> const &SS,
    std::vector<real> const &KK,
    std::vector<real> const &GG_,
    std::vector<real> const &MU_,
    std::vector<real> const &DMU_,
    real v0
  );
  // compute Vmax vector
  void compute_Vmax();
  // Forward step
  void FW();
  // Backward step
  void BW();
  // compute time
  [[nodiscard]] real compute_time() const;
  // compute the distance with sign.
  // [[nodiscard]] real signed_distance(real ax, real ay, real v) const;
  // add g argument
  // add slope argument for 3D ggv slope-aware lookup
  [[nodiscard]] real signed_distance(real ax, real ay, real v, real g, real slope) const;
  // check if a point is in the range
  // [[nodiscard]] bool is_in_range(real ax, real ay, real v) const;
  // add g argument
  // add slope argument for 3D ggv slope-aware lookup
  [[nodiscard]] bool is_in_range(real ax, real ay, real v, real g, real slope) const;
  // evaluation
  void evaluate(
    std::vector<real> const &SS, std::vector<real> &AX, std::vector<real> &AY, std::vector<real> &V
  );
  [[nodiscard]] real evalV(real s) const;
  [[nodiscard]] real evalAx(real s) const;
  [[nodiscard]] real evalAy(real s) const;
  [[nodiscard]] integer get_seg_idx(real s) const;
  [[nodiscard]] real evalVmax(const real s) const;

  // get dump
  [[nodiscard]] std::vector<int> get_dump() const { return this->dump_seg_id; }

  // 2.5d_vel_planner slope_correction setter
  void set_slope_corr(real c) { this->slope_corr_ = c; }

  void check_segments();
};

} // namespace GG

#endif // FWBW_HH
