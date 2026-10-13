// pybind11 binding for FBGA's GIGI/FWBW.
// Goal: eliminate the per-solve subprocess fork by exposing FWBW as a Python
// module. params.txt and gg.bin are loaded ONCE in the FBGANative ctor; each
// solve() call is a direct in-process function call (no fork, no CSV, no pipe).
//
// The VehicleParams / GGTable / GGModel helpers are copied verbatim from
// src_tests/GIGI_test_unicorn.cc (the CLI driver). If you ever change one
// side, also change the other. Marked with `// from GIGI_test_unicorn.cc`.

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "GIGI/FWBW.hxx"
#include "GIGI/gg_utils.hxx"
#include "GIGI/types.hxx"

namespace py = pybind11;

// -------------------------------------------------------------------------
// from GIGI_test_unicorn.cc  — VehicleParams + key=value loader
// -------------------------------------------------------------------------
struct VehicleParams {
  GG::real m       = 4.38;
  GG::real P_max   = 267.0;
  GG::real P_brake = 50.0;
  GG::real mu_x    = 1.0;
  GG::real mu_y    = 1.0;
  GG::real v_max   = 12.0;
};

static VehicleParams read_params(const std::string& path) {
  VehicleParams p;
  std::ifstream f(path);
  if (!f.is_open()) {
    throw std::runtime_error("read_params: cannot open " + path);
  }
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    auto eq = line.find('=');
    if (eq == std::string::npos) continue;
    std::string key = line.substr(0, eq);
    GG::real    val = std::stod(line.substr(eq + 1));
    if      (key == "m")       p.m       = val;
    else if (key == "P_max")   p.P_max   = val;
    else if (key == "P_brake") p.P_brake = val;
    else if (key == "mu_x")    p.mu_x    = val;
    else if (key == "mu_y")    p.mu_y    = val;
    else if (key == "v_max")   p.v_max   = val;
  }
  return p;
}

// -------------------------------------------------------------------------
// from GIGI_test_unicorn.cc  — GGTable + binary loader (2D / 3D auto)
// -------------------------------------------------------------------------
struct GGTable {
  int n_v = 0, n_g = 0, n_s = 0;
  std::vector<GG::real> v_list, g_list, slope_list;
  std::vector<GG::real> ax_max, ax_min, ay_max, gg_exp;

  GG::real interp2d(const std::vector<GG::real>& tbl, GG::real v, GG::real g) const {
    if (v < v_list.front()) v = v_list.front();
    if (v > v_list.back())  v = v_list.back();
    if (g < g_list.front()) g = g_list.front();
    if (g > g_list.back())  g = g_list.back();
    int iv = 0;
    while (iv + 1 < n_v - 1 && v_list[iv + 1] < v) iv++;
    int ig = 0;
    while (ig + 1 < n_g - 1 && g_list[ig + 1] < g) ig++;
    GG::real v0 = v_list[iv],     v1 = v_list[iv + 1];
    GG::real g0 = g_list[ig],     g1 = g_list[ig + 1];
    GG::real tv = (v1 > v0) ? (v - v0) / (v1 - v0) : GG::real(0.0);
    GG::real tg = (g1 > g0) ? (g - g0) / (g1 - g0) : GG::real(0.0);
    GG::real q00 = tbl[iv     * n_g + ig    ];
    GG::real q01 = tbl[iv     * n_g + ig + 1];
    GG::real q10 = tbl[(iv+1) * n_g + ig    ];
    GG::real q11 = tbl[(iv+1) * n_g + ig + 1];
    return (1-tv)*(1-tg)*q00 + (1-tv)*tg*q01 + tv*(1-tg)*q10 + tv*tg*q11;
  }

  GG::real interp3d(const std::vector<GG::real>& tbl,
                    GG::real v, GG::real g, GG::real slope) const {
    if (v < v_list.front()) v = v_list.front();
    if (v > v_list.back())  v = v_list.back();
    if (g < g_list.front()) g = g_list.front();
    if (g > g_list.back())  g = g_list.back();
    if (slope < slope_list.front()) slope = slope_list.front();
    if (slope > slope_list.back())  slope = slope_list.back();
    int iv = 0;
    while (iv + 1 < n_v - 1 && v_list[iv + 1] < v) iv++;
    int ig = 0;
    while (ig + 1 < n_g - 1 && g_list[ig + 1] < g) ig++;
    int is_ = 0;
    while (is_ + 1 < n_s - 1 && slope_list[is_ + 1] < slope) is_++;
    GG::real tv = (v_list[iv+1] > v_list[iv]) ? (v - v_list[iv]) / (v_list[iv+1] - v_list[iv]) : 0.0;
    GG::real tg = (g_list[ig+1] > g_list[ig]) ? (g - g_list[ig]) / (g_list[ig+1] - g_list[ig]) : 0.0;
    GG::real ts = (slope_list[is_+1] > slope_list[is_]) ? (slope - slope_list[is_]) / (slope_list[is_+1] - slope_list[is_]) : 0.0;
    auto idx = [&](int vi, int gi, int si) -> size_t {
      return static_cast<size_t>(vi) * n_g * n_s + gi * n_s + si;
    };
    GG::real c000 = tbl[idx(iv,   ig,   is_  )];
    GG::real c001 = tbl[idx(iv,   ig,   is_+1)];
    GG::real c010 = tbl[idx(iv,   ig+1, is_  )];
    GG::real c011 = tbl[idx(iv,   ig+1, is_+1)];
    GG::real c100 = tbl[idx(iv+1, ig,   is_  )];
    GG::real c101 = tbl[idx(iv+1, ig,   is_+1)];
    GG::real c110 = tbl[idx(iv+1, ig+1, is_  )];
    GG::real c111 = tbl[idx(iv+1, ig+1, is_+1)];
    GG::real c00 = c000*(1-ts) + c001*ts;
    GG::real c01 = c010*(1-ts) + c011*ts;
    GG::real c10 = c100*(1-ts) + c101*ts;
    GG::real c11 = c110*(1-ts) + c111*ts;
    GG::real c0 = c00*(1-tg) + c01*tg;
    GG::real c1 = c10*(1-tg) + c11*tg;
    return c0*(1-tv) + c1*tv;
  }

  GG::real interp(const std::vector<GG::real>& tbl,
                  GG::real v, GG::real g, GG::real slope = 0.0) const {
    if (n_s <= 1) return interp2d(tbl, v, g);
    return interp3d(tbl, v, g, slope);
  }

  GG::real ax_max_at(GG::real v, GG::real g, GG::real slope = 0.0) const { return interp(ax_max, v, g, slope); }
  GG::real ax_min_at(GG::real v, GG::real g, GG::real slope = 0.0) const { return interp(ax_min, v, g, slope); }
  GG::real ay_max_at(GG::real v, GG::real g, GG::real slope = 0.0) const { return interp(ay_max, v, g, slope); }
  GG::real gg_exp_at(GG::real v, GG::real g, GG::real slope = 0.0) const { return interp(gg_exp, v, g, slope); }
};

static GGTable read_gg_binary(const std::string& path) {
  GGTable t;
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error("read_gg_binary: cannot open " + path);
  }
  uint32_t nv = 0, ng = 0;
  f.read(reinterpret_cast<char*>(&nv), sizeof(uint32_t));
  f.read(reinterpret_cast<char*>(&ng), sizeof(uint32_t));
  t.n_v = static_cast<int>(nv);
  t.n_g = static_cast<int>(ng);

  auto read_vec = [&](std::vector<GG::real>& v, size_t n) {
    v.resize(n);
    f.read(reinterpret_cast<char*>(v.data()), n * sizeof(double));
  };

  f.seekg(0, std::ios::end);
  size_t file_size = f.tellg();
  size_t expected_2d = 8 + 8*nv + 8*ng + 4*8*static_cast<size_t>(nv)*ng;
  f.seekg(8, std::ios::beg);

  if (file_size > expected_2d) {
    uint32_t ns = 0;
    f.read(reinterpret_cast<char*>(&ns), sizeof(uint32_t));
    t.n_s = static_cast<int>(ns);
    read_vec(t.v_list, nv);
    read_vec(t.g_list, ng);
    read_vec(t.slope_list, ns);
    size_t total = static_cast<size_t>(nv) * ng * ns;
    read_vec(t.ax_max, total);
    read_vec(t.ax_min, total);
    read_vec(t.ay_max, total);
    read_vec(t.gg_exp, total);
  } else {
    t.n_s = 0;
    read_vec(t.v_list, nv);
    read_vec(t.g_list, ng);
    size_t total = static_cast<size_t>(nv) * ng;
    read_vec(t.ax_max, total);
    read_vec(t.ax_min, total);
    read_vec(t.ay_max, total);
    read_vec(t.gg_exp, total);
  }
  return t;
}

// -------------------------------------------------------------------------
// from GIGI_test_unicorn.cc  — GGModel factory (lambdas captured by FWBW)
// -------------------------------------------------------------------------
struct GGModel {
  std::function<GG::real(GG::real, GG::real, GG::real, GG::real)> upper;
  std::function<GG::real(GG::real, GG::real, GG::real, GG::real)> lower;
  GG::gg_range_max_min range;
  std::function<GG::real(GG::real, GG::real, GG::real)> exp_func;
};

static GGModel make_lookup(const GGTable* tbl) {
  // tbl is captured by pointer; the FBGANative wrapper owns it for the
  // lifetime of FWBW. (Captures by ref to a member would also work but
  // pointer makes the ownership explicit.)
  auto upper = [tbl](GG::real ay, GG::real v, GG::real g, GG::real slope) -> GG::real {
    GG::real ax_max_val = tbl->ax_max_at(v, g, slope);
    GG::real ay_max_val = tbl->ay_max_at(v, g, slope);
    GG::real p          = tbl->gg_exp_at(v, g, slope);
    GG::real ratio  = std::min(std::abs(ay) / ay_max_val, GG::real(1.0));
    GG::real factor = std::pow(std::max(GG::real(1.0) - std::pow(ratio, p), GG::real(0.0)),
                               GG::real(1.0) / p);
    return ax_max_val * factor;
  };
  auto lower = [tbl](GG::real ay, GG::real v, GG::real g, GG::real slope) -> GG::real {
    GG::real ax_min_val = tbl->ax_min_at(v, g, slope);
    GG::real ay_max_val = tbl->ay_max_at(v, g, slope);
    GG::real p          = tbl->gg_exp_at(v, g, slope);
    GG::real ratio  = std::min(std::abs(ay) / ay_max_val, GG::real(1.0));
    GG::real factor = std::pow(std::max(GG::real(1.0) - std::pow(ratio, p), GG::real(0.0)),
                               GG::real(1.0) / p);
    return ax_min_val * factor;
  };
  GG::gg_range_max_min range{
    [tbl](GG::real v, GG::real g, GG::real slope) { return -tbl->ay_max_at(v, g, slope); },
    [tbl](GG::real v, GG::real g, GG::real slope) { return  tbl->ay_max_at(v, g, slope); }
  };
  auto exp_func = [tbl](GG::real v, GG::real g, GG::real slope) -> GG::real {
    return tbl->gg_exp_at(v, g, slope);
  };
  return {upper, lower, range, exp_func};
}

// -------------------------------------------------------------------------
// FBGANative — the class exposed to Python.  Holds params + gg table +
// FWBW solver in memory for the entire lifetime of the Python object.
// -------------------------------------------------------------------------
class FBGANative {
public:
  FBGANative(const std::string& params_path, const std::string& gg_path) {
    params_   = read_params(params_path);
    gg_table_ = read_gg_binary(gg_path);
    model_    = make_lookup(&gg_table_);
    fwbw_     = std::make_unique<GG::FWBW>(model_.upper, model_.lower,
                                           model_.range, model_.exp_func);
  }

  // Returns (v, ax) as a pair of numpy arrays, both length N.
  // Throws on bad input. Caller (Python) catches and treats as failure.
  std::pair<py::array_t<double>, py::array_t<double>>
  solve(py::array_t<double, py::array::c_style | py::array::forcecast> s,
        py::array_t<double, py::array::c_style | py::array::forcecast> kappa,
        py::array_t<double, py::array::c_style | py::array::forcecast> g_tilde,
        py::array_t<double, py::array::c_style | py::array::forcecast> mu,
        py::array_t<double, py::array::c_style | py::array::forcecast> dmu_ds,
        double v0, double slope_corr) {
    auto s_buf  = s.unchecked<1>();
    auto k_buf  = kappa.unchecked<1>();
    auto g_buf  = g_tilde.unchecked<1>();
    auto m_buf  = mu.unchecked<1>();
    auto dm_buf = dmu_ds.unchecked<1>();
    const py::ssize_t n = s_buf.shape(0);
    if (k_buf.shape(0) != n || g_buf.shape(0) != n ||
        m_buf.shape(0) != n || dm_buf.shape(0) != n) {
      throw std::invalid_argument("FBGANative.solve: array length mismatch");
    }
    if (n < 5) {
      throw std::invalid_argument("FBGANative.solve: n < 5");
    }

    std::vector<GG::real> SS(n), KK(n), GG_(n), MU(n), DMU(n);
    for (py::ssize_t i = 0; i < n; ++i) {
      SS[i]  = s_buf(i);
      KK[i]  = k_buf(i);
      GG_[i] = g_buf(i);
      MU[i]  = m_buf(i);
      DMU[i] = dm_buf(i);
    }

    fwbw_->set_slope_corr(slope_corr);
    // Release the GIL while solving — compute() is pure C++, no Python
    // touched. Lets a parallel Python thread proceed during the solve.
    {
      py::gil_scoped_release release;
      (void)fwbw_->compute(SS, KK, GG_, MU, DMU, v0);
    }

    py::array_t<double> v_out(n);
    py::array_t<double> ax_out(n);
    auto v_mut  = v_out.mutable_unchecked<1>();
    auto ax_mut = ax_out.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < n; ++i) {
      v_mut(i)  = fwbw_->evalV(SS[i]);
      ax_mut(i) = fwbw_->evalAx(SS[i]);
    }
    return {std::move(v_out), std::move(ax_out)};
  }

  double v_max() const { return params_.v_max; }
  int n_v()      const { return gg_table_.n_v; }
  int n_g()      const { return gg_table_.n_g; }
  int n_s()      const { return gg_table_.n_s; }

private:
  VehicleParams                params_;
  GGTable                      gg_table_;
  GGModel                      model_;
  std::unique_ptr<GG::FWBW>    fwbw_;
};

PYBIND11_MODULE(fbga_native, m) {
  m.doc() = "FBGA native bindings (pybind11 wrapper around GIGI/FWBW). "
            "Loads params + gg.bin once; solve() is in-process, no subprocess.";

  py::class_<FBGANative>(m, "FBGANative")
    .def(py::init<const std::string&, const std::string&>(),
         py::arg("params_path"), py::arg("gg_path"),
         "Load vehicle params (key=value text) and gg.bin (binary) once.")
    .def("solve", &FBGANative::solve,
         py::arg("s"), py::arg("kappa"), py::arg("g_tilde"),
         py::arg("mu"), py::arg("dmu_ds"),
         py::arg("v0"), py::arg("slope_corr") = 1.0,
         "Run FWBW on the given path. Returns (v, ax) as numpy arrays.")
    .def("v_max", &FBGANative::v_max, "Vehicle v_max from params.txt.")
    .def("n_v",   &FBGANative::n_v,   "Number of velocity grid points in gg.bin.")
    .def("n_g",   &FBGANative::n_g,   "Number of g-load grid points in gg.bin.")
    .def("n_s",   &FBGANative::n_s,   "Slope axis size (0 = 2D table).");
}
