#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>

#include "GIGI/FWBW.hxx"
#include "GIGI/gg_utils.hxx"
#include "GIGI/types.hxx"

#include "rapidcsv.h"

#include "cxxopts.hpp"
                                                                        
// uses the precomputed ggv values
//      ax_max_engine, ax_min_adherence, ax_min_stoppie, ax_max_wheeling, ax_max_adherence).
//      Steps C/D/E below add: VehicleParams, read_params, GGTable, read_gg_binary, GGModel, make_lookup.

// vehicle parameters + simple key=value parser
struct VehicleParams {
  GG::real m       = 4.38;   // mass [kg]
  GG::real P_max   = 267.0;  // max engine power [W]
  GG::real P_brake = 50.0;   // max brake power [W]
  GG::real mu_x    = 1.0;    // longitudinal friction coefficient
  GG::real mu_y    = 1.0;    // lateral friction coefficient
  GG::real v_max   = 12.0;   // max velocity [m/s]
};

VehicleParams read_params(const std::string& path) {
  VehicleParams p;
  std::ifstream f(path);
  if (!f.is_open()) {
    std::cerr << "read_params: cannot open " << path << "\n";
    std::exit(1);
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

// GG diagram lookup table loaded from binary dump produced by Python.
//      Supports both 2D (V,g) and 3D (V,g,slope) formats.
//      2D layout: [uint32 nv][uint32 ng] + grids + 4×[nv*ng] arrays
//      3D layout: [uint32 nv][uint32 ng][uint32 ns] + grids + 4×[nv*ng*ns] arrays
//      Detection: file size determines 2D vs 3D.
struct GGTable {
  int n_v = 0, n_g = 0, n_s = 0;  // n_s=0 → 2D mode
  std::vector<GG::real> v_list, g_list, slope_list;
  std::vector<GG::real> ax_max, ax_min, ay_max, gg_exp;

  // 2D bilinear interpolation
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

  // 3D trilinear interpolation — idx = iv*n_g*n_s + ig*n_s + is
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
    // trilinear blend
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

GGTable read_gg_binary(const std::string& path) {
  GGTable t;
  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    std::cerr << "read_gg_binary: cannot open " << path << "\n";
    std::exit(1);
  }
  // Read nv, ng first
  uint32_t nv = 0, ng = 0;
  f.read(reinterpret_cast<char*>(&nv), sizeof(uint32_t));
  f.read(reinterpret_cast<char*>(&ng), sizeof(uint32_t));
  t.n_v = static_cast<int>(nv);
  t.n_g = static_cast<int>(ng);

  auto read_vec = [&](std::vector<GG::real>& v, size_t n) {
    v.resize(n);
    f.read(reinterpret_cast<char*>(v.data()), n * sizeof(double));
  };

  // Detect 2D vs 3D by file size
  f.seekg(0, std::ios::end);
  size_t file_size = f.tellg();
  size_t expected_2d = 8 + 8*nv + 8*ng + 4*8*static_cast<size_t>(nv)*ng;
  f.seekg(8, std::ios::beg);  // back to after nv,ng

  if (file_size > expected_2d) {
    // 3D format: read ns
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
    std::cout << "read_gg_binary: 3D (nv=" << nv << " ng=" << ng
              << " ns=" << ns << ")\n";
  } else {
    // 2D format
    t.n_s = 0;
    read_vec(t.v_list, nv);
    read_vec(t.g_list, ng);
    size_t total = static_cast<size_t>(nv) * ng;
    read_vec(t.ax_max, total);
    read_vec(t.ax_min, total);
    read_vec(t.ay_max, total);
    read_vec(t.gg_exp, total);
    std::cout << "read_gg_binary: 2D (nv=" << nv << " ng=" << ng << ")\n";
  }
  return t;
}

// GGModel struct + lookup model factory
//      Same diamond constraint as our NLP point_mass_model.py:
//        ax_upper = ax_max(v,g,slope) * (1 - (|ay|/ay_max(v,g,slope))^p)^(1/p)
//      slope parameter defaults to 0.0 for 2D backward compatibility.
struct GGModel {
  std::function<GG::real(GG::real, GG::real, GG::real, GG::real)> upper;   // (ay, v, g, slope) -> ax_max
  std::function<GG::real(GG::real, GG::real, GG::real, GG::real)> lower;   // (ay, v, g, slope) -> ax_min
  GG::gg_range_max_min range;                                    // (v, g, slope) -> ay_min/max
  std::function<GG::real(GG::real, GG::real, GG::real)> exp_func; // (v, g, slope) -> exponent p
};

GGModel make_lookup(const VehicleParams& /*params*/, const GGTable& tbl) {
  auto upper = [&tbl](GG::real ay, GG::real v, GG::real g, GG::real slope = 0.0) -> GG::real {
    GG::real ax_max_val = tbl.ax_max_at(v, g, slope);
    GG::real ay_max_val = tbl.ay_max_at(v, g, slope);
    GG::real p          = tbl.gg_exp_at(v, g, slope);
    GG::real ratio  = std::min(std::abs(ay) / ay_max_val, GG::real(1.0));
    GG::real factor = std::pow(std::max(GG::real(1.0) - std::pow(ratio, p), GG::real(0.0)),
                               GG::real(1.0) / p);
    return ax_max_val * factor;
  };
  auto lower = [&tbl](GG::real ay, GG::real v, GG::real g, GG::real slope = 0.0) -> GG::real {
    GG::real ax_min_val = tbl.ax_min_at(v, g, slope);
    GG::real ay_max_val = tbl.ay_max_at(v, g, slope);
    GG::real p          = tbl.gg_exp_at(v, g, slope);
    GG::real ratio  = std::min(std::abs(ay) / ay_max_val, GG::real(1.0));
    GG::real factor = std::pow(std::max(GG::real(1.0) - std::pow(ratio, p), GG::real(0.0)),
                               GG::real(1.0) / p);
    return ax_min_val * factor;
  };
  // propagate slope to range/exp_func lambdas as well
  GG::gg_range_max_min range{
    [&tbl](GG::real v, GG::real g, GG::real slope) { return -tbl.ay_max_at(v, g, slope); },
    [&tbl](GG::real v, GG::real g, GG::real slope) { return  tbl.ay_max_at(v, g, slope); }
  };
  // exponent function for Vmax mu correction
  // propagate slope
  auto exp_func = [&tbl](GG::real v, GG::real g, GG::real slope) -> GG::real {
    return tbl.gg_exp_at(v, g, slope);
  };
  return {upper, lower, range, exp_func};
}


// ███╗   ███╗ █████╗ ██╗███╗   ██╗
// ████╗ ████║██╔══██╗██║████╗  ██║
// ██╔████╔██║███████║██║██╔██╗ ██║
// ██║╚██╔╝██║██╔══██║██║██║╚██╗██║
// ██║ ╚═╝ ██║██║  ██║██║██║ ╚████║
// ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝
                                

int main(int argc, char *argv[])
{
  // CLI options
  //   --model   : GG model name (currently only "lookup"; friction_circle/aero to be added later)
  //   --input   : input CSV with columns (s, kappa, g_tilde)
  //   --params  : vehicle params text file (key=value)
  //   --gg      : GG table binary (lookup mode only)
  //   --output  : output CSV (s, v, ax, ay) with # laptime= header
  //   --v0      : initial velocity [m/s]
  cxxopts::Options options("GIGI_test_unicorn", "FWBW runner for our 3D GG lookup");
  options.add_options()
    ("h,help",  "Print help")
    ("model",   "GG model: lookup (friction_circle/aero to be added)",
                cxxopts::value<std::string>()->default_value("lookup"))
    ("input",   "Input CSV (s, kappa, g_tilde)",          cxxopts::value<std::string>())
    ("params",  "Vehicle params file (key=value)",        cxxopts::value<std::string>())
    ("gg",      "GG table binary (lookup mode)",          cxxopts::value<std::string>()->default_value(""))
    ("output",  "Output CSV (s, v, ax, ay)",              cxxopts::value<std::string>())
    ("v0",      "Initial velocity [m/s]",                 cxxopts::value<GG::real>()->default_value("1.0"))
    // 2.5d_vel_planner slope_correction (1.0 = physics)
    ("slope-corr", "Slope ax_grav scale (1=physics, higher=more conservative)",
                cxxopts::value<GG::real>()->default_value("1.0"));

  auto result = options.parse(argc, argv);
  // Print help
  if (result.count("help"))
  {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  // -- load input --
  // read CLI options + load input CSV (s, kappa, g_tilde) + params + GG table
  const std::string model_name  = result["model"].as<std::string>();
  const std::string input_path  = result["input"].as<std::string>();
  const std::string params_path = result["params"].as<std::string>();
  const std::string output_path = result["output"].as<std::string>();
  const std::string gg_path     = result["gg"].as<std::string>();
  const GG::real    v_initial   = result["v0"].as<GG::real>();
  // 2.5d_vel_planner slope_correction
  const GG::real    slope_corr  = result["slope-corr"].as<GG::real>();

  // input CSV: comma-separated, header row, '#' = comment
  rapidcsv::Document doc(
    input_path,
    rapidcsv::LabelParams(0, -1),
    rapidcsv::SeparatorParams(',', true),
    rapidcsv::ConverterParams(),
    rapidcsv::LineReaderParams(true, '#'));

  std::vector<GG::real> SS_vec = doc.GetColumn<GG::real>("s");
  std::vector<GG::real> KK_vec = doc.GetColumn<GG::real>("kappa");
  std::vector<GG::real> G_vec  = doc.GetColumn<GG::real>("g_tilde");
  // read mu column for ax_tilde = ax - g*sin(mu) correction
  std::vector<GG::real> MU_vec;
  try {
    MU_vec = doc.GetColumn<GG::real>("mu");
  } catch (...) {
    MU_vec.assign(SS_vec.size(), 0.0);
    std::cout << " > WARNING: no 'mu' column in input CSV, slope correction disabled\n";
  }
  // read dmu_ds column for g_tilde Vmax constraint
  std::vector<GG::real> DMU_vec;
  try {
    DMU_vec = doc.GetColumn<GG::real>("dmu_ds");
  } catch (...) {
    DMU_vec.assign(SS_vec.size(), 0.0);
    std::cout << " > WARNING: no 'dmu_ds' column in input CSV, g_tilde Vmax constraint disabled\n";
  }

  if (SS_vec.size() != KK_vec.size() || SS_vec.size() != G_vec.size()
      || SS_vec.size() != MU_vec.size() || SS_vec.size() != DMU_vec.size()) {
    std::cerr << "Input CSV column size mismatch: "
              << "s=" << SS_vec.size()
              << " kappa=" << KK_vec.size()
              << " g_tilde=" << G_vec.size()
              << " mu=" << MU_vec.size()
              << " dmu_ds=" << DMU_vec.size() << "\n";
    return 1;
  }

  // vehicle params (key=value text)
  VehicleParams params = read_params(params_path);

  // GG table binary (lookup model only)
  GGTable gg_table;
  if (model_name == "lookup") {
    if (gg_path.empty()) {
      std::cerr << "lookup model requires --gg <binary path>\n";
      return 1;
    }
    gg_table = read_gg_binary(gg_path);
  }

  // -- build GG model --
  // select GG model based on --model
  GGModel model;
  if (model_name == "lookup") {
    model = make_lookup(params, gg_table);
  } else {
    std::cerr << "Model not implemented: " << model_name
              << " (available: lookup)\n";
    return 1;
  }

  // -- print info --
  std::cout << "FBGA - Forward Backward Generic Acceleration constraints\n";
  std::cout << " > model:   " << model_name << "\n";
  std::cout << " > input:   " << input_path << "\n";
  std::cout << " > params:  " << params_path << "\n";
  if (model_name == "lookup") {
    std::cout << " > gg:      " << gg_path << "\n";
  }
  std::cout << " > output:  " << output_path << "\n";
  std::cout << " > v0:      " << v_initial << " m/s\n";
  std::cout << " > Vehicle params:\n";
  std::cout << " >   m       = " << params.m       << " kg\n";
  std::cout << " >   P_max   = " << params.P_max   << " W\n";
  std::cout << " >   P_brake = " << params.P_brake << " W\n";
  std::cout << " >   mu_x    = " << params.mu_x    << "\n";
  std::cout << " >   mu_y    = " << params.mu_y    << "\n";
  std::cout << " >   v_max   = " << params.v_max   << " m/s\n";
  std::cout << " > Track info:\n";
  std::cout << " >   N points = " << SS_vec.size() << "\n";
  std::cout << " >   Length   = " << SS_vec.back() - SS_vec.front() << " m\n";

  // -- run FWBW --
  // instantiate FWBW with the selected model and compute
  // pass gg_Exp for Vmax mu correction
  GG::FWBW fwbw(model.upper, model.lower, model.range, model.exp_func);
  // push 2.5d_vel_planner slope_correction into FWBW
  fwbw.set_slope_corr(slope_corr);
  std::cout << " > slope_corr: " << slope_corr << "\n";

  std::cout << " + FWBW computation started\n";
  auto start = std::chrono::steady_clock::now();
  // pass MU_vec + DMU_vec for 3D slope corrections
  GG::real T = fwbw.compute(SS_vec, KK_vec, G_vec, MU_vec, DMU_vec, v_initial);
  auto end   = std::chrono::steady_clock::now();
  std::cout << " + FWBW computation finished\n";

  auto elapsed_microsec = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  const GG::integer numpts = static_cast<GG::integer>(SS_vec.size());

  // -- results --
  std::cout << "\n============ FBGA Results ============\n";
  std::cout << " > CPU time:                " << elapsed_microsec.count() / 1000.0 << " ms\n";
  std::cout << " > Total lap time:          " << T << " s\n";
  std::cout << " > Number of segments:      " << numpts << "\n";
  std::cout << " > Average time/segment:    " << static_cast<double>(elapsed_microsec.count()) / numpts << " μs\n";
  std::cout << "======================================\n\n";

  // -- write output csv --
  // output CSV: comma-separated, first lines are # comments with metadata
  //      columns: s, v, ax, ay
  std::ofstream output_file(output_path);
  if (!output_file.is_open()) {
    std::cerr << "Unable to open file for writing: " << output_path << "\n";
    return 1;
  }
  output_file.precision(17);
  output_file << "# model="   << model_name << "\n";
  output_file << "# v0="      << v_initial  << "\n";
  output_file << "# laptime=" << T          << "\n";
  output_file << "# cpu_ms="  << elapsed_microsec.count() / 1000.0 << "\n";
  output_file << "# numpts="  << numpts     << "\n";
  output_file << "s,v,ax,ay\n";
  for (GG::integer i = 0; i < numpts; ++i) {
    output_file << SS_vec[i]                  << ","
                << fwbw.evalV(SS_vec[i])      << ","
                << fwbw.evalAx(SS_vec[i])     << ","
                << fwbw.evalAy(SS_vec[i])     << "\n";
  }
  output_file.close();

  return 0;
}
