#include <chrono>
#include <fstream>
#include <iostream>

#include "GIGI/FWBW.hxx"
#include "GIGI/gg_utils.hxx"
#include "GIGI/types.hxx"

#include "rapidcsv.h"

#include "cxxopts.hpp"


//  █████╗ ███████╗██████╗  ██████╗    ███╗   ███╗ ██████╗ ██████╗ ███████╗██╗     
// ██╔══██╗██╔════╝██╔══██╗██╔═══██╗   ████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║     
// ███████║█████╗  ██████╔╝██║   ██║   ██╔████╔██║██║   ██║██║  ██║█████╗  ██║     
// ██╔══██║██╔══╝  ██╔══██╗██║   ██║   ██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║     
// ██║  ██║███████╗██║  ██║╚██████╔╝   ██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗
// ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝    ╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝
                                                                             
// data structure to hold the aerodynamic data
struct AeroData{
  GG::real b = 0.73; // wheelbase
  GG::real L_W = 1.5; // length of the wheel
  GG::real h = 0.69; // height of the center of mass
  GG::real mu_X = 1.30; // longitudinal friction coefficient
  GG::real mu_Y = 1.40; // lateral friction coefficient
  GG::real c_a_0 = 0.00; // aerodynamic coefficient
  GG::real c_a_1 = 0.00; // aerodynamic coefficient
  GG::real c_a_2 = 0.5*1.2*0.25/(250.0*9.81); // aerodynamic coefficient
  GG::real h_a = 0.51; // height of the aerodynamic center
  GG::real g = 9.81; // gravitational acceleration
  GG::real m = 250.0; // mass of the motorcycle
  GG::real P = 145.0 * 1000.0; // maximum power of the engine in Watts
};

// maximum acceelration due to the engine

GG::real ax_max_engine(GG::real v, const AeroData& aero_data) {
  const auto b     = aero_data.b;
  const auto L_W   = aero_data.L_W;
  const auto h     = aero_data.h; 
  const auto mu_X  = aero_data.mu_X;
  const auto mu_Y  = aero_data.mu_Y;
  const auto c_a_0 = aero_data.c_a_0; 
  const auto c_a_1 = aero_data.c_a_1; 
  const auto c_a_2 = aero_data.c_a_2; 
  const auto h_a   = aero_data.h_a;
  const auto g     = aero_data.g;
  const auto m     = aero_data.m;
  const auto P     = aero_data.P;

  return  P / (m * v) - (c_a_2*v*v+c_a_1*v+c_a_0)*g;
}

// minimum acceleration from adherence ellipse

GG::real ax_min_adherence(GG::real ayg, GG::real v, const AeroData& aero_data) {
  const auto b     = aero_data.b;
  const auto L_W   = aero_data.L_W;
  const auto h     = aero_data.h; 
  const auto mu_X  = aero_data.mu_X;
  const auto mu_Y  = aero_data.mu_Y;
  const auto c_a_0 = aero_data.c_a_0; 
  const auto c_a_1 = aero_data.c_a_1; 
  const auto c_a_2 = aero_data.c_a_2; 
  const auto h_a   = aero_data.h_a;
  const auto g     = aero_data.g;
  const auto m     = aero_data.m;
  const auto P     = aero_data.P;
  //
  const auto ay = ayg / g;
  //
  return g*((-c_a_2 * mu_Y * v *v - c_a_1 * mu_Y * v - c_a_0 * mu_Y - sqrt(-std::pow(ay * mu_X ,2)+ std::pow(mu_Y * mu_X ,2))) / mu_Y);
}

// minimum acceleration from stoppie constraint

GG::real ax_min_stoppie(GG::real ayg, GG::real v, const AeroData& aero_data) {
  const auto b     = aero_data.b;
  const auto L_W   = aero_data.L_W;
  const auto h     = aero_data.h; 
  const auto mu_X  = aero_data.mu_X;
  const auto mu_Y  = aero_data.mu_Y;
  const auto c_a_0 = aero_data.c_a_0; 
  const auto c_a_1 = aero_data.c_a_1; 
  const auto c_a_2 = aero_data.c_a_2; 
  const auto h_a   = aero_data.h_a;
  const auto g     = aero_data.g;
  const auto m     = aero_data.m;
  const auto P     = aero_data.P;
  //
  const auto ay = ayg / g;
  //
  return g*(-(-c_a_2 * h_a * v * v + c_a_2 * h * v * v - c_a_1 * h_a * v + c_a_1 * h * v + L_W * sqrt((ay * ay + 1)) - b * sqrt((ay * ay + 1)) - c_a_0 * h_a + c_a_0 * h) / h);
}

// maximum acceleration from wheeling constraint

GG::real ax_max_wheeling(GG::real ayg, GG::real v, const AeroData& aero_data) {
  const auto b     = aero_data.b;
  const auto L_W   = aero_data.L_W;
  const auto h     = aero_data.h; 
  const auto mu_X  = aero_data.mu_X;
  const auto mu_Y  = aero_data.mu_Y;
  const auto c_a_0 = aero_data.c_a_0; 
  const auto c_a_1 = aero_data.c_a_1; 
  const auto c_a_2 = aero_data.c_a_2; 
  const auto h_a   = aero_data.h_a;
  const auto g     = aero_data.g;
  const auto m     = aero_data.m;
  const auto P     = aero_data.P;
  //
  const auto ay = ayg / g;
  //
  return g*(c_a_2 * h_a * v * v - c_a_2 * h * v * v + c_a_1 * h_a * v - c_a_1 * h * v + b * sqrt((ay * ay + 1)) + c_a_0 * h_a - c_a_0 * h) / h;
}

// maximum acceleration from adherence ellipse

GG::real ax_max_adherence(GG::real ayg, GG::real v, const AeroData& aero_data) {
  const auto b     = aero_data.b;
  const auto L_W   = aero_data.L_W;
  const auto h     = aero_data.h; 
  const auto mu_X  = aero_data.mu_X;
  const auto mu_Y  = aero_data.mu_Y;
  const auto c_a_0 = aero_data.c_a_0; 
  const auto c_a_1 = aero_data.c_a_1; 
  const auto c_a_2 = aero_data.c_a_2; 
  const auto h_a   = aero_data.h_a;
  const auto g     = aero_data.g;
  const auto m     = aero_data.m;
  const auto P     = aero_data.P;
  //
  const auto ay = ayg / g;
  //
  return g*(h *std::pow(mu_X,2) * (ay - mu_Y) * (ay + mu_Y) * (b - L_W) * sqrt((std::pow(ay,2) + 1)) - (std::pow(L_W,2) * (std::pow(mu_Y,2)) + h *std::pow(mu_X,2) * (h - h_a)) * (c_a_2 * std::pow(v,2) + c_a_1 * v + c_a_0) * (std::pow(ay,2)) + (-std::pow(L_W,2) + h *std::pow(mu_X,2) * (h - h_a)) * (c_a_2 * std::pow(v,2) + c_a_1 * v + c_a_0) * (std::pow(mu_Y,2)) + sqrt(-(ay + mu_Y) * std::pow(L_W,2) * (std::pow(ay,2) + 1) * (0.2e1 * h_a * (c_a_2 * std::pow(v,2) + c_a_1 * v + c_a_0) * (b - L_W) * sqrt((std::pow(ay,2) + 1)) + std::pow(b - L_W,2) * (std::pow(ay,2)) + b * b - 0.2e1 * L_W * b + std::pow(L_W,2) + std::pow(c_a_2 * std::pow(v,2) + c_a_1 * v + c_a_0,2) * h_a * h_a) *std::pow(mu_X,2) * (ay - mu_Y) * (std::pow(mu_Y,2)))) / ((std::pow(L_W,2) * (std::pow(mu_Y,2)) + std::pow(h,2) *std::pow(mu_X,2)) * (std::pow(ay,2)) + (-std::pow(h,2) *std::pow(mu_X,2) + std::pow(L_W,2)) * (std::pow(mu_Y,2)));
}


// ███╗   ███╗ █████╗ ██╗███╗   ██╗
// ████╗ ████║██╔══██╗██║████╗  ██║
// ██╔████╔██║███████║██║██╔██╗ ██║
// ██║╚██╔╝██║██╔══██║██║██║╚██╗██║
// ██║ ╚═╝ ██║██║  ██║██║██║ ╚████║
// ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝
                                

int main(int argc, char *argv[])
{
  // Parse command line arguments
  cxxopts::Options options("GIGI");
  options.add_options()
    ("h,help", "Print help")
    ("c,circuit", "Circuit name", cxxopts::value<std::string>()->default_value("Catalunya"));
  
  auto result = options.parse(argc, argv); 
  // Print help
  if (result.count("help"))
  {
    std::cout << options.help() << std::endl;
    exit(0);
  }

  // ██████╗  ██████╗  █████╗ ██████╗
  // ██╔══██╗██╔═══██╗██╔══██╗██╔══██╗
  // ██████╔╝██║   ██║███████║██║  ██║
  // ██╔══██╗██║   ██║██╔══██║██║  ██║
  // ██║  ██║╚██████╔╝██║  ██║██████╔╝
  // ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝

  const std::string circuit_name = result["circuit"].as<std::string>();

  const std::string comp_file_path = "./data/paper/ard_2d_fwbw_moto_" + circuit_name + ".txt";


  rapidcsv::Document doc_1D(
    comp_file_path, 
    rapidcsv::LabelParams(0, -1), // Ensure the first non-comment line is treated as headers
    rapidcsv::SeparatorParams('\t',true),
    rapidcsv::ConverterParams(),
    rapidcsv::LineReaderParams(true, '#')); // Skip only comment lines 

  // extract
  std::vector<GG::real> SS_vec_1D = doc_1D.GetColumn<GG::real>("zeta");
  std::vector<GG::real> KK_vec_1D = doc_1D.GetColumn<GG::real>("curvature");
  std::vector<GG::real> VX_vec_1D = doc_1D.GetColumn<GG::real>("v__x");
  std::vector<GG::real> AX_vec_1D = doc_1D.GetColumn<GG::real>("a__x");
  // ay is curvature times v^2
  std::vector<GG::real> AY_vec_1D;
  AY_vec_1D.resize(VX_vec_1D.size());
  for (size_t i = 0; i < VX_vec_1D.size(); ++i) {
    AY_vec_1D[i] = KK_vec_1D[i] * VX_vec_1D[i] * VX_vec_1D[i];
  }

  GG::LinearInterpolatorSet traj_spline_1D(
    SS_vec_1D, 
    { KK_vec_1D, VX_vec_1D, AX_vec_1D, AY_vec_1D},
    {"kappa", "v_x", "a_x", "a_y"});
  GG::real mesh_size_1D = 0.0;
  for (size_t i = 0; i < SS_vec_1D.size() - 1; ++i) {
    mesh_size_1D += (SS_vec_1D[i + 1] - SS_vec_1D[i]) / static_cast<GG::real>(SS_vec_1D.size() - 1);
  }

  // ███████╗███╗   ██╗██╗   ██╗███████╗██╗      ██████╗ ██████╗ ███████╗
  // ██╔════╝████╗  ██║██║   ██║██╔════╝██║     ██╔═══██╗██╔══██╗██╔════╝
  // █████╗  ██╔██╗ ██║██║   ██║█████╗  ██║     ██║   ██║██████╔╝█████╗
  // ██╔══╝  ██║╚██╗██║╚██╗ ██╔╝██╔══╝  ██║     ██║   ██║██╔═══╝ ██╔══╝
  // ███████╗██║ ╚████║ ╚████╔╝ ███████╗███████╗╚██████╔╝██║     ███████╗
  // ╚══════╝╚═╝  ╚═══╝  ╚═══╝  ╚══════╝╚══════╝ ╚═════╝ ╚═╝     ╚══════╝


  AeroData aero_data;

  // create the function handles for the FBGA

  // auto GG_shape1 = [&aero_data](GG::real ay, GG::real v) -> GG::real {
  //   return std::min(
  //     std::min(
  //       ax_max_wheeling(ay, v, aero_data),
  //       ax_max_adherence(ay, v, aero_data)),
  //     ax_max_engine(v, aero_data));
  // };
  // auto GG_shape2 = [&aero_data](GG::real ay, GG::real v) -> GG::real {
  //   return std::max(
  //     ax_min_stoppie(ay, v, aero_data),
  //     ax_min_adherence(ay, v, aero_data));
  // };
  // auto gg_range_min = [&aero_data](GG::real v) -> GG::real {
  //   return -aero_data.mu_Y * aero_data.g * 0.999;
  // };
  // auto gg_range_max = [&aero_data](GG::real v) -> GG::real {
  //   return +aero_data.mu_Y * aero_data.g * 0.999;
  // };
  // add g_apparent + slope arguments (unused, flat track) for compatibility with the 3D API
  auto GG_shape1 = [&aero_data](GG::real ay, GG::real v, GG::real /*g_apparent*/, GG::real /*slope*/) -> GG::real {
    return std::min(
      std::min(
        ax_max_wheeling(ay, v, aero_data),
        ax_max_adherence(ay, v, aero_data)),
      ax_max_engine(v, aero_data));
  };
  auto GG_shape2 = [&aero_data](GG::real ay, GG::real v, GG::real /*g_apparent*/, GG::real /*slope*/) -> GG::real {
    return std::max(
      ax_min_stoppie(ay, v, aero_data),
      ax_min_adherence(ay, v, aero_data));
  };
  auto gg_range_min = [&aero_data](GG::real v, GG::real /*g_apparent*/, GG::real /*slope*/) -> GG::real {
    return -aero_data.mu_Y * aero_data.g * 0.999;
  };
  auto gg_range_max = [&aero_data](GG::real v, GG::real /*g_apparent*/, GG::real /*slope*/) -> GG::real {
    return +aero_data.mu_Y * aero_data.g * 0.999;
  };

  GG::gg_range_max_min gg_range = {gg_range_min, gg_range_max};

  GG::real Vminenv = 5.0;
  GG::real Vmaxenv = 100.0;


  //  ██████╗ ██████╗ ██╗   ██╗████████╗
  // ██╔════╝██╔═══██╗██║   ██║╚══██╔══╝
  // ██║     ██║   ██║██║   ██║   ██║   
  // ██║     ██║   ██║██║   ██║   ██║   
  // ╚██████╗╚██████╔╝╚██████╔╝   ██║   
  //  ╚═════╝ ╚═════╝  ╚═════╝    ╚═╝   
                                      

  std::cout << "FBGA - Forward Backward Generic Acceleration constraints\n";
  std::cout << " > Running test for circuit: " << result["circuit"].as<std::string>() << "\n";
  std::cout << " > Motorcycle ggv envelope\n";
  // print aero data
  std::cout << " > Aero data:\n";
  std::cout << " >   b:     " << aero_data.b << "\n";
  std::cout << " >   L_W:   " << aero_data.L_W << "\n";
  std::cout << " >   h:     " << aero_data.h << "\n";
  std::cout << " >   mu_X:  " << aero_data.mu_X << "\n";
  std::cout << " >   mu_Y:  " << aero_data.mu_Y << "\n";
  std::cout << " >   c_a_0: " << aero_data.c_a_0 << "\n";
  std::cout << " >   c_a_1: " << aero_data.c_a_1 << "\n";
  std::cout << " >   c_a_2: " << aero_data.c_a_2 << "\n";
  std::cout << " >   h_a:   " << aero_data.h_a << "\n";
  std::cout << " >   g:     " << aero_data.g << "\n";
  std::cout << " >   m:     " << aero_data.m << "\n";
  std::cout << " >   P:     " << aero_data.P << "\n";

  std::cout << " > Road info:\n";
  std::cout << " >   mesh size: " << mesh_size_1D << "\n";
  std::cout << " >   Length: " << SS_vec_1D.back() - SS_vec_1D.front() << "\n";
                                    
  // ███████╗██████╗  ██████╗  █████╗ 
  // ██╔════╝██╔══██╗██╔════╝ ██╔══██╗
  // █████╗  ██████╔╝██║  ███╗███████║
  // ██╔══╝  ██╔══██╗██║   ██║██╔══██║
  // ██║     ██████╔╝╚██████╔╝██║  ██║
  // ╚═╝     ╚═════╝  ╚═════╝ ╚═╝  ╚═╝
                                                         
  // create object

  GG::FWBW fwbw(GG_shape1, GG_shape2, gg_range);

  // create the vector for the solution

  std::vector<GG::real> SS_gg_vec = SS_vec_1D;
  std::vector<GG::real> KK_gg_vec = KK_vec_1D;
  GG::integer numpts_gg = SS_gg_vec.size();

  GG::real v_initial = VX_vec_1D.front();

  std::cout << " + FWBW computation started\n";

  auto start = std::chrono::steady_clock::now();
  // GG::real T = fwbw.compute(SS_gg_vec, KK_gg_vec, v_initial);
  // add G_gg_vec (flat track, g = 9.81) for new 3D API
  std::vector<GG::real> G_gg_vec(SS_gg_vec.size(), 9.81);
  std::vector<GG::real> MU_gg_vec(SS_gg_vec.size(), 0.0);   // flat track, no slope
  std::vector<GG::real> DMU_gg_vec(SS_gg_vec.size(), 0.0);  // flat track, no slope rate
  GG::real T = fwbw.compute(SS_gg_vec, KK_gg_vec, G_gg_vec, MU_gg_vec, DMU_gg_vec, v_initial);
  auto end = std::chrono::steady_clock::now();

  // ██████╗  ██████╗ ███████╗████████╗██████╗ ██████╗  ██████╗  ██████╗
  // ██╔══██╗██╔═══██╗██╔════╝╚══██╔══╝██╔══██╗██╔══██╗██╔═══██╗██╔════╝
  // ██████╔╝██║   ██║███████╗   ██║   ██████╔╝██████╔╝██║   ██║██║     
  // ██╔═══╝ ██║   ██║╚════██║   ██║   ██╔═══╝ ██╔══██╗██║   ██║██║     
  // ██║     ╚██████╔╝███████║   ██║   ██║     ██║  ██║╚██████╔╝╚██████╗
  // ╚═╝      ╚═════╝ ╚══════╝   ╚═╝   ╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚═════╝


  std::cout << " + FWBW computation finished\n";
                                                                      
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  auto elapsed_microsec = std::chrono::duration_cast<std::chrono::microseconds>(end - start);


  std::cout << "\n============ FBGA Results ============\n";
  std::cout << " > CPU time:                " << elapsed_microsec.count() / 1000.0 << " ms\n";
  std::cout << " > Total lap time:          " << T << " s\n";
  std::cout << " > Number of segments:      " << numpts_gg << "\n";
  std::cout << " > Average time/segment:    " << static_cast<double>(elapsed_microsec.count()) / numpts_gg << " μs\n";
  std::cout << "======================================\n\n";



  //  ██████╗ ██╗   ██╗████████╗██████╗ ██╗   ██╗████████╗    ███████╗██╗██╗     ███████╗
  // ██╔═══██╗██║   ██║╚══██╔══╝██╔══██╗██║   ██║╚══██╔══╝    ██╔════╝██║██║     ██╔════╝
  // ██║   ██║██║   ██║   ██║   ██████╔╝██║   ██║   ██║       █████╗  ██║██║     █████╗  
  // ██║   ██║██║   ██║   ██║   ██╔═══╝ ██║   ██║   ██║       ██╔══╝  ██║██║     ██╔══╝  
  // ╚██████╔╝╚██████╔╝   ██║   ██║     ╚██████╔╝   ██║       ██║     ██║███████╗███████╗
  //  ╚═════╝  ╚═════╝    ╚═╝   ╚═╝      ╚═════╝    ╚═╝       ╚═╝     ╚═╝╚══════╝╚══════╝
                                                                                      

  // get the solution of the forward backward in terms of VX, AX, AY
  std::vector<GG::real> VX_gg_vec(numpts_gg);
  std::vector<GG::real> AX_gg_vec(numpts_gg);
  std::vector<GG::real> AY_gg_vec(numpts_gg);
  for (GG::integer i = 0; i < numpts_gg; ++i) {
    VX_gg_vec[i] = fwbw.evalV(SS_gg_vec[i]);
    AX_gg_vec[i] = fwbw.evalAx(SS_gg_vec[i]);
    AY_gg_vec[i] = fwbw.evalAy(SS_gg_vec[i]);
  }

  // write an output csv file with the solution
  std::ofstream output_file("./data/output_" + circuit_name + "_fwbw_moto.txt");
  if (output_file.is_open()) {
    output_file << "# circuit_name = " << circuit_name << "\n";
    output_file << "# v_initial = " << v_initial << "\n";
    output_file << "# FWBW solution\n";
    output_file << "# \n";
    output_file << "# cpu_time = " << elapsed_microsec.count() / 1000.0 << "ms \n";
    output_file << "# iterations = " << numpts_gg << "\n";
    output_file << "# time\n";
    output_file << "# headers\n";
    output_file << "zeta\tcurvature\tv_x\ta_x\ta_y\n";
    for (GG::integer i = 0; i < numpts_gg; ++i) {
      output_file << SS_gg_vec[i] << "\t" 
                  << KK_gg_vec[i] << "\t" 
                  << VX_gg_vec[i] << "\t" 
                  << AX_gg_vec[i] << "\t" 
                  << AY_gg_vec[i] << "\n";
    }
    output_file.close();
  } else {
    std::cerr << "Unable to open file for writing: ./data/FWBW/" + circuit_name + "_fwbw.txt\n";
  }
  

  return 0;
}
