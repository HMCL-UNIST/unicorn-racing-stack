#ifndef SEGMENT_HXX
#define SEGMENT_HXX

#include "types.hxx"
#include <string>
#include <cmath>

#include "gg_utils.hxx"

namespace GG
{

  template<typename T>
  constexpr T eval_v(const T& S, const T& A, const T& V0) {
      return std::sqrt(static_cast<T>(2) * A * S + std::pow(V0, 2));
  }

  template<typename T>
  constexpr T eval_v2(const T& S, const T& A, const T& V0) {
      return static_cast<T>(2) * A * S + std::pow(V0, 2);
  }
  class segment
  {
  private:
    real m_s0{0};
    real m_s1{0};
    real m_v0{0};
    real m_a{0};
    real m_L{0};
    real m_k0{0};
    real m_k1{0};
    SegmentType m_type{UNKNOWN};
  public:
    // constructors
    segment() = default;
    //
    segment(const real L, const real v0, const real k0, const real k1) 
    : m_s1(L), m_v0(v0), m_L(L), m_k0(k0), m_k1(k1) {}
    //
    segment(const real s0, const real L, const real v0, const real k0, const real k1)
    : m_s0(s0), m_s1(s0 + L), m_v0(v0), m_L(L), m_k0(k0), m_k1(k1) {}
    // setters
    void set_s0(const real s0){ this->m_s0 = s0; }
    void set_s1(const real s1){ this->m_s1 = s1; }
    void set_a(const real a){ this->m_a = a; }
    void set_v0(const real v0){ this->m_v0 = v0; }
    void set_L(const real L){ this->m_L = L; }
    void set_k0(const real k0){ this->m_k0 = k0; }
    void set_k1(const real k1){ this->m_k1 = k1; }
    void set_type(const SegmentType &type){ this->m_type = type; }
    // getters
    [[nodiscard]] real s0() const { return this->m_s0; }
    [[nodiscard]] real s1() const { return this->m_s1; }
    [[nodiscard]] real v0() const { return this->m_v0; }
    [[nodiscard]] real a() const { return this->m_a; }
    [[nodiscard]] real L() const { return this->m_L; }
    [[nodiscard]] real k0() const { return this->m_k0; }
    [[nodiscard]] real k1() const { return this->m_k1; }
    [[nodiscard]] SegmentType type() const { return this->m_type; }
    [[nodiscard]] real V(const real s) const { return eval_v(s, this->m_a, this->m_v0); }
    // evaluate v at L (final velocity)
    [[nodiscard]] real VF() const { return eval_v(this->m_L, this->m_a, this->m_v0); }
    // evaluate v (final) given a (forward propagation)
    [[nodiscard]] real VA(const real a) const { return eval_v(this->m_L, a, this->m_v0); }
    // evaluate v (initial) given b and vf (backward propagation)
    [[nodiscard]] real VB(const real b, const real vf) const { return sqrt(-static_cast<real>(2)*b*this->m_L+std::pow(vf,2)); }
    // evaluate t at s
    [[nodiscard]] real t(const real s) const { return static_cast<real>(2)*s / (this->m_v0 + eval_v(s,this->m_a,this->m_v0)); }
    // evaluate Total time
    [[nodiscard]] real T() const { return static_cast<real>(2)*this->m_L / (this->m_v0 + eval_v(this->m_L,this->m_a,this->m_v0)); }
    // evaluate curvature at s
    [[nodiscard]] real K(const real s) const { return this->m_k0 + s*(this->m_k1-this->m_k0)/this->m_L; }
    // evaluate lateral acceleration at s (ay = k*v^2)
    [[nodiscard]] real AY(const real s) const { return this->K(s)*std::pow(this->V(s),2); }
    // evaluate lateral acceleration at s (constant)
    [[nodiscard]] real AX(const real s) const { return this->m_a + 0*s; }
    // evaluate lateral acceleration at final point (forward)
    [[nodiscard]] real AYA(const real a) const { return this->m_k1 * std::pow(this->VA(a),2); }
    // evaluate lateral acceleration at initial point (backward)
    [[nodiscard]] real AYB(const real b, const real vf) const { return this->m_k0 * std::pow(this->VB(b,vf),2); }
    // retrieve initial lateral acceleration
    [[nodiscard]] real AY0() const { return this->m_k0 * std::pow(this->m_v0,2); }
    // retrieve final lateral acceleration
    [[nodiscard]] real AYF() const { return this->m_k1 * std::pow(this->VF(),2); }
    // eval t (generic)
    static real eval_t(const real s, const real a, const real v0) { return static_cast<real>(2)*s / (v0 + eval_v(s,a,v0)); }
  };



  struct NodeStruct3D
  {
    // STATIC MEMBERS (GIVEN)
    //// Length 
    real s{0};
    //// geometry (given)
    ///// euler angles
    real mu{0.0};
    real phi{0.0};
    real theta{0.0};
    ///// euler derivatives
    real mu_prime{0.0};
    real phi_prime{0.0};
    real theta_prime{0.0};
    ///// elurer secodn derivatives
    real mu_double_prime{0.0};
    real phi_double_prime{0.0};
    real theta_double_prime{0.0};
    ////// offset
    real n{0.0};
    real chi{0.0};
    ////// additional offsets
    real chi_prime{0.0};
    // STATIC MEMBERS (COMPUTED)
    //// Gravity corrections
    real g_x{0.0};
    real g_y{0.0};
    real g_z{0.0};
    //// Geometric Omegas 
    real Omega_x{0.0};
    real Omega_y{0.0};
    real Omega_z{0.0};
    //// Geometric Omegas prime
    real Omega_x_prime{0.0};
    real Omega_y_prime{0.0};
    real Omega_z_prime{0.0};
    //
    real V_max{130.0};
    // Adherence scaling
    real alpha{1.0};
  };

  struct CellStruct3D
  {
    real s_0{0};
    real s_1{0};
    real L{0};
    size_type ID0{0};
    size_type ID1{0};
    SegmentType m_type{UNKNOWN};
    real V_max{130.0};
    real V_dot{QUIET_NAN};
    real V_0{0.0};
    real V_1{0.0};
  };

}


#endif