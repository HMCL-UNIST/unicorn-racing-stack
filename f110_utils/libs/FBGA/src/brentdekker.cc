#include "./GIGI/brentdekker.hxx"


#define DEBUG_IO 0

#if DEBUG_IO
#include <iostream>
#endif

namespace GG
{

brentdekker::brentdekker(const real tol) : tol(tol), verbose("iter"){ }

brentdekker::brentdekker(const real tol, const int max_iter) : tol(tol), max_iter(max_iter), verbose("iter"){}

brentdekker::brentdekker(const real tol, const int max_iter, const std::string &verbose) : tol(tol), max_iter(max_iter), verbose(verbose){}

void brentdekker::setup(const real tol, const int max_iter, const std::string &verbose)
{
  this->set_tolerance(tol);
  this->set_max_iter(max_iter);
  this->set_verbose(verbose);
}

bool brentdekker::solve(const std::function<real(real)> &f, real a, real b, real &x0)
{
  this->flag = false;
  real x1    = a;
  real f1    = f(x1);
  if (f1 == 0)
  {
    x0         = x1;
    this->flag = true;
#if DEBUG_IO
    std::cout << "Brent-Dekker: Root at borders [a, b].\n";
#endif
    return true;
  }
  real x2 = b;
  real f2 = f(x2);
  if (f2 == 0)
  {
    x0         = x2;
    this->flag = true;
#if DEBUG_IO
    std::cout << "Brent-Dekker: Root at borders [a, b].\n";
#endif
    return true;
  }
  if (f1 * f2 > 0.0)
  {
#if DEBUG_IO
    std::cout << "Brent-Dekker: Root is not bracketed in [a, b].\n";
#endif
    x0 = std::numeric_limits<real>::quiet_NaN();
    return false;
  }
  real x3 = (x1 + x2) / static_cast<real>(2);
  for (integer ith = 0; ith <= this->max_iter; ith++)
  {
    const real f3 = f(x3);
    if (std::abs(f3) < this->tol)
    {
      x0         = x3;
      this->flag = true;
#if DEBUG_IO
      std::cout << "Convergence reached at iter " << ith << "\n";
#endif
      break;
    }
    if (f1 * f3 < 0.0)
      b = x3;
    else
      a = x3;
    if (b - a < this->tol * std::max(std::abs(b), 1.0))
    {
      x0         = (x1 + x2) / static_cast<real>(2);
      this->flag = true;
#if DEBUG_IO
      std::cout << "Convergence reached at iter " << ith << "\n";
#endif
      break;
    }
    const real denom = (f2 - f1) * (f3 - f1) * (f2 - f3);
    const real numer = x3 * (f1 - f2) * (f2 - f3 + f1) + f2 * x1 * (f2 - f3) + f1 * x2 * (f3 - f1);
    real dx          = denom != 0.0 ? f3 * numer / denom : b - a;
    real x           = x3 + dx;
    if ((b - x) * (x - a) < 0.0)
    {
      dx = (b - a) / static_cast<real>(2);
      x  = a + dx;
    }
    if (x1 < x3)
    {
      x2 = x3;
      f2 = f3;
    }
    else
    {
      x1 = x3;
      f1 = f3;
    }
    if (std::abs(x - x3) < this->tol)
    {
      x0         = x;
      this->flag = true;
#if DEBUG_IO
      std::cout << "Convergence reached at iter " << ith << "\n";
#endif
      break;
    }
    x3 = x;
  }
  this->iter   = this->max_iter;
  this->x_last = x0;
  return this->flag;
}


} // namespace GG
