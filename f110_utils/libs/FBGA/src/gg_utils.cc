#include "./GIGI/gg_utils.hxx"
#include "GIGI/types.hxx"
#include <algorithm>
#include <stdexcept>

#include <iostream>


namespace GG
{
  real piramid(const real x, const real y)
  {
    //        piramid function 
    //        + + + + + + +
    //        +   _____   +
    //        +  | --- |  +
    //        +  | - - |  +
    //        +  | --- |  +
    //        +  |_____|  +
    //        + + + + + + +
    return std::max({x - 1.0, -1.0 - x, y - 1.0, -1.0 - y});
  }

  real clip(const real x, const real a, const real b)
  {
    //        Clip function
    //      (return)  ^    .
    //                |   . (b)
    //                |  _____
    //                | /     
    //         _______ /_________> (x)
    //                /|
    //         _____/ |    
    //         (a) .  |
    //            .
    return std::min(std::max(x, a), b);
  }

  real square_conversion(const real a, const real amin, const real amax)
  {
      constexpr real eps = 1e-6;
      constexpr real max_output = 10.0; // or 1.0, 5.0 — adjust based on how extreme values you tolerate

      real norm = 2.0 * (a - amin) / std::max(amax - amin, eps) - 1.0;

      return std::clamp(norm, -max_output, max_output);;
  }

  real signed_distance(const real ax, const real axmin, const real axmax, const real ay, const real aymin, const real aymax)
  {
    return piramid(
      square_conversion(ax, axmin, axmax),
      square_conversion(ay, aymin, aymax));
  }




  void 
  computeFiniteDifference(const std::vector<double>& X, const std::vector<double>& Y, std::vector<real> & dY_dx)
  {
    dY_dx.resize(X.size());
    #if STD_NUM_DIFF
    // Compute the finite difference of Y with respect to X
    for (size_type i = 0; i < X.size() - 1; ++i)
    {
      dY_dx[i] = (Y[i + 1] - Y[i]) / (X[i + 1] - X[i]);
    }
    dY_dx.back() = dY_dx[dY_dx.size() - 2]; // Last value is the same as the second last
    #else
    dY_dx[0] = (Y[1] - Y[0]) / (X[1] - X[0]); // Forward difference for the first point
    dY_dx.back() = (Y[Y.size() - 1] - Y[Y.size() - 2]) / (X[X.size() - 1] - X[X.size() - 2]); // Backward difference for the last point
    for (size_type i = 0; i < X.size() - 1; ++i)
    {
      dY_dx[i] = (Y[i + 1] - Y[i- 1]) / (X[i + 1] - X[i - 1]);
    }

    #endif
  }


  std::vector<real> 
  computeFiniteDifference(const std::vector<double>& X, const std::vector<double>& Y)
  {
    std::vector<real> dY_dx;
    dY_dx.resize(X.size());

    // Compute the finite difference of Y with respect to X
    for (size_type i = 0; i < X.size() - 1; ++i)
    {
      dY_dx[i] = (Y[i + 1] - Y[i]) / (X[i + 1] - X[i]);
    }
    dY_dx.back() = dY_dx[dY_dx.size() - 2]; // Last value is the same as the second last
    
    return dY_dx;
  }


  void
  BilinearInterpolator::setup(
    const std::vector<real>& X,
    const std::vector<real>& Y,
    const std::vector<real>& Z,
    bool transpose)
  {
    // Check if the input vectors are valid
    if (X.empty() || Y.empty() || Z.empty())
    {
      throw std::invalid_argument("BilinearInterpolator::BilinearInterpolator() >>> X, Y and Z must not be empty.");
    }
    this->m_x = X;
    this->m_y = Y;
    this->m_z = Z;
    this->m_nx = (integer) X.size();
    this->m_ny = (integer) Y.size();


    if (this->m_x.size() < 2 || this->m_y.size() < 2)
    {
      throw std::invalid_argument("BilinearInterpolator::BilinearInterpolator() >>> X and Y must have at least two elements each.");
    }
    if (this->m_z.size() != this->m_x.size() * this->m_y.size())
    {
      throw std::invalid_argument("BilinearInterpolator::BilinearInterpolator() >>> Z must have same number of rows as size of X.");
    }

    if(transpose)
    {
      // transpose the z elements
      std::vector<real> z_transposed(static_cast<size_type>(this->m_nx * this->m_ny));
      for (integer i = 0; i < this->m_nx; ++i)
      {
        for (integer j = 0; j < this->m_ny; ++j)
        {
          z_transposed[(j * this->m_nx) + i] = this->m_z[(i * this->m_ny) + j];
        }
      }
      this->m_z = z_transposed;
    }


  }


  BilinearInterpolator::BilinearInterpolator(
    const std::vector<real>& X,
    const std::vector<real>& Y,
    const std::vector<real>& Z)
  {
    this->setup(X, Y, Z);
  }

  real 
  BilinearInterpolator::eval(double xi, double yi) const 
  {
    // Find i such that x[i] <= xi < x[i+1]
    const size_type i = findInterval(this->m_x, xi);
    const size_type j = findInterval(this->m_y, yi);

    // const size_type i = findIntervalWithGuess(this->m_x, xi, this->m_memory_idx_x);
    // const size_type j = findIntervalWithGuess(this->m_y, yi, this->m_memory_idx_y);
    // // update memory indices
    // this->m_memory_idx_x = i;
    // this->m_memory_idx_y = j;



    // Corners
    const real x0 = this->m_x[i];
    const real x1 = this->m_x[i + 1];
    const real y0 = this->m_y[j];
    const real y1 = this->m_y[j + 1];
    const real z00 = this->m_z[(i * this->m_nx) + j];
    const real z10 = this->m_z[((i + 1) * this->m_nx) + j];
    const real z01 = this->m_z[(i * this->m_nx) + (j + 1)];
    const real z11 = this->m_z[((i + 1) * this->m_nx) + (j + 1)];

    // delta normalized
    const real tx = (xi - x0) / (x1 - x0);
    const real ty = (yi - y0) / (y1 - y0);

    // Bilinear interpolation formula
    return (1 - tx) * (1 - ty) * z00 + tx * (1 - ty) * z10 + (1 - tx) * ty * z01 + tx * ty * z11;
  }


  // Find index i such that vec[i] <= value < vec[i+1]
  size_type 
  findInterval(const std::vector<real>& vec, real value) 
  {
   
    size_type seg = 0;
    
    if (value >= vec.front() && value <= vec.back())
    {
      for (integer i = 0; i < (integer) vec.size() - 1; ++i) 
      {
        if (value >= vec[i] && value <= vec[i+1])
        {
          seg = i;
          break;
        }
      }
    }
    else 
    {
      seg = value < vec.front() ? 0 : (integer) vec.size() - 2;
    }
    
    return seg;

  }


  // Find index i such that vec[i] <= value < vec[i+1]
  size_type 
  findIntervalWithGuess(const std::vector<real>& vec, real value, size_type guess) 
  {
    size_type seg = 0;
    // Check if the guess is within bounds
    if (guess >= vec.size() - 1 || guess < 0)
    {
      guess = 0;
    }

    // Check if the value is within the range of vec
    if (value >= vec.front() && value <= vec.back()) 
    {
      for (size_type i = guess; i < vec.size() - 1; ++i) 
      {
        if (value >= vec[i] && value <= vec[i + 1]) 
        {
          seg = i;
          break;
        }
      }
      // If not found, check the previous elements
      for(size_type i = guess; i > 0; --i) 
      {
        if (value >= vec[i - 1] && value <= vec[i]) 
        {
          seg = i - 1;
          break;
        }
      }
    }
    else 
    {
      seg = value < vec.front() ? 0 : (integer) vec.size() - 2;
    }

    return seg;


    // Use the guess to find the interval
   
  }
  

  size_type
  findIntervalBinarySearch(const std::vector<real>& vec, real value)
  {
    size_type low = 0;
    size_type high = vec.size() - 1;

    while (high - low > 1) 
    {
      size_type mid = low + (high - low) / 2;

      if (vec[mid] < value) 
      {
        low = mid;
      } else if (vec[mid] > value)
      {
        high = mid;
      } else
      {
        return mid; // Found exact match
      }
    }

    return high; // Return the index of the largest element less than or equal to value
  }

  
  size_type 
  findIntervalbyEquispaced(const std::vector<real>& vec, real value)
  {
    if (value < vec.front() || value > vec.back()) 
    {
      throw std::out_of_range("findIntervalbyEquispaced >> Interpolation point is outside the grid.");
    }

    auto i = (integer) ((value - vec.front()) / (vec.back() - vec.front()) * (static_cast<real>(vec.size()) - 1));
    return static_cast<size_type>(std::clamp(i, 0, (integer) vec.size() - 2) );
  }


  LinearInterpolator::LinearInterpolator(
    const std::vector<real>& X,
    const std::vector<real>& Y)
  {
    this->setup(X, Y);
  }

  void
  LinearInterpolator::setup(
    const std::vector<real>& X,
    const std::vector<real>& Y)
  {
    // Check if the input vectors are valid
    if (X.empty() || Y.empty())
    {
      throw std::invalid_argument("LinearInterpolator::LinearInterpolator() >>> X and Y must not be empty.");
    }
    this->m_x = X;
    this->m_y = Y;
    this->m_nx = (integer) X.size();

    if (this->m_x.size() < 2)
    {
      throw std::invalid_argument("LinearInterpolator::LinearInterpolator() >>> X and Y must have at least two elements each.");
    }
    // check if x and y are the same size
    if (this->m_x.size() != this->m_y.size())
    {
      throw std::invalid_argument("LinearInterpolator::LinearInterpolator() >>> X and Y must have the same size.");
    }
  }

  real
  LinearInterpolator::eval(real xi) const
  {
    // Find i such that x[i] <= xi < x[i+1]
    size_type i = findInterval(this->m_x, xi);

    // Linear interpolation formula
    real x0 = this->m_x[i];
    real x1 = this->m_x[i + 1];
    real y0 = this->m_y[i];
    real y1 = this->m_y[i + 1];

    // delta normalized
    real tx = (xi - x0) / (x1 - x0);

    // Linear interpolation formula
    return (1 - tx) * y0 + tx * y1;
  }


  LinearInterpolatorSet::LinearInterpolatorSet(
    const std::vector<real>& X,
    const std::vector<std::vector<real>>& Y,
    const std::vector<std::string>& headers)
  {
    this->setup(X, Y, headers);
  }

  void
  LinearInterpolatorSet::setup(
    const std::vector<real>& X,
    const std::vector<std::vector<real>>& Y,
    const std::vector<std::string>& headers)
  {
    // Check if the input vectors are valid
    if (X.empty() || Y.empty())
    {
      throw std::invalid_argument("LinearInterpolatorSet::LinearInterpolatorSet() >>> X and Y must not be empty.");
    }
    this->m_x = X;
    this->m_y = Y;
    this->m_headers = headers;
    this->m_nx = (integer) X.size();
    this->m_ny = (integer) Y.size();

    if (this->m_x.size() < 2)
    {
      throw std::invalid_argument("LinearInterpolatorSet::LinearInterpolatorSet() >>> X and Y must have at least two elements each.");
    }
    // check if x and y are the same size
    for (const auto& y : this->m_y)
    {
      if (y.size() != this->m_x.size())
      {
        throw std::invalid_argument("LinearInterpolatorSet::LinearInterpolatorSet() >>> X and Y must have the same size.");
      }
    }
    // check that header and y are the same size
    if (this->m_headers.size() != this->m_y.size())
    {
      throw std::invalid_argument("LinearInterpolatorSet::LinearInterpolatorSet() >>> Y and headers must have the same size.");
    }

  }

  real
  LinearInterpolatorSet::eval(const std::string& name, real xi) const
  {
    // Find the index of the header
    auto it = std::find(this->m_headers.begin(), this->m_headers.end(), name);
    if (it == this->m_headers.end())
    {
      std::cout << "LinearInterpolatorSet::eval() >>> Header not found: " << name << std::endl;
      throw std::invalid_argument("LinearInterpolatorSet::eval() >>> Header not found.");
    }
    auto index = static_cast<integer>(std::distance(this->m_headers.begin(), it));

    // Create a LinearInterpolator for the specified header
    size_type i = findInterval(this->m_x, xi);

    // Linear interpolation formula
    real x0 = this->m_x[i];
    real x1 = this->m_x[i + 1];
    real y0 = this->m_y[index][i];
    real y1 = this->m_y[index][i + 1];
    // delta normalized
    real tx = (xi - x0) / (x1 - x0);

    // Linear interpolation formula
    return (1 - tx) * y0 + tx * y1;
  }

}
