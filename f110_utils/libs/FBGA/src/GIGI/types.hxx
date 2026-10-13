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

#pragma once

#include <cstddef>
#include <limits>
#ifndef TYPES_HXX
#define TYPES_HXX

namespace GG
{

  /*\
   |   _____                     _       __
   |  |_   _|   _ _ __   ___  __| | ___ / _|___
   |    | || | | | '_ \ / _ \/ _` |/ _ \ |_/ __|
   |    | || |_| | |_) |  __/ (_| |  __/  _\__ \
   |    |_| \__, | .__/ \___|\__,_|\___|_| |___/
   |        |___/|_|
  \*/

  using real = double;             //!< Real number type
  using floating = float;          //!< Real number type
  using integer = int;             //!< Integer number type
  using size_type = std::size_t;   //!< Size type

  /*\
   |    ____                _              _
   |   / ___|___  _ __  ___| |_ __ _ _ __ | |_ ___
   |  | |   / _ \| '_ \/ __| __/ _` | '_ \| __/ __|
   |  | |__| (_) | | | \__ \ || (_| | | | | |_\__ \
   |   \____\___/|_| |_|___/\__\__,_|_| |_|\__|___/
   |
  \*/

  static constexpr real GRAVITY = 9.81;                                   //!< Gravity static constant value

  static constexpr real EPSILON_MACHINE = std::numeric_limits<real>::epsilon(); //!< Machine epsilon epsilon static constant value
  static constexpr real EPSILON_HIGH = 1.0E-16;                                 //!< High precision epsilon static constant value
  static constexpr real EPSILON_MEDIUM = 1.0E-10;                               //!< Medium precision epsilon static constant value
  static constexpr real EPSILON_LOW = 1.0E-07;                                  //!< Low precision epsilon static constant value
  static constexpr real EPSILON = EPSILON_MEDIUM;                               //!< Standard precision epsilon static constant value
  static real const INFTY = std::numeric_limits<real>::infinity();          //!< Infinity static constant value
  static real const QUIET_NAN = std::numeric_limits<real>::quiet_NaN();     //!< Not-a-Number static constant value
  static constexpr real PI = 3.141592653589793238462643383279500;         //!< Pi static constant value
  static constexpr real PIDIV180 = 0.017453292519943295769236907684886;   //!< Pi/180 static constant value
  static constexpr real DEG2RAD = PIDIV180;                               //!< Degrees to Gradians static constant value
  static constexpr real RAD2DEG = 1.0 / DEG2RAD;                          //!< Gradians to Degrees static constant value

} // namespace GIGI

#endif // TYPES_HXX