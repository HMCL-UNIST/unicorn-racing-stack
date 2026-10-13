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

///
/// file: GIGI.hh
///

#pragma once

#ifndef INCLUDE_GIGI
#define INCLUDE_GIGI


// Print GIGI errors
#ifndef GIGI_ERROR
#define GIGI_ERROR(MSG)                                                                           \
  {                                                                                                \
    throw std::runtime_error(std::to_string(MSG));                                                 \
  }
#endif

// Check for GIGI errors
#ifndef GIGI_ASSERT
#define GIGI_ASSERT(COND, MSG)                                                                    \
  if (!(COND))                                                                                     \
  GIGI_ERROR(MSG)
#endif


#endif

///
/// eof: GIGI.hh
///
