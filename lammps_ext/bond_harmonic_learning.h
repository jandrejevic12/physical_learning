/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef BOND_CLASS
// clang-format off
BondStyle(harmonic/learning,BondHarmonicLearning);
// clang-format on
#else

#ifndef LMP_BOND_HARMONIC_LEARNING_H
#define LMP_BOND_HARMONIC_LEARNING_H

#include "bond.h"

namespace LAMMPS_NS {

class BondHarmonicLearning : public Bond {
 public:
  BondHarmonicLearning(class LAMMPS *);
  ~BondHarmonicLearning() override;
  void compute(int, int) override;
  void coeff(int, char **) override;
  double equilibrium_distance(int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_data(FILE *) override;
  double single(int, double, int, int, double &) override;
  void born_matrix(int, double, int, int, double &, double &) override;
  void *extract(const char *, int &) override;

 protected:
  double *k, *r0, *e_t, *eta, *alpha; // stiffness, rest length, target strain, learning rates
  int *train, *mode, *phase, *target; // train trainable bond indicator, training mode, phase, target bond indicator

  virtual void allocate();

};

}    // namespace LAMMPS_NS

#endif
#endif
