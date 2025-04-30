#ifndef EVOLUTION_H_INCLUDED
#define EVOLUTION_H_INCLUDED

#include "particle.h"

double compute_dt(Particle* P, int npart, double dt_max, double eta, double epsilon);
double Evolution(Particle* P, int npart, double dt_max, double eta, double epsilon);
double Max(double a, double b);
double Min(double a, double b);

#endif
