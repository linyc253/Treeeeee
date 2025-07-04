#ifndef EVOLUTION_H_INCLUDED
#define EVOLUTION_H_INCLUDED

#include "particle.h"

double Evolution(Particle* P, int npart, double dt_max, double dt);
double Max(double a, double b);
double Min(double a, double b);
void Energy(Particle* P, int npart);

#endif
