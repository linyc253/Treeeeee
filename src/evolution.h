#ifndef EVOLUTION_H_INCLUDED
#define EVOLUTION_H_INCLUDED

#include "particle.h"

void Evolution(Particle* P, int npart, double dt, double t);
double Max(double a, double b);
double Min(double a, double b);

#endif
