#ifndef EVOLUTION_H_INCLUDED
#define EVOLUTION_H_INCLUDED

#include "particle.h"
#include <stdlib.h>  
#include <string.h>  

double compute_dt(Particle* P, int npart, double dt_max, double eta, double epsilon);
void force(Particle* P, int npart, int force_method, double epsilon);
double integrator(Particle* P, int npart, double dt_max, int integrator_method, int force_method, double epsilon, double eta);
double Evolution(Particle* P, int npart, double dt_max);
double Max(double a, double b);
double Min(double a, double b);

#endif
