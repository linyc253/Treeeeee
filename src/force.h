#ifndef FORCE_H_INCLUDED
#define FORCE_H_INCLUDED

#include "particle.h"

int Force(int a);
void two_particle_force(Particle* a, Particle* b, double* force, double epsilon);
void total_force(Particle* p, int npart);
 
#endif
