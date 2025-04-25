#ifndef FORCE_H_INCLUDED
#define FORCE_H_INCLUDED

int Force(int a);
void two_particle_force(Particle* a, Particle* b, double* force);
void total_force(Particle* p, int npart);
 
#endif
