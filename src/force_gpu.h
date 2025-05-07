#ifndef FORCE_GPU_H_INCLUDED
#define FORCE_GPU_H_INCLUDED

#include "tree.h"

void Particle_Cell_Force_gpu(Coord4* P, int ng, Coord4* C, int nl, Coord3* F, double epsilon);
 
#endif