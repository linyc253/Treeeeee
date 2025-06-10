#ifndef FORCE_GPU_H_INCLUDED
#define FORCE_GPU_H_INCLUDED

#include "tree.h"

double Particle_Cell_Force_gpu(InteractionList particle_list, InteractionList cell_list, Coord3* force_xyz, double epsilon, int threadsPerBlock, int compute_energy);
 
#endif