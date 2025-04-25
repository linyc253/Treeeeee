#include <stdio.h>
#include "force.h"
#include "parameter.h"
#include "particle.h"
#include "math.h"

int Force(int a){
    double THETA = get_double("Tree.THETA", 0.001); // Local variable
    printf("This is from force.c with THETA = %f\n", THETA);
    return 0;
}

void two_particle_force(Particle* a, Particle* b, double* force){
    double m_a = a->m;
    double m_b = b->m;
    double dx[DIM];
    double r_sq = 0.0; // r^2
    for (int i = 0; i < DIM; i++) {
        dx[i] = a->x[i] - b->x[i];
        r_sq += dx[i]*dx[i];
    }

    // calculate gravitational force (assume G = 1.0) 
    double epsilon = 1e-10;
    double F_mag = -(m_a * m_b) / pow(r_sq + epsilon*epsilon, 1.5);

    for (int i = 0; i < DIM; i++) {
        force[i] = F_mag * dx[i];
    }
}

void total_force(Particle* p, int npart){
    
    for (int i = 0; i < npart; i++){
        for (int k = 0; k < DIM; k++) {
            p[i].f[k] = 0.0;
        }
    }

    double f_tmp[DIM];
    for (int i = 0; i < npart; i++) {
        for (int j = 0; j < npart; j++) {
            if (i == j) continue;

            two_particle_force(&p[i], &p[j], f_tmp);

            for (int k = 0; k < DIM; k++) {
                p[i].f[k] += f_tmp[k];
            }
        }
    }
}

