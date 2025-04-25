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
    double dx[3];
    for (int i = 0; i < 3; i++) {
        dx[i] = a->x[i] - b->x[i];
    }

    // calculate gravitational force (assume G = 1.0)
    double r = sqrt(dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2]);
    double F_mag;
    double epsilon = 1e-10;
    if (DIM == 2 || DIM == 3) {
        F_mag = -(m_a * m_b) / (r*r + epsilon*epsilon);
    } else {
        printf("Dimension error\n");
        force[0] = force[1] = force[2] = 0.0;
        return;
    }

    for (int i = 0; i < 3; i++) {
        force[i] = F_mag * (dx[i] / r);
    }

    if (DIM == 2) {
        force[2] = 0.0;
    }
}

void total_force(Particle* p, int npart){
    
    for (int i = 0; i < npart; i++){
        for (int k = 0; k < 3; k++) {
            p[i].f[k] = 0.0;
        }
    }

    double f_tmp[3];
    for (int i = 0; i < npart; i++) {
        for (int j = 0; j < npart; j++) {
            if (i == j) continue;

            two_particle_force(&p[i], &p[j], f_tmp);

            for (int k = 0; k < 3; k++) {
                p[i].f[k] += f_tmp[k];
            }
        }
    }
}

