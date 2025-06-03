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

// Calculate gravitational force by (G = 1)
//  force = -(a.m * b.m / (|r|^2 + epsilon^2)^{3/2}) r
// where the vector r = a.x - b.x
void two_particle_force(Particle* a, Particle* b, double* force, double epsilon){
    double m_a = a->m;
    double m_b = b->m;
    double dx[DIM];
    double r_sq = 0.0; // r^2
    for (int i = 0; i < DIM; i++) {
        dx[i] = a->x[i] - b->x[i];
        r_sq += dx[i]*dx[i];
    }

    // calculate gravitational force (assume G = 1.0) 
    double F_mag = -(m_a * m_b) / pow(r_sq + epsilon*epsilon, 1.5);

    for (int i = 0; i < DIM; i++) {
        force[i] = F_mag * dx[i];
    }
}

double two_particle_potential(Particle* a, Particle* b, double epsilon){
    double m_a = a->m;
    double m_b = b->m;
    double dx[DIM];
    double r_sq = 0.0; // r^2
    for (int i = 0; i < DIM; i++) {
        dx[i] = a->x[i] - b->x[i];
        r_sq += dx[i]*dx[i];
    }
    return -(m_a * m_b) / pow(r_sq + epsilon*epsilon, 0.5);
}

// Update p.f[:] by gravitational force (brute force)
void total_force(Particle* p, int npart, double epsilon){
    // double epsilon = get_double("BasicSetting.epsilon", 1e-10);
    for (int i = 0; i < npart; i++){
        for (int k = 0; k < DIM; k++) {
            p[i].f[k] = 0.0;
        }
    }

    double f_tmp[DIM];
    for (int i = 0; i < npart; i++) {
        for (int j = 0; j < i; j++) {

            two_particle_force(&p[i], &p[j], f_tmp, epsilon);

            for (int k = 0; k < DIM; k++) {
                p[i].f[k] += f_tmp[k];
                p[j].f[k] -= f_tmp[k];
            }
        }
    }
}

double total_potential(Particle* p, int npart, double epsilon){
    // double epsilon = get_double("BasicSetting.epsilon", 1e-10);
    double V = 0.0;

    for (int i = 0; i < npart; i++) {
        for (int j = 0; j < i; j++) {
            V += two_particle_potential(&p[i], &p[j], epsilon);
        }
    }
    return V;
}

