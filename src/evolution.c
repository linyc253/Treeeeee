#include <stdio.h>
#include "evolution.h"
#include "parameter.h"
#include "particle.h"
#include "force.h"
#include "tree.h"

double Max(double a, double b){
    if(a >= b) return a;
    return b;
}

double Min(double a, double b){
    if(a <= b) return a;
    return b;
}

void Evolution(Particle* P, int npart, double dt) {
    // (a) Drift by 0.5*dt for all particles   
    for (int i = 0; i < npart; i++) {
        for (int j = 0; j < DIM; j++) {
            P[i].x[j] += P[i].v[j] * 0.5 * dt;
        }
    }
    // (b) Kick: update velocity using force    
    total_force(P, npart);
    for (int i = 0; i < npart; i++) {
        for (int j = 0; j < DIM; j++) {
            P[i].v[j] += P[i].f[j] * dt / P[i].m;
        }
    }
    // (c) Drift again by 0.5*dt               
    for (int i = 0; i < npart; i++) {
        for (int j = 0; j < DIM; j++) {
            P[i].x[j] += P[i].v[j] * 0.5 * dt;
        }
    }
}