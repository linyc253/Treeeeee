#include "evolution.h"
#include "parameter.h"
#include "particle.h"
#include "vec3.h"
#include "force.h"

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
    for (int i = 0; i < npart; i++) vec3_add_scaled(P[i].x, P[i].x, P[i].v, 0.5 * dt);
    // (b) Kick: update velocity using force
    total_force(P, npart);
    for (int i = 0; i < npart; i++) vec3_add_scaled(P[i].v, P[i].v, P[i].f, dt / P[i].m);
    // (c) Drift again by 0.5*dt
    for (int i = 0; i < npart; i++) vec3_add_scaled(P[i].x, P[i].x, P[i].v, 0.5 * dt);
}