#include <stdio.h>
#include <math.h>
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

double compute_dt(Particle* P, int npart, double dt_max, double eta, double epsilon) {
    double sum_a2 = 0.0;

    for (int i = 0; i < npart; i++) {
        // Compute |a|^2 = |f/m|^2
        double a2 = 0.0;
        for (int j = 0; j < 3; j++) {
            double a_j = P[i].f[j] / P[i].m;
            a2 += a_j * a_j;
        }
        sum_a2 += a2;
    }

    double rms_a = sqrt(sum_a2 / npart);
    double dt = sqrt(2.0 * eta * epsilon / rms_a);
    return (dt < dt_max) ? dt : dt_max;
}

double Evolution(Particle* P, int npart, double dt_max, double eta, double epsilon) {
    double dt = compute_dt(P, npart, dt_max, eta, epsilon);
    // (a) Drift by 0.5*dt for all particles   
    for (int i = 0; i < npart; i++) {
        for (int j = 0; j < DIM; j++) {
            P[i].x[j] += P[i].v[j] * 0.5 * dt;
        }
    }
    // (b) Kick: update velocity using force    
    int METHOD = get_int("BasicSetting.METHOD", 2);
    if(METHOD == 1) total_force(P, npart, epsilon);
    else if(METHOD == 2) total_force_tree(P, npart);
    
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
    return dt;
}