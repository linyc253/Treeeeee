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
    return Min(dt, dt_max);
}

double Evolution(Particle* P, int npart, double dt_max, double dt1) {
    int METHOD = get_int("BasicSetting.METHOD", 2);
    double eta = get_double("BasicSetting.ETA", 1.0e100); // 1.0e100 literally disable adaptive time step
    double epsilon = get_double("BasicSetting.EPSILON", 1e-4);

    // Drift by dt1 / 2
    for (int i = 0; i < npart; i++) {
        for (int j = 0; j < DIM; j++) {
            P[i].x[j] += P[i].v[j] * (dt1 / 2.0);
        }
    }

    // Calculate force and dt2
    if(METHOD == 1) total_force(P, npart, epsilon);
    else if(METHOD == 2) total_force_tree(P, npart);
    double dt2 = compute_dt(P, npart, dt_max, eta, epsilon);

    // Kick by (dt1 + dt2) / 2    (KDK scheme)
    for (int i = 0; i < npart; i++) {
        for (int j = 0; j < DIM; j++) {
            P[i].v[j] += P[i].f[j] * ((dt1 + dt2) / 2.0) / P[i].m;
        }
    }

    // Drift by dt2 / 2
    for (int i = 0; i < npart; i++) {
        for (int j = 0; j < DIM; j++) {
            P[i].x[j] += P[i].v[j] * (dt2 / 2.0);
        }
    }

    return dt2;
}