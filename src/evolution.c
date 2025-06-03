#include <stdio.h>
#include <math.h>
#include "evolution.h"
#include "parameter.h"
#include "particle.h"
#include "force.h"
#include <stdlib.h>
#include <string.h>
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

void force(Particle* P, int npart, int force_method, double epsilon) {
    if (force_method == 1)
        total_force(P, npart, epsilon);
    else if (force_method == 2)
        total_force_tree(P, npart);
    else {
        fprintf(stderr, "Unknown force method: %d\n", force_method);
        exit(1);
    }
}

double integrator(Particle* P, int npart, double dt_max, int integrator_method, int force_method, double epsilon, double eta) {
    double dt;
    switch (integrator_method) {
        case 1:  // DKD
            dt = compute_dt(P, npart, dt_max, eta, epsilon);
            // drift by dt / 2
            for (int i = 0; i < npart; i++){
                for (int j = 0; j < DIM; j++){
                    P[i].x[j] += P[i].v[j] * (dt / 2.0);
                }
            }
        
            // Kick by dt
            force(P, npart, force_method, epsilon);
            for (int i = 0; i < npart; i++) {
                for (int j = 0; j < DIM; j++) {
                    P[i].v[j] += P[i].f[j] * (dt) / P[i].m;
                }
            }

            // Drift by dt / 2
            for (int i = 0; i < npart; i++) {
                for (int j = 0; j < DIM; j++) {
                    P[i].x[j] += P[i].v[j] * (dt / 2.0);
                }
            }
            return dt;
            break;

        case 2:  // KDK
            dt = compute_dt(P, npart, dt_max, eta, epsilon);
            // Kick by dt / 2
            force(P, npart, force_method, epsilon);
            for (int i = 0; i < npart; i++) {
                for (int j = 0; j < DIM; j++) {
                    P[i].v[j] += P[i].f[j] * (dt / 2.0) / P[i].m;
                }
            }

            // drift by dt
            for (int i = 0; i < npart; i++) {
                for (int j = 0; j < DIM; j++) {
                    P[i].x[j] += P[i].v[j] * (dt);
                }
            }

            // Kick by dt2 / 2
            force(P, npart, force_method, epsilon);
            for (int i = 0; i < npart; i++) {
                for (int j = 0; j < DIM; j++) {
                    P[i].v[j] += P[i].f[j] * (dt / 2.0) / P[i].m;
                }
            }
            return dt;
            break;

        case 3: { // RK4
            
            double k1x[DIM], k1v[DIM], k2x[DIM], k2v[DIM], k3x[DIM], k3v[DIM], k4x[DIM], k4v[DIM];
            dt = compute_dt(P, npart, dt_max, eta, epsilon);
            Particle* temp = malloc(sizeof(Particle));

            for (int i = 0; i < npart; i++) {
                
                // Step 1
                memcpy(temp, P + i, sizeof(Particle));
                force(P, npart, force_method, epsilon);
                for (int j = 0; j < DIM; j++) {
                    k1x[j] = P[i].v[j] * dt;
                    k1v[j] = P[i].f[j] / P[i].m * dt;
                    P[i].x[j] = P[i].x[j] + k1x[j] / 2;
                    P[i].v[j] = P[i].v[j] + k1v[j] / 2;
                }

                // Step 2
                force(P, npart, force_method, epsilon);
                for (int j = 0; j < DIM; j++) {
                    k2x[j] = P[i].v[j] * dt;
                    k2v[j] = P[i].f[j] / P[i].m * dt;
                    P[i].x[j] = P[i].x[j] + k2x[j] / 2 - k1x[j] / 2;
                    P[i].v[j] = P[i].v[j] + k2v[j] / 2 - k1v[j] / 2;
                }

                // Step 3
                force(P, npart, force_method, epsilon);
                for (int j = 0; j < DIM; j++) {
                    k3x[j] = P[i].v[j] * dt;
                    k3v[j] = P[i].f[j] / P[i].m * dt;
                    P[i].x[j] = P[i].x[j] + k3x[j] - k2x[j] / 2;
                    P[i].v[j] = P[i].v[j] + k3v[j] - k2v[j] / 2;
                }

                // Step 4
                force(P, npart, force_method, epsilon);
                for (int j = 0; j < DIM; j++) {
                    k4x[j] = P[i].v[j] * dt;
                    k4v[j] = P[i].f[j] / P[i].m * dt;

                    // Final RK4 update
                    P[i].x[j] = temp->x[j] + dt / 6.0 * (k1x[j] + 2*k2x[j] + 2*k3x[j] + k4x[j]);
                    P[i].v[j] = temp->v[j] + dt / 6.0 * (k1v[j] + 2*k2v[j] + 2*k3v[j] + k4v[j]);
                }
            }

            free(temp);
            return dt;
            break;
        }

        default:
            fprintf(stderr, "Error: Unknown integrator METHOD=%d\n", integrator_method);
            exit(1);
    }
}

double Evolution(Particle* P, int npart, double dt_max) {
    int METHOD = get_int("BasicSetting.METHOD", 2);
    int INTEGRATOR = get_int("BasicSetting.INTEGRATOR", 2);
    double eta = get_double("BasicSetting.ETA", 1.0e100); // 1.0e100 literally disable adaptive time step
    double epsilon = get_double("BasicSetting.EPSILON", 1e-4);

    double dt = integrator(P, npart, dt_max, INTEGRATOR, METHOD, epsilon, eta);
    return dt;
}


// double Evolution(Particle* P, int npart, double dt_max, double dt1) {
//     int METHOD = get_int("BasicSetting.METHOD", 2);
//     double eta = get_double("BasicSetting.ETA", 1.0e100); // 1.0e100 literally disable adaptive time step
//     double epsilon = get_double("BasicSetting.EPSILON", 1e-4);

//     // Drift by dt1 / 2
//     for (int i = 0; i < npart; i++) {
//         for (int j = 0; j < DIM; j++) {
//             P[i].x[j] += P[i].v[j] * (dt1 / 2.0);
//         }
//     }

//     // Calculate force and dt2
//     if(METHOD == 1) total_force(P, npart, epsilon);
//     else if(METHOD == 2) total_force_tree(P, npart);
//     double dt2 = compute_dt(P, npart, dt_max, eta, epsilon);

//     // Kick by (dt1 + dt2) / 2    (KDK scheme)
//     for (int i = 0; i < npart; i++) {
//         for (int j = 0; j < DIM; j++) {
//             P[i].v[j] += P[i].f[j] * ((dt1 + dt2) / 2.0) / P[i].m;
//         }
//     }

//     // Drift by dt2 / 2
//     for (int i = 0; i < npart; i++) {
//         for (int j = 0; j < DIM; j++) {
//             P[i].x[j] += P[i].v[j] * (dt2 / 2.0);
//         }
//     }

//     return dt2;
// }