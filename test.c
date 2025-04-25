# include <stdio.h>
# include <math.h>

typedef struct {
    double m;
    double x[3];
    double v[3];
    double f[3];
} Particle;


void two_particle_force(Particle* a, Particle* b, double* force);
void total_force(Particle* p, int npart);
void Evolution(Particle* P, int npart, double dt, double t);
int DIM = 3;

static inline void vec3_add_scaled(double* out, const double* a, const double* b, double s) {
    for (int i = 0; i < 3; i++) out[i] = a[i] + s * b[i];
}

int main(){ 
    Particle p1 = {
        .m = 1.0,
        .x = {1.0, 2.0, 1.0},
        .v = {1.0, 2.0, 1.0},
    };
    Particle p2 = {
        .m = 2.0,
        .x = {2.0, 1.0, 2.0},
        .v = {2.0, 1.0, 2.0},
    };
    
    double dt = 0.1;
    double t = 0.0;
    Particle P[2];
    P[0] = p1;
    P[1] = p2;    
    for (int i = 0; i < 10; i++) {
        t += dt;
        printf("Particle %2d: x = (%f, %f, %f)\n", i, P[0].x[0], P[0].x[1], P[0].x[2]);
        Evolution(P, 2, dt, t);
    }


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
    double epsilon = 1e-12;
    if (DIM == 2 || DIM == 3) {
        F_mag = -(m_a * m_b) / (DIM == 2 ? r + epsilon : r*r + epsilon*epsilon);
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

void Evolution(Particle* P, int npart, double dt, double t) {
    if (t == 0) total_force(P, npart);
    // (a) Drift by 0.5*dt for all particles
    for (int i = 0; i < npart; i++) vec3_add_scaled(P[i].x, P[i].x, P[i].v, 0.5 * dt);
    // (b) Kick: update velocity using force
    total_force(P, npart);
    for (int i = 0; i < npart; i++) vec3_add_scaled(P[i].v, P[i].v, P[i].f, dt / P[i].m);
    // (c) Drift again by 0.5*dt
    for (int i = 0; i < npart; i++) vec3_add_scaled(P[i].x, P[i].x, P[i].v, 0.5 * dt);
}