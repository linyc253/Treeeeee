#ifndef PARTICLE_H_INCLUDED
#define PARTICLE_H_INCLUDED

typedef struct {
    double m;
    double x[3];
    double v[3];
} Particle;

int Read_Particle_File(Particle** P);
void Write_Particle_File(Particle* P, int npart, const char* filename);
#endif
