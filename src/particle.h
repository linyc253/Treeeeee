#ifndef PARTICLE_H_INCLUDED
#define PARTICLE_H_INCLUDED

typedef struct {
    // mass of particle
    double m;
    
    // coordinate of particle
    double x[3];

    // velocity of particle
    double v[3];

    // force on the particle
    double f[3];

    int zone;
} Particle;

int Read_Particle_File(Particle** P, const char* PARTICLE_FILE);
void Write_Particle_File(Particle* P, int npart, const char* filename);
void Write_Energy_File(double K, double V, double E, const char* filename);
void Initialize_Energy_File(const char* filename);
#endif
