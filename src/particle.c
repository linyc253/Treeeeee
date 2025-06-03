#include <stdio.h>
#include <stdlib.h>
#include "parameter.h"
#include "particle.h"

int Read_Particle_File(Particle** P, const char* PARTICLE_FILE){
    FILE* ptr = fopen(PARTICLE_FILE, "r");
    if (ptr == NULL) {
        printf("Failed to open %s\n", PARTICLE_FILE);
        exit(EXIT_FAILURE);
    }
    
    int npart;
    int status = fscanf(ptr, "%d", &npart);
    if (status != 1) {
        printf("Failed to read %s (detect EOF)\n", PARTICLE_FILE);
        exit(EXIT_FAILURE);
    }


    *P = (Particle*) malloc(npart * sizeof(Particle));
    Particle* PP = *P;
    for(int i = 0; i < npart; i++) status = fscanf(ptr, "%lf", &PP[i].m);
    for(int i = 0; i < npart; i++) status = fscanf(ptr, "%lf %lf %lf", &PP[i].x[0], &PP[i].x[1], &PP[i].x[2]);
    for(int i = 0; i < npart; i++) status = fscanf(ptr, "%lf %lf %lf", &PP[i].v[0], &PP[i].v[1], &PP[i].v[2]);
    for(int i = 0; i < npart; i++) PP[i].zone = 0;
    int RESTART = get_int("BasicSetting.RESTART", 0);
    if(RESTART != 0){
        double buff[3];
        // Read the force (useless)
        for(int i = 0; i < npart; i++) status = fscanf(ptr, "%lf %lf %lf", &buff[0], &buff[1], &buff[2]);
    }

    // Check status
    if (status != 3) {
        printf("Failed to read %s (detect EOF)\n", PARTICLE_FILE);
        exit(EXIT_FAILURE);
    }
    int buff;
    if (fscanf(ptr, "%d", &buff) != EOF) {
        printf("Failed to read %s (detect extra text)\n", PARTICLE_FILE);
        exit(EXIT_FAILURE);
    }
    fclose(ptr);
    printf("Read %s successfully\n", PARTICLE_FILE);
    return npart;
}

void Write_Particle_File(Particle* P, int npart, const char* filename){
    FILE* ptr = fopen(filename, "w");
    fprintf(ptr, "%d\n", npart);
    for(int i = 0; i < npart; i++) fprintf(ptr, "%lf\n", P[i].m);
    for(int i = 0; i < npart; i++) fprintf(ptr, "%lf %lf %lf\n", P[i].x[0], P[i].x[1], P[i].x[2]);
    for(int i = 0; i < npart; i++) fprintf(ptr, "%lf %lf %lf\n", P[i].v[0], P[i].v[1], P[i].v[2]);
    for(int i = 0; i < npart; i++) fprintf(ptr, "%lf %lf %lf\n", P[i].f[0], P[i].f[1], P[i].f[2]);
    fclose(ptr);
    return;
}

void Initialize_Energy_File(const char* filename){
    FILE* ptr = fopen(filename, "w");
    fprintf(ptr, "Kinetic     Potential     Total_Energy\n");
    fclose(ptr);
    return;
}


void Write_Energy_File(double K, double V, double E, const char* filename){
    FILE* ptr = fopen(filename, "a");
    fprintf(ptr, "%lf %lf %lf\n", K, V, E);
    fclose(ptr);
    return;
}