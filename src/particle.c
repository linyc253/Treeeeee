#include <stdio.h>
#include <stdlib.h>
#include "parameter.h"
#include "particle.h"

int Read_Particle_File(Particle** P){
    const char* PARTICLE_FILE = get_string("BasicSetting.PARTICLE_FILE", "Initial.dat");
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


    *P = malloc(npart * sizeof(Particle));
    Particle* PP = *P;
    for(int i = 0; i < npart; i++) status = fscanf(ptr, "%lf", &PP[i].m);
    for(int i = 0; i < npart; i++) status = fscanf(ptr, "%lf %lf %lf", &PP[i].x[0], &PP[i].x[1], &PP[i].x[2]);
    for(int i = 0; i < npart; i++) status = fscanf(ptr, "%lf %lf %lf", &PP[i].v[0], &PP[i].v[1], &PP[i].v[2]);

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
    fclose(ptr);
    return;
}