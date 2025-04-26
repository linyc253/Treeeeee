#include <stdio.h>
#include <stdlib.h>
#include "evolution.h"
#include "parameter.h"
#include "particle.h"

int main(){
    Read_Input_Parameter("Input_Parameter.ini");

    // Some examples about how to use get_*
    DIM = get_int("BasicSetting.DIM", 3); // global variable defined in parameter.c
    printf("Perform calculation in %d dimension\n", DIM);

    // Read Particle file
    Particle* P;
    int npart = Read_Particle_File(&P);
    Write_Particle_File(P, npart, "00000.dat");

    // Main Calculation
    double T_TOT = get_double("BasicSetting.T_TOT", 0.0);
    double DT = get_double("BasicSetting.DT", 0.1);
    double t = 0.0;
    int STEP_PER_OUT = get_int("BasicSetting.STEP_PER_OUT", 10);
    int step = 0;
    while(t < T_TOT){
        step++;
        // Update P[:].x & P[:].v & P[:].f
        double dt = Min(DT, T_TOT - t);
        Evolution(P, npart, dt);
        printf("Step %6d: x = (%lf, %lf, %lf)\n", step, P[0].x[0], P[0].x[1], P[0].x[2]);
        
        // Write files to 00001.dat 00002.dat ...
        if(step % STEP_PER_OUT == 0){
            char filename[16];
            sprintf(filename,"%05d.dat", step / STEP_PER_OUT);
            Write_Particle_File(P, npart, filename);
        }
        t = Min(t + DT, T_TOT);
        // printf("step %d:   t = %lf  ", step, t);
    }
    

    free(P);
    return 0;
}