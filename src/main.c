#include <stdio.h>
#include <stdlib.h>
#include "evolution.h"
#include "parameter.h"
#include "particle.h"
#include <sys/time.h>
#include <math.h>

int main(){
    Read_Input_Parameter("Input_Parameter.ini");

    // Some examples about how to use get_*
    DIM = get_int("BasicSetting.DIM", 3); // global variable defined in parameter.c
    printf("Perform calculation in %d dimension\n", DIM);

    // Read Particle file
    Particle* P;
    int npart = Read_Particle_File(&P);

    // Main Calculation
    double T_TOT = get_double("BasicSetting.T_TOT", 0.0);
    double DT = get_double("BasicSetting.DT", 0.1);
    double t = 0.0;
    int STEP_PER_OUT = get_int("BasicSetting.STEP_PER_OUT", -1);
    double TIME_PER_OUT = get_double("BasicSetting.TIME_PER_OUT", -0.1);
    int step = 0;
    int time_step = 0;
    while(t < T_TOT){
        struct timeval t0, t1;
        gettimeofday(&t0, 0);
        step++;

        // Update P[:].x & P[:].v & P[:].f
        double dt = Evolution(P, npart, Min(DT, T_TOT - t));
        //printf("time step dt: %.10f\n", dt);
        t = Min(t + dt + 1e-15, T_TOT);

        gettimeofday(&t1, 0);
        // Print info
#ifdef DEBUG
        printf("Step %5d: x = (%lf, %lf, %lf)\n", step, P[0].x[0], P[0].x[1], P[0].x[2]);
        printf("Step %5d: f = (%lf, %lf, %lf)\n", step, P[0].f[0], P[0].f[1], P[0].f[2]);
#endif
        printf("Step %5d: t = %lf,  dt = %lf  timeElapsed: %lu ms\n", step, t, dt, (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_usec - t0.tv_usec) / 1000);


        // Write files to 00001.dat 00002.dat ...
        if (((STEP_PER_OUT == -1) && (t / TIME_PER_OUT >= (double)time_step + 1.0)) || 
            ((STEP_PER_OUT != -1) && (step % STEP_PER_OUT == 0)) ) {
            time_step++;
            char filename[32];
            sprintf(filename, "%05d.dat", time_step);  
            Write_Particle_File(P, npart, filename);
            printf("Data written to %s (t = %.3f)\n", filename, t);
        }
        

    }

    Write_Particle_File(P, npart, "Final.dat");
    printf("Data written to Final.dat\n");
    
    free(P);
    return 0;
}