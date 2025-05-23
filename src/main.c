#include <stdio.h>
#include <stdlib.h>
#include "evolution.h"
#include "parameter.h"
#include "particle.h"
#include <sys/time.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

int main(){
    Read_Input_Parameter("Input_Parameter.ini");

    // Create data directory (if not exist)
    struct stat st = {0};
    const char* OUTDIR = get_string("BasicSetting.OUTDIR", ".");
    if (stat(OUTDIR, &st) == -1) {
        mkdir(OUTDIR, 0700);
    }

    // Set up dimension
    DIM = get_int("BasicSetting.DIM", 3); // global variable defined in parameter.c
    
    printf("Perform calculation in %d dimension\n", DIM);

    // Read Particle file
    Particle* P;
    int RESTART = get_int("BasicSetting.RESTART", 0);
    int npart;
    if(RESTART == 0){
        const char* PARTICLE_FILE = get_string("BasicSetting.PARTICLE_FILE", "Initial.dat");
        npart = Read_Particle_File(&P, PARTICLE_FILE);
    }
    else{
        char PARTICLE_FILE[32];
        sprintf(PARTICLE_FILE, "%s/%05d.dat", OUTDIR, RESTART);
        printf("===== Restart from %s =====\n", PARTICLE_FILE);
        npart = Read_Particle_File(&P, PARTICLE_FILE);
    }
    

    // Main Calculation
    double T_TOT = get_double("BasicSetting.T_TOT", 0.0);
    double DT = get_double("BasicSetting.DT", 0.1);
    double t = 0.0;
    int STEP_PER_OUT = get_int("BasicSetting.STEP_PER_OUT", -1);
    double TIME_PER_OUT = get_double("BasicSetting.TIME_PER_OUT", -0.1);
    int step = 0;
    int time_step = 0;
    double dt = 0.0;
    while(t < T_TOT){
        struct timeval t0, t1;
        gettimeofday(&t0, 0);
        step++;

        // Update P[:].x & P[:].v & P[:].f
        dt = Evolution(P, npart, Min(DT, T_TOT - t), dt);
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
            sprintf(filename, "%s/%05d.dat", OUTDIR, time_step + RESTART);  
            Write_Particle_File(P, npart, filename);
            printf("Data written to %s (t = %.3f)\n", filename, t);
        }
        

    }

    Write_Particle_File(P, npart, "Final.dat");
    printf("Data written to Final.dat\n");
    
    free(P);
    return 0;
}