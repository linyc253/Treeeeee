#include <stdio.h>
#include <stdlib.h>
#include "force.h"
#include "parameter.h"
#include "particle.h"

int main(){
    Read_Input_Parameter("Input_Parameter.ini");

    // Some examples about how to use get_*
    DIM = get_int("BasicSetting.DIM", 3); // global variable defined in parameter.c
    int METHOD = get_int("BasicSetting.METHOD", 2); // local variable
    printf("DIM = %d\n", DIM);
    printf("METHOD  = %d\n", METHOD);

    // An example demonstrates how to call a function from force.c
    Force(0);

    // Read Particle file
    Particle* P;
    int npart = Read_Particle_File(&P);
    Write_Particle_File(P, npart, "000.dat");

    free(P);
    return 0;
}