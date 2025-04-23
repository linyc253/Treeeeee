#include <stdio.h>
#include "force.h"
#include "parameter.h"

int Force(int a){
    double THETA = get_double("Tree.THETA", 0.001); // Local variable
    printf("This is from force.c with THETA = %f\n", THETA);
    return 0;
}