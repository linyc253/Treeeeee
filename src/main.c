#include <stdio.h>
#include "force.h"
#include "parameter.h"

int main(){
    Read_Input_Parameter("Input_Parameter.ini");

    DIM = get_int("basic setting.DIM", 3); // global variable defined in parameter.c
    int METHOD = get_int("basic setting.METHOD", 1); // Local variable
    
    printf("DIM = %d\n", DIM);
    printf("METHOD  = %d\n", METHOD);
    Force(0);
    return 0;
}