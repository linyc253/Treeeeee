#include <stdio.h>
#include "force.h"

int main(){
    printf("This is from main.c\n");
    int a;
    scanf("%d", &a);
    Force(a);
    return 0;
}