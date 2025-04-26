#ifndef TREE_H_INCLUDED
#define TREE_H_INCLUDED
#include "parameter.h"

typedef struct node{
    struct node** children; // Array to be allocate later
    struct node* parent;
    int npart;
    int i;
    double D;

    double x[3];
    double m;
} Node;

typedef struct{
    Node* root;
    double box_min[3];
    double box_max[3];
} Tree;

void total_force_tree(Particle* P, int npart);

#endif
