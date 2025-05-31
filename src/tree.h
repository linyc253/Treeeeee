#ifndef TREE_H_INCLUDED
#define TREE_H_INCLUDED
#include "parameter.h"
#include "particle.h"

typedef struct node{
    struct node* children[8]; // Array to be allocate later
    struct node* parent;
    int npart;
    long long cost;
    int i;
    double D; // cell size

    double x[3];
    double c[3]; // cell center
    double m;

    // for quadrupole 
    double p2_x[3][3];

} Node;

typedef struct{
    Node* root;
    double box_min[3];
    double box_max[3];
} Tree;

// This should be compatible with cuda data type: float4 (a.x, a.y, a.z, a.w)
typedef struct{
    float x[3];
    float m;
} Coord4;

// This should be compatible with cuda data type: float3 (a.x, a.y, a.z)
typedef struct{
    float x[3];
} Coord3;

void total_force_tree(Particle* P, int npart);

#endif
